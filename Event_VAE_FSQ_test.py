#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pyarrow.dataset as ds
from torch.utils.data import IterableDataset, DataLoader
import lightning as L
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import pathlib

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# In[2]:


# TODO: use Event_Number to group the particles
feature_cols = ["L1T_PFPart_Eta", "L1T_PFPart_Phi", "L1T_PFPart_PT"]
group_col = "Event_Number"


# In[3]:


# Data transformer: pre- and post-processing

import numpy as np
import torch

class PreprocessTranformer:
    def __init__(self, log_column_name, feature_names, epsilon=1e-8):
        self.col_name = log_column_name
        self.epsilon = epsilon
        # Find the integer index of the transformed column for tensor operations later
        self.col_idx = feature_names.index(log_column_name) if log_column_name else None

    def forward_dataframe(self, df):
        """Applies the forward transform to the Pandas DataFrame before training."""
        if self.col_name and self.col_name in df.columns:
            df[self.col_name] = np.log(df[self.col_name] + self.epsilon)
        return df

    def inverse_tensor(self, tensor):
        """Applies the inverse transform to the PyTorch prediction tensor."""
        if self.col_idx is not None:
            # Create a clone to avoid in-place modification issues during backprop
            tensor_inv = tensor.clone() 
            tensor_inv[:, self.col_idx] = torch.exp(tensor[:, self.col_idx]) - self.epsilon
            return tensor_inv
        return tensor


# In[4]:


batch_size = 256

class ParquetFeatureDataset(IterableDataset):
    def __init__(self, parquet_dir, features, max_particles=256, batch_size=32):
        self.dataset = ds.dataset(parquet_dir, format="parquet")
        self.features = features
        self.max_particles = max_particles
        self.batch_size = batch_size

    def __iter__(self):
        # Read only the 3 physical features
        batches = self.dataset.to_batches(columns=self.features, batch_size=self.batch_size)

        for batch in batches:
            # Convert to Pandas. Each cell now contains a numpy array of particles.
            df = batch.to_pandas()

            event_tensors = []

            # Zip the columns. This iterates row-by-row (event-by-event)
            for eta_arr, phi_arr, pt_arr in zip(df[self.features[0]], df[self.features[1]], df[self.features[2]]):

                # Skip empty events (e.g., zero particles passed the trigger)
                if len(eta_arr) == 0:
                    continue

                # Stack the 1D arrays into a [N, 3] matrix for this specific event
                coords = np.column_stack([eta_arr, phi_arr, pt_arr]).astype(np.float32)

                # Apply Log transform safely directly to the PT column (Index 2)
                coords[:, 2] = np.log(coords[:, 2] + 1e-8)

                # Enforce the maximum particles limit
                coords = coords[:self.max_particles]
                event_tensors.append(torch.tensor(coords))

            if not event_tensors:
                continue

            # Pad the variable-length events with 0.0 to create a square batch tensor
            padded_events = pad_sequence(event_tensors, batch_first=True, padding_value=0.0)

            # Force shape to [Batch, 256, 3] in case the largest event in this specific batch was < 256
            pad_len = self.max_particles - padded_events.shape[1]
            if pad_len > 0:
                padded_events = F.pad(padded_events, (0, 0, 0, pad_len), value=0.0)

            # Create Mask: True for REAL particles, False for PADDING
            mask = padded_events[:, :, 2] != 0.0 

            yield padded_events, mask

class ParquetDataModule(L.LightningDataModule):
    def __init__(self, parquet_dir, features=feature_cols):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.features = features

    def train_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dir, self.features)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn 
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)


# In[5]:


class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))

    def forward(self, z):
        z_bound = torch.tanh(z)
        half_l = (self.levels - 1) / 2
        z_scaled = z_bound * half_l
        z_rounded = torch.round(z_scaled)

        # We skip the STE here and do it in the main LightningModule forward pass
        # for cleaner loss tracking.
        return z_rounded / half_l


# In[6]:


class Phi(nn.Module):
    def __init__(self, dim_in, dim_alpha: int, dim_mu: int):
        super().__init__()
        self.ff_alpha = nn.Sequential(
            nn.Linear(dim_in, 2*dim_in),
            nn.GELU(),
            nn.Linear(2*dim_in , dim_alpha)
        )
        self.ff_mu = nn.Sequential(
            nn.Linear(dim_in, 2*dim_in),
            nn.GELU(),
            nn.Linear(2*dim_in, dim_mu)
        )

    def forward(self, z):
        # WELD: Actually pass the data through the layers
        z_alpha = self.ff_alpha(z)
        z_mu = self.ff_mu(z)
        return z_mu, z_alpha


# In[7]:


class Psi(nn.Module):
    def __init__(self, dim_mu: int, dim_alpha: int, dim_out):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim_mu + dim_alpha, (dim_mu + dim_alpha) * 2),
            nn.GELU(),
            nn.Linear((dim_mu + dim_alpha) * 2 , dim_out)
        )

    def forward(self, z_mu, z_alpha):
        # WELD: Replaced NumPy concat with PyTorch cat
        z = torch.cat([z_mu, z_alpha], dim=-1)
        z = self.ff(z)
        return z


# In[8]:


class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y, key_padding_mask=None):
        attn_out, _ = self.mha(query=x, key=y, value=y, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# In[9]:


class ParticleSetEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, latent_nodes=32, out_channels=9, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.self_attn_layers = nn.ModuleList([MAB(hidden_dim, num_heads) for _ in range(num_layers)])

        self.latent_queries = nn.Parameter(torch.randn(1, latent_nodes, hidden_dim))
        self.cross_attn_pool = MAB(hidden_dim, num_heads)
        self.to_latent = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, mask=None):
        x = self.embed(x)
        # PyTorch expects True for ignored padding. Our mask is True for real data.
        pad_mask = ~mask if mask is not None else None

        for layer in self.self_attn_layers:
            x = layer(x, x, key_padding_mask=pad_mask)

        q = self.latent_queries.expand(x.size(0), -1, -1)
        latent = self.cross_attn_pool(q, x, key_padding_mask=pad_mask)
        return self.to_latent(latent)


# In[10]:


class ParticleSetDecoder(nn.Module):
    def __init__(self, latent_channels=9, hidden_dim=64, out_nodes=256, out_channels=3, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(latent_channels, hidden_dim)
        self.out_queries = nn.Parameter(torch.randn(1, out_nodes, hidden_dim))
        self.cross_attn_expand = MAB(hidden_dim, num_heads)
        self.self_attn_layers = nn.ModuleList([MAB(hidden_dim, num_heads) for _ in range(num_layers)])
        self.to_physical = nn.Linear(hidden_dim, out_channels)

    def forward(self, z):
        z = self.embed(z)
        q = self.out_queries.expand(z.size(0), -1, -1)
        x = self.cross_attn_expand(q, z) 

        for layer in self.self_attn_layers:
            x = layer(x, x)

        return self.to_physical(x)


# In[11]:


from torch.optim.lr_scheduler import LinearLR

class PHA_FSQ_VAE(L.LightningModule):
    def __init__(self, input_dim=3, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ParticleSetEncoder(in_channels=3, hidden_dim=64, latent_nodes=32, out_channels=9)
        self.phi = Phi(dim_in=9, dim_alpha=1, dim_mu=8)

        self.quantizer_mu = FSQ(levels=[5,4,4,3,3,3,2,2])
        self.quantizer_alpha = FSQ(levels=[1024])

        self.psi = Psi(dim_mu=8, dim_alpha=1, dim_out=9)
        self.decoder = ParticleSetDecoder(latent_channels=9, hidden_dim=64, out_nodes=256, out_channels=3)

    def configure_optimizers(self):
        # Tip: AdamW (with weight decay) is vastly superior to Adam for Transformers
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

        # Warmup Phase: 
        # Start at 1% of the base LR (0.01 * 1e-3 = 1e-5)
        # Linearly increase to 100% of the base LR over the first 1,000 batches
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=1000
        )

        # Lightning requires a specific dictionary format if you want the 
        # scheduler to update every batch ("step") instead of every epoch.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step", # CRITICAL: Updates every batch
                "frequency": 1
            }
        }

    # WELD: Added the missing forward method orchestrating the split latent space
    def forward(self, x, mask):
        # 1. Encode
        z_encoded = self.encoder(x, mask)

        # 2. Split
        z_mu, z_alpha = self.phi(z_encoded)

        # 3. Quantize
        z_hat_mu = self.quantizer_mu(z_mu)
        z_hat_alpha = self.quantizer_alpha(z_alpha)

        # 4. Straight Through Estimator (STE)
        z_ste_mu = z_mu + (z_hat_mu - z_mu).detach()
        z_ste_alpha = z_alpha + (z_hat_alpha - z_alpha).detach()

        # 5. Merge and Decode
        z_decoded = self.psi(z_ste_mu, z_ste_alpha)
        x_hat = self.decoder(z_decoded)

        return x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha

    def training_step(self, batch, batch_idx):
        # WELD: Unpack the yielded tuple
        x, mask = batch 

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"CRITICAL ERROR: Bad data reached the GPU at step {self.global_step}")

        # Forward Pass
        x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)

        beta = 0.25

        # --- Masked Physics Losses ---
        # Expand 2D mask [B, 256] to 3D [B, 256, 3] to match coordinates
        mask_3d = mask.unsqueeze(-1).expand_as(x)

        loss_abs_full = F.l1_loss(x_hat, x, reduction='none')
        loss_abs = (loss_abs_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

        loss_l2_full = F.mse_loss(x_hat, x, reduction='none')
        loss_l2 = (loss_l2_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

        # --- Unmasked Latent Losses ---
        # The latent space [B, 32, X] is dense (no padding), so no mask is needed here
        loss_commitment = F.mse_loss(z_mu, z_hat_mu.detach())
        loss_amplitude = F.mse_loss(z_alpha, z_hat_alpha.detach())

        # Total Loss formulation
        loss_pha = loss_abs + (beta * loss_commitment) + loss_amplitude 

        # Logging
        self.log_dict({
            "train_loss": loss_pha,
            "mse_loss": loss_l2,
            "l1_recon": loss_abs,
            "commit_mu": loss_commitment,
            "commit_alpha": loss_amplitude
        }, prog_bar=True)

        return loss_pha



import os
import wandb
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor

# 1. Initialize the callback
lr_monitor = LearningRateMonitor(logging_interval='step')

# ==========================================
# 1. ENVIRONMENT & WANDB SETUP
# ==========================================
# Loads the key, project, and entity into the environment
#load_dotenv(dotenv_path="/eos/user/y/yelberke/wandb_api_key.env", override=True) 
load_dotenv(dotenv_path="../wandb_api_key.env", override=True) 
print(os.getenv("WANDB_PROJECT"))

# Explicit login (relies on WANDB_API_KEY being in the env)
wandb.login()

# Initialize the logger
wandb_logger = L.pytorch.loggers.WandbLogger(
    project=os.getenv("WANDB_PROJECT"),
    entity=os.getenv("WANDB_ENTITY"),
    #name="whole_event_pha_fsq",
    log_model="all" # Note: If checkpoints become too large, set this to False
    log_graph=True
)

# ==========================================
# 2. DATA & MODEL INITIALIZATION
# ==========================================

# Source - https://stackoverflow.com/a/3925701
# Posted by Corey Goldberg, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-08, License - CC BY-SA 4.0

with open('./pq_files.txt') as f:
    lines = f.read().splitlines()

parquet_path = lines[:3]
datamodule = ParquetDataModule(parquet_path)

# Initialize Model 
# The latent space is now handled internally via the split Phase (mu) and Amplitude (alpha)
model = PHA_FSQ_VAE(input_dim=3, hidden_dim=64, lr=1e-3)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
# Tell the logger to "watch" the model's gradients and weights
# Note: If gradients don't appear in WandB, move this line inside the LightningModule's `on_train_start`
wandb_logger.watch(model, log="all", log_freq=10, log_graph=True)


# Initialize Trainer
trainer = L.Trainer(
    max_epochs=10, 
    accelerator="auto", 
    logger=wandb_logger,
    gradient_clip_val=1.0, # Crucial: Prevents early divergence from log(0) or padding math
    callbacks = [lr_monitor]
)

# Train the model
trainer.fit(model, datamodule=datamodule)

# ==========================================
# 4. CLEANUP
# ==========================================
wandb_logger.experiment.unwatch(model)
wandb.finish() # Forces WandB to sync the final data and close the run cleanly


# In[ ]:


# 1. Set model to evaluation mode
model.eval()

# 2. Extract a large batch of flattened data from the pipeline
dataloader = datamodule.train_dataloader()
x_batch = next(iter(dataloader)) 
print(x_batch)

# 3. Run inference (no gradients needed)
with torch.no_grad():
    x_hat_batch, z_q_batch = model(x_batch)

print(x_hat_batch)

# 4. Calculate standard VAE metrics (e.g., MSE per feature)
mse_per_feature = F.mse_loss(x_hat_batch, x_batch, reduction='none').mean(dim=0)
print(f"MSE per feature: {mse_per_feature.cpu().numpy()}")

# Convert to NumPy for plotting
x_np = x_batch.cpu().numpy()
x_hat_np = x_hat_batch.cpu().numpy()


# In[ ]:


# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("FSQ-VAE: Original vs. Reconstructed Features", fontsize=16)

feature_names = feature_cols

for i in range(3):
    # Plot Original
    sns.histplot(x_np[:, i], bins=50, color="blue", alpha=0.5, 
                 label="Original", kde=False, stat="density", ax=axes[i])

    # Plot Reconstructed
    sns.histplot(x_hat_np[:, i], bins=50, color="orange", alpha=0.5, 
                 label="Reconstructed", kde=False, stat="density", ax=axes[i])

    axes[i].set_title(f"{feature_names[i]} (MSE: {mse_per_feature[i]:.4f})")
    axes[i].legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Set up the figure
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
#fig.suptitle("FSQ-VAE: Original vs. Reconstructed mass", fontsize=16)

feature_names = feature_cols

masses_orig = x_np[:,2] * np.cosh(x_np[:,0])
masses_reco = x_hat_np[:,2] * np.cosh(x_hat_np[:,0])

min_val = max(min(masses_orig.min(), masses_reco.min()), 1e-8) 
max_val = max(masses_orig.max(), masses_reco.max()).max()
log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)

axs[0].set_title("FSQ-VAE: Original vs. Reconstructed mass", fontsize=16)
sns.histplot(masses_orig, bins=log_bins, color="blue", alpha=0.5, 
             label="Original", kde=False, stat="density", ax=axs[0])

    # Plot Reconstructed
sns.histplot(masses_reco, bins=log_bins, color="orange", alpha=0.5, 
             label="Reconstructed", kde=False, stat="density", ax=axs[0])

axs[1].set_title("FSQ-VAE: m_reco - m_original", fontsize=16)
sns.histplot(masses_reco - masses_orig, bins=50, color="blue", alpha=0.5, 
             label="Original", kde=False, stat="density", ax=axs[1])
axs[0].set_xscale('log')


axs[0].set_title(f"mass (assume. m_0 = 0) (MSE: {mse_per_feature[i]:.4f})")
axs[0].legend()

plt.tight_layout()
wandb.run.log({"Evaluation/Log_Binned_Histograms": wandb.Image(fig)})

plt.show()


# In[ ]:


plt.close(fig)
wandb.finish()
os.system(f"python -m wandb sync --sync-all")


# In[ ]:




