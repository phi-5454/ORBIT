import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

import torch_modules as tm
import wandb


# Model evaluation
# TODO: Move to separate file
class PhysicsEvaluator:
    def __init__(self, feature_names=["Eta", "Phi", "pT (log)"]):
        self.feature_names = feature_names

    def evaluate_reconstruction(self, x, x_hat, mask):
        """
        Takes raw tensors, filters out padding, and returns a dict of metrics and figures.
        """
        results = {}

        # 1. Apply the mask! Extract only the REAL particles.
        # Shape goes from [Batch, 256, 3] -> [Total_Real_Particles, 3]
        x_real = x[mask]
        x_hat_real = x_hat[mask]

        # 2. Calculate true physical MSE per feature
        mse_per_feature = F.mse_loss(x_hat_real, x_real, reduction="none").mean(dim=0)

        # Log the numeric metrics
        for i, name in enumerate(self.feature_names):
            results[f"metrics/mse_{name.replace(' ', '_')}"] = mse_per_feature[i].item()

        # Convert to NumPy for plotting
        x_np = x_real.cpu().numpy()
        x_hat_np = x_hat_real.cpu().numpy()

        # ==========================================
        # FIGURE 1: Kinematic Features
        # ==========================================
        fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig1.suptitle("FSQ-VAE: Original vs. Reconstructed Features", fontsize=16)

        for i in range(3):
            # Plot Original
            sns.histplot(
                x_np[:, i],
                bins=50,
                color="blue",
                alpha=0.5,
                label="Original",
                kde=False,
                stat="density",
                ax=axes[i],
            )
            # Plot Reconstructed
            sns.histplot(
                x_hat_np[:, i],
                bins=50,
                color="orange",
                alpha=0.5,
                label="Reconstructed",
                kde=False,
                stat="density",
                ax=axes[i],
            )

            axes[i].set_title(
                f"{self.feature_names[i]} (MSE: {mse_per_feature[i]:.4f})"
            )
            axes[i].legend()

        plt.tight_layout()
        results["plots/kinematics"] = fig1  # Save Figure 1

        # ==========================================
        # FIGURE 2: Energy / Momentum (Mass assuming m=0)
        # ==========================================
        fig2, axs = plt.subplots(1, 2, figsize=(16, 5))

        # Assuming indices: 0 = Eta, 1 = Phi, 2 = log(pT)
        # We MUST use np.exp() on column 2 to get physical pT before calculating E = pT * cosh(eta)
        pt_orig = np.exp(x_np[:, 2])
        pt_reco = np.exp(x_hat_np[:, 2])

        energy_orig = pt_orig * np.cosh(x_np[:, 0])
        energy_reco = pt_reco * np.cosh(x_hat_np[:, 0])

        min_val = max(min(energy_orig.min(), energy_reco.min()), 1e-8)
        max_val = max(energy_orig.max(), energy_reco.max())
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)

        # Plot 1: Energy Distribution
        axs[0].set_title(
            "FSQ-VAE: Original vs. Reconstructed Energy (m=0)", fontsize=14
        )
        sns.histplot(
            energy_orig,
            bins=log_bins,
            color="blue",
            alpha=0.5,
            label="Original",
            kde=False,
            stat="density",
            ax=axs[0],
        )
        sns.histplot(
            energy_reco,
            bins=log_bins,
            color="orange",
            alpha=0.5,
            label="Reconstructed",
            kde=False,
            stat="density",
            ax=axs[0],
        )
        axs[0].set_xscale("log")
        axs[0].legend()

        # Plot 2: Residuals (Reco - Orig)
        # We calculate MSE of the physical energy to display in the title
        energy_mse = np.mean((energy_reco - energy_orig) ** 2)

        axs[1].set_title(
            f"Residuals: E_reco - E_original (MSE: {energy_mse:.2f})", fontsize=14
        )
        sns.histplot(
            energy_reco - energy_orig,
            bins=50,
            color="green",
            alpha=0.5,
            kde=False,
            stat="density",
            ax=axs[1],
        )

        plt.tight_layout()
        results["plots/energy_residuals"] = fig2  # Save Figure 2

        return results


class PHA_FSQ_VAE(L.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.save_hyperparameters()

        dim_mu = len(model_cfg["fsq_mu_levels"])
        dim_alpha = len(model_cfg["fsq_alpha_levels"])

        dim_quantized = dim_mu + dim_alpha
        in_ch = model_cfg["input_dim"]
        hidden_dim = model_cfg["hidden_dim"]
        latent_nodes = model_cfg["latent_nodes"]

        self.encoder = tm.ParticleSetEncoder(
            in_channels=in_ch,
            hidden_dim=hidden_dim,
            latent_nodes=latent_nodes,
            out_channels=dim_quantized,
        )
        self.phi = tm.Phi(dim_in=dim_quantized, dim_alpha=dim_alpha, dim_mu=dim_mu)

        self.quantizer_mu = tm.FSQ(levels=model_cfg["fsq_mu_levels"])
        self.quantizer_alpha = tm.FSQ(levels=model_cfg["fsq_alpha_levels"])

        self.psi = tm.Psi(dim_mu=dim_mu, dim_alpha=dim_alpha, dim_out=dim_quantized)
        self.decoder = tm.ParticleSetDecoder(
            latent_channels=dim_quantized,
            hidden_dim=hidden_dim,
            out_nodes=model_cfg["window_particles"],
            out_channels=in_ch,
        )

        self.evaluator = PhysicsEvaluator()

    def configure_optimizers(self):
        # Tip: AdamW (with weight decay) is vastly superior to Adam for Transformers
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.model_cfg["lr"],
            weight_decay=self.hparams.model_cfg["weight_decay"],
        )

        # Warmup Phase:
        # Start at 1% of the base LR (0.01 * 1e-3 = 1e-5)
        # Linearly increase to 100% of the base LR over the first 1,000 batches
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000
        )

        # Lightning requires a specific dictionary format if you want the
        # scheduler to update every batch ("step") instead of every epoch.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step",  # CRITICAL: Updates every batch
                "frequency": 1,
            },
        }

    def compute_losses(self, x, mask, beta=0.25):
        x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)

        mask_3d = mask.unsqueeze(-1).expand_as(x)

        loss_abs_full = F.l1_loss(x_hat, x, reduction="none")
        loss_abs = (loss_abs_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

        loss_l2_full = F.mse_loss(x_hat, x, reduction="none")
        loss_l2 = (loss_l2_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

        loss_commitment = F.mse_loss(z_mu, z_hat_mu.detach())
        loss_amplitude = F.mse_loss(z_alpha, z_hat_alpha.detach())

        loss_pha = loss_abs + (beta * loss_commitment) + loss_amplitude

        return loss_pha, loss_l2, loss_abs, loss_commitment, loss_amplitude, x_hat

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

    def _evaluate_and_log(self, sample_tuple, prefix="val"):
        """Handles evaluator routing for both validation and testing."""
        if sample_tuple is None:
            return

        x, x_hat, mask = sample_tuple
        results = self.evaluator.evaluate_reconstruction(x, x_hat, mask)

        for key, value in results.items():
            # 1. Route Scalars
            if isinstance(value, (int, float)):
                self.log(f"{prefix}_metrics/{key}", value, sync_dist=True)

            # 2. Route Figures
            elif hasattr(value, "savefig"):
                fig = value

                if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                    self.logger.experiment.log(
                        {f"{prefix}_plots/{key}": wandb.Image(fig)},
                        step=self.global_step,
                    )

                elif isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
                    self.logger.experiment.add_figure(
                        f"{prefix}_plots/{key}", fig, global_step=self.global_step
                    )
                else:
                    os.makedirs("local_debug_plots", exist_ok=True)
                    fig.savefig(
                        f"local_debug_plots/{prefix}_{key.replace('/', '_')}_step_{self.global_step}.png"
                    )

                plt.close(fig)

    def training_step(self, batch, batch_idx):
        # WELD: Unpack the yielded tuple
        x, mask = batch

        # Forward Pass
        # x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)
        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, _ = self.compute_losses(
            x, mask, beta=0.25
        )

        # Logging
        self.log_dict(
            {
                "train_loss": loss_pha,
                "mse_loss": loss_l2,
                "l1_recon": loss_abs,
                "commit_mu": loss_commit,
                "commit_alpha": loss_amp,
            },
            prog_bar=True,
        )

        return loss_pha

    def validation_step(self, batch, batch_idx):
        # TODO: Code duplication between training and validation.
        x, mask = batch

        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, x_hat = self.compute_losses(
            x, mask, beta=self.hparams.model_cfg["commit_beta"]
        )

        # Logging
        self.log_dict(
            {
                "val_loss": loss_pha,
                "val_mse_loss": loss_l2,
                "val_l1_recon": loss_abs,
                "val_commit_mu": loss_commit,
                "val_commit_alpha": loss_amp,
            },
            prog_bar=True,
            sync_dist=True,
        )

        # if batch_idx == 0:
        # self.val_sample = (x.detach(), x_hat.detach(), mask.detach())
        self.val_sample = (x.detach(), x_hat.detach(), mask.detach())

    def test_step(self, batch, batch_idx):
        # TODO: Code duplication between training and validation.
        x, mask = batch

        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, x_hat = self.compute_losses(
            x, mask, beta=0.25
        )

        # Logging
        self.log_dict(
            {
                "val_loss": loss_pha,
                "val_mse_loss": loss_l2,
                "val_l1_recon": loss_abs,
                "val_commit_mu": loss_commit,
                "val_commit_alpha": loss_amp,
            },
            prog_bar=True,
            sync_dist=True,
        )

        # if batch_idx == 0:
        # self.test_sample = (x.detach(), x_hat.detach(), mask.detach())
        self.test_sample = (x.detach(), x_hat.detach(), mask.detach())

    # Log the validation metrics and plots
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        # Pass the test sample and tell the router to use the "test" prefix
        self._evaluate_and_log(getattr(self, "val_sample", None), prefix="val")

    def on_test_epoch_end(self):
        # Pass the test sample and tell the router to use the "test" prefix
        self._evaluate_and_log(getattr(self, "test_sample", None), prefix="test")
