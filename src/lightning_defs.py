import os
import math

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

from eval_metrics import PhysicsEvaluator
import torch_modules as tm
import wandb


class PHA_FSQ_VAE(L.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.save_hyperparameters()

        dim_mu = len(model_cfg["fsq_mu_levels"])
        dim_alpha = len(model_cfg["fsq_alpha_levels"])

        codebook_dim = dim_mu + dim_alpha
        in_dim = model_cfg["input_dim"]
        hidden_dim = model_cfg["hidden_dim"]
        num_heads = model_cfg["num_heads"]
        num_enc_dec_layers = model_cfg["num_enc_dec_layers"]
        nf_mlp_expansion_factor = model_cfg["normformer_mlp_expansion_factor"]
        nf_dropout = model_cfg["normformer_dropout"]
        batch_size = model_cfg["batch_size"]
        window_particles = model_cfg["window_particles"]

        self.example_input_array = (torch.randn(batch_size, window_particles, in_dim),
                                    torch.ones(batch_size, window_particles, dtype=torch.bool)

                                    )

        self.input_proj = tm.MLP(in_dim, hidden_dim, [2 * hidden_dim, 2*hidden_dim])
        self.latent_proj = tm.MLP(hidden_dim, codebook_dim, [2 * hidden_dim, 2*hidden_dim])

        self.latent_proj_out = tm.MLP(codebook_dim, hidden_dim, [2 * hidden_dim, 2*hidden_dim])
        self.output_proj = tm.MLP(hidden_dim, in_dim, [2 * hidden_dim, 2*hidden_dim])

        self.encoder = tm.NormformerEncoder(num_layers=num_enc_dec_layers, model_dim=hidden_dim, nhead=num_heads, mlp_expansion_factor=nf_mlp_expansion_factor, dropout=nf_dropout)
        '''
        self.encoder = tm.ParticleSetEncoder(
            in_channels=in_dim,
            hidden_dim=hidden_dim,
            latent_nodes=num_heads,
            out_channels=dim_quantized,
        )
        '''

        # TODO: Rework phi
        self.phi = tm.Phi(dim_in=hidden_dim, dim_alpha=dim_alpha, dim_mu=dim_mu)

        self.quantizer_mu = tm.FSQ(levels=model_cfg["fsq_mu_levels"])
        self.quantizer_alpha = tm.FSQ(levels=model_cfg["fsq_alpha_levels"])

        # TODO: Rework psi
        self.psi = tm.Psi(dim_mu=dim_mu, dim_alpha=dim_alpha, dim_out=hidden_dim)

        '''
        self.decoder = tm.ParticleSetDecoder(
            latent_channels=codebook_dim,
            hidden_dim=hidden_dim,
            out_nodes=model_cfg["window_particles"],
            out_channels=in_dim,
        )
        '''
        self.decoder = tm.NormformerEncoder(num_layers=num_enc_dec_layers, model_dim=hidden_dim, nhead=num_heads, mlp_expansion_factor=nf_mlp_expansion_factor, dropout=nf_dropout)

        self.evaluator = PhysicsEvaluator()

    def forward(self, x, mask):
        # 1. Encode
        x_proj = self.input_proj(x)

        if(not torch.isfinite(x_proj).all().item()):
            print("111_")
        z_encoded = self.encoder(x_proj, mask, use_attention=self.model_cfg["use_attention"])
        if(not torch.isfinite(z_encoded).all().item()):
            print("222_")

        # 2. Split
        z_mu, z_alpha = self.phi(z_encoded)
        if(not torch.isfinite(z_mu).all().item()):
            print("333_")
        if(not torch.isfinite(z_mu).all().item()):
            print("333_")
        if(not torch.isfinite(z_alpha).all().item()):
            print("444_")

        # 3. Quantize
        if(self.model_cfg["skip_quantization"]==True):
            z_hat_mu = self.quantizer_mu(z_mu)
            z_hat_alpha = self.quantizer_alpha(z_alpha)

            z_decoded = self.psi(z_mu, z_alpha)
        else:
            z_hat_mu = self.quantizer_mu(z_mu)
            if(not torch.isfinite(z_hat_mu).all().item()):
                print("eeeee_")
            z_hat_alpha = self.quantizer_alpha(z_alpha)
            if(not torch.isfinite(z_hat_alpha).all().item()):
                print("aaaaa_")

            # 4. Straight Through Estimator (STE)
            # TODO: abstract this away 
            z_ste_mu = z_mu + (z_hat_mu - z_mu).detach()
            if(not torch.isfinite(z_ste_mu).all().item()):
                print("ooooo_")
            z_ste_alpha = z_alpha + (z_hat_alpha - z_alpha).detach()
            if(not torch.isfinite(z_ste_alpha).all().item()):
                print("iiii_")

            # 5. Merge and Decode
            z_decoded = self.psi(z_ste_mu, z_ste_alpha)
            if(not torch.isfinite(z_decoded).all().item()):
                print("uuuu_")

        x_hat_lat = self.decoder(z_decoded, mask, self.model_cfg["use_attention"])
        if(not torch.isfinite(x_hat_lat).all().item()):
            print("kkkkk_")
        x_hat = self.output_proj(x_hat_lat)

        return x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha

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

    def compute_losses(self, x, mask, beta=0.25, phi_idx=1):
            """
            Calculates losses while respecting the periodicity of the phi angle.
            Assuming x shape is [Batch, Particles, Features] and phi is at phi_idx.
            """
            x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)
            if(not torch.isfinite(x_hat).all().item()):
                print(x)
                print(x_hat)
                print("111")
            if(not torch.isfinite(z_mu).all().item()):
                print("222")
            if(not torch.isfinite(z_hat_mu).all().item()):
                print("333")
            if(not torch.isfinite(z_alpha).all().item()):
                print("444")
            if(not torch.isfinite(z_hat_alpha).all().item()):
                print("555")
            if(not torch.isfinite(x).all().item()):
                print("666")
            
            mask_3d = mask.unsqueeze(-1).expand_as(x)

            # 1. Calculate the raw difference
            diff = x_hat - x

            # 2. Wrap the periodic feature phi. Here, rescaled to [-1, 1] 
            # CRITICAL: We use .clone() here before slice assignment to prevent 
            # PyTorch from throwing an "in-place operation" Autograd error!
            diff_wrapped = diff.clone()
            diff_wrapped[..., phi_idx] = (diff[..., phi_idx] + 1) % (2 * 1) - 1
            if(not torch.isfinite(diff_wrapped).all().item()):
                print("AAA")
                print("AAA")
                print("AAA")
                print("AAA")
                print("AAA")
                print("AAA")

            # 3. Calculate the actual feature losses using the wrapped difference
            loss_abs_full = torch.abs(diff_wrapped)
            if(not torch.isfinite(loss_abs_full).all().item()):
                print("BBB")
                print("BBB")
                print("BBB")
                print("BBB")
                print("BBB")
                print("BBB")
            loss_l2_full = diff_wrapped ** 2  # Equivalent to F.mse_loss under the hood

            # 4. Apply the mask and mean (unchanged from your original code)
            loss_abs = (loss_abs_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)
            loss_l2 = (loss_l2_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)
            if(not torch.isfinite(loss_l2).all().item()):
                print("CCC")
                print("CCC")
                print("CCC")
                print("CCC")
                print("CCC")
            if(not torch.isfinite(loss_abs).all().item()):
                print("DDD")
                print("DDD")
                print("DDD")
                print("DDD")
                print("DDD")
                print("DDD")

            # 5. Calculate latent losses (unchanged)
            loss_commitment = F.mse_loss(z_mu, z_hat_mu.detach())
            loss_amplitude = F.mse_loss(z_alpha, z_hat_alpha.detach())

            # 6. Total loss
            loss_pha = loss_abs + (beta * loss_commitment) + loss_amplitude

            return loss_pha, loss_l2, loss_abs, loss_commitment, loss_amplitude, x_hat



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
                        #step=self.global_step,
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

    '''
    def on_fit_start(self) -> None:
        super().on_fit_start()
        """
        Triggered automatically right before the first training epoch.
        This ensures the model is fully initialized and the logger is attached.
        """
        # 1. Verify we actually have a WandB logger attached
        if self.logger is None or not isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            return

    '''

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
        self._evaluate_and_log(getattr(self, "jet_reco", None), prefix="val")

    def on_test_epoch_end(self):

        # Pass the test sample and tell the router to use the "test" prefix
        self._evaluate_and_log(getattr(self, "test_sample", None), prefix="test")
