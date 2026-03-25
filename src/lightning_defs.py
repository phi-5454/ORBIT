import os

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
        self._evaluate_and_log(getattr(self, "jet_reco", None), prefix="val")

    def on_test_epoch_end(self):
        # Pass the test sample and tell the router to use the "test" prefix
        self._evaluate_and_log(getattr(self, "test_sample", None), prefix="test")
