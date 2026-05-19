from line_profiler import profile
import os
import math

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import numpy as np

from eval_metrics import PhysicsEvaluator
import torch_modules as tm
import wandb


class PHA_FSQ_VAE(L.LightningModule):
    def __init__(self, model_cfg, output_dir):
        super().__init__()
        self.model_cfg = model_cfg
        self.output_dir = output_dir
        self.save_hyperparameters()

        self.total_train_events_seen = 0
        self.test_step_outputs = []
        self.val_correlation_diagnostics = None

        self.dim_mu = len(model_cfg["fsq_mu_levels"])
        self.dim_alpha = len(model_cfg["fsq_alpha_levels"])

        self.dim_mu = len(model_cfg["fsq_mu_levels"])
        self.dim_alpha = len(model_cfg["fsq_alpha_levels"])

        # Calculate total possible codes safely (default to 1 if empty to avoid div by zero)
        self.total_codes_mu = np.prod(model_cfg["fsq_mu_levels"]) if self.dim_mu > 0 else 1
        self.total_codes_alpha = np.prod(model_cfg["fsq_alpha_levels"]) if self.dim_alpha > 0 else 1
        self.total_codes_combined = self.total_codes_mu * self.total_codes_alpha

        # Create persistent sets for Validation
        self.val_used_codes_mu = set()
        self.val_used_codes_alpha = set()
        self.val_used_codes_combined = set()

        # Create persistent sets for Testing
        self.test_used_codes_mu = set()
        self.test_used_codes_alpha = set()
        self.test_used_codes_combined = set()

        codebook_dim = self.dim_mu + self.dim_alpha
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
        self.phi = tm.Phi(dim_in=hidden_dim, dim_alpha=self.dim_alpha, dim_mu=self.dim_mu)

        self.quantizer_mu = tm.FSQ(levels=model_cfg["fsq_mu_levels"])
        self.quantizer_alpha = tm.FSQ(levels=model_cfg["fsq_alpha_levels"])

        # TODO: Rework psi
        self.psi = tm.Psi(dim_mu=self.dim_mu, dim_alpha=self.dim_alpha, dim_out=hidden_dim)

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

    @profile
    def _track_codebook(self, z_hat_mu, z_hat_alpha, mask, prefix="val"):
        """Extracts unique codes from the current batch and adds them to the global epoch sets."""
        z_mu_valid = z_hat_mu[mask]
        z_alpha_valid = z_hat_alpha[mask]

        # Route to the correct sets
        if prefix == "val":
            set_mu, set_alpha, set_comb = self.val_used_codes_mu, self.val_used_codes_alpha, self.val_used_codes_combined
        else:
            set_mu, set_alpha, set_comb = self.test_used_codes_mu, self.test_used_codes_alpha, self.test_used_codes_combined

        # 1. Track Mu
        if self.dim_mu > 0:
            uniq_mu = torch.unique(z_mu_valid, dim=0).detach().cpu().numpy()
            for vec in np.round(uniq_mu, decimals=4):
                set_mu.add(tuple(vec))

        # 2. Track Alpha
        if self.dim_alpha > 0:
            uniq_alpha = torch.unique(z_alpha_valid, dim=0).detach().cpu().numpy()
            for vec in np.round(uniq_alpha, decimals=4):
                set_alpha.add(tuple(vec))

        # 3. Track Combined Space (The true cross-product utilization)
        if self.dim_mu > 0 and self.dim_alpha > 0:
            z_combined = torch.cat([z_mu_valid, z_alpha_valid], dim=-1)
            uniq_combined = torch.unique(z_combined, dim=0).detach().cpu().numpy()
            for vec in np.round(uniq_combined, decimals=4):
                set_comb.add(tuple(vec))

    @profile
    def _log_and_clear_utilization(self, prefix="val"):
        """Calculates utilization percentages, logs them, and safely clears the sets."""
        if prefix == "val":
            set_mu, set_alpha, set_comb = self.val_used_codes_mu, self.val_used_codes_alpha, self.val_used_codes_combined
        else:
            set_mu, set_alpha, set_comb = self.test_used_codes_mu, self.test_used_codes_alpha, self.test_used_codes_combined

        if self.dim_mu > 0:
            self.log(f"{prefix}_metrics/utilization_mu", len(set_mu) / self.total_codes_mu, sync_dist=True)
            self.log(f"{prefix}_metrics/active_codes_mu", float(len(set_mu)), sync_dist=True)
            set_mu.clear()

        if self.dim_alpha > 0:
            self.log(f"{prefix}_metrics/utilization_alpha", len(set_alpha) / self.total_codes_alpha, sync_dist=True)
            self.log(f"{prefix}_metrics/active_codes_alpha", float(len(set_alpha)), sync_dist=True)
            set_alpha.clear()

        if self.dim_mu > 0 and self.dim_alpha > 0:
            self.log(f"{prefix}_metrics/utilization_combined", len(set_comb) / self.total_codes_combined, sync_dist=True)
            self.log(f"{prefix}_metrics/active_codes_combined", float(len(set_comb)), sync_dist=True)
            set_comb.clear()

    @profile
    def forward(self, x, mask):
        # 1. Encode
        x_proj = self.input_proj(x)

        #if(not torch.isfinite(x_proj).all().item()):
            #print("111_")
        z_encoded = self.encoder(x_proj, mask, use_attention=self.model_cfg["use_attention"])

        # 2. Split
        z_mu, z_alpha = self.phi(z_encoded)

        # 3. Quantize
        if(self.model_cfg["skip_quantization"]==True):
            z_hat_mu = self.quantizer_mu(z_mu)
            z_hat_alpha = self.quantizer_alpha(z_alpha)

            z_decoded = self.psi(z_mu, z_alpha)
        else:
            z_hat_mu = self.quantizer_mu(z_mu)
            z_hat_alpha = self.quantizer_alpha(z_alpha)

            # 4. Straight Through Estimator (STE)
            # TODO: abstract this away 
            z_ste_mu = z_mu + (z_hat_mu - z_mu).detach()
            z_ste_alpha = z_alpha + (z_hat_alpha - z_alpha).detach()

            # 5. Merge and Decode
            z_decoded = self.psi(z_ste_mu, z_ste_alpha)

        x_hat_lat = self.decoder(z_decoded, mask, self.model_cfg["use_attention"])
        x_hat = self.output_proj(x_hat_lat)

        return x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha

    @profile
    def forward_with_diagnostics(self, x, mask):
        x_proj = self.input_proj(x)
        z_encoded, encoder_diags = self.encoder(
            x_proj,
            mask,
            use_attention=self.model_cfg["use_attention"],
            return_diagnostics=True,
        )

        z_mu, z_alpha = self.phi(z_encoded)
        z_hat_mu = self.quantizer_mu(z_mu)
        z_hat_alpha = self.quantizer_alpha(z_alpha)

        if self.model_cfg["skip_quantization"] == True:
            z_decoded = self.psi(z_mu, z_alpha)
        else:
            z_ste_mu = z_mu + (z_hat_mu - z_mu).detach()
            z_ste_alpha = z_alpha + (z_hat_alpha - z_alpha).detach()
            z_decoded = self.psi(z_ste_mu, z_ste_alpha)

        x_hat_lat, decoder_diags = self.decoder(
            z_decoded,
            mask,
            self.model_cfg["use_attention"],
            return_diagnostics=True,
        )
        x_hat = self.output_proj(x_hat_lat)

        return x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha, {
            "encoder": encoder_diags,
            "decoder": decoder_diags,
        }

    def _first_valid_indices(self, mask):
        has_valid = mask.any(dim=1)
        indices = mask.float().argmax(dim=1)
        return indices, has_valid

    @profile
    def _attention_diagnostic_stats(self, diagnostics, mask, prefix):
        stats = {}
        figures = {}
        valid = mask.bool()

        for layer in diagnostics:
            if "attn_weights" not in layer:
                continue

            layer_idx = layer["layer_idx"]
            weights = layer["attn_weights"].detach()
            # Shape: [batch, heads, query_particles, key_particles]
            query_mask = valid[:, None, :, None]
            key_mask = valid[:, None, None, :]
            pair_mask = query_mask & key_mask
            denom = pair_mask.sum().clamp(min=1)

            masked_weights = weights.masked_fill(~pair_mask, 0.0)
            diag = torch.diagonal(masked_weights, dim1=-2, dim2=-1)
            diag_mask = valid[:, None, :]
            diag_mass = diag[diag_mask.expand_as(diag)].mean()

            row_sums = masked_weights.sum(dim=-1).clamp(min=1e-12)
            probs = masked_weights / row_sums.unsqueeze(-1)
            entropy = -(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log()).sum(dim=-1)
            n_keys = valid.sum(dim=1).clamp(min=2).float()
            normalized_entropy = entropy / n_keys.log()[:, None, None]
            query_values = normalized_entropy[valid[:, None, :].expand_as(normalized_entropy)]

            max_mass = probs.max(dim=-1).values
            max_values = max_mass[valid[:, None, :].expand_as(max_mass)]

            attn_delta = layer.get("attn_delta")
            ff_delta = layer.get("ff_delta")
            if attn_delta is not None and ff_delta is not None:
                token_mask = valid[:, :, None]
                attn_norm = attn_delta.detach()[token_mask.expand_as(attn_delta)].norm()
                ff_norm = ff_delta.detach()[token_mask.expand_as(ff_delta)].norm()
                stats[f"{prefix}/layer_{layer_idx}_attn_to_ff_delta_norm"] = (
                    attn_norm / ff_norm.clamp(min=1e-12)
                ).detach().cpu()

            stats[f"{prefix}/layer_{layer_idx}_diag_attention_mass"] = diag_mass.detach().cpu()
            stats[f"{prefix}/layer_{layer_idx}_offdiag_attention_mass"] = (1.0 - diag_mass).detach().cpu()
            stats[f"{prefix}/layer_{layer_idx}_max_attention_mass"] = max_values.mean().detach().cpu()
            stats[f"{prefix}/layer_{layer_idx}_normalized_attention_entropy"] = query_values.mean().detach().cpu()
            stats[f"{prefix}/layer_{layer_idx}_valid_pair_count"] = denom.detach().float().cpu()

            if layer_idx in (0, len(diagnostics) - 1):
                first_event = int(valid.any(dim=1).nonzero(as_tuple=False)[0].item()) if valid.any() else 0
                n_valid = int(valid[first_event].sum().item())
                if n_valid > 1:
                    matrix = weights[first_event, :, :n_valid, :n_valid].mean(dim=0).detach().cpu().numpy()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(matrix, vmin=0.0, vmax=max(float(matrix.max()), 1e-6), aspect="auto")
                    ax.set_title(f"{prefix} layer {layer_idx} attention")
                    ax.set_xlabel("key particle")
                    ax.set_ylabel("query particle")
                    fig.colorbar(im, ax=ax)
                    figures[f"{prefix}/layer_{layer_idx}_attention_map"] = fig

        return stats, figures

    @profile
    def _build_context_probes(self, x, mask):
        target_idx, has_valid = self._first_valid_indices(mask)
        batch_idx = torch.arange(x.shape[0], device=x.device)

        self_only_x = torch.zeros_like(x)
        self_only_mask = torch.zeros_like(mask)
        self_only_x[batch_idx, target_idx] = x[batch_idx, target_idx]
        self_only_mask[batch_idx, target_idx] = has_valid

        swapped_x = torch.roll(x, shifts=1, dims=0)
        swapped_mask = torch.roll(mask, shifts=1, dims=0)
        swapped_x[batch_idx, target_idx] = x[batch_idx, target_idx]
        swapped_mask[batch_idx, target_idx] = has_valid

        return target_idx, has_valid, self_only_x, self_only_mask, swapped_x, swapped_mask

    @profile
    def _collect_correlation_diagnostics(self, x, mask):
        max_events = int(self.model_cfg.get("diagnostic_max_events", 8))
        x = x[:max_events]
        mask = mask[:max_events]

        with torch.no_grad():
            x_hat, _, _, _, _, diagnostics = self.forward_with_diagnostics(x, mask)

            target_idx, has_valid, self_only_x, self_only_mask, swapped_x, swapped_mask = self._build_context_probes(x, mask)
            batch_idx = torch.arange(x.shape[0], device=x.device)

            self_only_x_hat = self(self_only_x, self_only_mask)[0]
            swapped_x_hat = self(swapped_x, swapped_mask)[0]

            target_orig = x_hat[batch_idx, target_idx][has_valid]
            target_self_only = self_only_x_hat[batch_idx, target_idx][has_valid]
            target_swapped = swapped_x_hat[batch_idx, target_idx][has_valid]

            stats = {
                "context/self_only_target_l1": F.l1_loss(target_orig, target_self_only).detach().cpu(),
                "context/self_only_target_l2": F.mse_loss(target_orig, target_self_only).detach().cpu(),
                "context/swapped_context_target_l1": F.l1_loss(target_orig, target_swapped).detach().cpu(),
                "context/swapped_context_target_l2": F.mse_loss(target_orig, target_swapped).detach().cpu(),
            }

            perm = torch.stack([torch.randperm(x.shape[1], device=x.device) for _ in range(x.shape[0])])
            inv_perm = torch.argsort(perm, dim=1)
            gather_idx = perm[:, :, None].expand(-1, -1, x.shape[-1])
            x_perm = torch.gather(x, 1, gather_idx)
            mask_perm = torch.gather(mask, 1, perm)
            x_hat_perm = self(x_perm, mask_perm)[0]
            x_hat_unpermuted = torch.gather(x_hat_perm, 1, inv_perm[:, :, None].expand(-1, -1, x.shape[-1]))
            stats["context/permutation_equivariance_l1"] = F.l1_loss(
                x_hat[mask],
                x_hat_unpermuted[mask],
            ).detach().cpu()

            figures = {}
            for block_name in ("encoder", "decoder"):
                block_stats, block_figures = self._attention_diagnostic_stats(
                    diagnostics[block_name],
                    mask,
                    f"attention/{block_name}",
                )
                stats.update(block_stats)
                figures.update(block_figures)

        return stats, figures

    @profile
    def _log_correlation_diagnostics(self):
        diagnostics = self.val_correlation_diagnostics
        self.val_correlation_diagnostics = None
        if diagnostics is None:
            return

        stats, figures = diagnostics
        for key, value in stats.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
            self.log(f"val_diagnostics/{key}", value, sync_dist=True)

        if not self.trainer.is_global_zero:
            for fig in figures.values():
                plt.close(fig)
            return

        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            payload = {f"val_diagnostics/{key}": wandb.Image(fig) for key, fig in figures.items()}
            if payload:
                self.logger.experiment.log(payload, step=self.global_step)
        else:
            save_dir = os.path.join(self.output_dir, "local_debug_plots")
            os.makedirs(save_dir, exist_ok=True)
            for key, fig in figures.items():
                clean_key = key.replace("/", "_")
                fig.savefig(os.path.join(save_dir, f"val_diagnostics_{clean_key}_step_{self.global_step}.png"))

        for fig in figures.values():
            plt.close(fig)

    @profile
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

    @profile
    def compute_losses(self, x, mask, beta=0.25, phi_idx=1):
            """
            Calculates losses while respecting the periodicity of the phi angle.
            Assuming x shape is [Batch, Particles, Features] and phi is at phi_idx.
            """
            x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)
            
            mask_3d = mask.unsqueeze(-1).expand_as(x)

            # 1. Calculate the raw difference
            diff = x_hat - x

            # 2. Wrap the periodic feature phi. Here, rescaled to [-1, 1] 
            # CRITICAL: We use .clone() here before slice assignment to prevent 
            # PyTorch from throwing an "in-place operation" Autograd error!
            diff_wrapped = diff.clone()
            diff_wrapped[..., phi_idx] = (diff[..., phi_idx] + 1) % (2 * 1) - 1

            # 3. Calculate the actual feature losses using the wrapped difference
            loss_abs_full = torch.abs(diff_wrapped)
            loss_l2_full = diff_wrapped ** 2  # Equivalent to F.mse_loss under the hood

            # 4. Apply the mask and mean (unchanged from your original code)
            loss_abs = (loss_abs_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)
            loss_l2 = (loss_l2_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)

            # 5. Calculate latent losses (unchanged)
            loss_commitment = F.mse_loss(z_mu, z_hat_mu.detach()) if self.dim_mu > 0 else 0

            loss_amplitude = F.mse_loss(z_alpha, z_hat_alpha.detach()) if self.dim_alpha > 0 else 0

            # 6. Total loss
            loss_pha = loss_abs + (beta * loss_commitment) + loss_amplitude

            return loss_pha, loss_l2, loss_abs, loss_commitment, loss_amplitude, x_hat, z_hat_mu, z_hat_alpha



    @profile
    def _evaluate_and_log(self, sample_tuple, prefix="val"):
        """Handles evaluator routing for both validation and testing."""
        if sample_tuple is None:
            return

        x, x_hat, mask = sample_tuple
        results = self.evaluator.evaluate_reconstruction(x, x_hat, mask)

        # Initialize a dictionary to catch all the raw histogram arrays
        histograms_to_save = {}

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
                    os.makedirs(self.output_dir + "/" +"local_debug_plots", exist_ok=True)
                    fig.savefig(
                        f"local_debug_plots/{prefix}_{key.replace('/', '_')}_step_{self.global_step}.png"
                    )

                plt.close(fig)
                
            # 3. Route Raw Data (NumPy Arrays for Histograms)
            elif isinstance(value, np.ndarray):
                # Strip the "histograms/" prefix so the internal file keys are clean
                clean_key = key.replace("histograms/", "").replace("/", "_")
                histograms_to_save[clean_key] = value

        # ==========================================
        # 4. Save the collected histograms to disk
        # ==========================================
        if histograms_to_save:
            save_dir = self.output_dir + "/" + "saved_histograms"
            os.makedirs(save_dir, exist_ok=True)
            
            # Format: saved_histograms/val_hists_step_15000.npz
            filepath = f"{save_dir}/{prefix}_hists_step_{self.global_step}.npz"
            
            # Save all arrays into a single compressed binary file
            np.savez_compressed(filepath, **histograms_to_save)
            
            # Optional but highly recommended: Backup the raw data to WandB!
            if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                artifact = wandb.Artifact(
                    name=f"{prefix}_histograms_step_{self.global_step}", 
                    type="histogram_data"
                )
                artifact.add_file(filepath)
                self.logger.experiment.log_artifact(artifact)
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

    @profile
    def training_step(self, batch, batch_idx):
        # WELD: Unpack the yielded tuple
        x, mask = batch

        # Forward Pass
        # x_hat, z_mu, z_hat_mu, z_alpha, z_hat_alpha = self(x, mask)
        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, _, _, _ = self.compute_losses(
            x, mask, beta=0.25
        )

        # TODO: Make the increment only for the first epoch
        self.total_train_events_seen += x.shape[0]

        self.log_dict(
            {
                "train_loss": loss_pha,
                "mse_loss": loss_l2,
                "l1_recon": loss_abs,
                "commit_mu": loss_commit,
                "commit_alpha": loss_amp,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "global/events_seen",
            float(self.total_train_events_seen),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return loss_pha

    @profile
    def validation_step(self, batch, batch_idx):
        x, mask = batch
        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, x_hat, z_hat_mu, z_hat_alpha = self.compute_losses(
            x, mask, beta=self.hparams.model_cfg["commit_beta"]
        )

        # Track the codebook usage for this batch
        self._track_codebook(z_hat_mu, z_hat_alpha, mask, prefix="val")

        self.log_dict(
            {
                "val_loss": loss_pha,
                "val_mse_loss": loss_l2,
                "val_l1_recon": loss_abs,
                "val_commit_mu": loss_commit,
                "val_commit_alpha": loss_amp,
            },
            prog_bar=True, sync_dist=True,
        )

        if batch_idx == 0:
            self.val_sample = (x.detach(), x_hat.detach(), mask.detach())
            if not self.trainer.sanity_checking and self.val_correlation_diagnostics is None:
                self.val_correlation_diagnostics = self._collect_correlation_diagnostics(x.detach(), mask.detach())

    @profile
    def test_step(self, batch, batch_idx):
        x, mask = batch
        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, x_hat, z_hat_mu, z_hat_alpha = self.compute_losses(
            x, mask, beta=0.25
        )

        # Track the codebook usage for this batch
        self._track_codebook(z_hat_mu, z_hat_alpha, mask, prefix="test")

        self.log_dict(
            {
                "test_loss": loss_pha,
                "test_mse_loss": loss_l2,
                "test_l1_recon": loss_abs,
                "test_commit_mu": loss_commit,
                "test_commit_alpha": loss_amp,
            },
            prog_bar=True, sync_dist=True,
        )

        self.test_step_outputs.append({
            "x": x.detach().cpu(),
            "x_hat": x_hat.detach().cpu(),
            "mask": mask.detach().cpu()
        })

    @profile
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        # 1. Evaluate Physics
        self._evaluate_and_log(getattr(self, "val_sample", None), prefix="val")
        
        # 2. Log and clear codebook utilization
        self._log_and_clear_utilization(prefix="val")

        # 3. Log particle-correlation diagnostics
        self._log_correlation_diagnostics()

    @profile
    def on_test_epoch_end(self):
        # 1. Reconstruct giant tensor block
        x_all = torch.cat([b["x"] for b in self.test_step_outputs], dim=0)
        x_hat_all = torch.cat([b["x_hat"] for b in self.test_step_outputs], dim=0)
        mask_all = torch.cat([b["mask"] for b in self.test_step_outputs], dim=0)
        
        # 2. Evaluate Physics
        giant_tuple = (x_all, x_hat_all, mask_all)
        self._evaluate_and_log(giant_tuple, prefix="test")
        
        # 3. Log and clear codebook utilization
        self._log_and_clear_utilization(prefix="test")
        
        # 4. Clear memory
        self.test_step_outputs.clear()
