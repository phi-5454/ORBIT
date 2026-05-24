import json
import os
import time

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import wandb
from . import torch_modules as tm
from .eval_metrics import PhysicsEvaluator
from .plotting import attention_delta_eta_phi_figure, attention_map_figure, close_figure


class PHA_FSQ_VAE(L.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir,
        data_level="particle",
        jet_metric_workers=0,
        jet_metric_backend="process",
        jet_metric_start_method="forkserver",
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.output_dir = output_dir
        self.data_level = data_level
        self.save_hyperparameters()

        self.total_train_events_seen = 0
        self.test_step_outputs = []
        self.val_correlation_diagnostics = None

        default_quantizer = model_cfg.get("quantizer", "fsq")
        self.mu_quantizer_type = model_cfg.get("mu_quantizer") or default_quantizer
        self.alpha_quantizer_type = (
            model_cfg.get("alpha_quantizer") or default_quantizer
        )
        self.quantizer_type = (
            self.mu_quantizer_type
            if self.mu_quantizer_type == self.alpha_quantizer_type
            else f"mu:{self.mu_quantizer_type},alpha:{self.alpha_quantizer_type}"
        )

        self.dim_mu, self.total_codes_mu = self._branch_quantizer_shape("mu")
        self.dim_alpha, self.total_codes_alpha = self._branch_quantizer_shape("alpha")

        # Calculate total possible codes safely (default to 1 if empty to avoid div by zero)
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

        self.example_input_array = (
            torch.randn(batch_size, window_particles, in_dim),
            torch.ones(batch_size, window_particles, dtype=torch.bool),
        )

        self.input_proj = tm.MLP(in_dim, hidden_dim, [2 * hidden_dim, 2 * hidden_dim])
        self.latent_proj = tm.MLP(
            hidden_dim, codebook_dim, [2 * hidden_dim, 2 * hidden_dim]
        )

        self.latent_proj_out = tm.MLP(
            codebook_dim, hidden_dim, [2 * hidden_dim, 2 * hidden_dim]
        )
        self.output_proj = tm.MLP(hidden_dim, in_dim, [2 * hidden_dim, 2 * hidden_dim])

        self.encoder = tm.NormformerEncoder(
            num_layers=num_enc_dec_layers,
            model_dim=hidden_dim,
            nhead=num_heads,
            mlp_expansion_factor=nf_mlp_expansion_factor,
            dropout=nf_dropout,
        )
        """
        self.encoder = tm.ParticleSetEncoder(
            in_channels=in_dim,
            hidden_dim=hidden_dim,
            latent_nodes=num_heads,
            out_channels=dim_quantized,
        )
        """

        # TODO: Rework phi
        self.phi = tm.Phi(
            dim_in=hidden_dim, dim_alpha=self.dim_alpha, dim_mu=self.dim_mu
        )

        self.quantizer_mu = self._build_branch_quantizer("mu", self.dim_mu)
        self.quantizer_alpha = self._build_branch_quantizer("alpha", self.dim_alpha)

        # TODO: Rework psi
        self.psi = tm.Psi(
            dim_mu=self.dim_mu, dim_alpha=self.dim_alpha, dim_out=hidden_dim
        )

        """
        self.decoder = tm.ParticleSetDecoder(
            latent_channels=codebook_dim,
            hidden_dim=hidden_dim,
            out_nodes=model_cfg["window_particles"],
            out_channels=in_dim,
        )
        """
        self.decoder = tm.NormformerEncoder(
            num_layers=num_enc_dec_layers,
            model_dim=hidden_dim,
            nhead=num_heads,
            mlp_expansion_factor=nf_mlp_expansion_factor,
            dropout=nf_dropout,
        )

        self.evaluator = PhysicsEvaluator(
            data_level=self.data_level,
            jet_metric_workers=jet_metric_workers,
            jet_metric_backend=jet_metric_backend,
            jet_metric_start_method=jet_metric_start_method,
        )

    def _timing_enabled(self):
        return os.environ.get("ORBIT_PROFILE_EVAL", "0") == "1"

    def _timing_log(self, label, start_time):
        if self._timing_enabled() and self.trainer.is_global_zero:
            elapsed = time.perf_counter() - start_time
            print(f"[ORBIT_PROFILE_EVAL] {label}: {elapsed:.3f}s", flush=True)
        return time.perf_counter()

    def _branch_quantizer_type(self, branch):
        if branch == "mu":
            return self.mu_quantizer_type
        if branch == "alpha":
            return self.alpha_quantizer_type
        raise ValueError(f"Unknown quantizer branch: {branch}")

    def _branch_quantizer_shape(self, branch):
        quantizer_type = self._branch_quantizer_type(branch)
        if quantizer_type == "fsq":
            levels = self.model_cfg[f"fsq_{branch}_levels"]
            return len(levels), np.prod(levels) if len(levels) > 0 else 1
        if quantizer_type == "vq":
            dim = int(self.model_cfg.get(f"vq_{branch}_dim", 0))
            num_codes = (
                int(self.model_cfg.get(f"vq_{branch}_num_codes", 1)) if dim > 0 else 1
            )
            return dim, num_codes
        raise ValueError(f"Unsupported {branch} quantizer: {quantizer_type}")

    def _build_branch_quantizer(self, branch, dim):
        if dim <= 0:
            return None

        quantizer_type = self._branch_quantizer_type(branch)
        if quantizer_type == "fsq":
            return tm.FSQ(levels=self.model_cfg[f"fsq_{branch}_levels"])
        if quantizer_type == "vq":
            return tm.VQQuantizer(
                feature_size=dim,
                num_codes=int(self.model_cfg[f"vq_{branch}_num_codes"]),
                beta=float(self.model_cfg.get("vq_beta", 0.95)),
                gradient_estimator=self.model_cfg.get("vq_gradient_estimator", "ste"),
                kmeans_init=bool(self.model_cfg.get("vq_kmeans_init", False)),
                sync_nu=float(self.model_cfg.get("vq_sync_nu", 0.0)),
                affine_lr=float(self.model_cfg.get("vq_affine_lr", 0.0)),
                affine_groups=int(self.model_cfg.get("vq_affine_groups", 1)),
                replace_freq=int(self.model_cfg.get("vq_replace_freq", 0)),
            )
        raise ValueError(f"Unsupported {branch} quantizer: {quantizer_type}")

    def _empty_quantizer_loss(self):
        return torch.zeros((), device=self.device)

    def _quantize_one(self, branch, quantizer, z, dim):
        if dim <= 0:
            return z, z, z, self._empty_quantizer_loss()

        if self._branch_quantizer_type(branch) == "fsq":
            z_hat = quantizer(z)
            z_decoded = z + (z_hat - z).detach()
            return z_decoded, z_hat, z_hat, self._empty_quantizer_loss()

        z_decoded, z_hat, codes, loss = quantizer(z)
        return z_decoded, z_hat, codes, loss

    def _quantize_latents(self, z_mu, z_alpha):
        z_dec_mu, z_hat_mu, z_track_mu, loss_mu = self._quantize_one(
            "mu",
            self.quantizer_mu,
            z_mu,
            self.dim_mu,
        )
        z_dec_alpha, z_hat_alpha, z_track_alpha, loss_alpha = self._quantize_one(
            "alpha",
            self.quantizer_alpha,
            z_alpha,
            self.dim_alpha,
        )
        return (
            z_dec_mu,
            z_dec_alpha,
            z_hat_mu,
            z_hat_alpha,
            z_track_mu,
            z_track_alpha,
            loss_mu,
            loss_alpha,
        )

    def _should_apply_vq_commitment_loss(self):
        if not self.training:
            return True
        frequency = int(self.model_cfg.get("vq_commitment_update_frequency", 1))
        if frequency <= 1:
            return True
        return int(getattr(self, "global_step", 0)) % frequency == 0

    def _assemble_vq_loss(self, loss_parts, beta):
        if self.model_cfg.get("vq_loss_mode", "split") == "vqtorch":
            return beta * loss_parts["vqtorch"]

        commitment = loss_parts["commitment"]
        if not self._should_apply_vq_commitment_loss():
            commitment = commitment.detach().new_zeros(())
        return loss_parts["codebook"] + (beta * commitment)

    def _track_codebook(self, z_hat_mu, z_hat_alpha, mask, prefix="val"):
        """Extracts unique codes from the current batch and adds them to the global epoch sets."""
        # Route to the correct sets
        if prefix == "val":
            set_mu, set_alpha, set_comb = (
                self.val_used_codes_mu,
                self.val_used_codes_alpha,
                self.val_used_codes_combined,
            )
        else:
            set_mu, set_alpha, set_comb = (
                self.test_used_codes_mu,
                self.test_used_codes_alpha,
                self.test_used_codes_combined,
            )

        def add_unique_codes(codes, target_set):
            codes_valid = codes[mask]
            if codes_valid.ndim == 1:
                uniq = torch.unique(codes_valid).detach().cpu().numpy()
                for code in uniq:
                    target_set.add(int(code))
                return

            uniq = torch.unique(codes_valid, dim=0).detach().cpu().numpy()
            for vec in np.round(uniq, decimals=4):
                target_set.add(tuple(vec))

        # 1. Track Mu
        if self.dim_mu > 0:
            add_unique_codes(z_hat_mu, set_mu)

        # 2. Track Alpha
        if self.dim_alpha > 0:
            add_unique_codes(z_hat_alpha, set_alpha)

        # 3. Track Combined Space (The true cross-product utilization)
        if self.dim_mu > 0 and self.dim_alpha > 0:
            z_mu_valid = z_hat_mu[mask]
            z_alpha_valid = z_hat_alpha[mask]
            if z_mu_valid.ndim == 1:
                z_mu_valid = z_mu_valid[:, None]
            if z_alpha_valid.ndim == 1:
                z_alpha_valid = z_alpha_valid[:, None]
            z_combined = torch.cat([z_mu_valid, z_alpha_valid], dim=-1)
            uniq_combined = torch.unique(z_combined, dim=0).detach().cpu().numpy()
            for vec in np.round(uniq_combined, decimals=4):
                set_comb.add(tuple(vec))

    def _log_and_clear_utilization(self, prefix="val"):
        """Calculates utilization percentages, logs them, and safely clears the sets."""
        if prefix == "val":
            set_mu, set_alpha, set_comb = (
                self.val_used_codes_mu,
                self.val_used_codes_alpha,
                self.val_used_codes_combined,
            )
        else:
            set_mu, set_alpha, set_comb = (
                self.test_used_codes_mu,
                self.test_used_codes_alpha,
                self.test_used_codes_combined,
            )

        if self.dim_mu > 0:
            self.log(
                f"{prefix}_metrics/utilization_mu",
                len(set_mu) / self.total_codes_mu,
                sync_dist=True,
            )
            self.log(
                f"{prefix}_metrics/active_codes_mu", float(len(set_mu)), sync_dist=True
            )
            set_mu.clear()

        if self.dim_alpha > 0:
            self.log(
                f"{prefix}_metrics/utilization_alpha",
                len(set_alpha) / self.total_codes_alpha,
                sync_dist=True,
            )
            self.log(
                f"{prefix}_metrics/active_codes_alpha",
                float(len(set_alpha)),
                sync_dist=True,
            )
            set_alpha.clear()

        if self.dim_mu > 0 and self.dim_alpha > 0:
            self.log(
                f"{prefix}_metrics/utilization_combined",
                len(set_comb) / self.total_codes_combined,
                sync_dist=True,
            )
            self.log(
                f"{prefix}_metrics/active_codes_combined",
                float(len(set_comb)),
                sync_dist=True,
            )
            set_comb.clear()

    def forward(self, x, mask):
        # 1. Encode
        x_proj = self.input_proj(x)

        # if(not torch.isfinite(x_proj).all().item()):
        # print("111_")
        z_encoded = self.encoder(
            x_proj, mask, use_attention=self.model_cfg["use_attention"]
        )

        # 2. Split
        z_mu, z_alpha = self.phi(z_encoded)

        # 3. Quantize
        if self.model_cfg["skip_quantization"] == True:
            z_track_mu = z_mu
            z_track_alpha = z_alpha
            z_decoded = self.psi(z_mu, z_alpha)
            self._last_quantizer_loss_mu = self._empty_quantizer_loss()
            self._last_quantizer_loss_alpha = self._empty_quantizer_loss()
        else:
            (
                z_dec_mu,
                z_dec_alpha,
                _,
                _,
                z_track_mu,
                z_track_alpha,
                loss_mu,
                loss_alpha,
            ) = self._quantize_latents(z_mu, z_alpha)
            self._last_quantizer_loss_mu = loss_mu
            self._last_quantizer_loss_alpha = loss_alpha
            z_decoded = self.psi(z_dec_mu, z_dec_alpha)

        x_hat_lat = self.decoder(z_decoded, mask, self.model_cfg["use_attention"])
        x_hat = self.output_proj(x_hat_lat)

        return x_hat, z_mu, z_track_mu, z_alpha, z_track_alpha

    def forward_with_diagnostics(self, x, mask):
        x_proj = self.input_proj(x)
        z_encoded, encoder_diags = self.encoder(
            x_proj,
            mask,
            use_attention=self.model_cfg["use_attention"],
            return_diagnostics=True,
        )

        z_mu, z_alpha = self.phi(z_encoded)

        if self.model_cfg["skip_quantization"] == True:
            z_track_mu = z_mu
            z_track_alpha = z_alpha
            z_decoded = self.psi(z_mu, z_alpha)
            self._last_quantizer_loss_mu = self._empty_quantizer_loss()
            self._last_quantizer_loss_alpha = self._empty_quantizer_loss()
        else:
            (
                z_dec_mu,
                z_dec_alpha,
                _,
                _,
                z_track_mu,
                z_track_alpha,
                loss_mu,
                loss_alpha,
            ) = self._quantize_latents(z_mu, z_alpha)
            self._last_quantizer_loss_mu = loss_mu
            self._last_quantizer_loss_alpha = loss_alpha
            z_decoded = self.psi(z_dec_mu, z_dec_alpha)

        x_hat_lat, decoder_diags = self.decoder(
            z_decoded,
            mask,
            self.model_cfg["use_attention"],
            return_diagnostics=True,
        )
        x_hat = self.output_proj(x_hat_lat)

        return (
            x_hat,
            z_mu,
            z_track_mu,
            z_alpha,
            z_track_alpha,
            {
                "encoder": encoder_diags,
                "decoder": decoder_diags,
            },
        )

    def _first_valid_indices(self, mask):
        has_valid = mask.any(dim=1)
        indices = mask.float().argmax(dim=1)
        return indices, has_valid

    def _attention_diagnostic_stats(self, diagnostics, mask, prefix, x=None):
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
            entropy = -(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log()).sum(
                dim=-1
            )
            n_keys = valid.sum(dim=1).clamp(min=2).float()
            normalized_entropy = entropy / n_keys.log()[:, None, None]
            query_values = normalized_entropy[
                valid[:, None, :].expand_as(normalized_entropy)
            ]

            max_mass = probs.max(dim=-1).values
            max_values = max_mass[valid[:, None, :].expand_as(max_mass)]

            attn_delta = layer.get("attn_delta")
            ff_delta = layer.get("ff_delta")
            if attn_delta is not None and ff_delta is not None:
                token_mask = valid[:, :, None]
                attn_norm = attn_delta.detach()[token_mask.expand_as(attn_delta)].norm()
                ff_norm = ff_delta.detach()[token_mask.expand_as(ff_delta)].norm()
                stats[f"{prefix}/layer_{layer_idx}_attn_to_ff_delta_norm"] = (
                    (attn_norm / ff_norm.clamp(min=1e-12)).detach().cpu()
                )

            stats[f"{prefix}/layer_{layer_idx}_diag_attention_mass"] = (
                diag_mass.detach().cpu()
            )
            stats[f"{prefix}/layer_{layer_idx}_offdiag_attention_mass"] = (
                (1.0 - diag_mass).detach().cpu()
            )
            stats[f"{prefix}/layer_{layer_idx}_max_attention_mass"] = (
                max_values.mean().detach().cpu()
            )
            stats[f"{prefix}/layer_{layer_idx}_normalized_attention_entropy"] = (
                query_values.mean().detach().cpu()
            )
            stats[f"{prefix}/layer_{layer_idx}_valid_pair_count"] = (
                denom.detach().float().cpu()
            )

            if layer_idx in (0, len(diagnostics) - 1):
                first_event = (
                    int(valid.any(dim=1).nonzero(as_tuple=False)[0].item())
                    if valid.any()
                    else 0
                )
                n_valid = int(valid[first_event].sum().item())
                if n_valid > 1:
                    matrix = (
                        weights[first_event, :, :n_valid, :n_valid]
                        .mean(dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    fig = attention_map_figure(
                        matrix, f"{prefix} layer {layer_idx} attention"
                    )
                    figures[f"{prefix}/layer_{layer_idx}_attention_map"] = fig

                    if x is not None and x.shape[-1] >= 2:
                        delta_fig = attention_delta_eta_phi_figure(
                            weights,
                            x,
                            valid,
                            f"{prefix} layer {layer_idx} attention vs delta eta/phi",
                        )
                        if delta_fig is not None:
                            figures[
                                f"{prefix}/layer_{layer_idx}_delta_eta_phi_attention"
                            ] = delta_fig

                        offdiag_delta_fig = attention_delta_eta_phi_figure(
                            weights,
                            x,
                            valid,
                            f"{prefix} layer {layer_idx} attention vs delta eta/phi, i != j",
                            exclude_self=True,
                        )
                        if offdiag_delta_fig is not None:
                            figures[
                                f"{prefix}/layer_{layer_idx}_delta_eta_phi_attention_no_self"
                            ] = offdiag_delta_fig

        return stats, figures

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

        return (
            target_idx,
            has_valid,
            self_only_x,
            self_only_mask,
            swapped_x,
            swapped_mask,
        )

    def _collect_correlation_diagnostics(self, x, mask):
        max_events = int(self.model_cfg.get("diagnostic_max_events", 8))
        x = x[:max_events]
        mask = mask[:max_events]

        with torch.no_grad():
            x_hat, _, _, _, _, diagnostics = self.forward_with_diagnostics(x, mask)

            (
                target_idx,
                has_valid,
                self_only_x,
                self_only_mask,
                swapped_x,
                swapped_mask,
            ) = self._build_context_probes(x, mask)
            batch_idx = torch.arange(x.shape[0], device=x.device)

            self_only_x_hat = self(self_only_x, self_only_mask)[0]
            swapped_x_hat = self(swapped_x, swapped_mask)[0]

            target_orig = x_hat[batch_idx, target_idx][has_valid]
            target_self_only = self_only_x_hat[batch_idx, target_idx][has_valid]
            target_swapped = swapped_x_hat[batch_idx, target_idx][has_valid]

            stats = {
                "context/self_only_target_l1": F.l1_loss(target_orig, target_self_only)
                .detach()
                .cpu(),
                "context/self_only_target_l2": F.mse_loss(target_orig, target_self_only)
                .detach()
                .cpu(),
                "context/swapped_context_target_l1": F.l1_loss(
                    target_orig, target_swapped
                )
                .detach()
                .cpu(),
                "context/swapped_context_target_l2": F.mse_loss(
                    target_orig, target_swapped
                )
                .detach()
                .cpu(),
            }

            perm = torch.stack(
                [torch.randperm(x.shape[1], device=x.device) for _ in range(x.shape[0])]
            )
            inv_perm = torch.argsort(perm, dim=1)
            gather_idx = perm[:, :, None].expand(-1, -1, x.shape[-1])
            x_perm = torch.gather(x, 1, gather_idx)
            mask_perm = torch.gather(mask, 1, perm)
            x_hat_perm = self(x_perm, mask_perm)[0]
            x_hat_unpermuted = torch.gather(
                x_hat_perm, 1, inv_perm[:, :, None].expand(-1, -1, x.shape[-1])
            )
            stats["context/permutation_equivariance_l1"] = (
                F.l1_loss(
                    x_hat[mask],
                    x_hat_unpermuted[mask],
                )
                .detach()
                .cpu()
            )

            figures = {}
            for block_name in ("encoder", "decoder"):
                block_stats, block_figures = self._attention_diagnostic_stats(
                    diagnostics[block_name],
                    mask,
                    f"attention/{block_name}",
                    x=x,
                )
                stats.update(block_stats)
                figures.update(block_figures)

        return stats, figures

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
                close_figure(fig)
            return

        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            payload = {
                f"val_diagnostics/{key}": wandb.Image(fig)
                for key, fig in figures.items()
            }
            if payload:
                self.logger.experiment.log(payload, step=self.global_step)
        else:
            save_dir = os.path.join(self.output_dir, "local_debug_plots")
            os.makedirs(save_dir, exist_ok=True)
            for key, fig in figures.items():
                clean_key = key.replace("/", "_")
                fig.savefig(
                    os.path.join(
                        save_dir,
                        f"val_diagnostics_{clean_key}_step_{self.global_step}.png",
                    )
                )

        for fig in figures.values():
            close_figure(fig)

    def configure_optimizers(self):
        schedule_cfg = self._lr_schedule_config()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=schedule_cfg["max_lr"],
            weight_decay=self.hparams.model_cfg["weight_decay"],
        )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self._lr_at_epoch(epoch, schedule_cfg)
            / schedule_cfg["max_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _lr_schedule_config(self):
        model_cfg = self.hparams.model_cfg
        if model_cfg.get("use_attention", True):
            prefix = "transformer"
        else:
            prefix = "deepsets"

        try:
            total_epochs = int(getattr(self.trainer, "max_epochs", 0) or 0)
        except RuntimeError:
            total_epochs = 0

        return {
            "initial_lr": float(model_cfg[f"{prefix}_initial_lr"]),
            "max_lr": float(model_cfg[f"{prefix}_max_lr"]),
            "final_lr": float(model_cfg[f"{prefix}_final_lr"]),
            "warmup_epochs": int(model_cfg.get("lr_warmup_epochs", 4)),
            "decay_to_initial_epochs": int(
                model_cfg.get("lr_decay_to_initial_epochs", 20)
            ),
            "total_epochs": total_epochs,
        }

    @staticmethod
    def _smooth_interp(start, end, position, span):
        if span <= 0:
            return end
        fraction = min(max(position / span, 0.0), 1.0)
        fraction = 0.5 - 0.5 * np.cos(np.pi * fraction)
        return start + fraction * (end - start)

    def _lr_at_epoch(self, epoch, schedule_cfg):
        warmup_epochs = schedule_cfg["warmup_epochs"]
        decay_to_initial_epochs = schedule_cfg["decay_to_initial_epochs"]
        total_epochs = max(
            schedule_cfg["total_epochs"], warmup_epochs + decay_to_initial_epochs + 1
        )

        initial_lr = schedule_cfg["initial_lr"]
        max_lr = schedule_cfg["max_lr"]
        final_lr = schedule_cfg["final_lr"]

        if epoch <= warmup_epochs:
            return self._smooth_interp(
                initial_lr,
                max_lr,
                epoch,
                max(warmup_epochs, 1),
            )

        initial_decay_end = warmup_epochs + decay_to_initial_epochs
        if epoch <= initial_decay_end:
            return self._smooth_interp(
                max_lr,
                initial_lr,
                epoch - warmup_epochs,
                max(decay_to_initial_epochs, 1),
            )

        return self._smooth_interp(
            initial_lr,
            final_lr,
            epoch - initial_decay_end,
            max(total_epochs - initial_decay_end, 1),
        )

    def compute_losses(self, x, mask, beta=None, phi_idx=1):
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

        # 3. Calculate diagnostic reconstruction losses using the wrapped difference
        loss_abs_full = torch.abs(diff_wrapped)
        loss_l2_full = diff_wrapped**2  # Equivalent to F.mse_loss under the hood

        # 4. Apply the mask and mean
        loss_abs = (loss_abs_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)
        loss_l2 = (loss_l2_full * mask_3d).sum() / mask_3d.sum().clamp(min=1.0)
        loss_l2_distance = (diff_wrapped.norm(dim=-1) * mask).sum() / mask.sum().clamp(
            min=1.0
        )

        reconstruction_loss_name = self.model_cfg.get("reconstruction_loss", "l1")
        if reconstruction_loss_name == "l1":
            reconstruction_loss = loss_abs
        elif reconstruction_loss_name == "mse":
            reconstruction_loss = loss_l2
        elif reconstruction_loss_name in ("l2", "euclidean"):
            reconstruction_loss = loss_l2_distance
        else:
            raise ValueError(
                f"Unsupported reconstruction_loss: {reconstruction_loss_name}"
            )

        if self.mu_quantizer_type == "vq":
            loss_commitment = self._last_quantizer_loss_mu if self.dim_mu > 0 else 0
        else:
            loss_commitment = (
                F.mse_loss(z_mu, z_hat_mu.detach()) if self.dim_mu > 0 else 0
            )

        if self.alpha_quantizer_type == "vq":
            loss_amplitude = (
                self._last_quantizer_loss_alpha if self.dim_alpha > 0 else 0
            )
        else:
            loss_amplitude = (
                F.mse_loss(z_alpha, z_hat_alpha.detach()) if self.dim_alpha > 0 else 0
            )

        beta_mu = float(
            self.model_cfg.get(
                "beta_mu",
                self.model_cfg.get(
                    "mu_quant_loss_scale",
                    self.model_cfg.get("commit_beta", 0.25) if beta is None else beta,
                ),
            )
        )
        beta_alpha = float(
            self.model_cfg.get(
                "beta_alpha",
                self.model_cfg.get("alpha_quant_loss_scale", 1.0),
            )
        )

        if self.mu_quantizer_type == "vq" and self.dim_mu > 0:
            loss_commitment = self._assemble_vq_loss(loss_commitment, beta_mu)
        else:
            loss_commitment = beta_mu * loss_commitment

        if self.alpha_quantizer_type == "vq" and self.dim_alpha > 0:
            loss_amplitude = self._assemble_vq_loss(loss_amplitude, beta_alpha)
        else:
            loss_amplitude = beta_alpha * loss_amplitude

        loss_pha = reconstruction_loss + loss_commitment + loss_amplitude

        return (
            loss_pha,
            loss_l2,
            loss_abs,
            loss_commitment,
            loss_amplitude,
            x_hat,
            z_hat_mu,
            z_hat_alpha,
        )

    def _evaluate_and_log(self, sample_tuple, prefix="val"):
        """Handles evaluator routing for both validation and testing."""
        if sample_tuple is None:
            return

        x, x_hat, mask = sample_tuple
        t0 = time.perf_counter()
        results = self.evaluator.evaluate_reconstruction(x, x_hat, mask)
        t0 = self._timing_log(f"{prefix} evaluate_reconstruction", t0)

        # Initialize a dictionary to catch all the raw histogram arrays
        histograms_to_save = {}
        metrics_to_save = {}

        for key, value in results.items():
            # 1. Route Scalars
            if isinstance(value, (int, float)):
                self.log(f"{prefix}_metrics/{key}", value, sync_dist=True)
                metrics_to_save[key] = float(value)

            # 2. Route Figures
            elif hasattr(value, "savefig"):
                fig = value

                if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
                    self.logger.experiment.log(
                        {f"{prefix}_plots/{key}": wandb.Image(fig)},
                        # step=self.global_step,
                    )

                elif isinstance(self.logger, L.pytorch.loggers.TensorBoardLogger):
                    self.logger.experiment.add_figure(
                        f"{prefix}_plots/{key}", fig, global_step=self.global_step
                    )
                else:
                    os.makedirs(
                        self.output_dir + "/" + "local_debug_plots", exist_ok=True
                    )
                    fig.savefig(
                        f"local_debug_plots/{prefix}_{key.replace('/', '_')}_step_{self.global_step}.png"
                    )

                close_figure(fig)

            # 3. Route Raw Data (NumPy Arrays for Histograms)
            elif isinstance(value, np.ndarray):
                # Strip the "histograms/" prefix so the internal file keys are clean
                clean_key = key.replace("histograms/", "").replace("/", "_")
                histograms_to_save[clean_key] = value

        t0 = self._timing_log(f"{prefix} route/log results", t0)

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
                    type="histogram_data",
                )
                artifact.add_file(filepath)
                self.logger.experiment.log_artifact(artifact)

        t0 = self._timing_log(f"{prefix} save/log histograms", t0)

        if metrics_to_save:
            save_dir = self.output_dir + "/" + "saved_metrics"
            os.makedirs(save_dir, exist_ok=True)
            filepath = f"{save_dir}/{prefix}_metrics_step_{self.global_step}.json"
            with open(filepath, "w") as f:
                json.dump(metrics_to_save, f, indent=2, sort_keys=True)

        self._timing_log(f"{prefix} save metrics", t0)

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
        loss_pha, loss_l2, loss_abs, loss_commit, loss_amp, _, _, _ = (
            self.compute_losses(x, mask)
        )

        # TODO: Make the increment only for the first epoch
        self.total_train_events_seen += x.shape[0]

        self.log_dict(
            {
                "train_loss": loss_pha,
                "mse_loss": loss_l2,
                "l1_recon": loss_abs,
                "task_recon": loss_pha - loss_commit - loss_amp,
                "quant_loss_mu": loss_commit,
                "quant_loss_alpha": loss_amp,
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

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        (
            loss_pha,
            loss_l2,
            loss_abs,
            loss_commit,
            loss_amp,
            x_hat,
            z_hat_mu,
            z_hat_alpha,
        ) = self.compute_losses(x, mask)

        # Track the codebook usage for this batch
        self._track_codebook(z_hat_mu, z_hat_alpha, mask, prefix="val")

        self.log_dict(
            {
                "val_loss": loss_pha,
                "val_mse_loss": loss_l2,
                "val_l1_recon": loss_abs,
                "val_task_recon": loss_pha - loss_commit - loss_amp,
                "val_quant_loss_mu": loss_commit,
                "val_quant_loss_alpha": loss_amp,
            },
            prog_bar=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            self.val_sample = (x.detach(), x_hat.detach(), mask.detach())
            if (
                not self.trainer.sanity_checking
                and self.val_correlation_diagnostics is None
            ):
                self.val_correlation_diagnostics = (
                    self._collect_correlation_diagnostics(x.detach(), mask.detach())
                )

    def test_step(self, batch, batch_idx):
        x, mask = batch
        (
            loss_pha,
            loss_l2,
            loss_abs,
            loss_commit,
            loss_amp,
            x_hat,
            z_hat_mu,
            z_hat_alpha,
        ) = self.compute_losses(x, mask)

        # Track the codebook usage for this batch
        self._track_codebook(z_hat_mu, z_hat_alpha, mask, prefix="test")

        self.log_dict(
            {
                "test_loss": loss_pha,
                "test_mse_loss": loss_l2,
                "test_l1_recon": loss_abs,
                "test_task_recon": loss_pha - loss_commit - loss_amp,
                "test_quant_loss_mu": loss_commit,
                "test_quant_loss_alpha": loss_amp,
            },
            prog_bar=True,
            sync_dist=True,
        )

        self.test_step_outputs.append(
            {
                "x": x.detach().cpu(),
                "x_hat": x_hat.detach().cpu(),
                "mask": mask.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        # 1. Evaluate Physics
        self._evaluate_and_log(getattr(self, "val_sample", None), prefix="val")

        # 2. Log and clear codebook utilization
        self._log_and_clear_utilization(prefix="val")

        # 3. Log particle-correlation diagnostics
        self._log_correlation_diagnostics()

    def on_test_epoch_end(self):
        t0 = time.perf_counter()
        # 1. Reconstruct giant tensor block
        x_all = torch.cat([b["x"] for b in self.test_step_outputs], dim=0)
        x_hat_all = torch.cat([b["x_hat"] for b in self.test_step_outputs], dim=0)
        mask_all = torch.cat([b["mask"] for b in self.test_step_outputs], dim=0)
        t0 = self._timing_log("test concatenate outputs", t0)

        # 2. Evaluate Physics
        giant_tuple = (x_all, x_hat_all, mask_all)
        self._evaluate_and_log(giant_tuple, prefix="test")
        t0 = self._timing_log("test evaluate/log total", t0)

        # 3. Log and clear codebook utilization
        self._log_and_clear_utilization(prefix="test")
        t0 = self._timing_log("test codebook utilization", t0)

        # 4. Clear memory
        self.test_step_outputs.clear()
        self._timing_log("test clear outputs", t0)
