import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
import seaborn as sns
import torch

def exploratory_feature_histograms(x_np, x_hat_np, mse_per_feature, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("FSQ-VAE: Original vs. Reconstructed Features", fontsize=16)

    for i in range(3):
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

        axes[i].set_title(f"{feature_names[i]} (MSE: {mse_per_feature[i]:.4f})")
        axes[i].legend()

    plt.tight_layout()
    return fig


def exploratory_energy_histograms(x_np, x_hat_np, mse_per_feature):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    masses_orig = x_np[:, 2] * np.cosh(x_np[:, 0])
    masses_reco = x_hat_np[:, 2] * np.cosh(x_hat_np[:, 0])

    min_val = max(min(masses_orig.min(), masses_reco.min()), 1e-8)
    max_val = max(masses_orig.max(), masses_reco.max()).max()
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)

    axs[0].set_title("FSQ-VAE: Original vs. Reconstructed mass", fontsize=16)
    sns.histplot(
        masses_orig,
        bins=log_bins,
        color="blue",
        alpha=0.5,
        label="Original",
        kde=False,
        stat="density",
        ax=axs[0],
    )
    sns.histplot(
        masses_reco,
        bins=log_bins,
        color="orange",
        alpha=0.5,
        label="Reconstructed",
        kde=False,
        stat="density",
        ax=axs[0],
    )

    axs[1].set_title("FSQ-VAE: m_reco - m_original", fontsize=16)
    sns.histplot(
        masses_reco - masses_orig,
        bins=50,
        color="blue",
        alpha=0.5,
        label="Original",
        kde=False,
        stat="density",
        ax=axs[1],
    )
    axs[0].set_xscale("log")
    axs[0].set_title(f"mass (assume. m_0 = 0) (MSE: {mse_per_feature[2]:.4f})")
    axs[0].legend()

    plt.tight_layout()
    return fig


def add_reconstruction_plots(
    results,
    feature_names,
    mse_per_feature,
    x_np,
    x_hat_np,
    true_jet_pts,
    reco_jet_pts,
    true_jet_masses,
    reco_jet_masses,
    true_tau32s,
    reco_tau32s,
):
    mh.style.use(mh.style.ROOT)

    if len(true_jet_pts) > 0:
        true_jet_pts = np.array(true_jet_pts)
        reco_jet_pts = np.array(reco_jet_pts)

        fig, ax = plt.subplots(figsize=(8, 6))
        fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
        counts, bins = np.histogram(fractional_diff, bins=50, range=(-0.5, 0.5))

        results["histograms/jet_pt_resolution_counts"] = counts
        results["histograms/jet_pt_resolution_bins"] = bins

        mh.histplot(counts, bins=bins, ax=ax, histtype="step", color="indigo", linewidth=2)

        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"Fractional $p_T$ Resolution: $(p_T^{reco} - p_T^{true}) / p_T^{true}$")
        ax.set_ylabel("Number of Jets")
        ax.set_title("Jet Transverse Momentum Recovery")

        results["plots/jet_pt_resolution"] = fig

    fig1, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig1.suptitle("FSQ-VAE: Original vs. Reconstructed Features (per-particle)", fontsize=16)

    for i in range(3):
        feature_name = feature_names[i].replace(" ", "_")

        min_val = min(x_np[:, i].min(), x_hat_np[:, i].min())
        max_val = max(x_np[:, i].max(), x_hat_np[:, i].max())

        if i == 2:
            min_val = max(min_val, 1e-8)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            axes[i].set_xscale("log")
        else:
            bins = np.linspace(min_val, max_val, 50)

        counts_orig, _ = np.histogram(x_np[:, i], bins=bins, density=True)
        counts_reco, _ = np.histogram(x_hat_np[:, i], bins=bins, density=True)

        results[f"histograms/{feature_name}_orig_counts"] = counts_orig
        results[f"histograms/{feature_name}_reco_counts"] = counts_reco
        results[f"histograms/{feature_name}_bins"] = bins

        mh.histplot(
            [counts_orig, counts_reco],
            bins=bins,
            ax=axes[i],
            label=["Original", "Reconstructed"],
            color=["blue", "orange"],
            histtype="fill",
            alpha=0.5,
            edgecolor=["blue", "orange"],
        )

        axes[i].set_title(f"{feature_names[i]} (MSE: {mse_per_feature[i]:.4f})")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    plt.tight_layout()
    results["plots/kinematics"] = fig1

    fig2, axs = plt.subplots(1, 2, figsize=(16, 5))

    pt_orig = x_np[:, 2]
    pt_reco = x_hat_np[:, 2]

    energy_orig = pt_orig * np.cosh(x_np[:, 0])
    energy_reco = pt_reco * np.cosh(x_hat_np[:, 0])

    min_val = max(min(energy_orig.min(), energy_reco.min()), 1e-8)
    max_val = max(energy_orig.max(), energy_reco.max())
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)

    counts_e_orig, _ = np.histogram(energy_orig, bins=log_bins, density=True)
    counts_e_reco, _ = np.histogram(energy_reco, bins=log_bins, density=True)

    results["histograms/energy_orig_counts"] = counts_e_orig
    results["histograms/energy_reco_counts"] = counts_e_reco
    results["histograms/energy_bins"] = log_bins

    axs[0].set_title("FSQ-VAE: Original vs. Reconstructed Energy (m=0)", fontsize=14)
    mh.histplot(
        [counts_e_orig, counts_e_reco],
        bins=log_bins,
        ax=axs[0],
        label=["Original", "Reconstructed"],
        color=["blue", "orange"],
        histtype="fill",
        alpha=0.5,
        edgecolor=["blue", "orange"],
    )
    axs[0].set_xscale("log")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    energy_mse = np.mean((energy_reco - energy_orig) ** 2)
    res_counts, res_bins = np.histogram(energy_reco - energy_orig, bins=50, density=True)

    results["histograms/energy_residuals_counts"] = res_counts
    results["histograms/energy_residuals_bins"] = res_bins

    axs[1].set_title(f"Residuals: E_reco - E_original (MSE: {energy_mse:.2f})", fontsize=14)
    mh.histplot(res_counts, bins=res_bins, ax=axs[1], color="green", histtype="fill", alpha=0.5, edgecolor="green")
    axs[1].set_ylabel("Density")

    plt.tight_layout()
    results["plots/energy_residuals"] = fig2

    if len(true_jet_masses) > 0:
        fig3, axs3 = plt.subplots(1, 3, figsize=(18, 8))
        fig3.suptitle("FSQ-VAE: Jet Substructure", fontsize=16)

        true_jet_masses = np.array(true_jet_masses)
        reco_jet_masses = np.array(reco_jet_masses)
        true_tau32s = np.array(true_tau32s)
        reco_tau32s = np.array(reco_tau32s)

        mass_bins = np.linspace(0, 600, 50)

        counts_m_orig, _ = np.histogram(true_jet_masses, bins=mass_bins, density=True)
        counts_m_reco, _ = np.histogram(reco_jet_masses, bins=mass_bins, density=True)

        results["histograms/jet_mass_orig_counts"] = counts_m_orig
        results["histograms/jet_mass_reco_counts"] = counts_m_reco
        results["histograms/jet_mass_bins"] = mass_bins

        mh.histplot(
            [counts_m_orig, counts_m_reco],
            bins=mass_bins,
            ax=axs3[0],
            label=["Original", "Reconstructed"],
            color=["blue", "orange"],
            histtype="fill",
            alpha=0.5,
            edgecolor=["blue", "orange"],
        )
        axs3[0].set_xlabel("Jet Mass [GeV]")
        axs3[0].set_ylabel("Density")
        axs3[0].legend()

        mass_diff = reco_jet_masses - true_jet_masses
        diff_bins = np.linspace(-50, 50, 50)

        counts_mdiff, _ = np.histogram(mass_diff, bins=diff_bins, density=True)
        results["histograms/jet_mass_diff_counts"] = counts_mdiff
        results["histograms/jet_mass_diff_bins"] = diff_bins

        mh.histplot(counts_mdiff, bins=diff_bins, ax=axs3[1], histtype="fill", color="green", alpha=0.5, edgecolor="green")
        axs3[1].axvline(0, color="black", linestyle="--", alpha=0.5)
        axs3[1].set_xlabel(r"$m^{reco} - m^{orig}$ [GeV]")
        axs3[1].set_ylabel("Density")

        tau_diff = reco_tau32s - true_tau32s
        tau_bins = np.linspace(-0.4, 0.4, 50)

        counts_tdiff, _ = np.histogram(tau_diff, bins=tau_bins, density=True)
        results["histograms/tau32_diff_counts"] = counts_tdiff
        results["histograms/tau32_diff_bins"] = tau_bins

        mh.histplot(counts_tdiff, bins=tau_bins, ax=axs3[2], histtype="fill", color="purple", alpha=0.5, edgecolor="purple")
        axs3[2].axvline(0, color="black", linestyle="--", alpha=0.5)
        axs3[2].set_xlabel(r"$\tau_{32}^{reco} - \tau_{32}^{orig}$")
        axs3[2].set_ylabel("Density")

        plt.tight_layout()
        results["plots/jet_substructure"] = fig3

    return results


def attention_delta_eta_phi_figure(weights, x, valid, title, exclude_self=False):
    # Inputs are normalized as eta / 3 and phi / pi in the dataloader.
    attn = weights.mean(dim=1)
    eta = x[..., 0] * 3.0
    phi = x[..., 1] * math.pi

    deta = eta[:, :, None] - eta[:, None, :]
    dphi = phi[:, :, None] - phi[:, None, :]
    dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi

    pair_mask = valid[:, :, None] & valid[:, None, :]
    if exclude_self:
        self_mask = torch.eye(pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device)
        pair_mask = pair_mask & ~self_mask[None, :, :]
    if not pair_mask.any():
        return None

    deta_np = deta[pair_mask].detach().cpu().numpy()
    dphi_np = dphi[pair_mask].detach().cpu().numpy()
    weight_np = attn[pair_mask].detach().cpu().numpy()

    weight_sum, eta_edges, phi_edges = np.histogram2d(
        deta_np,
        dphi_np,
        bins=(60, 64),
        range=((-6.0, 6.0), (-math.pi, math.pi)),
        weights=weight_np,
    )
    pair_count, _, _ = np.histogram2d(
        deta_np,
        dphi_np,
        bins=(eta_edges, phi_edges),
    )
    hist = np.divide(
        weight_sum,
        pair_count,
        out=np.zeros_like(weight_sum),
        where=pair_count > 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta\eta = \eta_\mathrm{query} - \eta_\mathrm{key}$")
    ax.set_ylabel(r"$\Delta\phi = \phi_\mathrm{query} - \phi_\mathrm{key}$")
    fig.colorbar(im, ax=ax, label="mean attention weight per pair")
    return fig


def attention_map_figure(matrix, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0.0, vmax=max(float(matrix.max()), 1e-6), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("key particle")
    ax.set_ylabel("query particle")
    fig.colorbar(im, ax=ax)
    return fig


def close_figure(fig):
    plt.close(fig)


def show_figure():
    plt.show()


def _clean_metric_name(metric):
    return metric.replace("metrics/", "").replace("/", "_")


def plot_codebook_error_scatter(records, y_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mh.style.use(mh.style.ROOT)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for metric_idx, metric in enumerate(y_metrics):
        metric_records = [
            record
            for record in records
            if record.get(metric) is not None
            and record.get("total_codebook_size") is not None
            and record[metric] > 0
        ]
        if not metric_records:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, record in enumerate(metric_records):
            ax.scatter(
                record["total_codebook_size"],
                record[metric],
                color=colors[i % len(colors)],
                alpha=0.8,
                s=45,
                label=record.get("label", record.get("run_name", f"run_{i}")),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total codebook size")
        ax.set_ylabel(metric.replace("metrics/", ""))
        ax.set_title(f"Codebook size vs. {metric.replace('metrics/', '')}")

        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 12:
            ax.legend(prop={"size": 8})

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"codebook_size_vs_{_clean_metric_name(metric)}.png"))
        plt.close(fig)


def replot_jet_structure(npz_files, run_labels, output_dir="replot_outputs"):
    """
    Loads multiple .npz files and plots the superimposed histograms using mplhep.
    
    Args:
        npz_files (list of str): Paths to the .npz files.
        run_labels (list of str): Legend labels for each run (e.g., ["FSQ 10", "FSQ 21"]).
        output_dir (str): Directory to save the combined figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    mh.style.use(mh.style.ROOT)
    
    # Define distinct colors for the different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Load all data dictionaries
    runs_data = [np.load(f) for f in npz_files]
    
    # We can extract the bins and the "Original" truth from the very first run
    # (since the test set and bins are constant across all runs)
    ref_data = runs_data[0]

    # ==========================================
    # 1. Jet pT Resolution
    # ==========================================
    fig_pt_res, ax_pt_res = plt.subplots(figsize=(8, 6))
    bins_pt_res = ref_data["jet_pt_resolution_bins"]
    
    for i, data in enumerate(runs_data):
        mh.histplot(
            data["jet_pt_resolution_counts"], 
            bins=bins_pt_res, 
            ax=ax_pt_res, 
            label=run_labels[i], 
            histtype='step', 
            color=colors[i % len(colors)], 
            linewidth=2
        )
        
    ax_pt_res.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax_pt_res.set_xlabel(r"Fractional $p_T$ Resolution: $(p_T^{reco} - p_T^{true}) / p_T^{true}$")
    ax_pt_res.set_ylabel("Density / Number of Jets")
    ax_pt_res.set_title("Jet Transverse Momentum Recovery")
    ax_pt_res.legend(prop={'size': 10})
    fig_pt_res.savefig(os.path.join(output_dir, "combined_jet_pt_resolution.png"))
    plt.close(fig_pt_res)

    # ==========================================
    # 2. Kinematic Features (Eta, Phi, pT)
    # ==========================================
    fig_kin, axes_kin = plt.subplots(1, 3, figsize=(18, 6))
    fig_kin.suptitle("Kinematics: Original vs. Reconstructed Sweeps", fontsize=16)

    
    # log plot for pt
    axes_kin[2].set_yscale('log')
    features = ["Eta", "Phi", "pT"]
    for i, feat in enumerate(features):
        bins = ref_data[f"{feat}_bins"]
        
        # Plot the Original Truth once (Filled Grey)
        mh.histplot(
            ref_data[f"{feat}_orig_counts"], 
            bins=bins, ax=axes_kin[i], label="Original (Truth)", 
            color="grey", histtype='fill', alpha=0.3
        )
        
        # Overlay the Reconstructed runs
        for j, data in enumerate(runs_data):
            mh.histplot(
                data[f"{feat}_reco_counts"], 
                bins=bins, ax=axes_kin[i], label=f"{run_labels[j]}", 
                color=colors[j % len(colors)], histtype='step', linewidth=2
            )
            
        if feat == "pT":
            axes_kin[i].set_xscale("log")
            
        axes_kin[i].set_title(f"{feat} Distribution")
        axes_kin[i].set_ylabel("Density")
        axes_kin[i].legend(prop={'size': 10})

    plt.tight_layout()
    fig_kin.savefig(os.path.join(output_dir, "combined_kinematics.png"))
    plt.close(fig_kin)

    # ==========================================
    # 3. Energy and Residuals
    # ==========================================
    fig_energy, axs_e = plt.subplots(1, 2, figsize=(16, 5))
    
    # 3.1 Energy Distribution
    bins_e = ref_data["energy_bins"]
    mh.histplot(
        ref_data["energy_orig_counts"], bins=bins_e, ax=axs_e[0], 
        label="Original (Truth)", color="grey", histtype='fill', alpha=0.3
    )
    for j, data in enumerate(runs_data):
        mh.histplot(
            data["energy_reco_counts"], bins=bins_e, ax=axs_e[0], 
            label=f"Reco ({run_labels[j]})", color=colors[j % len(colors)], 
            histtype='step', linewidth=2
        )
    axs_e[0].set_title("Energy Distribution (m=0)")
    axs_e[0].set_xscale("log")
    axs_e[0].set_ylabel("Density")
    axs_e[0].legend(prop={'size': 10})

    # 3.2 Energy Residuals
    bins_e_res = ref_data["energy_residuals_bins"]
    for j, data in enumerate(runs_data):
        mh.histplot(
            data["energy_residuals_counts"], bins=bins_e_res, ax=axs_e[1], 
            label=run_labels[j], color=colors[j % len(colors)], 
            histtype='step', linewidth=2
        )
    axs_e[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs_e[1].set_title(r"Energy Residuals: $E^{reco} - E^{orig}$")
    axs_e[1].set_ylabel("Density")
    axs_e[1].legend(prop={'size': 10})

    plt.tight_layout()
    fig_energy.savefig(os.path.join(output_dir, "combined_energy.png"))
    plt.close(fig_energy)

    # ==========================================
    # 4. Jet Substructure (Mass, Mass Diff, Tau32 Diff)
    # ==========================================
    # Check if substructure data exists (in case a run crashed before substructure or had N<3)
    if "jet_mass_orig_counts" in ref_data.files:
        fig_sub, axs_sub = plt.subplots(1, 3, figsize=(18, 5))
        fig_sub.suptitle("Jet Substructure Sweep", fontsize=16)

        # 4.1 Jet Mass
        bins_mass = ref_data["jet_mass_bins"]
        mh.histplot(
            ref_data["jet_mass_orig_counts"], bins=bins_mass, ax=axs_sub[0], 
            label="Original (Truth)", color="grey", histtype='fill', alpha=0.3
        )
        for j, data in enumerate(runs_data):
            mh.histplot(
                data["jet_mass_reco_counts"], bins=bins_mass, ax=axs_sub[0], 
                label=f"Reco ({run_labels[j]})", color=colors[j % len(colors)], 
                histtype='step', linewidth=2
            )
        axs_sub[0].set_xlabel("Jet Mass [GeV]")
        axs_sub[0].set_ylabel("Density")
        axs_sub[0].legend(prop={'size': 10})

        # 4.2 Jet Mass Difference
        bins_mass_diff = ref_data["jet_mass_diff_bins"]
        for j, data in enumerate(runs_data):
            mh.histplot(
                data["jet_mass_diff_counts"], bins=bins_mass_diff, ax=axs_sub[1], 
                label=run_labels[j], color=colors[j % len(colors)], 
                histtype='step', linewidth=2
            )
        axs_sub[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axs_sub[1].set_xlabel(r"Jet $m^{reco} - m^{orig}$ [GeV]")
        axs_sub[1].set_ylabel("Density")
        axs_sub[1].legend(prop={'size': 10})

        # 4.3 Tau32 Difference
        bins_tau_diff = ref_data["tau32_diff_bins"]
        for j, data in enumerate(runs_data):
            mh.histplot(
                data["tau32_diff_counts"], bins=bins_tau_diff, ax=axs_sub[2], 
                label=run_labels[j], color=colors[j % len(colors)], 
                histtype='step', linewidth=2
            )
        axs_sub[2].axvline(0, color='black', linestyle='--', alpha=0.5)
        axs_sub[2].set_xlabel(r"$\tau_{32}^{reco} - \tau_{32}^{orig}$")
        axs_sub[2].set_ylabel("Density")
        axs_sub[2].legend(prop={'size': 10})

        plt.tight_layout()
        fig_sub.savefig(os.path.join(output_dir, "combined_substructure.png"))
        plt.close(fig_sub)
        
    print(f"Successfully generated comparison plots in '{output_dir}/'")
