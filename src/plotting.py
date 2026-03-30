import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

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
    fig_kin.suptitle("Kinematics: Original vs. Reconstructed Sweeps", fontsize=14)

    
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
