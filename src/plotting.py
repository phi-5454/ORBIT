import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

def replot_jet_structure(output_dir="", output_filename="combined_substructure.png", input_dirs=[""]):
    # Apply standard HEP styling
    mh.style.use(mh.style.ROOT)

    # Find all the run directories (e.g., fsq_10, fsq_21, fsq_42)
    run_dirs = sorted(glob.glob(os.path.join(base_dir, "fsq_*")))
    
    if not run_dirs:
        print(f"No run directories found in {base_dir}")
        return

    # Setup the 3-panel figure
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("FSQ-VAE Parameter Sweep: Jet Substructure", fontsize=20, y=1.05)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e'] # Distinct colors
    plotted_original = False

    for idx, run_dir in enumerate(run_dirs):
        run_name = os.path.basename(run_dir) # e.g., "fsq_21"
        
        # Find the latest .npz file in this run's saved_histograms folder
        npz_files = sorted(glob.glob(os.path.join(run_dir, "saved_histograms", "*.npz")))
        if not npz_files:
            print(f"Skipping {run_name} (No .npz files found)")
            continue
            
        latest_npz = npz_files[-1]
        data = np.load(latest_npz)
        color = colors[idx % len(colors)]

        # ==========================================
        # 1. Jet Mass (Overlaying Reco, Plotting Orig once)
        # ==========================================
        if not plotted_original and "jet_mass_orig_counts" in data:
            mh.histplot(
                data["jet_mass_orig_counts"], 
                bins=data["jet_mass_bins"], 
                ax=axs[0],
                label="True Jet Mass",
                color="grey", 
                histtype='fill', 
                alpha=0.3
            )
            plotted_original = True
        
        if "jet_mass_reco_counts" in data:
            mh.histplot(
                data["jet_mass_reco_counts"], 
                bins=data["jet_mass_bins"], 
                ax=axs[0],
                label=f"Reco ({run_name})",
                color=color, 
                histtype='step', 
                linewidth=2.5
            )

        # ==========================================
        # 2. Jet Mass Difference
        # ==========================================
        if "jet_mass_diff_counts" in data:
            mh.histplot(
                data["jet_mass_diff_counts"], 
                bins=data["jet_mass_diff_bins"], 
                ax=axs[1],
                label=f"{run_name}",
                color=color, 
                histtype='step', 
                linewidth=2.5
            )

        # ==========================================
        # 3. Tau32 Difference
        # ==========================================
        if "tau32_diff_counts" in data:
            mh.histplot(
                data["tau32_diff_counts"], 
                bins=data["tau32_diff_bins"], 
                ax=axs[2],
                label=f"{run_name}",
                color=color, 
                histtype='step', 
                linewidth=2.5
            )

    # --- Formatting Plot 1: Jet Mass ---
    axs[0].set_xlabel("Jet Mass [GeV]")
    axs[0].set_ylabel("Density")
    axs[0].legend(loc="upper right")

    # --- Formatting Plot 2: Mass Difference ---
    axs[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[1].set_xlabel(r"Jet Mass Difference: $M^{reco} - M^{orig}$ [GeV]")
    axs[1].set_ylabel("Density")
    axs[1].legend(loc="upper right")

    # --- Formatting Plot 3: Tau32 Difference ---
    axs[2].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[2].set_xlabel(r"$\tau_{32}$ Difference: $\tau_{32}^{reco} - \tau_{32}^{orig}$")
    axs[2].set_ylabel("Density")
    axs[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Successfully saved superimposed plots to {output_filename}")

if __name__ == "__main__":
    # Point this to where your Condor jobs dropped their folders
    plot_sweep_results(base_dir="WANTED_OUTPUT_DIR")
