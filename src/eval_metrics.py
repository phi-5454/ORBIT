import fastjet
import vector
import awkward as ak
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import mplhep as mh

from data_loading import PreprocessTranformer


# Model evaluation
class PhysicsEvaluator:
    def __init__(self, feature_names=["Eta", "Phi", "pT"]):
        self.feature_names = feature_names

    def evaluate_reconstruction(self, x, x_hat, mask):
        """
        Takes raw tensors, filters out padding, and returns a dict of metrics and figures.
        """
        results = {}

        # TODO: P_t_LOG -> P_t!!!
        # Apply inverse transform
        x = PreprocessTranformer().inverse_tensor(x)
        x_hat = PreprocessTranformer().inverse_tensor(x_hat)

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
        x_np = x_real.detach().cpu().numpy()
        x_hat_np = x_hat_real.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        x_np_tuple = x.detach().cpu().numpy()
        x_hat_np_tuple = x_hat.detach().cpu().numpy()

        jet_reco = EventJetReconstructor(R=0.8, min_jet_pt=0.0)
        true_jet_pts = []
        reco_jet_pts = []

        batch_size = x_np_tuple.shape[0]

        # Iterate through the batch event-by-event
        for i in range(batch_size):
            ev_mask = mask_np[i]
            
            # Apply the mask to extract ONLY real particles for this specific event
            # (indices: 0=eta, 1=phi, 2=pt)
            ev_x_eta, ev_x_phi, ev_x_pt = x_np_tuple[i, ev_mask, 0], x_np_tuple[i, ev_mask, 1], x_np_tuple[i, ev_mask, 2]
            ev_xhat_eta, ev_xhat_phi, ev_xhat_pt = x_hat_np_tuple[i, ev_mask, 0], x_hat_np_tuple[i, ev_mask, 1], x_hat_np_tuple[i, ev_mask, 2]

            # TODO: Make work for events that are split into many parts
            # Cluster the True and Reconstructed Jets
            true_jets = jet_reco(ev_x_pt, ev_x_eta, ev_x_phi)
            reco_jets = jet_reco(ev_xhat_pt, ev_xhat_eta, ev_xhat_phi)

            # Match by pT ordering: FastJet already sorts them descending by default!
            n_match = min(len(true_jets["pt"]), len(reco_jets["pt"]))
            
            if n_match > 0:
                true_jet_pts.extend(true_jets["pt"][:n_match])
                reco_jet_pts.extend(reco_jets["pt"][:n_match])

        # ==========================================
        # APPLY MPLHEP STYLE GLOBALLY FOR THESE PLOTS
        # ==========================================
        mh.style.use(mh.style.ROOT)

        # ==========================================
        # 3. PLOTTING THE JET DIFFERENCES
        # ==========================================
        if len(true_jet_pts) > 0:
            true_jet_pts = np.array(true_jet_pts)
            reco_jet_pts = np.array(reco_jet_pts)

            fig, ax = plt.subplots(figsize=(8, 6))
            
            fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
            
            # Use NumPy to calculate bins, then mplhep to plot
            counts, bins = np.histogram(fractional_diff, bins=50, range=(-0.5, 0.5))
            mh.histplot(counts, bins=bins, ax=ax, histtype='step', color='indigo', linewidth=2)
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel(r"Fractional $p_T$ Resolution: $(p_T^{reco} - p_T^{true}) / p_T^{true}$")
            ax.set_ylabel("Number of Jets")
            ax.set_title("Jet Transverse Momentum Recovery")
            
            results["plots/jet_pt_resolution"] = fig

        # ==========================================
        # FIGURE 1: Kinematic Features
        # ==========================================
        fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig1.suptitle("FSQ-VAE: Original vs. Reconstructed Features (per-particle)", fontsize=16)
        
        for i in range(3):
            # Calculate explicit shared bins to align the Original and Reco perfectly
            min_val = min(x_np[:, i].min(), x_hat_np[:, i].min())
            max_val = max(x_np[:, i].max(), x_hat_np[:, i].max())
            
            if i == 2:  # pT needs log bins
                min_val = max(min_val, 1e-8)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
                axes[i].set_xscale("log")
            else:
                bins = np.linspace(min_val, max_val, 50)

            # Histogram the data (density=True replaces seaborn's stat="density")
            counts_orig, _ = np.histogram(x_np[:, i], bins=bins, density=True)
            counts_reco, _ = np.histogram(x_hat_np[:, i], bins=bins, density=True)

            # Plot both simultaneously using mplhep
            mh.histplot(
                [counts_orig, counts_reco], 
                bins=bins, 
                ax=axes[i],
                label=["Original", "Reconstructed"],
                color=["blue", "orange"],
                histtype='fill', 
                alpha=0.5, 
                edgecolor=["blue", "orange"] # Keeps the step outlines sharp
            )

            axes[i].set_title(f"{self.feature_names[i]} (MSE: {mse_per_feature[i]:.4f})")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        plt.tight_layout()
        results["plots/kinematics"] = fig1

        # ==========================================
        # FIGURE 2: Energy / Momentum (Mass assuming m=0)
        # ==========================================
        fig2, axs = plt.subplots(1, 2, figsize=(16, 5))

        pt_orig = x_np[:, 2]
        pt_reco = x_hat_np[:, 2]

        energy_orig = pt_orig * np.cosh(x_np[:, 0])
        energy_reco = pt_reco * np.cosh(x_hat_np[:, 0])

        min_val = max(min(energy_orig.min(), energy_reco.min()), 1e-8)
        max_val = max(energy_orig.max(), energy_reco.max())
        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)

        # Plot 1: Energy Distribution
        axs[0].set_title("FSQ-VAE: Original vs. Reconstructed Energy (m=0)", fontsize=14)
        
        counts_e_orig, _ = np.histogram(energy_orig, bins=log_bins, density=True)
        counts_e_reco, _ = np.histogram(energy_reco, bins=log_bins, density=True)
        
        mh.histplot(
            [counts_e_orig, counts_e_reco],
            bins=log_bins,
            ax=axs[0],
            label=["Original", "Reconstructed"],
            color=["blue", "orange"],
            histtype='fill',
            alpha=0.5,
            edgecolor=["blue", "orange"]
        )
        axs[0].set_xscale("log")
        axs[0].set_ylabel("Density")
        axs[0].legend()

        # Plot 2: Residuals (Reco - Orig)
        energy_mse = np.mean((energy_reco - energy_orig) ** 2)
        axs[1].set_title(f"Residuals: E_reco - E_original (MSE: {energy_mse:.2f})", fontsize=14)
        
        res_counts, res_bins = np.histogram(energy_reco - energy_orig, bins=50, density=True)
        mh.histplot(
            res_counts, 
            bins=res_bins, 
            ax=axs[1], 
            color="green", 
            histtype='fill', 
            alpha=0.5,
            edgecolor="green"
        )
        axs[1].set_ylabel("Density")

        plt.tight_layout()
        results["plots/energy_residuals"] = fig2

        return results

class EventJetReconstructor:
    def __init__(self, R=0.8, min_jet_pt=200.0, max_jet_eta=None):
        """
        Initializes the FastJet evaluator.
        
        Args:
            R (float): Jet radius parameter (default 0.8 for AK8).
            min_jet_pt (float): Minimum pT threshold for reconstructed jets.
            max_jet_eta (float, optional): Maximum absolute pseudo-rapidity for jets.
        """
        # Register vector behaviors once
        vector.register_awkward()
        
        # Define algorithm once to save C++ initialization overhead
        self.jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        self.min_jet_pt = min_jet_pt
        self.max_jet_eta = max_jet_eta

    def __call__(self, pt, eta, phi, particle_mask=None):
        """
        Clusters a single event's particles into jets.
        
        Args:
            pt, eta, phi (array-like): 1D arrays of particle kinematics.
            particle_mask (array-like, optional): Boolean mask (e.g., puppi_weight > 0.05).
            
        Returns:
            Dictionary containing 1D numpy arrays of the jet 'pt', 'eta', and 'phi'.
        """
        # 1. Standardize inputs to flat numpy arrays (float64 required by FastJet)
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        # TODO: Define the PUPPI mask
        # 2. Apply particle-level cuts (like PUPPI weights) if provided
        if particle_mask is not None:
            mask = np.asarray(particle_mask, dtype=bool)
            pt, eta, phi = pt[mask], eta[mask], phi[mask]

        # 3. Guard against empty events (FastJet will crash on empty sequences)
        if len(pt) == 0:
            return {"pt": np.array([]), "eta": np.array([]), "phi": np.array([])}

        # 4. Zip into Awkward 4-vectors (assuming massless particles)
        particles = ak.zip(
            {
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": np.zeros_like(pt) 
            },
            with_name="Momentum4D"
        )

        # 5. Cluster
        cluster_sequence = fastjet.ClusterSequence(particles, self.jetdef)
        jets = cluster_sequence.inclusive_jets(min_pt=self.min_jet_pt)

        # 6. Apply Jet Eta Cut (if specified)
        if self.max_jet_eta is not None:
            eta_mask = np.abs(jets.eta) < self.max_jet_eta
            jets = jets[eta_mask]

        # 7. Return clean numpy arrays
        return {
            "pt": np.asarray(jets.pt),
            "eta": np.asarray(jets.eta),
            "phi": np.asarray(jets.phi)
        }


