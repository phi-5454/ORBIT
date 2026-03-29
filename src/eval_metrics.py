import fastjet
import vector
import awkward as ak
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import mplhep as mh
import logging

from data_loading import PreprocessTranformer

class PhysicsEvaluator:
    def __init__(self, feature_names=["Eta", "Phi", "pT"]):
        self.feature_names = feature_names

    def evaluate_reconstruction(self, x, x_hat, mask):
        results = {}

        # Apply inverse transform
        x = PreprocessTranformer().inverse_tensor(x)
        x_hat = PreprocessTranformer().inverse_tensor(x_hat)

        # Extract only the REAL particles
        x_real = x[mask]
        x_hat_real = x_hat[mask]

        # Calculate true physical MSE per feature
        mse_per_feature = F.mse_loss(x_hat_real, x_real, reduction="none").mean(dim=0)

        for i, name in enumerate(self.feature_names):
            results[f"metrics/mse_{name.replace(' ', '_')}"] = mse_per_feature[i].item()
        
        # Convert to NumPy
        x_np = x_real.detach().cpu().numpy()
        x_hat_np = x_hat_real.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        x_np_tuple = x.detach().cpu().numpy()
        x_hat_np_tuple = x_hat.detach().cpu().numpy()

        jet_reco = EventJetReconstructor(R=0.8, min_jet_pt=0.0)
        
        # Setup lists for kinematics and substructure
        true_jet_pts, reco_jet_pts = [], []
        true_jet_masses, reco_jet_masses = [], []
        true_tau32s, reco_tau32s = [], []

        batch_size = x_np_tuple.shape[0]

        for i in range(batch_size):
            ev_mask = mask_np[i]
            
            ev_x_eta, ev_x_phi, ev_x_pt = x_np_tuple[i, ev_mask, 0], x_np_tuple[i, ev_mask, 1], x_np_tuple[i, ev_mask, 2]
            ev_xhat_eta, ev_xhat_phi, ev_xhat_pt = x_hat_np_tuple[i, ev_mask, 0], x_hat_np_tuple[i, ev_mask, 1], x_hat_np_tuple[i, ev_mask, 2]

            true_jets = jet_reco(ev_x_pt, ev_x_eta, ev_x_phi)
            reco_jets = jet_reco(ev_xhat_pt, ev_xhat_eta, ev_xhat_phi)

            # 1. Matching Inclusive pT
            n_match = min(len(true_jets["pt"]), len(reco_jets["pt"]))
            if n_match > 0:
                true_jet_pts.extend(true_jets["pt"][:n_match])
                reco_jet_pts.extend(reco_jets["pt"][:n_match])
                
            # 2. Extracting Global Substructure metrics (if valid event)
            if true_jets["jet_n_constituents"] >= 3 and reco_jets["jet_n_constituents"] >= 3:
                true_jet_masses.append(true_jets["jet_mass"])
                reco_jet_masses.append(reco_jets["jet_mass"])
                true_tau32s.append(true_jets["tau32"])
                reco_tau32s.append(reco_jets["tau32"])

        # ==========================================
        # APPLY MPLHEP STYLE GLOBALLY FOR THESE PLOTS
        # ==========================================
        mh.style.use(mh.style.ROOT)

        # ==========================================
        # 3. PLOTTING AND SAVING THE JET PT DIFFERENCES
        # ==========================================
        if len(true_jet_pts) > 0:
            true_jet_pts = np.array(true_jet_pts)
            reco_jet_pts = np.array(reco_jet_pts)

            fig, ax = plt.subplots(figsize=(8, 6))
            fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
            
            counts, bins = np.histogram(fractional_diff, bins=50, range=(-0.5, 0.5))
            
            results["histograms/jet_pt_resolution_counts"] = counts
            results["histograms/jet_pt_resolution_bins"] = bins
            
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
            feature_name = self.feature_names[i].replace(' ', '_')
            
            min_val = min(x_np[:, i].min(), x_hat_np[:, i].min())
            max_val = max(x_np[:, i].max(), x_hat_np[:, i].max())
            
            if i == 2:  # pT
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
                histtype='fill', 
                alpha=0.5, 
                edgecolor=["blue", "orange"]
            )

            axes[i].set_title(f"{self.feature_names[i]} (MSE: {mse_per_feature[i]:.4f})")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        plt.tight_layout()
        results["plots/kinematics"] = fig1

        # ==========================================
        # FIGURE 2: Energy / Momentum
        # ==========================================
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
            bins=log_bins, ax=axs[0],
            label=["Original", "Reconstructed"],
            color=["blue", "orange"], histtype='fill', alpha=0.5, edgecolor=["blue", "orange"]
        )
        axs[0].set_xscale("log")
        axs[0].set_ylabel("Density")
        axs[0].legend()

        # Plot 2: Residuals
        energy_mse = np.mean((energy_reco - energy_orig) ** 2)
        res_counts, res_bins = np.histogram(energy_reco - energy_orig, bins=50, density=True)

        results["histograms/energy_residuals_counts"] = res_counts
        results["histograms/energy_residuals_bins"] = res_bins

        axs[1].set_title(f"Residuals: E_reco - E_original (MSE: {energy_mse:.2f})", fontsize=14)
        mh.histplot(res_counts, bins=res_bins, ax=axs[1], color="green", histtype='fill', alpha=0.5, edgecolor="green")
        axs[1].set_ylabel("Density")

        plt.tight_layout()
        results["plots/energy_residuals"] = fig2

        # ==========================================
        # FIGURE 3: Jet Substructure (Mass & Tau32)
        # ==========================================
        if len(true_jet_masses) > 0:
            fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
            fig3.suptitle("FSQ-VAE: Jet Substructure", fontsize=16)

            true_jet_masses = np.array(true_jet_masses)
            reco_jet_masses = np.array(reco_jet_masses)
            true_tau32s = np.array(true_tau32s)
            reco_tau32s = np.array(reco_tau32s)

            # ---------------- Plot 3.1: Jet Mass ----------------
            min_mass = max(0, min(true_jet_masses.min(), reco_jet_masses.min()))
            max_mass = max(true_jet_masses.max(), reco_jet_masses.max())
            mass_bins = np.linspace(min_mass, max_mass, 50)

            counts_m_orig, _ = np.histogram(true_jet_masses, bins=mass_bins, density=True)
            counts_m_reco, _ = np.histogram(reco_jet_masses, bins=mass_bins, density=True)

            results["histograms/jet_mass_orig_counts"] = counts_m_orig
            results["histograms/jet_mass_reco_counts"] = counts_m_reco
            results["histograms/jet_mass_bins"] = mass_bins

            mh.histplot(
                [counts_m_orig, counts_m_reco], bins=mass_bins, ax=axs3[0],
                label=["Original", "Reconstructed"],
                color=["blue", "orange"], histtype='fill', alpha=0.5, edgecolor=["blue", "orange"]
            )
            axs3[0].set_xlabel("Jet Mass [GeV]")
            axs3[0].set_ylabel("Density")
            axs3[0].legend()

            # ---------------- Plot 3.2: Jet Mass Diff ----------------
            mass_diff = reco_jet_masses - true_jet_masses
            # Using percentiles to ignore crazy outliers stretching the axes
            diff_bins = np.linspace(np.percentile(mass_diff, 1), np.percentile(mass_diff, 99), 50)

            counts_mdiff, _ = np.histogram(mass_diff, bins=diff_bins, density=True)
            results["histograms/jet_mass_diff_counts"] = counts_mdiff
            results["histograms/jet_mass_diff_bins"] = diff_bins

            mh.histplot(counts_mdiff, bins=diff_bins, ax=axs3[1], histtype='fill', color='green', alpha=0.5, edgecolor="green")
            axs3[1].axvline(0, color='black', linestyle='--', alpha=0.5)
            axs3[1].set_xlabel(r"$m^{reco} - m^{orig}$ [GeV]")
            axs3[1].set_ylabel("Density")

            # ---------------- Plot 3.3: Tau32 Diff ----------------
            tau_diff = reco_tau32s - true_tau32s
            tau_bins = np.linspace(-1, 1, 50) # Tau ranges [0,1], diff must be [-1,1]

            counts_tdiff, _ = np.histogram(tau_diff, bins=tau_bins, density=True)
            results["histograms/tau32_diff_counts"] = counts_tdiff
            results["histograms/tau32_diff_bins"] = tau_bins

            mh.histplot(counts_tdiff, bins=tau_bins, ax=axs3[2], histtype='fill', color='purple', alpha=0.5, edgecolor="purple")
            axs3[2].axvline(0, color='black', linestyle='--', alpha=0.5)
            axs3[2].set_xlabel(r"$\tau_{32}^{reco} - \tau_{32}^{orig}$")
            axs3[2].set_ylabel("Density")

            plt.tight_layout()
            results["plots/jet_substructure"] = fig3

        return results


# Ensure vector behaviors are registered
vector.register_awkward()

def calc_deltaR(particles, jet):
    """Helper to calculate DeltaR between particles and a specific jet."""
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)

class EventJetReconstructor:
    def __init__(
        self, 
        R=0.8, 
        min_jet_pt=0.0, 
        max_jet_eta=None, 
        beta=1.0, 
        use_wta_pt_scheme=False
    ):
        """
        Initializes the FastJet evaluator, combining inclusive clustering 
        with exclusive substructure calculation.
        
        Args:
            R (float): Jet radius parameter (default 0.8 for AK8).
            min_jet_pt (float): Minimum pT threshold for inclusive jets.
            max_jet_eta (float, optional): Maximum absolute pseudo-rapidity for jets.
            beta (float): Beta parameter for N-subjettiness (default 1.0).
            use_wta_pt_scheme (bool): Whether to use WTA pt scheme for clustering.
        """
        self.R = R
        self.min_jet_pt = min_jet_pt
        self.max_jet_eta = max_jet_eta
        self.beta = beta
        self.use_wta_pt_scheme = use_wta_pt_scheme
        
        # Define algorithm once to save C++ initialization overhead
        if use_wta_pt_scheme:
            self.jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme)
        else:
            self.jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, self.R)

    def __call__(self, pt, eta, phi, particle_mask=None):
        """
        Clusters a single event's particles into inclusive jets AND calculates 
        substructure metrics treating the entire input as a single fatjet.
        
        Args:
            pt, eta, phi (array-like): 1D arrays of particle kinematics.
            particle_mask (array-like, optional): Boolean mask (e.g., puppi_weight > 0.05).
            
        Returns:
            Dictionary containing inclusive jet kinematics (arrays) 
            and global substructure metrics (scalars).
        """
        # 1. Standardize inputs and apply mask
        pt = np.asarray(pt, dtype=np.float64)
        eta = np.asarray(eta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        if particle_mask is not None:
            mask = np.asarray(particle_mask, dtype=bool)
            pt, eta, phi = pt[mask], eta[mask], phi[mask]

        # 2. Guard against events with < 3 particles (N-subjettiness requires 3 for tau_3)
        if len(pt) < 3:
            return self._empty_result()

        # 3. Clip extreme pT (Replacing custom ak_select_and_preprocess)
        if np.max(pt) > 1e9:
            logging.warning("Particle pT > 1e9 detected. Clipping to 1e9.")
            pt = np.clip(pt, a_min=0.0, a_max=1e9)

        # 4. Zip into Awkward 4-vectors
        # Note: We wrap arrays in [] to create a batch dimension of size 1. 
        # This is required for fastjet and awkward reductions (like ak.sum(..., axis=1)).
        particles = ak.zip(
            {
                "pt": [pt],
                "eta": [eta],
                "phi": [phi],
                "mass": [np.zeros_like(pt)]
            },
            with_name="Momentum4D"
        )

        # 5. Global Kinematics
        particles_sum = ak.sum(particles, axis=1)
        
        # 6. Cluster
        cluster = fastjet.ClusterSequence(particles, self.jetdef)
        
        # --- A. INCLUSIVE JETS ---
        inclusive_jets = cluster.inclusive_jets(min_pt=self.min_jet_pt)
        if self.max_jet_eta is not None:
            eta_mask = np.abs(inclusive_jets.eta) < self.max_jet_eta
            inclusive_jets = inclusive_jets[eta_mask]

        # --- B. SUBSTRUCTURE (Exclusive Jets) ---
        d2 = cluster.exclusive_jets_energy_correlator(njets=1, func="d2")
        exclusive_jets_1 = cluster.exclusive_jets(n_jets=1)
        exclusive_jets_2 = cluster.exclusive_jets(n_jets=2)
        exclusive_jets_3 = cluster.exclusive_jets(n_jets=3)

        # Calculate N-subjettiness
        d0 = ak.sum(particles.pt * self.R**self.beta, axis=1)
        
        # Tau 1
        dr_1i = calc_deltaR(particles, exclusive_jets_1[:, :1])
        tau1 = ak.sum(particles.pt * dr_1i**self.beta, axis=1) / d0

        # Tau 2
        dr_1i_t2 = calc_deltaR(particles, exclusive_jets_2[:, :1])
        dr_2i_t2 = calc_deltaR(particles, exclusive_jets_2[:, 1:2])
        min_dr_t2 = ak.min(
            ak.concatenate([dr_1i_t2[..., np.newaxis]**self.beta, dr_2i_t2[..., np.newaxis]**self.beta], axis=-1), 
            axis=-1
        )
        tau2 = ak.sum(particles.pt * min_dr_t2, axis=1) / d0

        # Tau 3
        dr_1i_t3 = calc_deltaR(particles, exclusive_jets_3[:, :1])
        dr_2i_t3 = calc_deltaR(particles, exclusive_jets_3[:, 1:2])
        dr_3i_t3 = calc_deltaR(particles, exclusive_jets_3[:, 2:3])
        min_dr_t3 = ak.min(
            ak.concatenate([
                dr_1i_t3[..., np.newaxis]**self.beta, 
                dr_2i_t3[..., np.newaxis]**self.beta, 
                dr_3i_t3[..., np.newaxis]**self.beta
            ], axis=-1), 
            axis=-1
        )
        tau3 = ak.sum(particles.pt * min_dr_t3, axis=1) / d0

        # Ratios (Adding 1e-8 for division safety)
        tau21 = tau2 / (tau1 + 1e-8)
        tau32 = tau3 / (tau2 + 1e-8)

        # 7. Unpack and Return
        # We index [0] to unwrap the single event from the dummy batch dimension
        return {
            # Kinematics of found jets (can be multiple, so left as arrays)
            "pt": np.asarray(inclusive_jets.pt[0]) if len(inclusive_jets[0]) > 0 else np.array([]),
            "eta": np.asarray(inclusive_jets.eta[0]) if len(inclusive_jets[0]) > 0 else np.array([]),
            "phi": np.asarray(inclusive_jets.phi[0]) if len(inclusive_jets[0]) > 0 else np.array([]),
            
            # Global properties and Substructure (Scalars calculated over the whole point cloud)
            "jet_mass": float(particles_sum.mass[0]),
            "jet_pt": float(particles_sum.pt[0]),
            "jet_eta": float(particles_sum.eta[0]),
            "jet_phi": float(particles_sum.phi[0]),
            "jet_n_constituents": len(pt),
            "tau1": float(tau1[0]),
            "tau2": float(tau2[0]),
            "tau3": float(tau3[0]),
            "tau21": float(tau21[0]),
            "tau32": float(tau32[0]),
            "d2": float(d2[0]) if len(d2) > 0 else 0.0,
        }

    def _empty_result(self):
        """Returns safe default values for empty or rejected events."""
        return {
            "pt": np.array([]), "eta": np.array([]), "phi": np.array([]),
            "jet_mass": 0.0, "jet_pt": 0.0, "jet_eta": 0.0, "jet_phi": 0.0, 
            "jet_n_constituents": 0,
            "tau1": 0.0, "tau2": 0.0, "tau3": 0.0, "tau21": 0.0, "tau32": 0.0, "d2": 0.0
        }
