import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import awkward as ak
import fastjet
import numpy as np
import torch.nn.functional as F
import vector

try:
    from .data_loading import PreprocessTranformer
    from .plotting import add_reconstruction_plots
except ImportError:
    from data_loading import PreprocessTranformer
    from plotting import add_reconstruction_plots

_WORKER_JET_RECO = None


def _profile_eval(label, start_time):
    if os.environ.get("ORBIT_PROFILE_EVAL", "0") == "1":
        elapsed = time.perf_counter() - start_time
        print(f"[ORBIT_PROFILE_EVAL] {label}: {elapsed:.3f}s", flush=True)
    return time.perf_counter()


def _init_jet_reco_worker(R, min_jet_pt):
    global _WORKER_JET_RECO
    _WORKER_JET_RECO = EventJetReconstructor(R=R, min_jet_pt=min_jet_pt)


def _particle_jet_metrics_one_event(args):
    x_event, x_hat_event, ev_mask = args

    jet_reco = _WORKER_JET_RECO
    if jet_reco is None:
        jet_reco = EventJetReconstructor(R=0.8, min_jet_pt=0.0)

    ev_x_eta = x_event[ev_mask, 0]
    ev_x_phi = x_event[ev_mask, 1]
    ev_x_pt = x_event[ev_mask, 2]
    ev_xhat_eta = x_hat_event[ev_mask, 0]
    ev_xhat_phi = x_hat_event[ev_mask, 1]
    ev_xhat_pt = x_hat_event[ev_mask, 2]

    true_jets = jet_reco(ev_x_pt, ev_x_eta, ev_x_phi)
    reco_jets = jet_reco(ev_xhat_pt, ev_xhat_eta, ev_xhat_phi)

    true_jet_pts, reco_jet_pts = [], []
    true_jet_masses, reco_jet_masses = [], []
    true_tau32s, reco_tau32s = [], []

    n_match = min(len(true_jets["pt"]), len(reco_jets["pt"]))
    if n_match > 0:
        true_jet_pts.extend(true_jets["pt"][:n_match])
        reco_jet_pts.extend(reco_jets["pt"][:n_match])

    if true_jets["jet_n_constituents"] >= 3 and reco_jets["jet_n_constituents"] >= 3:
        true_jet_masses.append(true_jets["jet_mass"])
        reco_jet_masses.append(reco_jets["jet_mass"])
        true_tau32s.append(true_jets["tau32"])
        reco_tau32s.append(reco_jets["tau32"])

    return (
        true_jet_pts,
        reco_jet_pts,
        true_jet_masses,
        reco_jet_masses,
        true_tau32s,
        reco_tau32s,
    )


class PhysicsEvaluator:
    def __init__(
        self,
        feature_names=["Eta", "Phi", "pT"],
        data_level="particle",
        jet_metric_workers=0,
        jet_metric_backend="process",
        jet_metric_start_method="forkserver",
    ):
        self.feature_names = feature_names
        self.data_level = data_level
        self.jet_metric_workers = int(jet_metric_workers)
        self.jet_metric_backend = jet_metric_backend
        self.jet_metric_start_method = jet_metric_start_method
        if self.jet_metric_backend not in ("process", "thread"):
            raise ValueError(
                f"Unsupported eval worker backend: {self.jet_metric_backend}"
            )
        if self.jet_metric_start_method not in ("fork", "forkserver", "spawn"):
            raise ValueError(
                f"Unsupported eval worker start method: {self.jet_metric_start_method}"
            )

    def _collect_direct_jet_metrics(self, x_np_tuple, x_hat_np_tuple, mask_np):
        true_jet_pts, reco_jet_pts = [], []
        batch_size = x_np_tuple.shape[0]

        for i in range(batch_size):
            ev_mask = mask_np[i]
            ev_x_pt = x_np_tuple[i, ev_mask, 2]
            ev_xhat_pt = x_hat_np_tuple[i, ev_mask, 2]

            n_match = min(len(ev_x_pt), len(ev_xhat_pt))
            if n_match > 0:
                true_jet_pts.extend(ev_x_pt[:n_match])
                reco_jet_pts.extend(ev_xhat_pt[:n_match])

        return true_jet_pts, reco_jet_pts, [], [], [], []

    def _collect_particle_jet_metrics(self, x_np_tuple, x_hat_np_tuple, mask_np):
        true_jet_pts, reco_jet_pts = [], []
        true_jet_masses, reco_jet_masses = [], []
        true_tau32s, reco_tau32s = [], []

        batch_size = x_np_tuple.shape[0]

        if self.jet_metric_workers <= 1 or self.jet_metric_backend == "thread":
            _init_jet_reco_worker(0.8, 0.0)

        if self.jet_metric_workers > 1 and batch_size > 1:
            event_args = [
                (x_np_tuple[i], x_hat_np_tuple[i], mask_np[i])
                for i in range(batch_size)
            ]
            executor_cls = (
                ThreadPoolExecutor
                if self.jet_metric_backend == "thread"
                else ProcessPoolExecutor
            )
            executor_kwargs = {"max_workers": self.jet_metric_workers}
            if executor_cls is ProcessPoolExecutor:
                executor_kwargs.update(
                    {
                        "initializer": _init_jet_reco_worker,
                        "initargs": (0.8, 0.0),
                        "mp_context": mp.get_context(
                            self.jet_metric_start_method
                        ),
                    }
            )

            t0 = time.perf_counter()
            with executor_cls(**executor_kwargs) as executor:
                t0 = _profile_eval("eval jet metric pool start", t0)
                for result in executor.map(
                    _particle_jet_metrics_one_event, event_args
                ):
                    (
                        ev_true_jet_pts,
                        ev_reco_jet_pts,
                        ev_true_jet_masses,
                        ev_reco_jet_masses,
                        ev_true_tau32s,
                        ev_reco_tau32s,
                    ) = result
                    true_jet_pts.extend(ev_true_jet_pts)
                    reco_jet_pts.extend(ev_reco_jet_pts)
                    true_jet_masses.extend(ev_true_jet_masses)
                    reco_jet_masses.extend(ev_reco_jet_masses)
                    true_tau32s.extend(ev_true_tau32s)
                    reco_tau32s.extend(ev_reco_tau32s)
                t0 = _profile_eval("eval jet metric pool map", t0)

            _profile_eval("eval jet metric pool shutdown", t0)

            return (
                true_jet_pts,
                reco_jet_pts,
                true_jet_masses,
                reco_jet_masses,
                true_tau32s,
                reco_tau32s,
            )

        for i in range(batch_size):
            result = _particle_jet_metrics_one_event(
                (x_np_tuple[i], x_hat_np_tuple[i], mask_np[i])
            )
            true_jet_pts.extend(result[0])
            reco_jet_pts.extend(result[1])
            true_jet_masses.extend(result[2])
            reco_jet_masses.extend(result[3])
            true_tau32s.extend(result[4])
            reco_tau32s.extend(result[5])

        return (
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
        )

    def evaluate_reconstruction(self, x, x_hat, mask):
        results = {}
        t0 = time.perf_counter()

        # Apply inverse transform
        x = PreprocessTranformer().inverse_tensor(x)
        x_hat = PreprocessTranformer().inverse_tensor(x_hat)
        t0 = _profile_eval("eval inverse transform", t0)

        # Extract only the REAL particles
        x_real = x[mask]
        x_hat_real = x_hat[mask]

        # Calculate true physical MSE per feature
        mse_per_feature = F.mse_loss(x_hat_real, x_real, reduction="none").mean(dim=0)

        for i, name in enumerate(self.feature_names):
            results[f"metrics/mse_{name.replace(' ', '_')}"] = mse_per_feature[i].item()
        t0 = _profile_eval("eval mse", t0)

        # Convert to NumPy
        x_np = x_real.detach().cpu().numpy()
        x_hat_np = x_hat_real.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        x_np_tuple = x.detach().cpu().numpy()
        x_hat_np_tuple = x_hat.detach().cpu().numpy()
        t0 = _profile_eval("eval tensor to numpy", t0)

        if self.data_level == "jet":
            metric_inputs = self._collect_direct_jet_metrics(
                x_np_tuple, x_hat_np_tuple, mask_np
            )
        else:
            metric_inputs = self._collect_particle_jet_metrics(
                x_np_tuple, x_hat_np_tuple, mask_np
            )
        t0 = _profile_eval("eval collect jet metrics", t0)

        (
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
        ) = metric_inputs

        add_reconstruction_plots(
            results,
            self.feature_names,
            mse_per_feature,
            x_np,
            x_hat_np,
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
        )
        _profile_eval("eval build reconstruction plots", t0)

        return results


# Ensure vector behaviors are registered
vector.register_awkward()


def calc_deltaR(particles, jet):
    """Helper to calculate DeltaR between particles and a specific jet."""
    jet = ak.unflatten(ak.flatten(jet), counts=1)
    return particles.deltaR(jet)


class EventJetReconstructor:
    def __init__(
        self, R=0.8, min_jet_pt=0.0, max_jet_eta=None, beta=1.0, use_wta_pt_scheme=False
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
            self.jetdef = fastjet.JetDefinition(
                fastjet.kt_algorithm, self.R, fastjet.WTA_pt_scheme
            )
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
            {"pt": [pt], "eta": [eta], "phi": [phi], "mass": [np.zeros_like(pt)]},
            with_name="Momentum4D",
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
            ak.concatenate(
                [
                    dr_1i_t2[..., np.newaxis] ** self.beta,
                    dr_2i_t2[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        tau2 = ak.sum(particles.pt * min_dr_t2, axis=1) / d0

        # Tau 3
        dr_1i_t3 = calc_deltaR(particles, exclusive_jets_3[:, :1])
        dr_2i_t3 = calc_deltaR(particles, exclusive_jets_3[:, 1:2])
        dr_3i_t3 = calc_deltaR(particles, exclusive_jets_3[:, 2:3])
        min_dr_t3 = ak.min(
            ak.concatenate(
                [
                    dr_1i_t3[..., np.newaxis] ** self.beta,
                    dr_2i_t3[..., np.newaxis] ** self.beta,
                    dr_3i_t3[..., np.newaxis] ** self.beta,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        tau3 = ak.sum(particles.pt * min_dr_t3, axis=1) / d0

        # Ratios (Adding 1e-8 for division safety)
        tau21 = tau2 / (tau1 + 1e-8)
        tau32 = tau3 / (tau2 + 1e-8)

        # 7. Unpack and Return
        # We index [0] to unwrap the single event from the dummy batch dimension
        return {
            # Kinematics of found jets (can be multiple, so left as arrays)
            "pt": (
                np.asarray(inclusive_jets.pt[0])
                if len(inclusive_jets[0]) > 0
                else np.array([])
            ),
            "eta": (
                np.asarray(inclusive_jets.eta[0])
                if len(inclusive_jets[0]) > 0
                else np.array([])
            ),
            "phi": (
                np.asarray(inclusive_jets.phi[0])
                if len(inclusive_jets[0]) > 0
                else np.array([])
            ),
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
            "pt": np.array([]),
            "eta": np.array([]),
            "phi": np.array([]),
            "jet_mass": 0.0,
            "jet_pt": 0.0,
            "jet_eta": 0.0,
            "jet_phi": 0.0,
            "jet_n_constituents": 0,
            "tau1": 0.0,
            "tau2": 0.0,
            "tau3": 0.0,
            "tau21": 0.0,
            "tau32": 0.0,
            "d2": 0.0,
        }
