import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
import torch
from matplotlib.lines import Line2D

# =============================================================================
# Plot style macros
# =============================================================================

ORIGINAL_COLOR = "#0000ff"
RECONSTRUCTED_COLOR = ORIGINAL_COLOR
TRUTH_REFERENCE_COLOR = "grey"
RESIDUAL_COLOR = ORIGINAL_COLOR
RUN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
CODEBOOK_FAMILY_COLORS = {
    "fsq": "#1f77b4",
    "vq_ste": "#ff7f0e",
    "vq_rotation": "#2ca02c",
}
CODEBOOK_FAMILY_MARKERS = {
    "fsq": "o",
    "vq_ste": "s",
    "vq_rotation": "^",
}
CODEBOOK_FAMILY_LABELS = {
    "fsq": "FSQ",
    "vq_ste": "VQ STE",
    "vq_rotation": "VQ rotation",
}
SCATTER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p", "H")
HISTOGRAM_LINEWIDTH = 2
HISTOGRAM_FILL_ALPHA = 0.5
TRUTH_REFERENCE_FILL_ALPHA = 0.35
SCATTER_LINE_ALPHA = 0.55
SCATTER_POINT_ALPHA = 0.8
SCATTER_LINEWIDTH = 1.5
REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_STYLE = "--"
REFERENCE_LINE_ALPHA = 0.5
FEATURE_FIGSIZE = (21, 8)
ENERGY_FIGSIZE = (16, 5)
RESOLUTION_FIGSIZE = (8, 6)
SUBSTRUCTURE_FIGSIZE = (21, 8)
SUBSTRUCTURE_SIMPLE_FIGSIZE = (18, 8)
SUBSTRUCTURE_COMPARISON_FIGSIZE = (21, 7)
ATTENTION_DELTA_FIGSIZE = (6, 5)
ATTENTION_MAP_FIGSIZE = (5, 4)
SCATTER_FIGSIZE = (8, 6)
MONEY_TRIPLET_FIGSIZE = (21, 6)
RATIO_HEIGHT_RATIOS = (3, 1)
RATIO_HSPACE = 0.08
RATIO_WSPACE = 0.25
RATIO_YLIM = (0.5, 1.5)
UTILIZATION_YLIM = (0.0, 1.05)
KINEMATIC_LABELS = {
    "pT": r"$p_T$",
    "Eta": r"$\eta$",
    "Phi": r"$\phi$",
}
KINEMATIC_RESIDUAL_LABELS = {
    "pT": r"$p_T^{reco} - p_T^{orig}$",
    "Eta": r"$\eta^{reco} - \eta^{orig}$",
    "Phi": r"$\phi^{reco} - \phi^{orig}$",
}
PLOT_TITLES_ENABLED = True


# =============================================================================
# Shared plot styling and layout helpers
# =============================================================================


def set_plot_titles_enabled(enabled):
    global PLOT_TITLES_ENABLED
    PLOT_TITLES_ENABLED = enabled


def _set_title(ax, title, **kwargs):
    if PLOT_TITLES_ENABLED:
        ax.set_title(title, **kwargs)


def _set_suptitle(fig, title, **kwargs):
    if PLOT_TITLES_ENABLED:
        fig.suptitle(title, **kwargs)


def _plot_filled_histogram(
    ax,
    bins,
    counts,
    label,
    color=ORIGINAL_COLOR,
    alpha=HISTOGRAM_FILL_ALPHA,
):
    mh.histplot(
        counts,
        bins=bins,
        ax=ax,
        label=label,
        color=color,
        histtype="fill",
        alpha=alpha,
        edgecolor="none",
    )


def _plot_single_histogram(ax, bins, counts, label=None, color=ORIGINAL_COLOR):
    mh.histplot(
        counts,
        bins=bins,
        ax=ax,
        label=label,
        color=color,
        histtype="step",
        linewidth=HISTOGRAM_LINEWIDTH,
    )


def _plot_original_reconstructed_histograms(ax, bins, original, reconstructed):
    _plot_filled_histogram(
        ax,
        bins,
        original,
        label="Original",
        color=ORIGINAL_COLOR,
    )
    _plot_single_histogram(
        ax,
        bins,
        reconstructed,
        label="Reconstructed",
        color=RECONSTRUCTED_COLOR,
    )


def _plot_ratio_histogram(
    ax,
    bins,
    numerator,
    denominator,
    label=None,
    color=RECONSTRUCTED_COLOR,
):
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    _plot_single_histogram(ax, bins, ratio, label=label, color=color)


def _triplet_axes(fig, ratio_indices=(0,)):
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = []
    ratio_axes = {}
    for i in range(3):
        if i in ratio_indices:
            axis = fig.add_subplot(grid[0, i])
            ratio_axes[i] = fig.add_subplot(grid[1, i], sharex=axis)
        else:
            axis = fig.add_subplot(grid[:, i])
        axes.append(axis)
    return axes, ratio_axes


def _pair_axes(fig, with_first_ratio=False):
    if not with_first_ratio:
        return fig.subplots(1, 2), {}

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=RATIO_HEIGHT_RATIOS,
        hspace=RATIO_HSPACE,
        wspace=RATIO_WSPACE,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[:, 1]),
    ]
    return axes, {0: fig.add_subplot(grid[1, 0], sharex=axes[0])}


def _configure_ratio_axis(ax, xlabel):
    ax.axhline(
        1.0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.set_ylim(*RATIO_YLIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reco / orig")


def _adjust_ratio_layout(fig):
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.88)


def _validate_data_level(data_level):
    if data_level not in ("particle", "jet"):
        raise ValueError(f"Unsupported data level: {data_level}")


def _clean_feature_name(feature_name):
    return feature_name.replace(" ", "_")


def collect_reconstruction_histograms(
    feature_names,
    x_np,
    x_hat_np,
    true_jet_pts=(),
    reco_jet_pts=(),
    true_jet_masses=(),
    reco_jet_masses=(),
    true_tau32s=(),
    reco_tau32s=(),
    true_missing_ets=(),
    reco_missing_ets=(),
    data_level="particle",
):
    _validate_data_level(data_level)
    histograms = {}

    if len(true_jet_pts) > 0:
        true_jet_pts = np.asarray(true_jet_pts)
        reco_jet_pts = np.asarray(reco_jet_pts)
        fractional_diff = (reco_jet_pts - true_jet_pts) / (true_jet_pts + 1e-8)
        counts, bins = np.histogram(fractional_diff, bins=50, range=(-0.5, 0.5))
        histograms["jet_pt_resolution_counts"] = counts
        histograms["jet_pt_resolution_bins"] = bins

    for i, feature_name in enumerate(feature_names):
        min_val = min(x_np[:, i].min(), x_hat_np[:, i].min())
        max_val = max(x_np[:, i].max(), x_hat_np[:, i].max())
        if i == 2:
            min_val = max(min_val, 1e-8)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
        else:
            bins = np.linspace(min_val, max_val, 50)

        clean_name = _clean_feature_name(feature_name)
        histograms[f"{clean_name}_orig_counts"] = np.histogram(
            x_np[:, i], bins=bins, density=True
        )[0]
        histograms[f"{clean_name}_reco_counts"] = np.histogram(
            x_hat_np[:, i], bins=bins, density=True
        )[0]
        histograms[f"{clean_name}_bins"] = bins
        diff_counts, diff_bins = np.histogram(
            x_hat_np[:, i] - x_np[:, i], bins=50, density=True
        )
        histograms[f"{clean_name}_diff_counts"] = diff_counts
        histograms[f"{clean_name}_diff_bins"] = diff_bins

    energy_orig = x_np[:, 2] * np.cosh(x_np[:, 0])
    energy_reco = x_hat_np[:, 2] * np.cosh(x_hat_np[:, 0])
    min_val = max(min(energy_orig.min(), energy_reco.min()), 1e-8)
    max_val = max(energy_orig.max(), energy_reco.max())
    energy_bins = np.logspace(np.log10(min_val), np.log10(max_val), num=50)
    histograms["energy_orig_counts"] = np.histogram(
        energy_orig, bins=energy_bins, density=True
    )[0]
    histograms["energy_reco_counts"] = np.histogram(
        energy_reco, bins=energy_bins, density=True
    )[0]
    histograms["energy_bins"] = energy_bins
    histograms["energy_residuals_counts"], histograms["energy_residuals_bins"] = (
        np.histogram(energy_reco - energy_orig, bins=50, density=True)
    )

    if data_level == "particle" and len(true_missing_ets) > 0:
        true_missing_ets = np.asarray(true_missing_ets)
        reco_missing_ets = np.asarray(reco_missing_ets)
        max_missing_et = max(true_missing_ets.max(), reco_missing_ets.max(), 1e-8)
        missing_et_bins = np.linspace(0, max_missing_et, 50)
        histograms["missing_et_orig_counts"] = np.histogram(
            true_missing_ets, bins=missing_et_bins, density=True
        )[0]
        histograms["missing_et_reco_counts"] = np.histogram(
            reco_missing_ets, bins=missing_et_bins, density=True
        )[0]
        histograms["missing_et_bins"] = missing_et_bins

    # Tau32 and reconstructed fat-jet mass are available only when clustering particles.
    if data_level == "particle" and len(true_jet_masses) > 0:
        true_jet_masses = np.asarray(true_jet_masses)
        reco_jet_masses = np.asarray(reco_jet_masses)
        true_tau32s = np.asarray(true_tau32s)
        reco_tau32s = np.asarray(reco_tau32s)

        mass_bins = np.linspace(0, 600, 50)
        histograms["jet_mass_orig_counts"] = np.histogram(
            true_jet_masses, bins=mass_bins, density=True
        )[0]
        histograms["jet_mass_reco_counts"] = np.histogram(
            reco_jet_masses, bins=mass_bins, density=True
        )[0]
        histograms["jet_mass_bins"] = mass_bins

        mass_diff_bins = np.linspace(-50, 50, 50)
        histograms["jet_mass_diff_counts"] = np.histogram(
            reco_jet_masses - true_jet_masses, bins=mass_diff_bins, density=True
        )[0]
        histograms["jet_mass_diff_bins"] = mass_diff_bins

        tau_diff_bins = np.linspace(-0.4, 0.4, 50)
        histograms["tau32_diff_counts"] = np.histogram(
            reco_tau32s - true_tau32s, bins=tau_diff_bins, density=True
        )[0]
        histograms["tau32_diff_bins"] = tau_diff_bins

    return histograms


def plot_residual_histogram(
    ax,
    bins,
    series,
    xlabel,
    title=None,
    ylabel="Density",
):
    for counts, label, color in series:
        _plot_single_histogram(ax, bins, counts, label=label, color=color)
    ax.axvline(
        0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    if title:
        _set_title(ax, title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if any(label for _, label, _ in series):
        ax.legend(prop={"size": 10})


def _single_run_residual_series(counts):
    return [(counts, None, RESIDUAL_COLOR)]


def plot_feature_histograms(
    histograms,
    feature_names,
    reconstructed_series=None,
    mse_per_feature=None,
    include_all_ratios=False,
    title="Original vs. Reconstructed Features",
):
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=FEATURE_FIGSIZE)
    ratio_indices = (0, 1, 2) if include_all_ratios else (0,)
    axes, ratio_axes = _triplet_axes(fig, ratio_indices=ratio_indices)
    _set_suptitle(fig, title, fontsize=16)

    for axis_idx, feature_idx in enumerate((2, 0, 1)):
        axis = axes[axis_idx]
        feature_name = feature_names[feature_idx]
        clean_name = _clean_feature_name(feature_name)
        bins = histograms[f"{clean_name}_bins"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                histograms[f"{clean_name}_reco_counts"],
            )
        else:
            _plot_filled_histogram(
                axis,
                bins,
                histograms[f"{clean_name}_orig_counts"],
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
                alpha=TRUTH_REFERENCE_FILL_ALPHA,
            )
        for data, label, color in reconstructed_series:
            if not single_run:
                _plot_single_histogram(
                    axis,
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    label=label,
                    color=color,
                )
            if axis_idx in ratio_axes:
                _plot_ratio_histogram(
                    ratio_axes[axis_idx],
                    bins,
                    data[f"{clean_name}_reco_counts"],
                    histograms[f"{clean_name}_orig_counts"],
                    label=label,
                    color=color,
                )

        metric = ""
        if mse_per_feature is not None:
            metric = f" (MSE: {mse_per_feature[feature_idx]:.4f})"
        _set_title(axis, f"{feature_name}{metric}")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Density")
        axis.legend(prop={"size": 10})
        if feature_idx == 2:
            axis.set_xscale("log")
        if axis_idx in ratio_axes:
            _configure_ratio_axis(ratio_axes[axis_idx], xlabel=feature_name)
            axis.tick_params(labelbottom=False)

    _adjust_ratio_layout(fig)
    return fig


def plot_energy_histograms(
    histograms,
    reconstructed_series=None,
    include_ratio=False,
    title="Original vs. Reconstructed Energy (m=0)",
):
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig = plt.figure(figsize=ENERGY_FIGSIZE)
    axes, ratio_axes = _pair_axes(fig, with_first_ratio=include_ratio)
    bins = histograms["energy_bins"]
    if single_run:
        _plot_original_reconstructed_histograms(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            histograms["energy_reco_counts"],
        )
    else:
        _plot_filled_histogram(
            axes[0],
            bins,
            histograms["energy_orig_counts"],
            label="Original (Truth)",
            color=TRUTH_REFERENCE_COLOR,
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
        )
    for data, label, color in reconstructed_series:
        if not single_run:
            _plot_single_histogram(
                axes[0], bins, data["energy_reco_counts"], label=label, color=color
            )
        if include_ratio:
            _plot_ratio_histogram(
                ratio_axes[0],
                bins,
                data["energy_reco_counts"],
                histograms["energy_orig_counts"],
                label=label,
                color=color,
            )
    axes[0].set_xscale("log")
    _set_title(axes[0], title)
    axes[0].set_xlabel("Energy [GeV]")
    axes[0].set_ylabel("Density")
    axes[0].legend(prop={"size": 10})
    if include_ratio:
        _configure_ratio_axis(ratio_axes[0], xlabel="Energy [GeV]")
        axes[0].tick_params(labelbottom=False)

    residual_series = (
        _single_run_residual_series(histograms["energy_residuals_counts"])
        if single_run
        else [
            (data["energy_residuals_counts"], label, color)
            for data, label, color in reconstructed_series
        ]
    )
    plot_residual_histogram(
        axes[1],
        histograms["energy_residuals_bins"],
        residual_series,
        xlabel=r"$E^{reco} - E^{orig}$ [GeV]",
        title=r"Energy Residuals: $E^{reco} - E^{orig}$",
    )
    if include_ratio:
        _adjust_ratio_layout(fig)
    else:
        plt.tight_layout()
    return fig


def plot_missing_transverse_energy(histograms, reconstructed_series=None):
    """Plot event-level missing transverse energy for particle reconstruction."""
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
    bins = histograms["missing_et_bins"]
    if single_run:
        _plot_original_reconstructed_histograms(
            ax,
            bins,
            histograms["missing_et_orig_counts"],
            histograms["missing_et_reco_counts"],
        )
    else:
        _plot_filled_histogram(
            ax,
            bins,
            histograms["missing_et_orig_counts"],
            label="Original (Truth)",
            color=TRUTH_REFERENCE_COLOR,
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
        )
        for data, label, color in reconstructed_series:
            _plot_single_histogram(
                ax,
                bins,
                data["missing_et_reco_counts"],
                label=label,
                color=color,
            )
    ax.set_xlabel(r"Missing transverse energy $E_T^\mathrm{miss}$ [GeV]")
    ax.set_ylabel("Density")
    _set_title(ax, "Missing Transverse Energy")
    ax.legend(prop={"size": 10})
    plt.tight_layout()
    return fig


# =============================================================================
# Paper plots
# =============================================================================


def plot_paper_kinematic_distributions(
    histograms,
    data_level,
    reconstructed_series=None,
):
    """Plot the three core kinematic distributions with minimal decoration."""
    _validate_data_level(data_level)
    single_run = reconstructed_series is None
    reconstructed_series = reconstructed_series or [
        (histograms, "Reconstructed", RECONSTRUCTED_COLOR)
    ]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        bins = histograms[f"{feature_name}_bins"]
        original = histograms[f"{feature_name}_orig_counts"]
        if single_run:
            _plot_original_reconstructed_histograms(
                axis,
                bins,
                original,
                histograms[f"{feature_name}_reco_counts"],
            )
        else:
            _plot_filled_histogram(
                axis,
                bins,
                original,
                label="Original (Truth)",
                color=TRUTH_REFERENCE_COLOR,
                alpha=TRUTH_REFERENCE_FILL_ALPHA,
            )
            for data, label, color in reconstructed_series:
                _plot_single_histogram(
                    axis,
                    bins,
                    data[f"{feature_name}_reco_counts"],
                    label=label,
                    color=color,
                )
        if feature_name == "pT":
            axis.set_xscale("log")
        axis.set_xlabel(feature_label)
        axis.set_ylabel("Density")
        _set_title(axis, f"{feature_label} distribution")
        axis.legend(prop={"size": 10})

    _set_suptitle(fig, f"{data_level.capitalize()} kinematics: original vs. reconstructed")
    plt.tight_layout()
    return fig


def plot_paper_kinematic_differences(
    histograms,
    data_level,
    difference_series=None,
):
    """Plot reconstructed-minus-original residuals for the three core features."""
    _validate_data_level(data_level)
    difference_series = difference_series or [
        (histograms, None, RESIDUAL_COLOR)
    ]
    fig, axes = plt.subplots(1, 3, figsize=MONEY_TRIPLET_FIGSIZE)

    for axis, feature_name in zip(axes, ("pT", "Eta", "Phi")):
        feature_label = KINEMATIC_LABELS[feature_name]
        plot_residual_histogram(
            axis,
            histograms[f"{feature_name}_diff_bins"],
            [
                (data[f"{feature_name}_diff_counts"], label, color)
                for data, label, color in difference_series
            ],
            xlabel=KINEMATIC_RESIDUAL_LABELS[feature_name],
            title=f"{feature_label} residuals",
        )

    _set_suptitle(fig, f"{data_level.capitalize()} kinematic residuals")
    plt.tight_layout()
    return fig


def paper_reconstruction_plots(histograms, data_level):
    """Return the core single-run reconstruction figures."""
    return {
        "paper_kinematic_distributions": plot_paper_kinematic_distributions(
            histograms,
            data_level,
        ),
        "paper_kinematic_differences": plot_paper_kinematic_differences(
            histograms,
            data_level,
        ),
    }


def _has_paper_histograms(histograms):
    return all(
        f"{feature_name}_{suffix}" in histograms
        for feature_name in ("pT", "Eta", "Phi")
        for suffix in ("bins", "orig_counts", "reco_counts", "diff_bins", "diff_counts")
    )


def paper_reconstruction_comparison_plots(runs_data, run_labels, data_level):
    """Return the core multirun reconstruction figures."""
    reconstructed_series = [
        (data, run_labels[i], RUN_COLORS[i % len(RUN_COLORS)])
        for i, data in enumerate(runs_data)
    ]
    return {
        "paper_combined_kinematic_distributions.png": plot_paper_kinematic_distributions(
            runs_data[0],
            data_level,
            reconstructed_series=reconstructed_series,
        ),
        "paper_combined_kinematic_differences.png": plot_paper_kinematic_differences(
            runs_data[0],
            data_level,
            difference_series=reconstructed_series,
        ),
    }


# =============================================================================
# Single-run reconstruction plots
# =============================================================================


def exploratory_feature_histograms(
    x_np,
    x_hat_np,
    mse_per_feature,
    feature_names,
    include_all_ratios=False,
    data_level="particle",
):
    mh.style.use(mh.style.ROOT)
    histograms = collect_reconstruction_histograms(
        feature_names, x_np, x_hat_np, data_level=data_level
    )
    return plot_feature_histograms(
        histograms,
        feature_names,
        mse_per_feature=mse_per_feature,
        include_all_ratios=include_all_ratios,
        title=f"{data_level.capitalize()} Features: Original vs. Reconstructed",
    )


def exploratory_energy_histograms(
    x_np,
    x_hat_np,
    mse_per_feature,
    include_all_ratios=False,
    data_level="particle",
):
    mh.style.use(mh.style.ROOT)
    histograms = collect_reconstruction_histograms(
        ["Eta", "Phi", "pT"], x_np, x_hat_np, data_level=data_level
    )
    return plot_energy_histograms(
        histograms,
        include_ratio=include_all_ratios,
        title=f"{data_level.capitalize()} Energy: Original vs. Reconstructed (m=0)",
    )


def reconstruction_plots(
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
    include_all_ratios=False,
    data_level="particle",
    histograms=None,
):
    mh.style.use(mh.style.ROOT)
    _validate_data_level(data_level)
    if histograms is None:
        histograms = collect_reconstruction_histograms(
            feature_names,
            x_np,
            x_hat_np,
            true_jet_pts,
            reco_jet_pts,
            true_jet_masses,
            reco_jet_masses,
            true_tau32s,
            reco_tau32s,
            data_level=data_level,
        )
    figures = {}
    figures.update(paper_reconstruction_plots(histograms, data_level))

    if "jet_pt_resolution_counts" in histograms:
        fig, ax = plt.subplots(figsize=RESOLUTION_FIGSIZE)
        _plot_single_histogram(
            ax,
            histograms["jet_pt_resolution_bins"],
            histograms["jet_pt_resolution_counts"],
        )
        ax.axvline(
            0,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLE,
            alpha=REFERENCE_LINE_ALPHA,
        )
        ax.set_xlabel(r"Fractional $p_T$ Resolution: $(p_T^{reco} - p_T^{true}) / p_T^{true}$")
        ax.set_ylabel("Number of Jets")
        _set_title(ax, "Jet Transverse Momentum Recovery")
        figures["jet_pt_resolution"] = fig

    figures["kinematics"] = plot_feature_histograms(
        histograms,
        feature_names,
        mse_per_feature=mse_per_feature,
        include_all_ratios=include_all_ratios,
        title=f"{data_level.capitalize()} Kinematics: Original vs. Reconstructed",
    )
    figures["energy_residuals"] = plot_energy_histograms(
        histograms,
        include_ratio=include_all_ratios,
        title=f"{data_level.capitalize()} Energy: Original vs. Reconstructed (m=0)",
    )
    if data_level == "particle" and "missing_et_bins" in histograms:
        figures["paper_missing_transverse_energy"] = plot_missing_transverse_energy(
            histograms
        )

    if data_level == "particle" and "jet_mass_orig_counts" in histograms:
        if include_all_ratios:
            fig3 = plt.figure(figsize=SUBSTRUCTURE_FIGSIZE)
            axs3, ratio_axes = _triplet_axes(fig3, ratio_indices=(0,))
        else:
            fig3, axs3 = plt.subplots(1, 3, figsize=SUBSTRUCTURE_SIMPLE_FIGSIZE)
            ratio_axes = {}
        _set_suptitle(fig3, "FSQ-VAE: Jet Substructure", fontsize=16)

        _plot_original_reconstructed_histograms(
            axs3[0],
            histograms["jet_mass_bins"],
            histograms["jet_mass_orig_counts"],
            histograms["jet_mass_reco_counts"],
        )
        _set_title(axs3[0], "Jet Mass")
        axs3[0].set_xlabel("Jet Mass [GeV]")
        axs3[0].set_ylabel("Density")
        axs3[0].legend()
        if include_all_ratios:
            ratio_ax = ratio_axes[0]
            _plot_ratio_histogram(
                ratio_ax,
                histograms["jet_mass_bins"],
                histograms["jet_mass_reco_counts"],
                histograms["jet_mass_orig_counts"],
            )
            _configure_ratio_axis(ratio_ax, xlabel="Jet Mass [GeV]")
            axs3[0].tick_params(labelbottom=False)

        plot_residual_histogram(
            axs3[1],
            histograms["jet_mass_diff_bins"],
            _single_run_residual_series(histograms["jet_mass_diff_counts"]),
            xlabel=r"$m^{reco} - m^{orig}$ [GeV]",
            title="Jet Mass Residuals",
        )
        plot_residual_histogram(
            axs3[2],
            histograms["tau32_diff_bins"],
            _single_run_residual_series(histograms["tau32_diff_counts"]),
            xlabel=r"$\tau_{32}^{reco} - \tau_{32}^{orig}$",
            title=r"$\tau_{32}$ Residuals",
        )

        if include_all_ratios:
            _adjust_ratio_layout(fig3)
        else:
            plt.tight_layout()
        figures["jet_substructure"] = fig3

    return figures


# =============================================================================
# Single-run attention diagnostic plots
# =============================================================================


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

    fig, ax = plt.subplots(figsize=ATTENTION_DELTA_FIGSIZE)
    im = ax.imshow(
        hist.T,
        origin="lower",
        extent=[eta_edges[0], eta_edges[-1], phi_edges[0], phi_edges[-1]],
        aspect="auto",
    )
    _set_title(ax, title)
    ax.set_xlabel(r"$\Delta\eta = \eta_\mathrm{query} - \eta_\mathrm{key}$")
    ax.set_ylabel(r"$\Delta\phi = \phi_\mathrm{query} - \phi_\mathrm{key}$")
    fig.colorbar(im, ax=ax, label="mean attention weight per pair")
    return fig


def attention_map_figure(matrix, title):
    fig, ax = plt.subplots(figsize=ATTENTION_MAP_FIGSIZE)
    im = ax.imshow(matrix, vmin=0.0, vmax=max(float(matrix.max()), 1e-6), aspect="auto")
    _set_title(ax, title)
    ax.set_xlabel("key particle")
    ax.set_ylabel("query particle")
    fig.colorbar(im, ax=ax)
    return fig


def close_figure(fig):
    plt.close(fig)


def show_figure():
    plt.show()


# =============================================================================
# Multi-run comparison plots
# =============================================================================


def _clean_metric_name(metric):
    return metric.replace("metrics/", "").replace("/", "_")


def _codebook_family(record):
    label = str(record.get("label", record.get("run_name", ""))).lower()
    if "mlp" in label:
        return None
    if label.startswith("fsq"):
        return "fsq"
    if label.startswith("vq") and "rotation" in label:
        return "vq_rotation"
    if label.startswith("vq") and "ste" in label:
        return "vq_ste"
    return None


def _codebook_family_styles():
    return (
        CODEBOOK_FAMILY_COLORS,
        CODEBOOK_FAMILY_MARKERS,
        CODEBOOK_FAMILY_LABELS,
    )


def plot_scatter_figure(
    records,
    y_value,
    x_value,
    x_label,
    y_label,
    title,
    family_fn=None,
    family_colors=None,
    family_markers=None,
    family_labels=None,
    log_x=False,
    log_y=False,
):
    colors = RUN_COLORS
    markers = SCATTER_MARKERS
    family_colors = family_colors or {}
    family_markers = family_markers or {}
    family_labels = family_labels or {}
    family_fn = family_fn or (lambda record: None)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    present_families = []
    for family in family_labels:
        family_records = [record for record in records if family_fn(record) == family]
        if family_records:
            present_families.append(family)
        if len(family_records) < 2:
            continue
        family_records = sorted(family_records, key=x_value)
        ax.plot(
            [x_value(record) for record in family_records],
            [y_value(record) for record in family_records],
            color=family_colors[family],
            alpha=SCATTER_LINE_ALPHA,
            linewidth=SCATTER_LINEWIDTH,
            zorder=1,
        )

    for i, record in enumerate(records):
        family = family_fn(record)
        ax.scatter(
            x_value(record),
            y_value(record),
            color=family_colors.get(family, colors[i % len(colors)]),
            marker=family_markers.get(family, markers[i % len(markers)]),
            alpha=SCATTER_POINT_ALPHA,
            s=45,
            zorder=2,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _set_title(ax, title)
    family_handles = [
        Line2D(
            [0],
            [0],
            color=family_colors[family],
            marker=family_markers[family],
            linewidth=SCATTER_LINEWIDTH,
            markersize=7,
            label=family_labels[family],
        )
        for family in present_families
    ]
    if family_handles:
        ax.legend(handles=family_handles, prop={"size": 9})
    plt.tight_layout()
    return fig


def plot_codebook_error_scatter(records, y_metrics):
    mh.style.use(mh.style.ROOT)

    family_colors, family_markers, family_labels = _codebook_family_styles()
    figures = {}

    for metric in y_metrics:
        metric_records = [
            record
            for record in records
            if record.get(metric) is not None
            and record.get("total_codebook_size") is not None
            and record[metric] > 0
        ]
        if not metric_records:
            continue

        metric_name = metric.replace("metrics/", "")
        clean_metric_name = _clean_metric_name(metric)

        figures[f"codebook_size_vs_{clean_metric_name}.png"] = plot_scatter_figure(
            records=metric_records,
            y_value=lambda record, metric=metric: record[metric],
            x_value=lambda record: record["total_codebook_size"],
            x_label="Total codebook size",
            y_label=metric_name,
            title=f"Codebook size vs. {metric_name}",
            family_fn=_codebook_family,
            family_colors=family_colors,
            family_markers=family_markers,
            family_labels=family_labels,
            log_x=True,
            log_y=True,
        )
        figures[f"codebook_bits_vs_{clean_metric_name}.png"] = plot_scatter_figure(
            records=metric_records,
            y_value=lambda record, metric=metric: record[metric],
            x_value=lambda record: math.ceil(math.log2(record["total_codebook_size"])),
            x_label="Bits required to represent codebook",
            y_label=metric_name,
            title=f"Codebook bits vs. {metric_name}",
            family_fn=_codebook_family,
            family_colors=family_colors,
            family_markers=family_markers,
            family_labels=family_labels,
            log_y=True,
        )

    return figures


def plot_codebook_utilization_scatter(records):
    mh.style.use(mh.style.ROOT)

    figures = {}
    family_colors, family_markers, family_labels = _codebook_family_styles()
    metrics = (
        ("metrics/utilization_mu", "Mu codebook utilization"),
        ("metrics/utilization_alpha", "Alpha codebook utilization"),
        ("metrics/utilization_combined", "Combined codebook utilization"),
    )
    for metric, label in metrics:
        metric_records = [
            record
            for record in records
            if record.get(metric) is not None
            and record.get("total_codebook_size") is not None
        ]
        if not metric_records:
            continue
        clean_metric_name = _clean_metric_name(metric)
        figures[f"codebook_size_vs_{clean_metric_name}.png"] = plot_scatter_figure(
            records=metric_records,
            x_value=lambda record: record["total_codebook_size"],
            y_value=lambda record, metric=metric: record[metric],
            x_label="Total codebook size",
            y_label=label,
            title=f"Codebook size vs. {label.lower()}",
            family_fn=_codebook_family,
            family_colors=family_colors,
            family_markers=family_markers,
            family_labels=family_labels,
            log_x=True,
        )
        figures[f"codebook_size_vs_{clean_metric_name}.png"].axes[0].set_ylim(
            *UTILIZATION_YLIM
        )

    return figures


def plot_theoretical_codebook_bits():
    """Plot the theoretical relationship between codebook size and bit count."""
    mh.style.use(mh.style.ROOT)

    bits = np.linspace(0, 45, 181)
    codebook_sizes = np.exp2(bits)
    nominal_points = (
        (37, r"nominal size: 37 bits, $1.37 \times 10^{11}$", "-", CODEBOOK_FAMILY_COLORS["fsq"]),
        (40, r"nominal size, with PID: 40 bits, $1.10 \times 10^{12}$", "--", CODEBOOK_FAMILY_COLORS["vq_ste"]),
    )

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.plot(
        codebook_sizes,
        bits,
        color=REFERENCE_LINE_COLOR,
        linewidth=SCATTER_LINEWIDTH,
        label=r"$\log_2(\mathrm{codebook\ size})$",
    )
    for nominal_bits, label, linestyle, color in nominal_points:
        nominal_size = 2**nominal_bits
        ax.plot(
            nominal_size,
            nominal_bits,
            color=color,
            marker="o",
            markersize=7,
            linestyle="none",
            label=label,
            zorder=2,
        )
        ax.axvline(
            nominal_size,
            color=color,
            linestyle=linestyle,
            alpha=SCATTER_LINE_ALPHA,
            linewidth=SCATTER_LINEWIDTH,
            zorder=1,
        )
    ax.set_xscale("log")
    ax.set_xlim(1, 1e14)
    ax.set_xticks(
        [1, 1e2, 1e5, 1e8, 1e11, 1e14],
        ["1", r"$10^2$", r"$10^5$", r"$10^8$", r"$10^{11}$", r"$10^{14}$"],
    )
    ax.set_xlabel("Codebook size")
    ax.set_ylabel("Required bits")
    ax.set_ylim(bottom=0)
    _set_title(ax, "Theoretical codebook size vs. bits")
    ax.legend(prop={"size": 9})
    plt.tight_layout()
    return fig


def _record_total_mse(record):
    total = record.get("metrics/mse_total")
    if total is not None:
        return total
    components = [
        record.get("metrics/mse_Eta"),
        record.get("metrics/mse_Phi"),
        record.get("metrics/mse_pT"),
    ]
    if any(value is None for value in components):
        return None
    return float(np.mean(components))


def _record_total_utilization(record):
    for metric in (
        "metrics/utilization_total",
        "metrics/utilization_combined",
        "metrics/utilization_mu",
        "metrics/utilization_alpha",
    ):
        value = record.get(metric)
        if value is not None:
            return value
    return None


def plot_paper_codebook_scans(records):
    """Plot the two headline codebook-size scans."""
    family_colors, family_markers, family_labels = _codebook_family_styles()
    figures = {}

    mse_records = [
        record
        for record in records
        if record.get("total_codebook_size") is not None
        and _record_total_mse(record) is not None
        and _record_total_mse(record) > 0
    ]
    if mse_records:
        figures["paper_codebook_size_vs_mse_total.png"] = plot_scatter_figure(
            records=mse_records,
            x_value=lambda record: record["total_codebook_size"],
            y_value=_record_total_mse,
            x_label="Total codebook size",
            y_label="Total MSE",
            title="Codebook size vs. total MSE",
            family_fn=_codebook_family,
            family_colors=family_colors,
            family_markers=family_markers,
            family_labels=family_labels,
            log_x=True,
            log_y=True,
        )

    utilization_records = [
        record
        for record in records
        if record.get("total_codebook_size") is not None
        and _record_total_utilization(record) is not None
    ]
    if utilization_records:
        figures["paper_codebook_size_vs_utilization_total.png"] = plot_scatter_figure(
            records=utilization_records,
            x_value=lambda record: record["total_codebook_size"],
            y_value=_record_total_utilization,
            x_label="Total codebook size",
            y_label="Total codebook utilization",
            title="Codebook size vs. total utilization",
            family_fn=_codebook_family,
            family_colors=family_colors,
            family_markers=family_markers,
            family_labels=family_labels,
            log_x=True,
        )
        figures["paper_codebook_size_vs_utilization_total.png"].axes[0].set_ylim(
            *UTILIZATION_YLIM
        )

    return figures


def replot_reconstruction_comparison(
    runs_data,
    run_labels,
    include_all_ratios=False,
    data_level="particle",
):
    """
    Plots superimposed reconstruction histograms for multiple runs.
    
    Args:
        runs_data (list of mapping): Histogram arrays for each run.
        run_labels (list of str): Legend labels for each run (e.g., ["FSQ 10", "FSQ 21"]).
    """
    mh.style.use(mh.style.ROOT)
    _validate_data_level(data_level)
    figures = {}
    
    # Define distinct colors for the different runs
    colors = RUN_COLORS
    
    # We can extract the bins and the "Original" truth from the very first run
    # (since the test set and bins are constant across all runs)
    ref_data = runs_data[0]
    if _has_paper_histograms(ref_data):
        figures.update(
            paper_reconstruction_comparison_plots(runs_data, run_labels, data_level)
        )

    # ==========================================
    # 1. Jet pT Resolution
    # ==========================================
    fig_pt_res, ax_pt_res = plt.subplots(figsize=RESOLUTION_FIGSIZE)
    bins_pt_res = ref_data["jet_pt_resolution_bins"]
    
    for i, data in enumerate(runs_data):
        _plot_single_histogram(
            ax_pt_res,
            bins_pt_res,
            data["jet_pt_resolution_counts"],
            label=run_labels[i],
            color=colors[i % len(colors)],
        )
        
    ax_pt_res.axvline(
        0,
        color=REFERENCE_LINE_COLOR,
        linestyle=REFERENCE_LINE_STYLE,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax_pt_res.set_xlabel(r"Fractional $p_T$ Resolution: $(p_T^{reco} - p_T^{true}) / p_T^{true}$")
    ax_pt_res.set_ylabel("Density / Number of Jets")
    _set_title(ax_pt_res, "Jet Transverse Momentum Recovery")
    ax_pt_res.legend(prop={'size': 10})
    figures["combined_jet_pt_resolution.png"] = fig_pt_res

    reconstructed_series = [
        (data, run_labels[i], colors[i % len(colors)])
        for i, data in enumerate(runs_data)
    ]
    figures["combined_kinematics.png"] = plot_feature_histograms(
        ref_data,
        ["Eta", "Phi", "pT"],
        reconstructed_series=reconstructed_series,
        include_all_ratios=include_all_ratios,
        title=f"{data_level.capitalize()} Kinematics: Reconstruction Sweeps",
    )
    figures["combined_energy.png"] = plot_energy_histograms(
        ref_data,
        reconstructed_series=reconstructed_series,
        include_ratio=include_all_ratios,
        title=f"{data_level.capitalize()} Energy: Reconstruction Sweeps (m=0)",
    )
    if data_level == "particle" and all(
        "missing_et_bins" in data for data in runs_data
    ):
        figures["paper_combined_missing_transverse_energy.png"] = (
            plot_missing_transverse_energy(
                ref_data,
                reconstructed_series=reconstructed_series,
            )
        )

    # ==========================================
    # 4. Jet Substructure (Mass, Mass Diff, Tau32 Diff)
    # ==========================================
    # Check if substructure data exists (in case a run crashed before substructure or had N<3)
    if data_level == "particle" and "jet_mass_orig_counts" in ref_data:
        fig_sub = plt.figure(figsize=SUBSTRUCTURE_COMPARISON_FIGSIZE)
        axs_sub, ratio_axes = _triplet_axes(fig_sub, ratio_indices=(0,))
        _set_suptitle(fig_sub, "Jet Substructure Sweep", fontsize=16)

        # 4.1 Jet Mass
        bins_mass = ref_data["jet_mass_bins"]
        _plot_filled_histogram(
            axs_sub[0],
            bins_mass,
            ref_data["jet_mass_orig_counts"],
            label="Original (Truth)",
            color=TRUTH_REFERENCE_COLOR,
            alpha=TRUTH_REFERENCE_FILL_ALPHA,
        )
        for j, data in enumerate(runs_data):
            _plot_single_histogram(
                axs_sub[0],
                bins_mass,
                data["jet_mass_reco_counts"],
                label=f"Reco ({run_labels[j]})",
                color=colors[j % len(colors)],
            )
            _plot_ratio_histogram(
                ratio_axes[0],
                bins_mass,
                data["jet_mass_reco_counts"],
                ref_data["jet_mass_orig_counts"],
                label=run_labels[j],
                color=colors[j % len(colors)],
            )
        _configure_ratio_axis(ratio_axes[0], xlabel="Jet Mass [GeV]")
        axs_sub[0].tick_params(labelbottom=False)
        axs_sub[0].set_xlabel("Jet Mass [GeV]")
        axs_sub[0].set_ylabel("Density")
        axs_sub[0].legend(prop={'size': 10})

        # 4.2 Jet Mass Difference
        plot_residual_histogram(
            axs_sub[1],
            ref_data["jet_mass_diff_bins"],
            [
                (data["jet_mass_diff_counts"], run_labels[j], colors[j % len(colors)])
                for j, data in enumerate(runs_data)
            ],
            xlabel=r"Jet $m^{reco} - m^{orig}$ [GeV]",
        )
        plot_residual_histogram(
            axs_sub[2],
            ref_data["tau32_diff_bins"],
            [
                (data["tau32_diff_counts"], run_labels[j], colors[j % len(colors)])
                for j, data in enumerate(runs_data)
            ],
            xlabel=r"$\tau_{32}^{reco} - \tau_{32}^{orig}$",
        )

        _adjust_ratio_layout(fig_sub)
        figures["combined_substructure.png"] = fig_sub

    return figures


def replot_jet_structure(runs_data, run_labels, include_all_ratios=False, data_level="particle"):
    """Backward-compatible alias for replot_reconstruction_comparison."""
    return replot_reconstruction_comparison(
        runs_data,
        run_labels,
        include_all_ratios=include_all_ratios,
        data_level=data_level,
    )
