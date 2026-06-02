#!/usr/bin/env python3
"""Generate a gallery of every plot in src.plotting using dummy data."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import (
    attention_delta_eta_phi_figure,
    attention_map_figure,
    collect_reconstruction_histograms,
    exploratory_energy_histograms,
    exploratory_feature_histograms,
    plot_codebook_error_scatter,
    plot_codebook_utilization_scatter,
    plot_paper_codebook_scans,
    plot_theoretical_codebook_bits,
    replot_reconstruction_comparison,
    reconstruction_plots,
    set_plot_titles_enabled,
)


FEATURE_NAMES = ["Eta", "Phi", "pT"]
GENERATED_FILENAMES = set()


def save_figure(fig, output_dir, filename):
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)
    GENERATED_FILENAMES.add(filename)


def gallery_sections(filenames):
    sections = (
        (
            "Paper Plots",
            lambda filename: "paper_" in filename,
        ),
        (
            "Exploratory Reconstruction",
            lambda filename: "_exploratory_" in filename and "paper_" not in filename,
        ),
        (
            "Single-Run Reconstruction",
            lambda filename: filename.startswith(("particle_", "jet_"))
            and "_combined_" not in filename
            and "_exploratory_" not in filename
            and "paper_" not in filename,
        ),
        (
            "Multirun Reconstruction",
            lambda filename: "_combined_" in filename and "paper_" not in filename,
        ),
        (
            "Attention Diagnostics",
            lambda filename: filename.startswith("attention_"),
        ),
        (
            "Codebook Scans",
            lambda filename: filename.startswith("codebook_"),
        ),
    )
    unassigned = set(filenames)
    grouped = []
    for title, belongs_to_section in sections:
        section_filenames = sorted(
            filename for filename in filenames if belongs_to_section(filename)
        )
        grouped.append((title, section_filenames))
        unassigned.difference_update(section_filenames)
    if unassigned:
        grouped.append(("Other Plots", sorted(unassigned)))
    return grouped


def save_gallery_pdf(output_dir, sections, pdf_filename):
    with PdfPages(output_dir / pdf_filename) as pdf:
        for section_title, filenames in sections:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.text(
                0.5,
                0.5,
                section_title,
                ha="center",
                va="center",
                fontsize=28,
                weight="bold",
            )
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for filename in filenames:
                image = plt.imread(output_dir / filename)
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.imshow(image)
                ax.set_title(filename)
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def dummy_particles(rng, num_particles, noise_scale):
    x = np.column_stack(
        (
            rng.normal(0.0, 1.0, num_particles),
            rng.uniform(-np.pi, np.pi, num_particles),
            rng.lognormal(mean=3.2, sigma=0.8, size=num_particles),
        )
    )
    x_hat = x.copy()
    x_hat[:, 0] += rng.normal(0.0, 0.10 * noise_scale, num_particles)
    x_hat[:, 1] += rng.normal(0.0, 0.08 * noise_scale, num_particles)
    x_hat[:, 1] = (x_hat[:, 1] + np.pi) % (2 * np.pi) - np.pi
    x_hat[:, 2] *= rng.lognormal(mean=0.0, sigma=0.10 * noise_scale, size=num_particles)
    return x, x_hat


def dummy_jet_features(rng, num_jets, noise_scale):
    x = np.column_stack(
        (
            rng.normal(0.0, 1.0, num_jets),
            rng.uniform(-np.pi, np.pi, num_jets),
            rng.lognormal(mean=5.5, sigma=0.45, size=num_jets),
        )
    )
    x_hat = x.copy()
    x_hat[:, 0] += rng.normal(0.0, 0.06 * noise_scale, num_jets)
    x_hat[:, 1] += rng.normal(0.0, 0.05 * noise_scale, num_jets)
    x_hat[:, 1] = (x_hat[:, 1] + np.pi) % (2 * np.pi) - np.pi
    x_hat[:, 2] *= rng.lognormal(mean=0.0, sigma=0.06 * noise_scale, size=num_jets)
    return x, x_hat


def dummy_jets(rng, num_jets, noise_scale):
    true_pts = rng.lognormal(mean=5.5, sigma=0.45, size=num_jets)
    reco_pts = true_pts * rng.normal(1.0, 0.06 * noise_scale, size=num_jets)
    true_masses = np.clip(rng.normal(170.0, 35.0, size=num_jets), 5.0, None)
    reco_masses = true_masses + rng.normal(0.0, 10.0 * noise_scale, size=num_jets)
    true_tau32s = np.clip(rng.beta(3.0, 4.0, size=num_jets), 0.0, 1.0)
    reco_tau32s = np.clip(
        true_tau32s + rng.normal(0.0, 0.05 * noise_scale, size=num_jets),
        0.0,
        1.0,
    )
    return true_pts, reco_pts, true_masses, reco_masses, true_tau32s, reco_tau32s


def dummy_missing_transverse_energy(rng, num_events, noise_scale):
    true_missing_ets = rng.lognormal(mean=3.7, sigma=0.65, size=num_events)
    reco_missing_ets = np.clip(
        true_missing_ets + rng.normal(0.0, 5.0 * noise_scale, size=num_events),
        0.0,
        None,
    )
    return true_missing_ets, reco_missing_ets


def reconstruction_results(
    rng,
    num_particles,
    num_jets,
    noise_scale,
    include_all_ratios=False,
    data_level="particle",
):
    if data_level == "particle":
        x, x_hat = dummy_particles(rng, num_particles, noise_scale)
        jets = dummy_jets(rng, num_jets, noise_scale)
        missing_ets = dummy_missing_transverse_energy(rng, num_jets, noise_scale)
    else:
        x, x_hat = dummy_jet_features(rng, num_jets, noise_scale)
        jets = (x[:, 2], x_hat[:, 2], (), (), (), ())
        missing_ets = ((), ())
    mse_per_feature = np.mean((x_hat - x) ** 2, axis=0)
    histograms = histogram_data(
        x,
        x_hat,
        jets,
        missing_ets=missing_ets,
        data_level=data_level,
    )
    figures = reconstruction_plots(
        FEATURE_NAMES,
        mse_per_feature,
        x,
        x_hat,
        *jets,
        include_all_ratios=include_all_ratios,
        data_level=data_level,
        histograms=histograms,
    )
    return x, x_hat, mse_per_feature, figures, histograms


def histogram_data(x, x_hat, jets, missing_ets=((), ()), data_level="particle"):
    return collect_reconstruction_histograms(
        FEATURE_NAMES,
        x,
        x_hat,
        *jets,
        true_missing_ets=missing_ets[0],
        reco_missing_ets=missing_ets[1],
        data_level=data_level,
    )


def save_reconstruction_plots(figures, output_dir, prefix):
    for filename, fig in figures.items():
        save_figure(fig, output_dir, f"{prefix}_{filename}.png")


def save_mode_reconstruction_suite(
    rng,
    output_dir,
    data_level,
    num_particles,
    num_jets,
    include_all_ratios,
):
    x, x_hat, mse_per_feature, figures, histograms = reconstruction_results(
        rng,
        num_particles,
        num_jets,
        noise_scale=1.0,
        include_all_ratios=include_all_ratios,
        data_level=data_level,
    )
    save_figure(
        exploratory_feature_histograms(
            x,
            x_hat,
            mse_per_feature,
            FEATURE_NAMES,
            include_all_ratios=include_all_ratios,
            data_level=data_level,
        ),
        output_dir,
        f"{data_level}_exploratory_features.png",
    )
    save_figure(
        exploratory_energy_histograms(
            x,
            x_hat,
            mse_per_feature,
            include_all_ratios=include_all_ratios,
            data_level=data_level,
        ),
        output_dir,
        f"{data_level}_exploratory_energy.png",
    )
    save_reconstruction_plots(figures, output_dir, prefix=data_level)

    runs_data = [histograms]
    for noise_scale in (0.75, 1.35):
        _, _, _, comparison_figures, comparison_histograms = reconstruction_results(
            rng,
            num_particles,
            num_jets,
            noise_scale=noise_scale,
            include_all_ratios=include_all_ratios,
            data_level=data_level,
        )
        runs_data.append(comparison_histograms)
        for fig in comparison_figures.values():
            plt.close(fig)

    combined_figures = replot_reconstruction_comparison(
        runs_data,
        run_labels=["baseline", "lower noise", "higher noise"],
        include_all_ratios=include_all_ratios,
        data_level=data_level,
    )
    for filename, fig in combined_figures.items():
        save_figure(fig, output_dir, f"{data_level}_{filename}")


def save_attention_plots(rng, output_dir):
    batch_size, num_heads, num_particles = 3, 4, 16
    weights = torch.tensor(
        rng.random((batch_size, num_heads, num_particles, num_particles)),
        dtype=torch.float32,
    )
    weights = weights.softmax(dim=-1)
    x = torch.tensor(
        np.stack(
            (
                rng.uniform(-1.0, 1.0, (batch_size, num_particles)),
                rng.uniform(-1.0, 1.0, (batch_size, num_particles)),
                rng.lognormal(2.0, 0.6, (batch_size, num_particles)),
            ),
            axis=-1,
        ),
        dtype=torch.float32,
    )
    valid = torch.tensor(
        rng.random((batch_size, num_particles)) > 0.15,
        dtype=torch.bool,
    )

    delta_fig = attention_delta_eta_phi_figure(
        weights,
        x,
        valid,
        title="Dummy attention vs. angular separation",
        exclude_self=True,
    )
    if delta_fig is not None:
        save_figure(delta_fig, output_dir, "attention_delta_eta_phi.png")

    matrix = weights[0].mean(dim=0).numpy()
    save_figure(
        attention_map_figure(matrix, title="Dummy attention map"),
        output_dir,
        "attention_map.png",
    )


def save_codebook_plots(output_dir):
    save_figure(
        plot_theoretical_codebook_bits(),
        output_dir,
        "paper_codebook_size_vs_bits_theoretical.png",
    )

    records = []
    families = (
        ("FSQ", 0.85),
        ("VQ STE", 1.0),
        ("VQ rotation", 0.75),
    )
    for family, factor in families:
        for size in (64, 512, 4096):
            records.append(
                {
                    "label": f"{family} {size}",
                    "total_codebook_size": size,
                    "metrics/mse_Eta": factor * 0.3 / np.sqrt(size),
                    "metrics/mse_Phi": factor * 0.5 / np.sqrt(size),
                    "metrics/mse_pT": factor * 18.0 / np.sqrt(size),
                    "metrics/mse_total": factor * 6.4 / np.sqrt(size),
                    "metrics/utilization_mu": min(0.95, factor * 16.0 / np.sqrt(size)),
                    "metrics/utilization_total": min(0.95, factor * 16.0 / np.sqrt(size)),
                }
            )

    figures = plot_codebook_error_scatter(
        records,
        y_metrics=["metrics/mse_Eta", "metrics/mse_Phi", "metrics/mse_pT"],
    )
    for filename, fig in figures.items():
        save_figure(fig, output_dir, filename)
    figures = plot_codebook_utilization_scatter(records)
    for filename, fig in figures.items():
        save_figure(fig, output_dir, filename)
    figures = plot_paper_codebook_scans(records)
    for filename, fig in figures.items():
        save_figure(fig, output_dir, filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plot_showcase"),
        help="Directory for generated PNG files (default: outputs/plot_showcase)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-particles", type=int, default=4000)
    parser.add_argument("--num-jets", type=int, default=1200)
    parser.add_argument(
        "--pdf-name",
        default="plot_showcase.pdf",
        help="Filename for the multi-page PDF gallery (default: plot_showcase.pdf)",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Suppress titles inside plots; PDF page filename headings remain visible",
    )
    parser.add_argument(
        "--all-ratios",
        action="store_true",
        help="Add ratio panels to every original-versus-reconstructed histogram",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_titles_enabled(not args.no_titles)
    rng = np.random.default_rng(args.seed)

    for data_level in ("particle", "jet"):
        save_mode_reconstruction_suite(
            rng,
            args.output_dir,
            data_level=data_level,
            num_particles=args.num_particles,
            num_jets=args.num_jets,
            include_all_ratios=args.all_ratios,
        )

    save_attention_plots(rng, args.output_dir)
    save_codebook_plots(args.output_dir)

    generated = sorted(GENERATED_FILENAMES)
    sections = gallery_sections(generated)
    save_gallery_pdf(args.output_dir, sections, args.pdf_name)
    print(f"Generated {len(generated)} plots in {args.output_dir}:")
    for filename in generated:
        print(f"  {filename}")
    print("PDF sections:")
    for title, filenames in sections:
        print(f"  {title}: {len(filenames)} plots")
    print(f"Generated PDF gallery: {args.output_dir / args.pdf_name}")


if __name__ == "__main__":
    main()
