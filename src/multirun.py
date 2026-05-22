import copy
import csv
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import wandb

from plotting import plot_codebook_error_scatter, replot_jet_structure
from train_eval import TrainPipeline


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _codebook_size(levels):
    if not levels:
        return 1
    return int(np.prod(np.array(levels, dtype=np.int64)))


def _latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_metrics(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_summary_csv(path, records):
    if not records:
        return

    keys = sorted({key for record in records for key in record.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _build_run_config(base_config, codebook_config, seed):
    run_config = copy.deepcopy(base_config)
    model_config = run_config["model"]

    model_config["fsq_mu_levels"] = list(codebook_config.get("fsq_mu_levels", model_config["fsq_mu_levels"]))
    model_config["fsq_alpha_levels"] = list(codebook_config.get("fsq_alpha_levels", model_config["fsq_alpha_levels"]))

    for key, value in codebook_config.get("model_overrides", {}).items():
        model_config[key] = value

    for key, value in codebook_config.get("overrides", {}).items():
        run_config[key] = value

    run_config["split_random_seed"] = seed
    return run_config


def run_codebook_multirun(config, train_val_files, test_files):
    multirun_cfg = config.get("multirun", {})
    suite_name = _safe_name(multirun_cfg.get("name", "codebook_size_scan"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_id = f"{suite_name}_{timestamp}"

    output_root = Path(config["output_dir"])
    suite_dir = output_root / suite_id
    comparison_dir = suite_dir / "comparisons"

    codebook_configs = multirun_cfg.get("codebook_configs", [])
    seeds = multirun_cfg.get("seeds", [config.get("split_random_seed", 42)])
    eval_prefix = "test" if config.get("run_test", False) else "val"
    y_metrics = multirun_cfg.get(
        "y_metrics",
        ["metrics/mse_Eta", "metrics/mse_Phi", "metrics/mse_pT"],
    )

    if not codebook_configs:
        raise ValueError("multirun.codebook_configs must contain at least one codebook configuration.")

    records = []
    npz_files = []
    run_labels = []

    _write_json(suite_dir / "base_config.json", config)

    for codebook_idx, codebook_config in enumerate(codebook_configs):
        label = codebook_config.get("label")
        if label is None:
            label = f"cb_{codebook_idx}"
        label = _safe_name(label)

        for seed in seeds:
            run_config = _build_run_config(config, codebook_config, seed)
            mu_levels = run_config["model"]["fsq_mu_levels"]
            alpha_levels = run_config["model"]["fsq_alpha_levels"]
            mu_codebook_size = _codebook_size(mu_levels)
            alpha_codebook_size = _codebook_size(alpha_levels)
            total_codebook_size = mu_codebook_size * alpha_codebook_size

            run_name = f"{suite_id}_{label}_seed_{seed}"
            run_config["run_name"] = run_name

            L.seed_everything(seed, workers=True)

            pipeline = TrainPipeline(
                config=run_config,
                unique_run_name=run_name,
                train_val_files=train_val_files,
                test_files=test_files,
            )
            pipeline.run(run_config["run_validation"], run_config["run_test"])
            wandb.finish()

            run_dir = output_root / run_name
            _write_json(run_dir / "resolved_config.json", run_config)

            metrics_path = _latest_file(str(run_dir / "saved_metrics" / f"{eval_prefix}_metrics_step_*.json"))
            hist_path = _latest_file(str(run_dir / "saved_histograms" / f"{eval_prefix}_hists_step_*.npz"))
            metrics = _load_metrics(metrics_path)

            record = {
                "suite_id": suite_id,
                "run_name": run_name,
                "label": label,
                "seed": seed,
                "mu_levels": str(mu_levels),
                "alpha_levels": str(alpha_levels),
                "mu_codebook_size": mu_codebook_size,
                "alpha_codebook_size": alpha_codebook_size,
                "total_codebook_size": total_codebook_size,
                "metrics_path": metrics_path,
                "hist_path": hist_path,
            }
            record.update(metrics)
            records.append(record)

            if hist_path is not None:
                npz_files.append(hist_path)
                run_labels.append(f"{label}, seed {seed}")

            _write_summary_csv(suite_dir / "summary.csv", records)
            _write_json(suite_dir / "manifest.json", records)

    plot_codebook_error_scatter(records, y_metrics=y_metrics, output_dir=str(comparison_dir))

    if npz_files:
        replot_jet_structure(npz_files=npz_files, run_labels=run_labels, output_dir=str(comparison_dir))

    return {
        "suite_id": suite_id,
        "suite_dir": str(suite_dir),
        "comparison_dir": str(comparison_dir),
        "num_runs": len(records),
    }
