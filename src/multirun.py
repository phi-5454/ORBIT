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


def _make_suite_id(multirun_cfg):
    explicit_suite_id = multirun_cfg.get("suite_id")
    if explicit_suite_id:
        return _safe_name(explicit_suite_id)

    suite_name = _safe_name(multirun_cfg.get("name", "codebook_size_scan"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{suite_name}_{timestamp}"


def _build_task_specs(config, suite_id):
    multirun_cfg = config.get("multirun", {})
    codebook_configs = multirun_cfg.get("codebook_configs", [])
    seeds = multirun_cfg.get("seeds", [config.get("split_random_seed", 42)])

    if not codebook_configs:
        raise ValueError("multirun.codebook_configs must contain at least one codebook configuration.")

    tasks = []
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

            tasks.append(
                {
                    "task_index": len(tasks),
                    "label": label,
                    "seed": seed,
                    "run_name": run_name,
                    "run_config": run_config,
                    "mu_levels": mu_levels,
                    "alpha_levels": alpha_levels,
                    "mu_codebook_size": mu_codebook_size,
                    "alpha_codebook_size": alpha_codebook_size,
                    "total_codebook_size": total_codebook_size,
                }
            )

    return tasks


def _load_record_files(records_dir):
    records = []
    for path in sorted(records_dir.glob("*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def collect_codebook_multirun(config):
    multirun_cfg = config.get("multirun", {})
    suite_id = _make_suite_id(multirun_cfg)
    output_root = Path(config["output_dir"])
    suite_dir = output_root / suite_id
    comparison_dir = suite_dir / "comparisons"
    records_dir = suite_dir / "records"
    y_metrics = multirun_cfg.get(
        "y_metrics",
        ["metrics/mse_Eta", "metrics/mse_Phi", "metrics/mse_pT"],
    )

    records = _load_record_files(records_dir)
    if not records:
        raise ValueError(f"No multirun records found in {records_dir}")

    _write_summary_csv(suite_dir / "summary.csv", records)
    _write_json(suite_dir / "manifest.json", records)
    plot_codebook_error_scatter(records, y_metrics=y_metrics, output_dir=str(comparison_dir))

    npz_files = [record["hist_path"] for record in records if record.get("hist_path")]
    run_labels = [f"{record['label']}, seed {record['seed']}" for record in records if record.get("hist_path")]
    if npz_files:
        replot_jet_structure(npz_files=npz_files, run_labels=run_labels, output_dir=str(comparison_dir))

    return {
        "suite_id": suite_id,
        "suite_dir": str(suite_dir),
        "comparison_dir": str(comparison_dir),
        "num_runs": len(records),
    }


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
    if multirun_cfg.get("collect_only", False):
        return collect_codebook_multirun(config)

    suite_id = _make_suite_id(multirun_cfg)

    output_root = Path(config["output_dir"])
    suite_dir = output_root / suite_id
    records_dir = suite_dir / "records"

    eval_prefix = "test" if config.get("run_test", False) else "val"
    job_index = int(multirun_cfg.get("job_index", 0))
    job_count = int(multirun_cfg.get("job_count", 1))

    if job_count < 1:
        raise ValueError("multirun.job_count must be >= 1")
    if job_index < 0 or job_index >= job_count:
        raise ValueError(f"multirun.job_index must be in [0, {job_count}), got {job_index}")

    records = []
    tasks = _build_task_specs(config, suite_id)
    selected_tasks = [task for task in tasks if task["task_index"] % job_count == job_index]
    _write_json(suite_dir / "base_config.json", config)

    for task in selected_tasks:
        run_config = task["run_config"]
        L.seed_everything(task["seed"], workers=True)

        pipeline = TrainPipeline(
            config=run_config,
            unique_run_name=task["run_name"],
            train_val_files=train_val_files,
            test_files=test_files,
        )
        pipeline.run(run_config["run_validation"], run_config["run_test"])
        wandb.finish()

        run_dir = output_root / task["run_name"]
        _write_json(run_dir / "resolved_config.json", run_config)

        metrics_path = _latest_file(str(run_dir / "saved_metrics" / f"{eval_prefix}_metrics_step_*.json"))
        hist_path = _latest_file(str(run_dir / "saved_histograms" / f"{eval_prefix}_hists_step_*.npz"))
        metrics = _load_metrics(metrics_path)

        record = {
            "suite_id": suite_id,
            "task_index": task["task_index"],
            "job_index": job_index,
            "job_count": job_count,
            "run_name": task["run_name"],
            "label": task["label"],
            "seed": task["seed"],
            "mu_levels": str(task["mu_levels"]),
            "alpha_levels": str(task["alpha_levels"]),
            "mu_codebook_size": task["mu_codebook_size"],
            "alpha_codebook_size": task["alpha_codebook_size"],
            "total_codebook_size": task["total_codebook_size"],
            "metrics_path": metrics_path,
            "hist_path": hist_path,
        }
        record.update(metrics)
        records.append(record)
        _write_json(records_dir / f"task_{task['task_index']:04d}.json", record)

    if job_count == 1:
        return collect_codebook_multirun(config)

    return {
        "suite_id": suite_id,
        "suite_dir": str(suite_dir),
        "records_dir": str(records_dir),
        "num_runs": len(records),
        "num_total_tasks": len(tasks),
    }
