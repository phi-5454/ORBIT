import os
from pathlib import Path


def _eos_home_alias(path):
    parts = path.parts
    if len(parts) >= 5 and parts[:3] == ("/", "eos", "user"):
        initial = parts[3]
        return Path("/", "eos", f"home-{initial}", *parts[4:])
    return None


def _canonical_repo_root(path):
    alias = _eos_home_alias(path)
    if alias is not None and alias.exists():
        return alias
    return path


BASE_DIR = _canonical_repo_root(Path(__file__).resolve().parent.parent)

import hydra
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from .multirun import run_codebook_multirun
from .plotting import replot_reconstruction_comparison
from .train_eval import TrainPipeline
import datetime
import uuid

USERNAME = os.environ.get("USER")

def _as_list(value, field_name):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"{field_name} must be a string or a list of strings, got {type(value).__name__}")


def _resolve_path(path):
    path = Path(os.path.expanduser(path))
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _read_file_lists(file_lists, field_name):
    parquet_files = []
    for file_list in _as_list(file_lists, field_name):
        file_list_path = _resolve_path(file_list)
        with open(file_list_path) as f:
            parquet_files.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    return parquet_files


def make_run_name(base_name=None):
    if base_name is None:
        return None

    # 1. Generate a sortable timestamp (e.g., "20260329_174411")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Generate a short, unique random ID (e.g., "4f8a2c")
    short_id = uuid.uuid4().hex[:6]

    # 3. Construct the final name
    # Result looks like: "baseline_vae_20260329_174411_4f8a2c"
    unique_run_name = f"{base_name}_{timestamp}_{short_id}"
    return unique_run_name


def _save_figures(figures, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, fig in figures.items():
        fig.savefig(output_dir / filename)
        plt.close(fig)


def _load_npz_dict(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}

@hydra.main(
    version_base=None, config_path=str(BASE_DIR / "conf"), config_name="config.yaml"
)
def main(cfg: DictConfig):
    if(cfg["replot_only"]):
        npz_files = [
            f"{cfg['replot']['in_base_dir']}/{filename}"
            for filename in cfg["replot"]["in_files"]
        ]
        runs_data = [_load_npz_dict(path) for path in npz_files]
        _save_figures(
            replot_reconstruction_comparison(
                runs_data=runs_data,
                run_labels=cfg["replot"]["run_labels"],
                data_level=cfg["data"]["level"],
            ),
            f"{cfg['replot']['in_base_dir']}/combined_plots",
        )
        return

    # Hydra config to a python dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) or {}
    assert isinstance(config, dict), f"Could not convert OmegaConf to a dict."

    # environemnt setup
    load_dotenv(dotenv_path=str(BASE_DIR / config["wandb_env"]), override=True)

    lines_train_val = _read_file_lists(config["train_val_files"], "train_val_files")

    lines_test = None
    if(config["run_test"]):
        lines_test = _read_file_lists(config["test_files"], "test_files")

    # Explicit login (relies on WANDB_API_KEY being in the env)
    wandb.login()

    if config.get("multirun", {}).get("enabled", False):
        result = run_codebook_multirun(
            config=config,
            train_val_files=lines_train_val,
            test_files=lines_test,
        )
        print(f"Done running multirun suite: {result['suite_dir']}")
        return

    unique_run_name = make_run_name(cfg["run_name"])

    # TODO: pass a sub-part of the config
    p = TrainPipeline(
        config=config,
        unique_run_name=unique_run_name,
        train_val_files=lines_train_val,
        test_files=lines_test,
    )
    p.run(config["run_validation"], config["run_test"])

    print("Done running training pipeline")

    # TODO: log evaluation results with wandb

    wandb.finish()  # Forces WandB to sync the final data and close the run cleanly
    os.system(f"python -m wandb sync --sync-all")


if __name__ == "__main__":
    main()
