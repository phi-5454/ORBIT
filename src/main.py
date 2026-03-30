import argparse
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from plotting import replot_jet_structure
from train_eval import TrainPipeline
import datetime
import uuid

# from train_eval import TrainPipeline

USERNAME = os.environ.get("USER")
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RESOURCE_DIR = BASE_DIR / "resources"

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

@hydra.main(
    version_base=None, config_path=str(BASE_DIR / "conf"), config_name="config.yaml"
)
def main(cfg: DictConfig):
    if(cfg["replot_only"]):
        npz_files = [f"{cfg["replot"]["in_base_dir"]}/{f}" for f in cfg["replot"]["in_files"]]
        # TODO: Put the output plots into the original run's directory.
        replot_jet_structure(npz_files=npz_files, run_labels=cfg["replot"]["run_labels"], output_dir=f"{cfg["replot"]["in_base_dir"]}/combined_plots")
        return

    unique_run_name = make_run_name(cfg["run_name"])
    BASE_DIR = Path(__file__).resolve().parent.parent
    SRC_DIR = BASE_DIR / "src"
    RESOURCE_DIR = BASE_DIR / "resources"

    # Hydra config to a python dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) or {}
    assert isinstance(config, dict), f"Could not convert OmegaConf to a dict."

    # environemnt setup
    load_dotenv(dotenv_path=str(BASE_DIR / config["wandb_env"]), override=True)

    with open(str(BASE_DIR / config["train_val_files"])) as f:
        lines_train_val = f.read().splitlines()

    lines_test = None
    if(config["run_test"]):
        with open(str(BASE_DIR / config["test_files"])) as f:
            lines_test = f.read().splitlines()

    # Explicit login (relies on WANDB_API_KEY being in the env)
    wandb.login()

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
