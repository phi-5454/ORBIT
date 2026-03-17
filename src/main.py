import argparse
import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from train_eval import TrainPipeline

# from train_eval import TrainPipeline

USERNAME = os.environ.get("USER")
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RESOURCE_DIR = BASE_DIR / "resources"


@hydra.main(
    version_base=None, config_path=str(BASE_DIR / "conf"), config_name="config.yaml"
)
def main(cfg: DictConfig):
    BASE_DIR = Path(__file__).resolve().parent.parent
    SRC_DIR = BASE_DIR / "src"
    RESOURCE_DIR = BASE_DIR / "resources"

    # Hydra config to a python dict
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) or {}
    assert isinstance(config, dict), f"Could not convert OmegaConf to a dict."

    # environemnt setup
    load_dotenv(dotenv_path=str(BASE_DIR / config["wandb_env"]), override=True)

    with open(str(BASE_DIR / config["train_val_files"])) as f:
        lines = f.read().splitlines()

    train_val_split = config["train_val_split"]

    train_val_parquet_path = lines[:3]
    test_parquet_path = lines[:3]

    # Explicit login (relies on WANDB_API_KEY being in the env)
    wandb.login()

    # TODO: pass a sub-part of the config
    p = TrainPipeline(
        config=config,
        train_files=train_val_parquet_path,
        val_files=train_val_parquet_path,
    )
    p.run()

    print("Done running training pipeline")

    # TODO: log evaluation results with wandb

    wandb.finish()  # Forces WandB to sync the final data and close the run cleanly
    os.system(f"python -m wandb sync --sync-all")


if __name__ == "__main__":
    main()
