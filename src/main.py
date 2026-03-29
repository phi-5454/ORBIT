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
