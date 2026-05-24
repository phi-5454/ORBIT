import datetime
import os
import uuid

import lightning as L
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
from data_loading import ParquetDataModule, feature_cols
from lightning_defs import PHA_FSQ_VAE
from torch_modules import *
from lightning.pytorch.callbacks import ModelSummary

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

class TrainPipeline:
    # TODO: handle all the params
    def __init__(self, config, unique_run_name, train_val_files=[], test_files=[]) -> None:
        self.config = config
        train_val_split = np.array(self.config["train_val_split"])
        train_val_split_norm = train_val_split / np.sum(train_val_split)
        num_train_val = len(train_val_files)
        num_train = int(num_train_val * train_val_split_norm[0])

        #shuffle
        seed = config.get("split_random_seed", 42)
        np.random.seed(seed)
        indices = np.random.permutation(num_train_val)
        train_val_files = np.array(train_val_files)[indices]

        train_files = [os.path.expanduser(path) for path in train_val_files[:max(num_train,1)]]
        val_files = [os.path.expanduser(path) for path in train_val_files[min(num_train, num_train_val - 1):]]
        test_files = [os.path.expanduser(path) for path in test_files] if test_files else test_files
        data_level = config.get("data", {}).get("level", "particle")
        if data_level not in config.get("data", {}):
            raise ValueError(f"Unknown data.level '{data_level}'. Add a matching config.data.{data_level} block.")
        data_cfg = config.get("data", {}).get(data_level, {})

        self.datamodule = ParquetDataModule(
            train_files,
            val_files,
            test_files,
            features=data_cfg.get("features"),
            selected_features=data_cfg.get("selected_features"),
            window_particles=config["model"]["window_particles"],
            batch_size=config["model"]["batch_size"],
            num_workers=config["num_dataload_workers"],
            shuffle_train=config.get("shuffle_train", True),
            shuffle_seed=config.get("split_random_seed", 42),
            mask_column=data_cfg.get("mask_column"),
            mask_min_value=data_cfg.get("mask_min_value", 0.0),
        )

        self.unique_run_name = unique_run_name

        # TODO: We assume we want to log with WandB.
        self.logger = L.pytorch.loggers.WandbLogger(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=self.unique_run_name,
            log_model="all",  # Note: If checkpoints become too large, set this to False
            save_dir=config["output_dir"]
        )

        # Upload config to wandb.
        self.logger.experiment.config.update(self.config)


        # Init callbacks
        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        from lightning.pytorch.callbacks import EarlyStopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.get("earlystopping_patience", 5),
            verbose=True,
            mode="min"
        )

        # Initialize Trainer
        self.trainer = L.Trainer(
            logger=self.logger,
            callbacks=[lr_monitor, early_stop_callback, ModelSummary(max_depth=-1)],
            **config["trainer"],
            profiler="simple"
        )
    def run(self, run_validation=True, run_test=False):

        mode=self.config["mode"]
        # TODO: Assumes a wandb logger
        wandb_active_config = dict(self.logger.experiment.config)
        # Initialize Model
        # model = PHA_FSQ_VAE(input_dim=3, hidden_dim=64, lr=1e-3)
        model_cfg = self.config["model"]
        
        # Automate input_dim based on the features selected in the datamodule
        model_cfg["input_dim"] = len(self.datamodule.selected_features)

        # TODO: Handle the outpts more centrally
        output_dir = f"{self.config["output_dir"]}/{self.unique_run_name}"
        model = PHA_FSQ_VAE(
            model_cfg,
            output_dir=output_dir,
            data_level=self.config.get("data", {}).get("level", "particle"),
            jet_metric_workers=self.config.get("num_eval_workers", 0),
            jet_metric_backend=self.config.get("eval_worker_backend", "process"),
            jet_metric_start_method=self.config.get(
                "eval_worker_start_method", "forkserver"
            ),
        )
        # model = PHA_FSQ_VAE(
        # input_dim=len(feature_cols),
        # hidden_dim=model_cfg["hidden_dim"],
        # lr=model_cfg["lr"],
        # )

        # Tell the logger to "watch" the model's gradients and weights
        self.logger.watch(model, log="all", log_freq=10, log_graph=True)

        if mode == "train":

            # Train the model
            self.trainer.fit(model, datamodule=self.datamodule)

            if run_validation:
                self.trainer.validate(model, datamodule=self.datamodule)

            if run_test:
                self.trainer.test(model, datamodule=self.datamodule)

        if mode == "test_only":
            print(f"Skipping training. Loading weights from: {self.config["ckpt_path"]}")
        
            # PyTorch Lightning handles the weight loading automatically!
            # You do NOT need to do model.load_state_dict() manually.
            self.trainer.test(model, datamodule=self.datamodule, ckpt_path=self.config["ckpt_path"])



        # TODO: assumes a WandB logger
        self.logger.experiment.unwatch(model)
