import os

import fastjet
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pylorentz
import seaborn as sns
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb
from data_loading import ParquetDataModule, feature_cols
from lightning_defs import PHA_FSQ_VAE
from torch_modules import *
from lightning.pytorch.callbacks import ModelSummary

class TrainPipeline:
    # TODO: handle all the params
    def __init__(self, config, train_files=[], val_files=[], test_files=[]) -> None:
        self.config = config

        self.train_datamodule = ParquetDataModule(train_files, window_particles=config["model"]["window_particles"])
        if self.config["run_validation"]:
            self.val_datamodule = ParquetDataModule(val_files, window_particles=config["model"]["window_particles"])
        if self.config["run_test"]:
            self.test_datamodule = ParquetDataModule(test_files, window_particles=config["model"]["window_particles"])

        # TODO: We assume we want to log with WandB.
        self.logger = L.pytorch.loggers.WandbLogger(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            # name="whole_event_pha_fsq",
            log_model="all",  # Note: If checkpoints become too large, set this to False
        )

        # Init callbacks
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Initialize Trainer
        self.trainer = L.Trainer(
            logger=self.logger,
            callbacks=[lr_monitor, ModelSummary(max_depth=-1)],
            **config["trainer"],
        )

    def run(self, run_validation=True, run_test=False):

        # TODO: Assumes a wandb logger
        wandb_active_config = dict(self.logger.experiment.config)

        # Initialize Model
        # model = PHA_FSQ_VAE(input_dim=3, hidden_dim=64, lr=1e-3)
        model_cfg = self.config["model"]
        model_cfg["input_dim"] = 3 # TODO: automate this 
        model = PHA_FSQ_VAE(model_cfg)
        # model = PHA_FSQ_VAE(
        # input_dim=len(feature_cols),
        # hidden_dim=model_cfg["hidden_dim"],
        # lr=model_cfg["lr"],
        # )

        # Tell the logger to "watch" the model's gradients and weights
        self.logger.watch(model, log="all", log_freq=10, log_graph=True)

        # Train the model
        self.trainer.fit(model, datamodule=self.train_datamodule)

        if run_validation:
            self.trainer.validate(model, datamodule=self.val_datamodule)

        if run_test:
            self.trainer.test(model, datamodule=self.test_datamodule)

        # TODO: assumes a WandB logger
        self.logger.experiment.unwatch(model)
