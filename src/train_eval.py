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
from data_loading import ParquetDataModule
from lightning_defs import PHA_FSQ_VAE
from torch_modules import *


class TrainPipeline:
    # TODO: handle all the params
    def __init__(self, config, train_files=[], val_files=[], test_files=[]) -> None:
        self.config = config
        self.train_datamodule = ParquetDataModule(train_files)
        if self.config["run_validation"]:
            self.val_datamodule = ParquetDataModule(val_files)
        if self.config["run_test"]:
            self.test_datamodule = ParquetDataModule(test_files)
        self.logger = L.pytorch.loggers.WandbLogger(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            # name="whole_event_pha_fsq",
            log_model="all",  # Note: If checkpoints become too large, set this to False
        )

        print(self.logger)

        # Init callbacks
        lr_monitor = LearningRateMonitor(logging_interval="step")
        # assert isinstance(lr_monitor, L.Callback)

        # Initialize Trainer
        self.trainer = L.Trainer(
            # max_epochs=10,
            default_root_dir="outputs/"
            max_epochs=1,
            accelerator="auto",
            logger=self.logger,
            gradient_clip_val=1.0,  # Crucial: Prevents early divergence from log(0) or padding math
            callbacks=[lr_monitor],
            # limit_train_batches=1,
            # limit_test_batches=1,
            # limit_val_batches=1,
            # fast_dev_run=True,
        )

    def run(self, run_validation=True, run_test=False):

        # TODO: Assumes a wandb logger
        active_config = dict(self.logger.experiment.config)
        # Initialize Model
        model = PHA_FSQ_VAE(input_dim=3, hidden_dim=64, lr=1e-3)

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
