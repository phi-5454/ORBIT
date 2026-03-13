import keras
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

# This is a bit farouche
feature_cols = ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PT"]


class PreprocessTranformer:
    def __init__(self, log_column_name, feature_names, epsilon=1e-8):
        self.col_name = log_column_name
        self.epsilon = epsilon
        # Find the integer index of the transformed column for tensor operations later
        self.col_idx = feature_names.index(log_column_name) if log_column_name else None

    # TODO: trasnform to realistic bit depth.
    def truncate_quantize(self, df):
        # ..truncate an quantize the given features
        pass

    def forward_dataframe(self, df):
        """Applies the forward transform to the Pandas DataFrame before training."""
        if self.col_name and self.col_name in df.columns:
            df[self.col_name] = np.log(df[self.col_name] + self.epsilon)
        return df

    def inverse_tensor(self, tensor):
        """Applies the inverse transform to the PyTorch prediction tensor."""
        if self.col_idx is not None:
            # Create a clone to avoid in-place modification issues during backprop
            tensor_inv = tensor.clone()
            tensor_inv[:, self.col_idx] = (
                torch.exp(tensor[:, self.col_idx]) - self.epsilon
            )
            return tensor_inv
        return tensor


class ParquetFeatureDataset(IterableDataset):
    def __init__(self, parquet_dirs, features, max_particles=256, batch_size=32):
        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.features = features
        self.max_particles = max_particles
        self.batch_size = batch_size

    def __iter__(self):
        # Read only the 3 physical features
        batches = self.dataset.to_batches(
            columns=self.features, batch_size=self.batch_size
        )

        for batch in batches:
            # Convert to Pandas. Each cell now contains a numpy array of particles.
            df = batch.to_pandas()

            event_tensors = []

            # Zip the columns. This iterates row-by-row (event-by-event)
            for eta_arr, phi_arr, pt_arr in zip(
                df[self.features[0]], df[self.features[1]], df[self.features[2]]
            ):

                # Skip empty events (e.g., zero particles passed the trigger)
                if len(eta_arr) == 0:
                    continue

                # Stack the 1D arrays into a [N, 3] matrix for this specific event
                coords = np.column_stack([eta_arr, phi_arr, pt_arr]).astype(np.float32)

                # Apply Log transform safely directly to the PT column (Index 2)
                coords[:, 2] = np.log(coords[:, 2] + 1e-8)

                # Enforce the maximum particles limit
                coords = coords[: self.max_particles]
                event_tensors.append(torch.tensor(coords))

            if not event_tensors:
                continue

            # Pad the variable-length events with 0.0 to create a square batch tensor
            padded_events = pad_sequence(
                event_tensors, batch_first=True, padding_value=0.0
            )

            # Force shape to [Batch, 256, 3] in case the largest event in this specific batch was < 256
            pad_len = self.max_particles - padded_events.shape[1]
            if pad_len > 0:
                padded_events = F.pad(padded_events, (0, 0, 0, pad_len), value=0.0)

            # Create Mask: True for REAL particles, False for PADDING
            mask = padded_events[:, :, 2] != 0.0

            yield padded_events, mask


class ParquetDataModule(L.LightningDataModule):
    def __init__(self, parquet_dir, features=feature_cols):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.features = features

    def train_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dir, self.features)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dir, self.features)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dir, self.features)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)
