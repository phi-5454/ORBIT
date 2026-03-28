import lightning as L
import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

# TODO: This is a bit farouche
feature_cols = ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PT", "L1T_PUPPIPart_PID", "L1T_PUPPIPart_PuppiW"]

# For quantizing inputs
class UniformQuantizerSTE(nn.Module):
    def __init__(self, bit_depth: int, lsb: float = 1/500, signed: bool = True):
        super().__init__()
        self.bit_depth = bit_depth
        self.lsb = lsb
        self.signed = signed
        
        # Calculate the allowable integer ranges based on bit depth
        # Two's complement
        if self.signed:
            # e.g., 8-bit signed: -128 to 127
            self.q_min = -(2 ** (self.bit_depth - 1))
            self.q_max = (2 ** (self.bit_depth - 1)) - 1
        # Simple unsigned binary representation
        else:
            # e.g., 8-bit unsigned: 0 to 255
            self.q_min = 0
            self.q_max = (2 ** self.bit_depth) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Scale to the integer domain
        x_scaled = x / self.lsb
        
        # 2. Round to nearest integer
        x_rounded = torch.round(x_scaled)
        # 3. Clamp (clip) to the maximum/minimum allowable bit depth values
        x_clamped = torch.clamp(x_rounded, self.q_min, self.q_max)
        
        # 4. Scale back to the physical physics domain
        x_quantized = x_clamped * self.lsb
        
        # Straight-through estimator
        return x + (x_quantized - x).detach()

class PreprocessTranformer:
    # TODO: Integrate the PUPPI_weight cut
    # TODO: This is all hardcoded for now
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.col_name = "L1T_PUPPIPart_PT"
        # Find the integer index of the transformed column for tensor operations later
        self.col_idx = feature_cols.index(self.col_name) if self.col_name else None

    # TODO: trasnform to realistic bit depth.
    def truncate_quantize(self, df):
        # ..truncate an quantize the given features
        pass

    def forward_dataframe(self, df):


        """Applies the forward transform to the Pandas DataFrame before training."""
        #if self.col_name and self.col_name in df.columns:

        # Apply same transforms as OmniJet
        df["L1T_PUPPIPart_PT"] = df["L1T_PUPPIPart_PT"].apply(
                lambda x: np.log(np.asarray(x) + self.epsilon) - 1.8
                )
        df["L1T_PUPPIPart_Phi"] = df["L1T_PUPPIPart_Phi"] / np.pi
        df["L1T_PUPPIPart_Eta"] = df["L1T_PUPPIPart_Eta"] / 3

        return df

    def inverse_tensor(self, tensor):
        """Applies the inverse transform to the PyTorch prediction tensor."""
        # Create a clone to avoid in-place modification issues during backprop
        tensor_inv = tensor.clone()
        tensor_inv[..., 2] = (
            torch.exp(tensor[..., 2]) - self.epsilon + 1.8
        )
        tensor_inv[..., 0] = tensor[..., 0] * 3
        # Azimuthal angle, Modulo 2pi
        tensor_inv[..., 1] = (tensor[..., 1] * torch.pi + torch.pi) % (2 * torch.pi) - torch.pi
        
        return tensor_inv

        return tensor


class ParquetFeatureDataset(IterableDataset):
    def __init__(self, parquet_dirs, features, max_particles=256, batch_size=32):
        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.features = features
        self.max_particles = max_particles
        self.batch_size = batch_size

    def __iter__(self):
        # Read only the required physical features
        batches = self.dataset.to_batches(
            columns=self.features, batch_size=self.batch_size
        )

        for batch in batches:
            # Convert to Pandas. Each cell now contains a numpy array of particles.
            df = batch.to_pandas()
            df = PreprocessTranformer().forward_dataframe(df)

            event_tensors = []

            PUPPI_cutoff = 0.05

            # Zip the columns. This iterates row-by-row (event-by-event)
            # TODO: PID
            for eta_arr, phi_arr, pt_arr, pid_arr, puppiw_arr in zip(
                df[self.features[0]], df[self.features[1]], df[self.features[2]], df[self.features[3]], df[self.features[4]] 
            ):
                PUPPI_mask = puppiw_arr > PUPPI_cutoff


                # Skip empty events (e.g., zero particles passed the trigger)
                if len(eta_arr) == 0:
                    continue

                # Stack the 1D arrays into a [N, 3] matrix for this specific event
                coords = np.column_stack([eta_arr[PUPPI_mask], phi_arr[PUPPI_mask], pt_arr[PUPPI_mask]]).astype(np.float32)

                # Apply Log transform safely directly to the PT column (Index 2)
                # coords[:, 2] = np.log(coords[:, 2] + 1e-8)

                # Enforce the maximum particles limit
                # TODO: Split the event into many smaller if exceed max. particle count
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
    def __init__(self, parquet_dirs_train, parquet_dirs_val, parquet_dirs_test, features=feature_cols, window_particles=256):
        super().__init__()
        self.parquet_dirs_train = parquet_dirs_train
        self.parquet_dirs_val = parquet_dirs_val
        self.parquet_dirs_test = parquet_dirs_test
        self.features = features
        self.window_particles = window_particles

    def train_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_train, self.features, self.window_particles)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_val, self.features, self.window_particles)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_test, self.features, self.window_particles)
        # Note: If num_workers > 0 on IterableDataset, you need a custom worker_init_fn
        # to prevent data duplication. Kept at 0 for safe out-of-the-box running.
        return DataLoader(dataset, batch_size=None, num_workers=0)
