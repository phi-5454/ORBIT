from line_profiler import profile
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

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
    def __init__(self, parquet_dirs, features, selected_features=None, max_particles=256, batch_size=32):
        # We load the base dataset just to map the files
        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.features = features
        self.selected_features = selected_features or ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PT"]
        self.max_particles = max_particles
        self.batch_size = batch_size

    @profile
    def __iter__(self):
        # 1. GET WORKER INFO
        worker_info = get_worker_info()
        files = self.dataset.files

        # 2. SHARD THE FILES ACROSS WORKERS
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Slice the file list: start at worker_id, step by num_workers
            files = files[worker_id::num_workers]
            
            # Edge case: If there are more workers than files, some workers get nothing
            if not files:
                return

        # 3. CREATE A WORKER-SPECIFIC DATASET
        worker_dataset = ds.dataset(files, format="parquet")

        # Read only the required physical features for this specific worker
        batches = worker_dataset.to_batches(
            columns=self.features, batch_size=self.batch_size
        )

        for batch in batches:
            df = batch.to_pandas()
            df = PreprocessTranformer().forward_dataframe(df)

            event_tensors = []
            PUPPI_cutoff = 0.05
            
            # Map selected features to their data
            cols_to_zip = [df[f] for f in self.selected_features]
            
            # We always need PuppiW for the mask if available
            has_puppi = "L1T_PUPPIPart_PuppiW" in df.columns
            if has_puppi:
                puppiw_data = df["L1T_PUPPIPart_PuppiW"]
            else:
                puppiw_data = [None] * len(df)

            for i, zipped_row in enumerate(zip(*cols_to_zip)):
                if has_puppi:
                    puppiw_arr = puppiw_data.iloc[i]
                    PUPPI_mask = puppiw_arr > PUPPI_cutoff
                else:
                    # Fallback if PuppiW is not present
                    PUPPI_mask = np.ones_like(zipped_row[0], dtype=bool)

                if len(zipped_row[0]) == 0:
                    continue

                # Stack the selected features for masked particles
                coords = np.column_stack([
                    f_arr[PUPPI_mask] for f_arr in zipped_row
                ]).astype(np.float32)

                coords = coords[: self.max_particles]
                event_tensors.append(torch.tensor(coords))

            if not event_tensors:
                continue

            padded_events = pad_sequence(
                event_tensors, batch_first=True, padding_value=0.0
            )

            pad_len = self.max_particles - padded_events.shape[1]
            if pad_len > 0:
                padded_events = F.pad(padded_events, (0, 0, 0, pad_len), value=0.0)

            # Mask is True for real particles, False for padding
            # A particle is real if it has non-zero PT (assuming PT is one of the features)
            # or more generally if any of its features are non-zero.
            mask = (padded_events != 0).any(dim=-1)

            yield padded_events, mask


class ParquetDataModule(L.LightningDataModule):
    def __init__(self, parquet_dirs_train, parquet_dirs_val, parquet_dirs_test, features=feature_cols, selected_features=None, window_particles=256, num_workers=0):
        super().__init__()
        self.parquet_dirs_train = parquet_dirs_train
        self.parquet_dirs_val = parquet_dirs_val
        self.parquet_dirs_test = parquet_dirs_test
        self.features = features
        self.selected_features = selected_features or ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PT"]
        self.window_particles = window_particles
        self.num_workers = num_workers
        self.prefetch_factor = 4

    def train_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_train, self.features, self.selected_features, self.window_particles)
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers, 
            # FIX: Automatically disables persistence when debugging with 0 workers
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_val, self.features, self.selected_features, self.window_particles)
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers, 
            # FIX: Automatically disables persistence when debugging with 0 workers
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_test, self.features, self.selected_features, self.window_particles)
        # Test loaders generally shouldn't use persistent workers anyway, 
        # since they only run once at the very end.
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
