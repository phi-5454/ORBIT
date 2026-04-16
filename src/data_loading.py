from line_profiler import profile
import lightning as L
import numpy as np
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import awkward as ak

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
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def forward_awkward(self, ak_array):
        """Applies the forward transform to the Awkward Array (vectorized)."""
        # We need to make a copy if we don't want to modify the original Arrow-backed array
        # or just return a new record.
        new_fields = {}
        for field in ak_array.fields:
            if field == "L1T_PUPPIPart_PT":
                new_fields[field] = np.log(ak_array[field] + self.epsilon) - 1.8
            elif field == "L1T_PUPPIPart_Phi":
                new_fields[field] = ak_array[field] / np.pi
            elif field == "L1T_PUPPIPart_Eta":
                new_fields[field] = ak_array[field] / 3
            else:
                new_fields[field] = ak_array[field]
        
        return ak.Array(new_fields)

    def inverse_tensor(self, tensor):
        """Applies the inverse transform to the PyTorch prediction tensor."""
        # Create a clone to avoid in-place modification issues during backprop
        tensor_inv = tensor.clone()
        tensor_inv[..., 2] = (
            torch.exp(tensor[..., 2] + 1.8) - self.epsilon
        )
        tensor_inv[..., 0] = tensor[..., 0] * 3
        # Azimuthal angle, Modulo 2pi
        tensor_inv[..., 1] = (tensor[..., 1] * torch.pi + torch.pi) % (2 * torch.pi) - torch.pi
        
        return tensor_inv



class ParquetFeatureDataset(IterableDataset):
    def __init__(self, parquet_dirs, features, selected_features=None, max_particles=256, batch_size=32):
        # We load the base dataset just to map the files
        self.dataset = ds.dataset(parquet_dirs, format="parquet")
        self.features = features
        self.selected_features = selected_features or ["L1T_PUPPIPart_Eta", "L1T_PUPPIPart_Phi", "L1T_PUPPIPart_PT"]
        self.max_particles = max_particles
        self.batch_size = batch_size
        self.preprocess = PreprocessTranformer()

    @profile
    def __iter__(self):
        # 1. GET WORKER INFO
        worker_info = get_worker_info()
        files = self.dataset.files

        # 2. SHARD THE FILES ACROSS WORKERS
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = files[worker_id::num_workers]
            if not files: return

        # 3. CREATE A WORKER-SPECIFIC DATASET
        worker_dataset = ds.dataset(files, format="parquet")

        # Read only the required physical features for this specific worker
        batches = worker_dataset.to_batches(
            columns=self.features, batch_size=self.batch_size
        )

        for batch in batches:
            # 1. Convert Arrow RecordBatch to Awkward Array
            ak_batch = ak.from_arrow(batch)
            
            # 2. Apply Preprocessing (vectorized)
            ak_batch = self.preprocess.forward_awkward(ak_batch)
            
            # 3. Filtering by PUPPI weight
            PUPPI_cutoff = 0.05
            if "L1T_PUPPIPart_PuppiW" in ak_batch.fields:
                # This applies the mask to each list in the batch
                mask = ak_batch["L1T_PUPPIPart_PuppiW"] > PUPPI_cutoff
                ak_batch = ak_batch[mask]
            
            # Remove empty events
            ak_batch = ak_batch[ak.num(ak_batch[self.selected_features[0]]) > 0]
            if len(ak_batch) == 0:
                continue

            # 4. Select and Stack Features
            # Create a list of arrays, then stack them along a new last axis
            selected_data = [ak_batch[f][:, :, np.newaxis] for f in self.selected_features]
            stacked = ak.concatenate(selected_data, axis=-1)
            
            # 5. Pad / Truncate to max_particles
            # ak.pad_none pads along the specified axis. clip=True truncates if longer.
            padded = ak.pad_none(stacked, self.max_particles, axis=1, clip=True)
            # Fill Nones with 0.0 to get a regular NumPy-compatible array
            filled = ak.fill_none(padded, 0.0)
            
            # 6. Convert to PyTorch Tensor
            # Convert to numpy first, then torch.
            np_batch = ak.to_numpy(filled).astype(np.float32)
            tensor_batch = torch.from_numpy(np_batch)
            
            # 7. Create Mask
            # The mask should be True for real particles, False for padding.
            # We can generate it by padding a ones array.
            ones = ak.ones_like(ak_batch[self.selected_features[0]])
            mask_padded = ak.pad_none(ones, self.max_particles, axis=1, clip=True)
            mask_filled = ak.fill_none(mask_padded, 0.0)
            torch_mask = torch.from_numpy(ak.to_numpy(mask_filled).astype(bool))

            yield tensor_batch, torch_mask


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
        self.prefetch_factor = 4 if num_workers > 0 else None

    def train_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_train, self.features, self.selected_features, self.window_particles)
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers, 
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_val, self.features, self.selected_features, self.window_particles)
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers, 
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        dataset = ParquetFeatureDataset(self.parquet_dirs_test, self.features, self.selected_features, self.window_particles)
        return DataLoader(
            dataset, 
            batch_size=None, 
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
