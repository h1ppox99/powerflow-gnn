"""
PyTorch Geometric dataset wrapper for the PowerGraph benchmarks.
Simplified specifically for Optimal Power Flow (OPF) regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import mat73
import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data, InMemoryDataset

# Best practice: Load data on CPU
_DEVICE = torch.device("cpu")


class PowerGrid(InMemoryDataset):
    """
    PyG ``InMemoryDataset`` compatible with the original PowerGraph codebase
    but optimized and fixed for Optimal Power Flow (OPF).
    """

    names = {
        "uk": ["uk", "Uk", "UK", None],
        "ieee24": ["ieee24", "Ieee24", "IEEE24", None],
        "ieee39": ["ieee39", "Ieee39", "IEEE39", None],
        "ieee118": ["ieee118", "Ieee118", "IEEE118", None],
        "texas": ["texas", "texas", "texas", None],
    }

    def __init__(
        self,
        root: str | Path,
        name: str,
        datatype: str = "nodeopf", # Default to OPF, but accepts arguments
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.datatype = datatype.lower()
        self.name = name.lower()
        self._root = Path(root)
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        if self.name not in self.names:
            raise ValueError(f"Unknown dataset {name!r}. Expected one of {list(self.names)}.")
        
        # Compatibility Check: Warn if user tries to load non-OPF datatypes
        if self.datatype != "nodeopf":
            print(f"WARNING: This simplified class is optimized for 'nodeopf'. "
                  f"You requested '{self.datatype}', but it will process as OPF.")

        super().__init__(str(self._root), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------ #
    # Directory helpers (Kept consistent with your codebase)
    # ------------------------------------------------------------------ #

    @property
    def raw_dir(self) -> str:
        # Matches your previous snippet structure
        return str(self._root / self.name / self.name / "raw")

    @property
    def processed_dir(self) -> str:
        # Forces the use of the OPF processed directory to avoid conflicts
        return str(self._root / self.name / "processed_nodeopf")

    @property
    def raw_file_names(self) -> list[str]:
        # [cite_start]STRICTLY the files needed for OPF [cite: 914, 918, 910, 912]
        return [
            "edge_index_opf.mat",
            "edge_attr_opf.mat",
            "Xopf.mat",
            "Y_polar_opf.mat",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        raise RuntimeError(
            "This dataset expects the Figshare archive to be downloaded manually. "
            "Place the extracted folders under <root>/<dataset_name>/raw."
        )

    # ------------------------------------------------------------------ #
    # Processing (Fixed Logic)
    # ------------------------------------------------------------------ #

    def process(self) -> None:
        """
        Processing logic strictly for Optimal Power Flow node-level regression.
        Includes global normalization and full feature slicing.
        """
        
        # 1. Load Data Files
        try:
            edge_index_mat = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_index_opf.mat"))
            edge_attr_mat = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_attr_opf.mat"))
            X_mat = mat73.loadmat(str(Path(self.raw_dir) / "Xopf.mat"))
            Y_mat = mat73.loadmat(str(Path(self.raw_dir) / "Y_polar_opf.mat"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing OPF .mat files in {self.raw_dir}. Ensure you downloaded the correct data.")

        # 2. Extract Data (Safe Key Access)
        # Handles potential naming inconsistencies
        e_idx = edge_index_mat.get("edge_index_opf", edge_index_mat.get("edge_index"))
        e_attr = edge_attr_mat.get("edge_attr_opf", edge_attr_mat.get("edge_attr"))
        
        # Handle X and Y keys safely
        x_key = "Xopf" if "Xopf" in X_mat else "X"
        y_key = "Y_polar_opf" if "Y_polar_opf" in Y_mat else "Y_polar"

        # 3. Process Edges
        # Fix 0-based indexing for edges (MATLAB is 1-based)
        edge_index = torch.tensor(e_idx.astype(np.int32) - 1, dtype=torch.long).t().contiguous()
        
        # Normalize edge attributes (Conductance/Susceptance)
        edge_attr = torch.tensor(e_attr, dtype=torch.float)
        edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)

        # 4. Process Node Features (keep cols 0,1,3; drop col 2)
        fullX = [
            torch.tensor(sample[:, [0, 1, 3]], dtype=torch.float, device=_DEVICE)
            for sample in X_mat[x_key]
        ]
        
        # Targets
        fullY = [
            torch.tensor(sample, dtype=torch.float, device=_DEVICE) 
            for sample in Y_mat[y_key]
        ]

        # 5. Global Normalization Calculation
        # Concatenate all graphs to find the global max for scaling
        fullXcat = torch.cat(fullX, dim=0)
        fullYcat = torch.cat(fullY, dim=0)
        
        maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
        maxsX[maxsX == 0] = 1.0  # Prevent division by zero
        
        maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)
        maxsY[maxsY == 0] = 1.0

        data_list = []
        for sample_x, sample_y in zip(fullX, fullY):
            # Create mask: True where we have a non-zero target (the Unknowns)
            # [cite_start]This implements the masking strategy from the paper [cite: 133-137]
            mask = sample_y != 0
            
            data = Data(
                x=sample_x / maxsX,       # Normalize Inputs
                edge_index=edge_index,    # Topology
                edge_attr=edge_attr,      # Edge Features
                y=sample_y / maxsY,       # Normalize Targets
                maxs=maxsY,               # Store for de-normalization
                mask=mask                 # Boolean mask for loss calculation
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save processed data
        torch.save(self.collate(data_list), self.processed_paths[0])
