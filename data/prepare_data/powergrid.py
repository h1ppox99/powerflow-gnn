"""PyTorch Geometric dataset wrapper for the PowerGraph benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import mat73
import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data, InMemoryDataset


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PowerGrid(InMemoryDataset):
    """PyG ``InMemoryDataset`` for the PowerGraph node-level benchmarks."""

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
        datatype: str = "binary",
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

        super().__init__(str(self._root), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------ #
    # Directory helpers
    # ------------------------------------------------------------------ #

    @property
    def raw_dir(self) -> str:
        return str(self._root / self.name / "raw")

    @property
    def processed_dir(self) -> str:
        suffix = {
            "binary": "processed_b",
            "regression": "processed_r",
            "multiclass": "processed_m",
            "node": "processed_node",
            "nodeopf": "processed_nodeopf",
        }.get(self.datatype)
        if suffix is None:
            raise ValueError(
                f"Unsupported datatype {self.datatype!r}. "
                "Expected one of binary/regression/multiclass/node/nodeopf."
            )
        return str(self._root / self.name / suffix)

    @property
    def raw_file_names(self) -> list[str]:
        if self.datatype in {"binary", "regression", "multiclass"}:
            return [
                "Bf.mat",
                "blist.mat",
                "Ef.mat",
                "exp.mat",
                "of_bi.mat",
                "of_mc.mat",
                "of_reg.mat",
                "edge_index.mat",
                "edge_attr.mat",
            ]
        if self.datatype == "nodeopf":
            return [
                "edge_index_opf.mat",
                "edge_attr_opf.mat",
                "Xopf.mat",
                "Y_polar_opf.mat",
            ]
        # Default to node-level power flow data.
        return [
            "edge_index.mat",
            "edge_attr.mat",
            "X.mat",
            "Y_polar.mat",
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:  # pragma: no cover - manual download only
        raise RuntimeError(
            "This dataset expects the Figshare archive to be downloaded manually. "
            "Place the extracted folders under <root>/<dataset_name>/raw."
        )

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #

    def process(self) -> None:
        def th_delete(tensor: torch.Tensor, indices: list[int] | torch.Tensor) -> torch.Tensor:
            mask = torch.ones_like(tensor, dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        datatype = self.datatype

        if datatype in {"binary", "regression", "multiclass"}:
            edge_order = mat73.loadmat(str(Path(self.raw_dir) / "blist.mat"))
            edge_order = torch.tensor(edge_order["bList"] - 1)

            of_bi = mat73.loadmat(str(Path(self.raw_dir) / "of_bi.mat"))["output_features"]
            of_reg = mat73.loadmat(str(Path(self.raw_dir) / "of_reg.mat"))["dns_MW"]
            of_mc = mat73.loadmat(str(Path(self.raw_dir) / "of_mc.mat"))["category"]

            node_f = mat73.loadmat(str(Path(self.raw_dir) / "Bf.mat"))["B_f_tot"]
            edge_f = mat73.loadmat(str(Path(self.raw_dir) / "Ef.mat"))["E_f_post"]
            exp_mask = mat73.loadmat(str(Path(self.raw_dir) / "exp.mat"))["explainations"]

            data_list = []

            for i in range(len(node_f)):
                x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape(-1, 3).to(_DEVICE)
                features = torch.tensor(edge_f[i][0], dtype=torch.float32)
                mask = torch.zeros(len(features), 1)
                if exp_mask[i][0] is not None:
                    mask[exp_mask[i][0].astype("int") - 1] = 1

                cont = [j for j in range(len(features)) if np.all(np.array(features[j])) == 0]
                e_mask_post = th_delete(mask, cont)
                e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(_DEVICE)

                f_tot = th_delete(features, cont).reshape(-1, 4).type(torch.float32)
                f_totw = torch.cat((f_tot, f_tot), 0).to(_DEVICE)

                edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
                edge_iwr = torch.fliplr(edge_iw)
                edge_iw = torch.cat((edge_iw, edge_iwr), 0).t().contiguous().to(_DEVICE)

                if datatype == "binary":
                    y = torch.tensor(of_bi[i][0], dtype=torch.float, device=_DEVICE).view(1, -1)
                elif datatype == "regression":
                    y = torch.tensor(of_reg[i], dtype=torch.float, device=_DEVICE).view(1, -1)
                else:
                    y = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.float, device=_DEVICE).view(1, -1)

                data = Data(
                    x=x,
                    edge_index=edge_iw,
                    edge_attr=f_totw,
                    y=y.to(torch.float),
                    edge_mask=e_mask_post,
                    idx=i,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        elif datatype == "nodeopf":
            edge_index = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_index_opf.mat"))["edge_index"]
            edge_attr = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_attr_opf.mat"))["edge_attr"]
            X = mat73.loadmat(str(Path(self.raw_dir) / "Xopf.mat"))
            Y = mat73.loadmat(str(Path(self.raw_dir) / "Y_polar_opf.mat"))
            edge_order = torch.tensor(edge_index.astype(np.int32) - 1, dtype=torch.long).t().contiguous().to(_DEVICE)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)

            fullX = [torch.tensor(sample[:, [0, 1, 3]], dtype=torch.float, device=_DEVICE) for sample in X["X"]]
            fullY = [torch.tensor(sample, dtype=torch.float, device=_DEVICE) for sample in Y["Y_polar"]]
            fullXcat = torch.cat(fullX, dim=0)
            fullYcat = torch.cat(fullY, dim=0)
            maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)

            data_list = []
            for sample_x, sample_y in zip(fullX, fullY):
                mask = sample_y != 0
                data = Data(
                    x=sample_x / maxsX,
                    edge_index=edge_order,
                    y=sample_y / maxsY,
                    edge_attr=edge_attr,
                    maxs=maxsY,
                    mask=mask,
                ).to(_DEVICE)
                data_list.append(data)
        else:
            edge_index = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_index.mat"))["edge_index"]
            edge_attr = scipy.io.loadmat(str(Path(self.raw_dir) / "edge_attr.mat"))["edge_attr"]
            X = mat73.loadmat(str(Path(self.raw_dir) / "X.mat"))
            Y = mat73.loadmat(str(Path(self.raw_dir) / "Y_polar.mat"))
            edge_order = torch.tensor(edge_index.astype(np.int32) - 1, dtype=torch.long).t().contiguous().to(_DEVICE)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)

            fullX = [torch.tensor(sample, dtype=torch.float, device=_DEVICE) for sample in X["Xpf"]]
            fullY = [torch.tensor(sample, dtype=torch.float, device=_DEVICE) for sample in Y["Y_polarpf"]]
            fullXcat = torch.cat(fullX, dim=0)
            fullYcat = torch.cat(fullY, dim=0)
            maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)

            data_list = []
            for sample_x, sample_y in zip(fullX, fullY):
                mask = sample_y != 0
                data = Data(
                    x=sample_x / maxsX,
                    edge_index=edge_order,
                    y=sample_y / maxsY,
                    edge_attr=edge_attr,
                    maxs=maxsY,
                    mask=mask,
                ).to(_DEVICE)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
