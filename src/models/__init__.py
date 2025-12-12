"""Public models exposed by the `src.models` package."""

from .graphsage_pi import GraphSAGE_PI
from .transformer_baseline import TransformerBaseline
from .transformer_vn import TransformerConvVN
from .HeGGA import HeGGA, build_model as build_HeGGA
from .hegga_noattn import HeGGANoAttn, build_model as build_hegga_noattn
from .hhn_one_attention import HHNOneAttention, build_model as build_hhn_one_attention

__all__ = [
    "GraphSAGE_PI",
    "compute_degree_histogram",
    "TransformerBaseline",
    "TransformerConvVN",
    "HeGGA",
    "HeGGANoAttn",
    "HHNOneAttention",
]

def load_model(cfg: dict, dataset):
    """Load the model based on the configuration and dataset.

    Args:
        cfg (dict): Configuration dictionary.
        dataset (InMemoryDataset): The dataset to be used for training. 

    Raises:
        ValueError: If the model name is unknown.

    Returns:
        _type_: torch.nn.Module instance of the model.
    """
    name = cfg["model"]["name"].lower()

    if name == "graphsage_pi":
        data = dataset[0]  # Assuming dataset is an InMemoryDataset
        in_dim = data.x.size(-1)
        out_dim = data.y.size(-1)
        return GraphSAGE_PI(
            in_dim=in_dim,
            hidden_dim=cfg["model"]["hidden"],
            out_dim=out_dim,
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            aggr=cfg["model"]["agg"]
        )
    elif name == "transformer":
        data = dataset[0]
        in_dim = data.x.size(-1)
        out_dim = data.y.size(-1)
        edge_dim = data.edge_attr.size(-1) if data.edge_attr is not None else None
        return TransformerBaseline(
            in_dim=in_dim,
            hidden_dim=cfg["model"]["hidden"],
            out_dim=out_dim,
            num_layers=cfg["model"]["num_layers"],
            heads=cfg["model"].get("heads", 4),
            dropout=cfg["model"].get("dropout", 0.0),
            edge_dim=edge_dim,
        )
    elif name == "transformer_vn":
        data = dataset[0]
        in_dim = data.x.size(-1)
        out_dim = data.y.size(-1)
        edge_dim = data.edge_attr.size(-1) if data.edge_attr is not None else None
        return TransformerConvVN(
            in_dim=in_dim,
            hidden_dim=cfg["model"]["hidden"],
            out_dim=out_dim,
            num_layers=cfg["model"]["num_layers"],
            heads=cfg["model"].get("heads", 4),
            dropout=cfg["model"].get("dropout", 0.0),
            edge_dim=edge_dim,
        )
    elif name == "hegga":
        return build_HeGGA(cfg, dataset)
    elif name == "hegganoattn":
        return build_hegga_noattn(cfg, dataset)
    elif name == "hhn_one_attention":
        return build_hhn_one_attention(cfg, dataset)
    else:
        raise ValueError(f"Unknown model name: {cfg['model']['name']}")
