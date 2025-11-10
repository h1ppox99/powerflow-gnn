# Entry point for launching experiments

import argparse, yaml
from src.models.graphsage_pi import GraphSAGE_PI
from src.training.train import fit
from pathlib import Path

def load_dataset(cfg):
    if cfg["data"]["backend"] == "synthetic":
        from data.prepare_data import SyntheticPowerGrid
        return SyntheticPowerGrid(
            num_graphs=cfg["data"].get("num_graphs", 100),
            n=cfg["data"].get("n", 60),
            f_in=cfg["model"].get("in_dim", 16)
        )
    else:
        # Import your PowerGraph dataset
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
        from data.prepare_data import PowerGrid, discover_powergraph_root
        print("[Debug] Discovering PowerGraph root...")
        
        root = discover_powergraph_root()
        return PowerGrid(
            root=root,
            name=cfg["data"]["grid"],
            datatype=cfg["data"].get("task", "nodeopf")
        )

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
    if cfg["model"]["name"] == "graphsage_pi":
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
    else:
        raise ValueError(f"Unknown model name: {cfg['model']['name']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[Debug] Loading dataset...")
    dataset = load_dataset(cfg)
    print(f"[Debug] Dataset loaded with {len(dataset)} graphs.")
    model = load_model(cfg, dataset)
    print("[Debug] Model initialized")
    fit(model, dataset, cfg)

if __name__ == "__main__":
    main()