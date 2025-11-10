# Entry point for launching experiments

import argparse, yaml
from src.models.graphsage_pi import GraphSAGE_PI
from src.training.train import fit
from pathlib import Path

def load_dataset(cfg):
    if cfg["data"]["backend"] == "synthetic":
        # from src.data.synthetic_powergraph import make_synthetic_dataset
        # return make_synthetic_dataset(num_graphs=200, n=60, f_in=cfg["model"]["in_dim"])
        raise NotImplementedError("Synthetic dataset not yet implemented.")
    else:
        # Import your PowerGraph dataset
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))
        from prepare_data import PowerGrid, discover_powergraph_root
        print("[Debug] Discovering PowerGraph root...")
        
        root = discover_powergraph_root()
        return PowerGrid(
            root=root,
            name=cfg["data"]["grid"],
            datatype=cfg["data"].get("task", "nodeopf")
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("[Debug] Loading dataset...")
    dataset = load_dataset(cfg)
    print(f"[Debug] Dataset loaded with {len(dataset)} graphs.")
    mcfg = cfg["model"]
    model = GraphSAGE_PI(
        in_dim=mcfg["in_dim"], hidden=mcfg["hidden"], out_dim=mcfg["out_dim"],
        num_layers=mcfg["num_layers"], dropout=mcfg["dropout"], agg=mcfg["agg"]
    )
    print("[Debug] Model initialized")
    fit(model, dataset, cfg)

if __name__ == "__main__":
    main()