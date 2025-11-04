# Entry point for launching experiments

import argparse, yaml
from src.models.graphsage_pi import GraphSAGE_PI
from src.training.train import fit

def load_dataset(cfg):
    if cfg["data"]["backend"] == "synthetic":
        from src.data.synthetic_powergraph import make_synthetic_dataset
        return make_synthetic_dataset(num_graphs=200, n=60, f_in=cfg["model"]["in_dim"])
    else:
        # later: import your teammates' class, e.g.:
        # from src.data.powergraph_dataset import PowerGraphNodeLevel
        # return PowerGraphNodeLevel(root="data", grid=cfg["data"]["grid"], task=cfg["data"]["task"])
        raise NotImplementedError

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(cfg)
    mcfg = cfg["model"]
    model = GraphSAGE_PI(
        in_dim=mcfg["in_dim"], hidden=mcfg["hidden"], out_dim=mcfg["out_dim"],
        num_layers=mcfg["num_layers"], dropout=mcfg["dropout"], agg=mcfg["agg"]
    )
    fit(model, dataset, cfg)

if __name__ == "__main__":
    main()