# Entry point for launching experiments

import argparse, yaml #type: ignore[import]
from src.models.graphsage_pi import GraphSAGE_PI
from src.training.train import fit
from pathlib import Path
from src.models import load_model
from src.experiments.experiment_logger import log_experiment_run

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

    # Log configuration snapshot at launch (before training), to keep a record per run.
    try:
        log_experiment_run(args.config, cfg, {}, log_path="experiment_runs.csv")
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to record config snapshot: {exc}")

    dataset = load_dataset(cfg)
    model = load_model(cfg, dataset)
    test_metrics = fit(model, dataset, cfg)

    try:
        log_experiment_run(args.config, cfg, test_metrics)
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to record experiment in shared CSV: {exc}")

if __name__ == "__main__":
    main()
