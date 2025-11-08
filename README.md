# Physics-Informed GNN for Optimal Power Flow

This project implements physics-informed graph neural networks (GNNs) to improve the accuracy and interpretability of optimal power flow solutions in electrical grids.

## Overview

Optimal power flow (OPF) is a fundamental problem in power systems engineering, aiming to determine the most efficient operating conditions while satisfying physical and operational constraints. Traditional methods can be computationally intensive and may not fully leverage the underlying grid topology and physics. This project integrates physics-based regularization into GNN architectures to enforce Kirchhoff’s laws and power balance constraints, enhancing model reliability and generalization. The goal is to develop scalable, interpretable models that can assist in real-time power grid management.

## Setup Instructions

1. Install `uv` if not already installed:

   ```bash
   pip install uv
   ```

2. Create a new virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Synchronize dependencies and install requirements:

   ```bash
   uv sync
   ```

4. You are now ready to run experiments.

## Dataset

The repository ships with the PowerGraph benchmark installed under `data/raw`.
Each dataset follows the `<dataset_name>/<dataset_name>/raw` layout from the original Figshare
release ([PowerGraph](https://figshare.com/articles/dataset/PowerGraph/22820534?file=50081700)).
The helper utilities in `data/utils_data.py` expose convenience functions to
list the available datasets or instantiate the `PowerGrid` PyG dataset wrapper
without hard-coded paths.

Commands to install:

```bash
cd data/raw
wget -O data.tar.gz "https://figshare.com/ndownloader/files/46619152"
tar -xf data.tar.gz
rm data.tar.gz
```

## Running an Experiment

To launch a training run with the default GraphSAGE configuration, use:

```bash
uv run src/experiments/run_experiment.py --config src/config/graphsage.yaml
```

This command starts training with physics-informed regularization enabled and logs results for analysis.

## Repository Structure

```
powerflow-gnn/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── README.md           # Instructions to download PowerGraph dataset
│   ├── processed/
│   │   └── train_val_test_split.pt  # Preprocessed PyG InMemoryDataset
│   └── utils_data.py           # Dataset loading, normalization, splitting
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graphsage_pi.py     # Physics-informed GraphSAGE model (baseline)
│   │   ├── pna_pi.py           # PNA-based physics-informed model (extension)
│   │   └── layers.py           # Custom message passing, aggregation, normalization
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── physical_loss.py    # Kirchhoff regularization, power balance penalties
│   │   └── regression_loss.py  # RMSE, circular phase angle loss
│   │
│   ├── training/
│   │   ├── train.py            # Standard supervised training loop
│   │   ├── evaluate.py         # Validation, RMSE/KCL metrics
│   │   ├── physics_regularizer.py  # Add-on loss term (KCL term integration)
│   │   └── utils_training.py   # Learning rate scheduling, checkpointing, early stop
│   │
│   ├── config/
│   │   ├── default.yaml        # Default hyperparameters and paths
│   │   └── graphsage.yaml      # Specific configuration for GraphSAGE run
│   │
│   ├── experiments/
│   │   ├── run_experiment.py   # Entrypoint for launching experiments
│   │   ├── sweep_config.yaml   # Grid search or hyperparameter sweep definitions
│   │   └── logs/               # TensorBoard or WandB logs
│   │
│   ├── visualization/
│   │   ├── visualize_graphs.py # Plot small graphs and predictions vs true values
│   │   ├── visualize_losses.py # Training curve plots
│   │   └── visualize_results.ipynb # Jupyter notebook for figures
│   │
│   └── utils.py                # Shared helpers (metrics, logging, reproducibility)
│
├── notebooks/
│   ├── 00_dataset_exploration.ipynb  # Analyze PowerGraph node/edge features
│   ├── 01_train_graphsage.ipynb       # End-to-end baseline training notebook
│   ├── 02_physics_loss_analysis.ipynb # Visualize effect of Kirchhoff regularization
│   └── 03_comparison_pna.ipynb         # Compare GraphSAGE vs PNA models
│
├── tests/
│   ├── test_data_loading.py      # Ensure dataset processed correctly
│   ├── test_model_forward.py     # Sanity check model dimensions
│   ├── test_loss_terms.py        # Unit test physical and regression losses
│   └── test_training_loop.py     # Smoke test for training/evaluation
│
└── scripts/
    ├── download_data.sh          # Automated dataset download from figshare
    ├── preprocess_data.py        # Create and save normalized PyG dataset
    ├── train_graphsage.sh        # CLI run for reproducibility
    └── evaluate_model.sh
```

## Credits

- Yassine Guennoun
- Édouard Rabasse
- Hippolyte Wallaert  
