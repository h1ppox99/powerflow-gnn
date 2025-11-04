powerflow-gnn/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── README.md                # Instructions to download PowerGraph dataset
│   ├── processed/
│   │   └── train_val_test_split.pt  # Preprocessed PyG InMemoryDataset
│   └── utils_data.py                # Dataset loading, normalization, splitting
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── graphsage_pi.py          # Physics-informed GraphSAGE model (baseline)
│   │   ├── pna_pi.py                # PNA-based physics-informed model (extension)
│   │   └── layers.py                # Custom message passing, aggregation, normalization
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── physical_loss.py         # Kirchhoff regularization, power balance penalties
│   │   └── regression_loss.py       # RMSE, circular phase angle loss
│   │
│   ├── training/
│   │   ├── train.py                 # Standard supervised training loop
│   │   ├── evaluate.py              # Validation, RMSE/KCL metrics
│   │   ├── physics_regularizer.py   # Add-on loss term (KCL term integration)
│   │   └── utils_training.py        # Learning rate scheduling, checkpointing, early stop
│   │
│   ├── config/
│   │   ├── default.yaml             # Default hyperparameters and paths
│   │   └── graphsage.yaml           # Specific configuration for GraphSAGE run
│   │
│   ├── experiments/
│   │   ├── run_experiment.py        # Entrypoint for launching experiments
│   │   ├── sweep_config.yaml        # Grid search or hyperparameter sweep definitions
│   │   └── logs/                    # TensorBoard or WandB logs
│   │
│   ├── visualization/
│   │   ├── visualize_graphs.py      # Plot small graphs and predictions vs true values
│   │   ├── visualize_losses.py      # Training curve plots
│   │   └── visualize_results.ipynb  # Jupyter notebook for figures
│   │
│   └── utils.py                     # Shared helpers (metrics, logging, reproducibility)
│
├── notebooks/
│   ├── 00_dataset_exploration.ipynb # Analyze PowerGraph node/edge features
│   ├── 01_train_graphsage.ipynb     # End-to-end baseline training notebook
│   ├── 02_physics_loss_analysis.ipynb # Visualize effect of Kirchhoff regularization
│   └── 03_comparison_pna.ipynb      # Compare GraphSAGE vs PNA models
│
├── tests/
│   ├── test_data_loading.py         # Ensure dataset processed correctly
│   ├── test_model_forward.py        # Sanity check model dimensions
│   ├── test_loss_terms.py           # Unit test physical and regression losses
│   └── test_training_loop.py        # Smoke test for training/evaluation
│
└── scripts/
    ├── download_data.sh             # Automated dataset download from figshare
    ├── preprocess_data.py           # Create and save normalized PyG dataset
    ├── train_graphsage.sh           # CLI run for reproducibility
    └── evaluate_model.sh