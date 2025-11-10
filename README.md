# Physics-Informed GNN for Optimal Power Flow

This project implements physics-informed graph neural networks (GNNs) to improve the accuracy and interpretability of optimal power flow solutions in electrical grids.

## Overview

Optimal power flow (OPF) is a fundamental problem in power systems engineering, aiming to determine the most efficient operating conditions while satisfying physical and operational constraints. Traditional methods can be computationally intensive and may not fully leverage the underlying grid topology and physics. This project integrates physics-based regularization into GNN architectures to enforce Kirchhoff’s laws and power balance constraints, enhancing model reliability and generalization. The goal is to develop scalable, interpretable models that can assist in real-time power grid management.

## Work already completed

- Setup project 
- Built data processing pipeline 
- Written models scripts 
- Linked models and data processing pipelines

TODO : 

- Adjust losses scripts
- Train models

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
Or one can download the dataset "dataset_pf_opf.zip" from (https://figshare.com/articles/dataset/PowerGraph/22820534) and unzip it under `data/raw/`.
## Running an Experiment

To launch a training run with the default GraphSAGE configuration, use:

```bash
uv run src/experiments/run_experiment.py --config src/config/graphsage.yaml
```

This command starts training with physics-informed regularization enabled and logs results for analysis.

## Credits

- Yassine Guennoun
- Édouard Rabasse
- Hippolyte Wallaert  
