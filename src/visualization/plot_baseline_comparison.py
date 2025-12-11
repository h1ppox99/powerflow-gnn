import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# -----------------------------------------
# Automatic repo base path
# -----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]     # adjust depth if needed
CSV_PATH = REPO_ROOT / "experiment_runs.csv"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using repo root: {REPO_ROOT}")
print(f"Saving outputs to: {OUTPUT_DIR}")

# -----------------------------------------
# Load CSV
# -----------------------------------------
df = pd.read_csv(CSV_PATH)

def extract_config(row):
    try:
        cfg = json.loads(row["config_yaml"])
        m = cfg.get("model", {})
        return {
            "hidden": m.get("hidden"),
            "heads": m.get("heads"),
            "num_layers": m.get("num_layers"),
            "pe_dims": m.get("pe_dims"),
            "dropout": m.get("dropout"),
        }
    except:
        return {}

df = pd.concat([df, df.apply(extract_config, axis=1, result_type="expand")], axis=1)

# ----------------------------------------------------
# Rename models for nicer plotting
# ----------------------------------------------------
df["pretty_name"] = df.model_name.replace({
    "transformer": "Baseline",
    "hh_mpnn": "Our model",
})

df_baseline = df[df.pretty_name == "Baseline"]
df_ours = df[df.pretty_name == "Our model"]

# ----------------------------------------------------
# Quantities
# ----------------------------------------------------
quantities = ["test_mse_v", "test_mse_theta", "test_mse_p", "test_mse_q"]
labels = ["V", "θ", "Pg", "Qg"]

grids = ["IEEE24", "IEEE39", "UK", "IEEE118"]

# ----------------------------------------------------
# FIGURE
# ----------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 7), sharey=True)
plt.rcParams.update({"font.size": 22})

for ax, grid in zip(axes, grids):

    # PICK BEST RUN (lowest total_mse)
    base = (
        df_baseline[df_baseline.data_grid == grid]
        .sort_values("total_mse")
        .iloc[0]
    )
    ours = (
        df_ours[df_ours.data_grid == grid]
        .sort_values("total_mse")
        .iloc[0]
    )

    baseline_vals = [base[q] for q in quantities]
    ours_vals = [ours[q] for q in quantities]

    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, baseline_vals, width=w, label="Baseline", color="#4C72B0")
    ax.bar(x + w/2, ours_vals, width=w, label="Our Model", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22)
    ax.set_yscale("log")
    ax.set_title(grid, fontsize=22, pad=10)

    # 🔥 Increase Y tick font size
    ax.tick_params(axis='y', labelsize=22)

    ax.grid(True, which="both", linestyle="--", linewidth=1, alpha=0.35)

axes[0].set_ylabel("MSE (log scale)", fontsize=22)

handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="upper center", ncol=2, fontsize=22, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.90])

out_path = OUTPUT_DIR / "curve_baseline_vs_powergnn_4grids.png"
plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
