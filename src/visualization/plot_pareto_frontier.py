# src/visualization/pareto.py
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pareto_frontier(input_dir, x_metric="mean_kcl_residual", y_metric="loss_mse", show_plot=True):
    """
    Plots the Pareto frontier between physics error and data error
    from a set of experiment result JSON files.
    """
    input_path = Path(input_dir)
    # 1. Find all JSON files
    files = sorted(glob.glob(str(input_path / "*.json")))
    
    data_points = []
    print(f"[Pareto] Scanning {len(files)} files in {input_path}...")

    # 2. Extract Data
    for file in files:
        try:
            with open(file, 'r') as f:
                content = json.load(f)
                
            # Handle both structures (flattened or nested test_metrics)
            metrics = content.get("test_metrics", content)
            
            y = metrics.get(y_metric)
            x = metrics.get(x_metric)
            
            # Extract weight for coloring/labeling
            # Tries to find 'physics_weight', 'value', or falls back to 'parameter_value'
            weight = content.get("physics_weight", content.get("value", content.get("parameter_value")))

            if y is not None and x is not None and weight is not None:
                data_points.append({
                    "weight": float(weight),
                    "y": float(y),
                    "x": float(x)
                })
        except Exception as e:
            print(f"[Pareto] Warning: Skipping {file}: {e}")

    if not data_points:
        print("[Pareto] No valid data points found to plot.")
        return

    # Sort by weight to make the line connect logically
    data_points.sort(key=lambda d: d["weight"])

    # Convert to arrays for plotting
    weights = np.array([d["weight"] for d in data_points])
    ys = np.array([d["y"] for d in data_points])
    xs = np.array([d["x"] for d in data_points])

    # 3. Plotting
    plt.figure(figsize=(10, 7))
    
    # Plot the trajectory line
    plt.plot(xs, ys, linestyle='--', color='gray', alpha=0.5, zorder=1)
    
    # Scatter plot with color mapped to the physics_weight
    # Use LogNorm because weights usually vary by orders of magnitude
    sc = plt.scatter(xs, ys, c=weights, cmap='viridis', 
                     norm=plt.matplotlib.colors.LogNorm(), 
                     s=100, edgecolors='k', zorder=2)
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Physics Weight ($\lambda$)")

    # Annotate specific points (min, max, and median)
    indices_to_annotate = [0, len(weights)//2, len(weights)-1]
    for i in indices_to_annotate:
        label = f"$\lambda={weights[i]:.1e}$"
        plt.annotate(label, 
                     (xs[i], ys[i]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, 
                     arrowprops=dict(arrowstyle="->", color='gray'))

    # Formatting
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"Physics Error ({x_metric})")
    plt.ylabel(f"Data Error ({y_metric})")
    plt.title(f"Pareto Frontier: {y_metric} vs. {x_metric}")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    output_file = input_path / "pareto_frontier.png"
    plt.savefig(output_file, dpi=300)
    print(f"[Pareto] Plot saved to {output_file}")
    
    if show_plot:
        plt.show()

if __name__ == "__main__":
    # Allows running as: python src/visualization/pareto.py --dir experiment_curves/train_physics_weight
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="experiment_curves", help="Directory containing JSON results")
    parser.add_argument("--x", type=str, default="mean_kcl_residual", help="Metric for X-axis")
    parser.add_argument("--y", type=str, default="loss_mse", help="Metric for Y-axis")
    args = parser.parse_args()
    print(args.dir)
    
    plot_pareto_frontier(args.dir, args.x, args.y, show_plot=True)
