"""Synthetic power grid data generator for testing and development."""

from __future__ import annotations

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
from typing import Optional, Callable
import networkx as nx


def generate_power_grid_topology(num_nodes: int, avg_degree: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a realistic power grid topology using a spatial graph model.
    
    Args:
        num_nodes: Number of buses/nodes in the grid
        avg_degree: Average node degree (typical power grids: 2.5-3.5)
    
    Returns:
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, num_edge_features]
    """
    # Create a geometric graph to mimic spatial structure of power grids
    np.random.seed(42)
    positions = np.random.rand(num_nodes, 2)
    
    # Calculate distance threshold for desired average degree
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(positions))
    
    # Connect nodes within threshold distance
    threshold = np.percentile(distances[distances > 0], 100 * avg_degree / num_nodes * 10)
    
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distances[i, j] < threshold and distances[i, j] > 0:
                edges.append([i, j])
    
    # Ensure connectivity using MST
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i, j in edges:
        G.add_edge(i, j, weight=distances[i, j])
    
    if not nx.is_connected(G):
        # Add edges from MST to ensure connectivity
        mst_edges = list(nx.minimum_spanning_tree(nx.complete_graph(num_nodes)).edges())
        for i, j in mst_edges:
            if not G.has_edge(i, j):
                edges.append([i, j])
    
    edges = np.array(edges)
    
    # Create bidirectional edges
    edge_index = torch.cat([
        torch.tensor(edges, dtype=torch.long).t(),
        torch.tensor(edges[:, [1, 0]], dtype=torch.long).t()
    ], dim=1)
    
    num_edges = edge_index.shape[1]
    
    # Generate edge attributes: [R, X, B, capacity]
    # R (resistance), X (reactance), B (susceptance), capacity (thermal limit)
    edge_attr = torch.zeros(num_edges, 4)
    for i in range(num_edges // 2):
        # Typical transmission line parameters
        r = np.random.uniform(0.01, 0.1)  # Resistance (p.u.)
        x = np.random.uniform(0.05, 0.5)  # Reactance (p.u.)
        b = np.random.uniform(0.01, 0.2)  # Susceptance (p.u.)
        cap = np.random.uniform(50, 500)  # Capacity (MW)
        
        # Same attributes for both directions
        edge_attr[i] = torch.tensor([r, x, b, cap])
        edge_attr[i + num_edges // 2] = torch.tensor([r, x, b, cap])
    
    return edge_index, edge_attr


def generate_nodeopf_sample(num_nodes: int, edge_index: torch.Tensor, 
                            edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single OPF (Optimal Power Flow) sample.
    
    Node features (X): [P_load, Q_load, V_setpoint]
    Node targets (Y): [P_gen, Q_gen, V_magnitude, theta]
    
    Args:
        num_nodes: Number of nodes
        edge_index: Edge connectivity
        edge_attr: Edge attributes
    
    Returns:
        x: Node features [num_nodes, 3]
        y: Node targets [num_nodes, 4]
    """
    # Node features: [P_load, Q_load, V_setpoint]
    x = torch.zeros(num_nodes, 3)
    
    # Classify nodes: slack, PV (generator), PQ (load)
    num_generators = max(2, num_nodes // 10)  # ~10% generators
    generator_nodes = np.random.choice(num_nodes, num_generators, replace=False)
    slack_node = generator_nodes[0]
    pv_nodes = generator_nodes[1:]
    pq_nodes = [i for i in range(num_nodes) if i not in generator_nodes]
    
    # Generate loads (P_load, Q_load) for all nodes
    for i in range(num_nodes):
        if i in pq_nodes:
            # Load nodes have significant consumption
            p_load = np.random.uniform(10, 100)  # MW
            q_load = p_load * np.random.uniform(0.3, 0.5)  # Reactive power (power factor ~0.9)
        else:
            # Generator nodes typically have small/no load
            p_load = np.random.uniform(0, 10)
            q_load = p_load * np.random.uniform(0.1, 0.3)
        
        x[i, 0] = p_load
        x[i, 1] = q_load
        x[i, 2] = 1.0  # Voltage setpoint (p.u.)
    
    # Generate targets: [P_gen, Q_gen, V_magnitude, theta]
    y = torch.zeros(num_nodes, 4)
    
    # Slack bus (reference)
    y[slack_node, 0] = 0  # P_gen (will be determined by power balance)
    y[slack_node, 1] = 0  # Q_gen
    y[slack_node, 2] = 1.0  # V = 1.0 p.u.
    y[slack_node, 3] = 0.0  # theta = 0 (reference)
    
    # PV buses (generators)
    for node in pv_nodes:
        p_gen = np.random.uniform(50, 200)  # MW
        y[node, 0] = p_gen
        y[node, 1] = np.random.uniform(-50, 50)  # Q_gen (varies)
        y[node, 2] = np.random.uniform(0.98, 1.02)  # V controlled
        y[node, 3] = np.random.uniform(-0.1, 0.1)  # theta (small deviation)
    
    # PQ buses (loads)
    for node in pq_nodes:
        y[node, 0] = 0  # No generation
        y[node, 1] = 0
        y[node, 2] = np.random.uniform(0.95, 1.05)  # V varies
        y[node, 3] = np.random.uniform(-0.2, 0.2)  # theta varies more
    
    # Balance power: slack bus picks up the difference
    total_load_p = x[:, 0].sum()
    total_gen_p = y[:, 0].sum()
    y[slack_node, 0] = total_load_p - total_gen_p + total_load_p * 0.05  # +5% losses
    
    return x, y


def generate_powerflow_sample(num_nodes: int, edge_index: torch.Tensor,
                              edge_attr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single power flow sample (similar to node datatype).
    
    Args:
        num_nodes: Number of nodes
        edge_index: Edge connectivity
        edge_attr: Edge attributes
    
    Returns:
        x: Node features [num_nodes, n_features]
        y: Node targets [num_nodes, 4]
    """
    # For power flow, we might have more input features
    # X: [P_load, Q_load, P_gen, Q_gen, node_type, ...]
    x = torch.zeros(num_nodes, 6)
    
    num_generators = max(2, num_nodes // 10)
    generator_nodes = np.random.choice(num_nodes, num_generators, replace=False)
    
    for i in range(num_nodes):
        is_gen = i in generator_nodes
        
        # Loads
        p_load = np.random.uniform(0, 100) if not is_gen else np.random.uniform(0, 10)
        q_load = p_load * np.random.uniform(0.3, 0.5)
        
        # Generation
        p_gen = np.random.uniform(50, 200) if is_gen else 0
        q_gen = np.random.uniform(-50, 50) if is_gen else 0
        
        # Node type encoding
        node_type = 1.0 if is_gen else 0.0
        
        x[i] = torch.tensor([p_load, q_load, p_gen, q_gen, node_type, 1.0])
    
    # Targets: [P_flow, Q_flow, V, theta]
    y = torch.zeros(num_nodes, 4)
    
    for i in range(num_nodes):
        is_gen = i in generator_nodes
        y[i, 0] = x[i, 2] - x[i, 0]  # Net P injection
        y[i, 1] = x[i, 3] - x[i, 1]  # Net Q injection
        y[i, 2] = np.random.uniform(0.95, 1.05) if not is_gen else np.random.uniform(0.98, 1.02)
        y[i, 3] = np.random.uniform(-0.3, 0.3)
    
    return x, y


class SyntheticPowerGrid(InMemoryDataset):
    """
    Synthetic power grid dataset that mimics the structure of PowerGrid dataset.
    
    Args:
        root: Root directory to store data
        num_graphs: Number of graphs to generate
        num_nodes: Number of nodes per graph (can be list for variable sizes)
        datatype: Type of task ('nodeopf', 'node', 'binary', 'regression', 'multiclass')
        transform: Optional transform
        pre_transform: Optional pre-transform
        pre_filter: Optional pre-filter
    """
    
    def __init__(
        self,
        root: str | Path,
        num_graphs: int = 200,
        num_nodes: int | list[int] = 30,
        datatype: str = "nodeopf",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.datatype = datatype.lower()
        self.seed = seed
        
        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> list[str]:
        return []  # No raw files for synthetic data
    
    @property
    def processed_file_names(self) -> str:
        return f"synthetic_{self.datatype}_{self.num_graphs}.pt"
    
    def download(self):
        pass  # No download needed
    
    def process(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        data_list = []
        
        for i in range(self.num_graphs):
            # Determine number of nodes for this graph
            if isinstance(self.num_nodes, (list, tuple)):
                n_nodes = np.random.choice(self.num_nodes)
            else:
                n_nodes = self.num_nodes
            
            # Generate topology
            edge_index, edge_attr = generate_power_grid_topology(n_nodes)
            
            # Generate samples based on datatype
            if self.datatype == "nodeopf":
                x, y = generate_nodeopf_sample(n_nodes, edge_index, edge_attr)
                
                # Normalize edge attributes
                edge_attr_norm = torch.nn.functional.normalize(edge_attr, dim=0)
                
                # Compute max values for denormalization later
                maxsY = y.abs().max(dim=0)[0]
                maxsY[maxsY == 0] = 1.0  # Avoid division by zero
                
                # Create mask (non-zero outputs)
                mask = y != 0
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr_norm,
                    y=y / maxsY,  # Normalize targets
                    maxs=maxsY,
                    mask=mask,
                )
            
            elif self.datatype == "node":
                x, y = generate_powerflow_sample(n_nodes, edge_index, edge_attr)
                
                edge_attr_norm = torch.nn.functional.normalize(edge_attr, dim=0)
                maxsY = y.abs().max(dim=0)[0]
                maxsY[maxsY == 0] = 1.0
                mask = y != 0
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr_norm,
                    y=y / maxsY,
                    maxs=maxsY,
                    mask=mask,
                )
            
            elif self.datatype == "binary":
                # Graph-level binary classification (e.g., stable/unstable)
                x, y_node = generate_nodeopf_sample(n_nodes, edge_index, edge_attr)
                
                # Binary target: 1 if system is stable (voltage within limits)
                is_stable = ((y_node[:, 2] >= 0.95) & (y_node[:, 2] <= 1.05)).all()
                y = torch.tensor([float(is_stable)], dtype=torch.float).view(1, -1)
                
                # Create edge mask (for explainability)
                edge_mask = torch.ones(edge_index.shape[1], 1)
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    edge_mask=edge_mask,
                    idx=i,
                )
            
            elif self.datatype == "regression":
                # Graph-level regression (e.g., total power loss)
                x, y_node = generate_nodeopf_sample(n_nodes, edge_index, edge_attr)
                
                # Compute total system loss
                total_loss = (y_node[:, 0].sum() - x[:, 0].sum()).abs()
                y = torch.tensor([total_loss.item()], dtype=torch.float).view(1, -1)
                
                edge_mask = torch.ones(edge_index.shape[1], 1)
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    edge_mask=edge_mask,
                    idx=i,
                )
            
            elif self.datatype == "multiclass":
                # Graph-level multiclass (e.g., operating regime)
                x, y_node = generate_nodeopf_sample(n_nodes, edge_index, edge_attr)
                
                # Classify into regimes based on voltage profile
                v_mean = y_node[:, 2].mean()
                if v_mean < 0.97:
                    category = 0  # Low voltage
                elif v_mean > 1.03:
                    category = 1  # High voltage
                else:
                    category = 2  # Normal
                
                y = torch.tensor([float(category)], dtype=torch.float).view(1, -1)
                edge_mask = torch.ones(edge_index.shape[1], 1)
                
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    edge_mask=edge_mask,
                    idx=i,
                )
            
            else:
                raise ValueError(f"Unknown datatype: {self.datatype}")
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])


def make_synthetic_dataset(
    num_graphs: int = 200,
    num_nodes: int = 30,
    datatype: str = "nodeopf",
    root: Optional[str | Path] = None,
) -> SyntheticPowerGrid:
    """
    Convenience function to create a synthetic power grid dataset.
    
    Args:
        num_graphs: Number of graphs to generate
        num_nodes: Number of nodes per graph
        datatype: Task type
        root: Root directory (defaults to ./data/synthetic)
    
    Returns:
        SyntheticPowerGrid dataset
    """
    if root is None:
        root = Path("data") / "synthetic"
    
    return SyntheticPowerGrid(
        root=root,
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        datatype=datatype,
    )


# Example usage
if __name__ == "__main__":
    # Create synthetic datasets for different tasks
    print("Creating synthetic nodeopf dataset...")
    dataset_opf = make_synthetic_dataset(num_graphs=100, num_nodes=30, datatype="nodeopf")
    print(f"Created {len(dataset_opf)} graphs")
    print(f"Sample graph: {dataset_opf[0]}")
    
    print("\nCreating synthetic binary classification dataset...")
    dataset_bin = make_synthetic_dataset(num_graphs=100, num_nodes=30, datatype="binary")
    print(f"Created {len(dataset_bin)} graphs")
    print(f"Sample graph: {dataset_bin[0]}")