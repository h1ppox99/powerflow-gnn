# Description of the losses
## Physics-Informed Regularization : physical_loss.physics_loss

To ensure that the Graph Neural Network (GNN) predicts solutions that are not only statistically plausible but also physically valid, we incorporate a physics-informed regularization term into the loss function. This term, denoted as $\mathcal{L}_{\text{phy}}$, penalizes violations of Kirchhoff’s laws and the AC power flow equations.

The computation of this loss proceeds in three steps: denormalization, branch flow calculation, and nodal balance aggregation.

### 1. Denormalization

Since the neural network operates in a normalized feature space (typically $[-1, 1]$ or $[0, 1]$), the raw predictions must first be mapped back to their physical units to apply non-linear physical laws. Let $\hat{\mathbf{y}}_i = [\hat{P}_i, \hat{Q}_i, \hat{V}_i, \hat{\theta}_i]$ be the normalized prediction for node $i$. The physical variables are recovered as:

$$
V_i = \hat{V}_i \cdot V_{\text{max}}, \quad \theta_i = \hat{\theta}_i \cdot \theta_{\text{max}}
$$

where $V_{\text{max}}$ and $\theta_{\text{max}}$ are the dataset-specific scaling factors.

### 2. Branch Flow Calculation

For every edge $(i, j)$ in the graph, we calculate the active power ($P_{ij}$) and reactive power ($Q_{ij}$) flowing from node $i$ to node $j$. Using the line admittance $Y_{ij} = G_{ij} + jB_{ij}$ (where $G$ is conductance and $B$ is susceptance), the AC power flow equations are:

$$
\begin{aligned}
    P_{ij} &= V_i^2 G_{ij} - V_i V_j \left( G_{ij} \cos(\theta_{ij}) + B_{ij} \sin(\theta_{ij}) \right) \\
    Q_{ij} &= -V_i^2 B_{ij} - V_i V_j \left( G_{ij} \sin(\theta_{ij}) - B_{ij} \cos(\theta_{ij}) \right)
\end{aligned}
$$

where $\theta_{ij} = \theta_i - \theta_j$ is the phase angle difference. The term $V_i^2$ represents the self-consumption or charging of the line at the source node.

### 3. Nodal Balance and Loss Computation

According to Kirchhoff's Current Law (KCL), the net power injected at a node must equal the sum of powers flowing out of it through transmission lines. We aggregate the branch flows using a scatter-sum operation to obtain the calculated injection for each node:

$$
P_{i}^{\text{calc}} = \sum_{j \in \mathcal{N}(i)} P_{ij}, \quad Q_{i}^{\text{calc}} = \sum_{j \in \mathcal{N}(i)} Q_{ij}
$$

The physical loss is defined as the Mean Squared Error (MSE) of the power mismatch residuals, enforcing the conservation of energy across the entire grid:

$$
\mathcal{L}_{\text{phy}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \left( P_{i}^{\text{pred}} - P_{i}^{\text{calc}} \right)^2 + \left( Q_{i}^{\text{pred}} - Q_{i}^{\text{calc}} \right)^2 \right]
$$

By minimizing $\mathcal{L}_{\text{phy}}$, the model learns to respect the underlying topology and physical constraints of the electrical grid, effectively acting as a soft-constraint optimization solver.