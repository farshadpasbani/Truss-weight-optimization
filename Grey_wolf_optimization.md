# Technical Note on Grey Wolf Optimization (GWO)

## Introduction
Grey Wolf Optimization (GWO) is a nature-inspired optimization algorithm based on the social hierarchy and hunting behavior of grey wolves in the wild. You can find the original paper here: [Grey Wolf Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853)
This project employs GWO to optimize truss structures by minimizing weight while adhering to structural constraints such as displacement and stress limits.

## How GWO Works
GWO simulates the leadership hierarchy and hunting tactics of grey wolves, categorized into alpha, beta, delta, and omega wolves. In optimization terms:
- **Alpha (α)**: The best solution.
- **Beta (β)**: The second-best solution.
- **Delta (δ)**: The third-best solution.
- **Omega (ω)**: Other search agents.

The algorithm mimics wolves' behavior of encircling their prey, approaching it, and finally attacking. These behaviors are modeled mathematically in the optimization process:
1. **Encircling prey**
2. **Hunting**
3. **Attacking the prey**

## Algorithm Implementation in Truss Optimization
In the context of truss optimization, the algorithm aims to find the optimal cross-sectional areas of truss elements that minimize the overall weight subject to structural performance constraints.

### 1. Initialization
Generate an initial population of wolves where each position represents a potential solution (set of cross-sectional areas).
```python
positions = np.random.uniform(bounds[0], bounds[1], (num_wolves, dim))
```

### 2. Fitness Evaluation
Each wolf is assessed based on the structural performance of the truss configuration it represents, calculating weight, displacement, and stress.
```python
fitness, weight = cost_function(
    cross_sectional_areas,
    connectivity_matrix,
    forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag,
)
```

### 3. Update Positions
Using the hunting behavior-inspired formulae, wolves update their positions in the search space. The positions consider the alpha, beta, and delta to converge towards the best solution while maintaining diversity.
```python
for i in range(num_wolves):
    r1, r2 = np.random.rand(), np.random.rand()
    A1 = 2 * a * r1 - a
    C1 = 2 * r2
    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
    X1 = alpha_pos[j] - A1 * D_alpha

    r1, r2 = np.random.rand(), np.random.rand()
    A2 = 2 * a * r1 - a
    C2 = 2 * r2
    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
    X2 = beta_pos[j] - A2 * D_beta

    r1, r2 = np.random.rand(), np.random.rand()
    A3 = 2 * a * r1 - a
    C3 = 2 * r2
    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
    X3 = delta_pos[j] - A3 * D_delta

    positions[i, j] = (X1 + X2 + X3) / 3
```
### 4. Convergence
The process repeats until a stopping criterion (e.g., maximum number of iterations or a convergence threshold) is met.

## Results and Visualization
The alpha position, representing the best truss design, is taken as the optimal solution. Trends such as changes in fitness values and weights over iterations can be visualized to assess the algorithm's effectiveness and convergence behavior.
