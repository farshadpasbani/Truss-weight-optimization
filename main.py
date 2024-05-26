import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import *
from gwo import grey_wolf_optimization


# Truss definition:
A = np.ones(16)
material_density = 0.00000785  # kg/mm3
Connect = np.array(
    [
        [1, 2],
        [1, 4],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 5],
        [3, 6],
        [4, 5],
        [4, 7],
        [5, 6],
        [5, 7],
        [5, 8],
        [6, 9],
        [6, 8],
        [7, 8],
        [8, 9],
    ]
)
ForceNodes = np.array([1, 4])
Node = np.array(
    [
        [0, 0],
        [0, 1200],
        [0, 2400],
        [1200, 0],
        [1200, 1200],
        [1200, 2400],
        [2400, 0],
        [2400, 1200],
        [2400, 2400],
    ]
)
restrains = np.array([7, 8, 9])
num_wolves = 10
max_iter = 100
# Constraints
max_displacement = 30
allowable_stress = 355
bounds = (10, 10000)

optimized_truss, _ = grey_wolf_optimization(
    analyze_truss,
    num_wolves,
    max_iter,
    dim=len(Connect),
    bounds=bounds,
    connectivity_matrix=Connect,
    force_nodes=ForceNodes,
    node_coordinates=Node,
    restrained_nodes=restrains,
    material_density=material_density,
    max_displacement=max_displacement,
    allowable_stress=allowable_stress,
    plot_flag=0,
)

# plot
analyze_truss(
    optimized_truss,
    Connect,
    ForceNodes,
    Node,
    restrains,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag=1,
)
