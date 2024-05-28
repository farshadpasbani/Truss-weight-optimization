import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import analyze_truss
from gwo import grey_wolf_optimization
from differential_evolution import differential_evolution

connectivity_matrix = np.array(
    [
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 4],
        [3, 5],
        [4, 5],
        [4, 6],
        [5, 6],
    ]
)
applied_forces = [(0, 0, -10000), (2, 0, -10000)]
node_coordinates = np.array(
    [
        [0, 0],
        [0, 1200],
        [1200, 0],
        [1200, 1200],
        [2400, 0],
        [2400, 1200],
    ]
)
restrained_nodes = np.array([5, 6])
num_agents = 10
max_iterations = 100
cross_section_bounds = (10, 1000)
material_density = 0.0000078  # kg/mm3
max_allowable_displacement = 4
max_allowable_stress = 355

optimized_truss_gwo, fitness_gwo, trends_gwo = grey_wolf_optimization(
    analyze_truss,
    num_agents,
    max_iterations,
    len(connectivity_matrix),
    cross_section_bounds,
    connectivity_matrix,
    applied_forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_allowable_displacement,
    max_allowable_stress,
    0,
)

optimized_truss_de, fitness_de, trends_de = differential_evolution(
    analyze_truss,
    num_agents,
    max_iterations,
    len(connectivity_matrix),
    cross_section_bounds,
    connectivity_matrix,
    applied_forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_allowable_displacement,
    max_allowable_stress,
    0,
)

print("GWO Best Solution:", optimized_truss_gwo)
print("GWO Best Fitness:", fitness_gwo)
print("DE Best Solution:", optimized_truss_de)
print("DE Best Fitness:", fitness_de)


def plot_all_trends(trends_gwo, trends_de):
    iterations = range(max_iterations)
    gwo_penalized_weights, gwo_weights = zip(*trends_gwo)
    de_penalized_weights, de_weights = zip(*trends_de)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, gwo_penalized_weights, "b--", label="GWO Penalized Weight")
    plt.plot(iterations, gwo_weights, "b-", label="GWO Actual Weight")
    plt.plot(iterations, de_penalized_weights, "r--", label="DE Penalized Weight")
    plt.plot(iterations, de_weights, "r-", label="DE Actual Weight")
    plt.title("Comparison of Optimization Trends")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_all_trends(trends_gwo, trends_de)

print("Plotting GWO optimized structure...")
analyze_truss(
    optimized_truss_gwo,
    connectivity_matrix,
    applied_forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_allowable_displacement,
    max_allowable_stress,
    1,
)

print("Plotting DE optimized structure...")
analyze_truss(
    optimized_truss_de,
    connectivity_matrix,
    applied_forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_allowable_displacement,
    max_allowable_stress,
    1,
)
