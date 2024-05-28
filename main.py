import numpy as np
import matplotlib.pyplot as plt

from truss_analysis import analyze_truss
from gwo import grey_wolf_optimization
from differential_evolution import differential_evolution

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
forces = [(0, 6000, 10000), (3, 6000, 10000)]
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
num_agents = 10
max_iter = 100
bounds = (10, 10000)
material_density = 0.00000785
max_displacement = 30
allowable_stress = 355

optimized_truss_gwo, fitness_gwo, trends_gwo = grey_wolf_optimization(
    analyze_truss,
    num_agents,
    max_iter,
    len(Connect),
    bounds,
    Connect,
    forces,
    Node,
    restrains,
    material_density,
    max_displacement,
    allowable_stress,
    0,
)

optimized_truss_de, fitness_de, trends_de = differential_evolution(
    analyze_truss,
    num_agents,
    max_iter,
    len(Connect),
    bounds,
    Connect,
    forces,
    Node,
    restrains,
    material_density,
    max_displacement,
    allowable_stress,
    0,
)

print("GWO Best Solution:", optimized_truss_gwo)
print("GWO Best Fitness:", fitness_gwo)
print("DE Best Solution:", optimized_truss_de)
print("DE Best Fitness:", fitness_de)


def plot_all_trends(trends_gwo, trends_de):
    iterations = range(max_iter)
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
    Connect,
    forces,
    Node,
    restrains,
    material_density,
    max_displacement,
    allowable_stress,
    1,
)

print("Plotting DE optimized structure...")
analyze_truss(
    optimized_truss_de,
    Connect,
    forces,
    Node,
    restrains,
    material_density,
    max_displacement,
    allowable_stress,
    1,
)
