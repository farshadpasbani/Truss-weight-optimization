import numpy as np
import matplotlib.pyplot as plt


def grey_wolf_optimization(
    cost_function,
    num_wolves,
    max_iter,
    dim,
    bounds,
    connectivity_matrix,
    force_nodes,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag,
):
    """
    Grey Wolf Optimization algorithm to minimize truss_penalized_weight by changing cross-sectional areas of truss elements.

    Parameters:
    num_wolves (int): Number of wolves in the population.
    max_iter (int): Maximum number of iterations.
    dim (int): Number of dimensions (number of cross-sectional areas).
    bounds (tuple): Tuple containing lower and upper bounds for the cross-sectional areas.
    connectivity_matrix (np.ndarray): Connectivity matrix defining the elements and their nodes.
    force_nodes (np.ndarray): Array of nodes where external forces are applied.
    node_coordinates (np.ndarray): Array of coordinates of the nodes.
    restrained_nodes (np.ndarray): Array of nodes where displacements are restrained.
    max_displacement (float): Maximum allowable displacement (default is 3 mm).
    allowable_stress (float): Maximum allowable stress (default is 2160 N/mm2).

    Returns:
    np.ndarray: Best solution found (optimized cross-sectional areas).
    float: Best fitness value (minimum truss_penalized_weight).
    """
    # Initialize population
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")

    positions = np.random.uniform(bounds[0], bounds[1], (num_wolves, dim))

    # Optimization loop
    for iter in range(max_iter):
        for i in range(num_wolves):
            # Calculate fitness
            cross_sectional_areas = positions[i]

            fitness, weight = cost_function(
                cross_sectional_areas,
                connectivity_matrix,
                force_nodes,
                node_coordinates,
                restrained_nodes,
                material_density,
                max_displacement,
                allowable_stress,
                plot_flag,
            )

            # Update alpha, beta, and delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = cross_sectional_areas.copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = cross_sectional_areas.copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = cross_sectional_areas.copy()

        # Update positions
        a = 2 - iter * (2 / max_iter)  # a decreases linearly from 2 to 0

        for i in range(num_wolves):
            for j in range(dim):
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

        # Ensure positions are within bounds
        positions = np.clip(positions, bounds[0], bounds[1])
        print(alpha_score, weight)

    return alpha_pos, alpha_score


if __name__ == "__main__":
    # Example usage (the user should provide actual input values for these):
    # Define problem parameters
    evaluation_count = 0
    num_wolves = 10
    max_iter = 100
    dim = 10  # Number of truss elements (this should match the number of elements in your truss)
    bounds = (0.1, 10.0)  # Example bounds for cross-sectional areas
    connectivity_matrix = np.array([...])
    force_nodes = np.array([...])
    node_coordinates = np.array([...])
    restrained_nodes = np.array([...])

    # Run GWO
    best_solution, best_fitness = grey_wolf_optimization(
        num_wolves,
        max_iter,
        dim,
        bounds,
        connectivity_matrix,
        force_nodes,
        node_coordinates,
        restrained_nodes,
    )
    print("Best solution (cross-sectional areas):", best_solution)
    print("Best fitness (truss penalized weight):", best_fitness)
