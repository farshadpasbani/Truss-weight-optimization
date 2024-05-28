import numpy as np


def grey_wolf_optimization(
    cost_function,
    num_wolves,
    max_iter,
    dim,
    bounds,
    connectivity_matrix,
    forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag,
):
    alpha_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_pos = np.zeros(dim)
    beta_score = float("inf")
    delta_pos = np.zeros(dim)
    delta_score = float("inf")
    trends = []

    positions = np.random.uniform(bounds[0], bounds[1], (num_wolves, dim))

    for iter in range(max_iter):
        for i in range(num_wolves):
            cross_sectional_areas = positions[i]
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

        a = 2 - iter * (2 / max_iter)

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

        positions = np.clip(positions, bounds[0], bounds[1])
        trends.append((alpha_score, weight))
        print(f"Iteration {iter}: Best fitness = {alpha_score}")

    return alpha_pos, alpha_score, trends
