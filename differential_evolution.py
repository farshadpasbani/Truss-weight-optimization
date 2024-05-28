import numpy as np


def differential_evolution(
    cost_function,
    pop_size,
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
    CR = 0.9
    F = 0.8
    trends = []

    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = np.array(
        [
            cost_function(
                ind,
                connectivity_matrix,
                forces,
                node_coordinates,
                restrained_nodes,
                material_density,
                max_displacement,
                allowable_stress,
                plot_flag,
            )[0]
            for ind in pop
        ]
    )

    best_idx = np.argmin(fitness)
    best_vector = pop[best_idx]

    for iteration in range(max_iter):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
            trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
            trial_fitness, trial_weight = cost_function(
                trial,
                connectivity_matrix,
                forces,
                node_coordinates,
                restrained_nodes,
                material_density,
                max_displacement,
                allowable_stress,
                plot_flag,
            )

            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                pop[i] = trial
                if trial_fitness < fitness[best_idx]:
                    best_idx = i
                    best_vector = trial

        trends.append((fitness[best_idx], trial_weight))
        print(f"Iteration {iteration}: Best fitness = {fitness[best_idx]}")

    return best_vector, fitness[best_idx], trends
