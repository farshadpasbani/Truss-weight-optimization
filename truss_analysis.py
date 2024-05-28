import numpy as np
import matplotlib.pyplot as plt


def compute_element_stiffness(E, A, length, C, S):
    return (
        E
        * A
        / length
        * np.array(
            [
                [C**2, C * S, -(C**2), -C * S],
                [C * S, S**2, -C * S, -(S**2)],
                [-(C**2), -C * S, C**2, C * S],
                [-C * S, -(S**2), C * S, S**2],
            ]
        )
    )


def assemble_global_stiffness(
    global_stiffness_matrix, local_stiffness_matrices, element_dof_indices
):
    for i in range(len(local_stiffness_matrices)):
        indices = np.ix_(element_dof_indices[i], element_dof_indices[i])
        global_stiffness_matrix[indices] += local_stiffness_matrices[i]
    return global_stiffness_matrix


def apply_boundary_conditions(global_stiffness_matrix, restrained_nodes, avg_diagonal):
    dof_indices = np.array(
        [2 * np.array(restrained_nodes) - 2, 2 * np.array(restrained_nodes) - 1]
    ).flatten()
    global_stiffness_matrix[dof_indices, :] = 0
    global_stiffness_matrix[:, dof_indices] = 0
    global_stiffness_matrix[dof_indices, dof_indices] = avg_diagonal
    return global_stiffness_matrix


def analyze_truss(
    cross_sectional_areas,
    connectivity_matrix,
    forces,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag,
):
    num_dof = 2 * len(node_coordinates)
    youngs_modulus = 210000
    force_vector = np.zeros(num_dof)
    for node_idx, fx, fy in forces:
        force_vector[2 * node_idx] = fx
        force_vector[2 * node_idx + 1] = fy

    node_i_coords = node_coordinates[connectivity_matrix[:, 0] - 1, :]
    node_j_coords = node_coordinates[connectivity_matrix[:, 1] - 1, :]

    element_lengths = np.linalg.norm(node_j_coords - node_i_coords, axis=1)
    cos_theta = (node_j_coords[:, 0] - node_i_coords[:, 0]) / element_lengths
    sin_theta = (node_j_coords[:, 1] - node_i_coords[:, 1]) / element_lengths

    local_stiffness_matrices = np.array(
        [
            compute_element_stiffness(
                youngs_modulus,
                cross_sectional_areas[i],
                element_lengths[i],
                cos_theta[i],
                sin_theta[i],
            )
            for i in range(len(connectivity_matrix))
        ]
    )

    global_stiffness_matrix = np.zeros((num_dof, num_dof))
    element_dof_indices = np.array(
        [
            [
                2 * (connectivity_matrix[i, 0] - 1),
                2 * (connectivity_matrix[i, 0] - 1) + 1,
                2 * (connectivity_matrix[i, 1] - 1),
                2 * (connectivity_matrix[i, 1] - 1) + 1,
            ]
            for i in range(len(connectivity_matrix))
        ]
    )

    global_stiffness_matrix = assemble_global_stiffness(
        global_stiffness_matrix, local_stiffness_matrices, element_dof_indices
    )
    avg_diagonal = np.mean(np.diag(global_stiffness_matrix))
    global_stiffness_matrix = apply_boundary_conditions(
        global_stiffness_matrix, restrained_nodes, avg_diagonal
    )

    displacements = np.linalg.solve(global_stiffness_matrix, force_vector)

    if plot_flag == 1:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Plot undeformed truss
        for i in range(len(connectivity_matrix)):
            start_node = connectivity_matrix[i, 0] - 1
            end_node = connectivity_matrix[i, 1] - 1
            x_values = [node_coordinates[start_node, 0], node_coordinates[end_node, 0]]
            y_values = [node_coordinates[start_node, 1], node_coordinates[end_node, 1]]
            plt.plot(
                x_values,
                y_values,
                "ko-",
                lw=2,
                markersize=5,
                label="Undeformed" if i == 0 else "",
            )

        # Calculate deformed coordinates
        deformed_coordinates = node_coordinates + displacements.reshape(-1, 2)

        # Plot deformed truss
        for i in range(len(connectivity_matrix)):
            start_node = connectivity_matrix[i, 0] - 1
            end_node = connectivity_matrix[i, 1] - 1
            x_values = [
                deformed_coordinates[start_node, 0],
                deformed_coordinates[end_node, 0],
            ]
            y_values = [
                deformed_coordinates[start_node, 1],
                deformed_coordinates[end_node, 1],
            ]
            plt.plot(
                x_values,
                y_values,
                "ro-",
                lw=2,
                markersize=5,
                label="Deformed" if i == 0 else "",
            )

        for idx, coord in enumerate(node_coordinates):
            plt.text(
                coord[0],
                coord[1] + 30,
                str(idx + 1),
                color="blue",
                fontsize=9,
                horizontalalignment="center",
            )

        for node in restrained_nodes:
            node_idx = node - 1
            plt.plot(
                node_coordinates[node_idx, 0],
                node_coordinates[node_idx, 1],
                "rs",
                markersize=10,
                label=(
                    "Restrained Node"
                    if "Restrained Node" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )

        for node_idx, fx, fy in forces:
            plt.arrow(
                node_coordinates[node_idx, 0],
                node_coordinates[node_idx, 1],
                fx / 1000,
                fy / 1000,
                head_width=50,
                head_length=100,
                fc="green",
                ec="green",
            )
            plt.text(
                node_coordinates[node_idx, 0] + fx / 1000,
                node_coordinates[node_idx, 1] + fy / 1000,
                f"{fx}N, {fy}N",
                fontsize=10,
                color="darkgreen",
            )

        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("X Coordinate (mm)")
        plt.ylabel("Y Coordinate (mm)")
        plt.title("Truss Structure Visualization")
        plt.grid(True)
        plt.legend()
        plt.show()

    element_stresses = np.zeros((len(connectivity_matrix), 4))
    for i in range(len(connectivity_matrix)):
        dof_indices = element_dof_indices[i]
        element_stresses[i, :] = (
            1
            / cross_sectional_areas[i]
            * np.dot(local_stiffness_matrices[i], displacements[dof_indices].flatten())
        )

    element_stress_magnitudes = np.sqrt(np.sum(element_stresses[:, :2] ** 2, axis=1))
    demand_capacity_ratios = element_stress_magnitudes / allowable_stress

    demand_capacity = np.abs(demand_capacity_ratios)
    stress_penalty = np.maximum(1, demand_capacity**2)
    displacement_penalty = np.maximum(
        1, np.max(np.abs(displacements)) - max_displacement
    )
    truss_penalized_weight = (
        np.sum(
            displacement_penalty**2
            * stress_penalty.T
            * cross_sectional_areas
            * element_lengths
        )
        * material_density
    )

    truss_weight = np.sum(cross_sectional_areas * element_lengths) * material_density

    return truss_penalized_weight, truss_weight
