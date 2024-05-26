import numpy as np
import matplotlib.pyplot as plt


def compute_element_stiffness(E, A, length, C, S):
    """
    Compute the stiffness matrix for a truss element.

    Parameters:
    E (float): Young's modulus of the material.
    A (float): Cross-sectional area of the element.
    length (float): Length of the element.
    C (float): Cosine of the angle between the element and the x-axis.
    S (float): Sine of the angle between the element and the y-axis.

    Returns:
    np.ndarray: 4x4 local stiffness matrix for the element.
    """
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
    """
    Assemble the global stiffness matrix from the local stiffness matrices of all elements.

    Parameters:
    global_stiffness_matrix (np.ndarray): The global stiffness matrix to be updated.
    local_stiffness_matrices (np.ndarray): Array of local stiffness matrices of the elements.
    element_dof_indices (np.ndarray): Indices of the degrees of freedom associated with each element.

    Returns:
    np.ndarray: Updated global stiffness matrix.
    """
    for i in range(len(local_stiffness_matrices)):
        global_stiffness_matrix[
            np.ix_(element_dof_indices[i], element_dof_indices[i])
        ] += local_stiffness_matrices[i]
    return global_stiffness_matrix


def apply_boundary_conditions(global_stiffness_matrix, restrained_nodes, avg_diagonal):
    """
    Apply boundary conditions to the global stiffness matrix by modifying it to account for restrained nodes.

    Parameters:
    global_stiffness_matrix (np.ndarray): The global stiffness matrix to be updated.
    restrained_nodes (list): List of nodes where displacements are restrained.
    avg_diagonal (float): Average value of the diagonal elements of the global stiffness matrix.

    Returns:
    np.ndarray: Updated global stiffness matrix with boundary conditions applied.
    """
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
    force_nodes,
    node_coordinates,
    restrained_nodes,
    material_density,
    max_displacement,
    allowable_stress,
    plot_flag,
):
    """
    Analyze the truss structure to compute its weight considering stress and displacement penalties.

    Parameters:
    cross_sectional_areas (np.ndarray): Array of cross-sectional areas of the truss elements.
    connectivity_matrix (np.ndarray): Connectivity matrix defining the elements and their nodes.
    force_nodes (np.ndarray): Array of nodes where external forces are applied.
    node_coordinates (np.ndarray): Array of coordinates of the nodes.
    restrained_nodes (np.ndarray): Array of nodes where displacements are restrained.
    max_displacement (float): Maximum allowable displacement (default is 3).
    allowable_stress (float): Maximum allowable stress (default is 2160).

    Returns:
    tuple: Computed penalized weight of the truss structure and actual weight.
    """

    num_dof = 2 * len(node_coordinates)
    youngs_modulus = 20500
    force_vector = np.zeros((num_dof, 1))
    force_vector[2 * force_nodes - 1] = -10000

    # Extract node coordinates for elements
    node_i_coords = node_coordinates[connectivity_matrix[:, 0] - 1, :]
    node_j_coords = node_coordinates[connectivity_matrix[:, 1] - 1, :]

    # Compute element lengths and direction cosines
    element_lengths = np.linalg.norm(node_j_coords - node_i_coords, axis=1)
    cos_theta = (node_j_coords[:, 0] - node_i_coords[:, 0]) / element_lengths
    sin_theta = (node_j_coords[:, 1] - node_i_coords[:, 1]) / element_lengths

    # Compute local stiffness matrices for all elements
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

    # Prepare global stiffness matrix
    global_stiffness_matrix = np.zeros((num_dof, num_dof))

    # Prepare DOF indices for all elements
    element_dof_indices = np.array(
        [
            [
                2 * (connectivity_matrix[i, 0] - 1),
                2 * connectivity_matrix[i, 0] - 1,
                2 * (connectivity_matrix[i, 1] - 1),
                2 * connectivity_matrix[i, 1] - 1,
            ]
            for i in range(len(connectivity_matrix))
        ]
    )

    # Assemble global stiffness matrix
    global_stiffness_matrix = assemble_global_stiffness(
        global_stiffness_matrix, local_stiffness_matrices, element_dof_indices
    )

    # Apply boundary conditions
    avg_diagonal = np.mean(np.diag(global_stiffness_matrix))
    global_stiffness_matrix = apply_boundary_conditions(
        global_stiffness_matrix, restrained_nodes, avg_diagonal
    )

    # Solve for displacements
    displacements = np.linalg.solve(global_stiffness_matrix, force_vector)

    if plot_flag == 1:
        # Plot deformed truss
        plt.figure()
        for i in range(len(connectivity_matrix)):
            plt.plot(
                [
                    node_coordinates[connectivity_matrix[i, 0] - 1, 0]
                    + displacements[2 * connectivity_matrix[i, 0] - 2],
                    node_coordinates[connectivity_matrix[i, 1] - 1, 0]
                    + displacements[2 * connectivity_matrix[i, 1] - 2],
                ],
                [
                    node_coordinates[connectivity_matrix[i, 0] - 1, 1]
                    + displacements[2 * connectivity_matrix[i, 0] - 1],
                    node_coordinates[connectivity_matrix[i, 1] - 1, 1]
                    + displacements[2 * connectivity_matrix[i, 1] - 1],
                ],
                "k-",
            )
        plt.plot(
            node_coordinates[:, 0] + displacements[::2].T,
            node_coordinates[:, 1] + displacements[1::2].T,
            "ro",
        )
        for i in range(len(force_nodes)):
            plt.text(
                node_coordinates[force_nodes[i] - 1, 0]
                + displacements[2 * force_nodes[i] - 2],
                node_coordinates[force_nodes[i] - 1, 1]
                + displacements[2 * force_nodes[i] - 1],
                f"Fx = {force_vector[2 * force_nodes[i] - 2]}, Fy = {force_vector[2 * force_nodes[i] - 1]}",
            )
        plt.show()

    # Compute element stresses and demand-capacity ratios
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

    # Compute penalties and truss weight
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


# Example usage (the user should provide actual input values for these):
# cross_sectional_areas = np.array([...])
# connectivity_matrix = np.array([...])
# force_nodes = np.array([...])
# node_coordinates = np.array([...])
# restrained_nodes = np.array([...])
# penalized_weight, weight = analyze_truss(cross_sectional_areas, connectivity_matrix, force_nodes, node_coordinates, restrained_nodes, max_displacement=3, allowable_stress=2160)


# Example usage (the user should provide actual input values for these):
# cross_sectional_areas = np.array([...])
# connectivity_matrix = np.array([...])
# force_nodes = np.array([...])
# node_coordinates = np.array([...])
# restrained_nodes = np.array([...])
# penalized_weight, weight = analyze_truss(cross_sectional_areas, connectivity_matrix, force_nodes, node_coordinates, restrained_nodes, max_displacement=3, allowable_stress


if __name__ == "__main__":
    FunctionEvaluation = 0
    A = np.ones(16)

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

    # Constraints
    max_displacement = 3
    allowable_stress = 2160
    print(
        analyze_truss(
            A, Connect, ForceNodes, Node, restrains, max_displacement, allowable_stress
        )
    )
