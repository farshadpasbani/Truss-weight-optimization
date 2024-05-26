# Truss Analysis and Optimization Documentation
This documentation provides an overview of the functions and modules for analyzing and optimizing a truss structure using finite element analysis and the Grey Wolf Optimization algorithm.

## Theoretical Background for Truss Analysis

### 1. Introduction
Trusses are structures composed of straight members connected at joints, designed to bear loads through axial forces (tension or compression). They are commonly used in engineering applications such as bridges, towers, and buildings. Truss analysis involves determining the internal forces in the members and the displacements of the joints under given loads and boundary conditions.

### 2. Fundamental Assumptions
Truss analysis is based on several key assumptions:
1. **Members are Pin-Connected**: Members are connected by frictionless pins, allowing them to rotate freely.
2. **Loads are Applied at Joints**: External loads and reactions are applied only at the joints, not along the members.
3. **Members are Axially Loaded**: Members carry only axial forces, with no bending moments or shear forces.
4. **Small Deformations**: Deformations are small enough that members stay within their elastic limit.

### 3. Types of Trusses
Trusses can be classified into two main types based on their geometry and load-carrying behavior:
1. **Plane Trusses**: Lie in a single plane and are used in two-dimensional structural applications.
2. **Space Trusses**: Extend into three dimensions and are used for complex structures like space frames.

### 4. Structural Analysis of Trusses
The analysis of trusses typically involves the following steps:

#### 4.1. Calculation of Member Stiffness Matrix
Each truss member's stiffness matrix represents its resistance to deformation under axial loads. For a truss element, the local stiffness matrix $\( k \)$ in local coordinates can be derived as:

$$
k = \frac{EA}{L} \begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}
$$

where:
- $\( E \)$ is the Young's modulus of the material.
- $\( A \)$ is the cross-sectional area of the member.
- $\( L \)$ is the length of the member.

#### 4.2. Transformation to Global Coordinates
The local stiffness matrix must be transformed to the global coordinate system using the direction cosines of the member. For a truss element oriented at an angle $\theta$ with respect to the global $x$-axis, the transformation matrix $\( T \)$ is:

$$
T = \begin{bmatrix}
C & S & 0 & 0 \\
-S & C & 0 & 0 \\
0 & 0 & C & S \\
0 & 0 & -S & C
\end{bmatrix}
$$

where $C = \cos(\theta)$ and $S = \sin(\theta)$.

The global stiffness matrix $K_e$ for an element is obtained as:

$$
K_e = T^T k T
$$



#### 4.3. Assembly of Global Stiffness Matrix
The global stiffness matrix for the entire truss is assembled by summing the contributions of individual element stiffness matrices into the appropriate positions, based on the connectivity of the nodes.

#### 4.4. Application of Boundary Conditions
Boundary conditions are applied by modifying the global stiffness matrix and the load vector to account for supports and constraints. Nodes with specified displacements (restrained nodes) are handled by setting the corresponding rows and columns of the stiffness matrix to enforce the constraints.

#### 4.5. Solution of System Equations
The system of linear equations representing the equilibrium of the truss is solved to find the nodal displacements:
$$\[ \mathbf{K} \mathbf{d} = \mathbf{F} \]$$
where $\( \mathbf{K} \)$ is the global stiffness matrix, $\( \mathbf{d} \)$ is the vector of nodal displacements, and $\( \mathbf{F} \)$ is the vector of applied loads.

#### 4.6. Calculation of Member Forces
Once the nodal displacements are known, the internal axial forces in the truss members are calculated using:
$$\[ \mathbf{F}_e = k (\mathbf{T} \mathbf{d}_e) \]$$
where $\( \mathbf{d}_e \)$ is the vector of displacements for the element's nodes.

### 5. Penalty Approach for Constraints
Various methods exist for managing constraints, such as maximum displacement and allowable stress limits. In our approach, we employ a penalty method by augmenting the total cost function, representing the weight of the truss, with a penalty term proportional to the degree of constraint violation.

For instance, when the calculated stress in a truss member surpasses the allowable stress threshold, we modify the weight of that member by multiplying it with the difference between the allowable stress and the calculated stress. While this technique does not guarantee that the resulting truss fully satisfies all constraints, it ensures that the optimization process progresses toward the optimal solution by penalizing constraint violations.

### 6. Structural Optimization
Structural optimization involves finding the optimal configuration of a truss to meet certain criteria such as minimum weight, cost, or deflection. Methods like Grey Wolf Optimization (GWO) are used to iteratively improve the design by adjusting parameters like cross-sectional areas of the members while respecting constraints on stresses and displacements.
