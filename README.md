# Truss Analysis and Optimization Documentation

## Overview
This project focuses on optimizing truss structures using Grey Wolf Optimization (GWO) and Differential Evolution (DE) algorithms. It automates the design optimization process to minimize weight while adhering to structural constraints like displacement and stress limits.
- Theoretical background for Finite Element Analysis of trusses: [Truss-analysis.md](Truss-analysis.md)
- Inner workings of Grey Wolf Optimization algorithm: [Grey_wolf_optimization.md](Grey_wolf_optimization.md); Original paper: [Grey Wolf Optimizer](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853)
- Inner workings of Differential Evolution algorithm - original paper: [Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328)

## File Descriptions

### `main.py`
- **Purpose**: Serves as the central execution script, orchestrating the application of GWO and DE algorithms for truss optimization.
- **Key Components**:
  - `grey_wolf_optimization`: Implements the GWO algorithm to optimize truss configurations.
  - `differential_evolution`: Applies the DE algorithm for truss optimization.
  - `plot_all_trends`: Visualizes comparative trends between GWO and DE optimizations.
  - `analyze_truss`: Analyzes the optimized truss based on physical and structural parameters.

### `truss_analysis.py`
- **Purpose**: Handles the structural analysis of trusses, including computation of stiffness matrices and response under load.
- **Key Components**:
  - `compute_element_stiffness`: Calculates the stiffness matrix for truss elements based on material properties and geometry.
  - `assemble_global_stiffness`: Constructs the global stiffness matrix from individual element matrices.
  - `apply_boundary_conditions`: Enforces boundary conditions on nodes (supports) to simulate real-world constraints.
  - `analyze_truss`: Conducts a complete truss analysis, determining displacements, stresses, and overall system response.

### `gwo.py`
- **Purpose**: Contains the implementation of the Grey Wolf Optimization algorithm specifically tailored for truss design.
- **Key Components**:
  - `grey_wolf_optimization`: Executes the optimization process, adjusting truss parameters to find the minimal weight configuration that satisfies all design constraints.

### `differential_evolution.py`
- **Purpose**: Provides the implementation of the Differential Evolution algorithm for truss optimization.
- **Key Components**:
  - `differential_evolution`: Optimizes truss designs by evolving a population of potential solutions towards the best solution using genetic operators.

## Installation
To set up this project:
1. Ensure Python 3.x is installed.
2. Install necessary dependencies:
   ```bash
   pip install numpy matplotlib

## Usage
To run the application:
1. Navigate to the project directory.
2. Execute the script via the command line:
   ```bash
   python main.py
3. Review output graphs and terminal messages for optimization results.

## Contributing
Contributors are welcome to improve the algorithms or extend the application's capabilities:
- Fork the repository.
- Create your feature branch (`git checkout -b feature/fooBar`).
- Commit your changes (`git commit -am 'Add some fooBar'`).
- Push to the branch (`git push origin feature/fooBar`).
- Open a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- **Farshad Pasbani** - *Truss weight optimization* - [GitHub](https://github.com/farshadpasbani)


