# FEM Project
This repo is for the Vorlesung "Methoden der Feldberechnung" at the HKA.
It contains the code for 1D and 2D FEM solvers.

## Set up the environment
This project uses a `pyproject.toml` file to manage dependencies. 
To use the whole project, you can need following dependencies:
- `cmake` (for building the C++ code)
- `Eigen3` (for linear algebra in C++)
- `Intel MKL` (for hardware accelerated linear algebra on Intel Hardware; optional)

The whole FEM Solver is also available in python code, but it is not as optimized or fast as the C++ code. This C++ code is highly recommended.

To set up the environment, you can use `pip` to install the dependencies from the `pyproject.toml` file, but I recommend using `uv` to manage the environment. You can create a new environment and install the dependencies with the following commands:

```bash
uv sync
```

If you also want to use jupyter notebooks, run this:
```bash
uv sync --extra dev
```