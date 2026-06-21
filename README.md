# FEM Project
This repo is for the Vorlesung "Methoden der Feldberechnung" at the HKA.
It contains the code for 1D and 2D FEM solvers.

## Set up the environment
This project uses a `pyproject.toml` file to manage dependencies. 
To use the whole project, you can need following dependencies:
- `cmake` (for building the C++ code)
- `Ninja`
- `clang` 
- `Eigen3` (for linear algebra in C++)
- `Intel MKL` (for hardware accelerated linear algebra on Intel Hardware; optional)
    - see [the official Intel MKL installation guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) for installation instructions. 
    - This has only been tested on version `2026.0`

The whole FEM Solver is also available in python code, but it is not as optimized or fast as the C++ code. This C++ code is highly recommended.

To set up the environment, you can use `pip` to install the dependencies from the `pyproject.toml` file, but I recommend using `uv` to manage the environment. You can create a new environment and install the dependencies with the following commands:

```bash
uv sync
```

If you also want to use jupyter notebooks, run this:
```bash
uv sync --extra dev
```

## Usage
To use the FEM solvers, you can import the modules in your python code and call the functions.
```python
import fem_cpp  # CPU-based solver
```

If using [hardware acceleration](#hardware-acceleration), use this:
```python
import fem_cpp_mkl  # MKL-based solver
```

## Harware Acceleration
> **Warning:** \
> You cannot import and use hardware accelerated and CPU solvers in the same python session due to nanobind limitations. 

This project can use Intel MKL for hardware accelerated linear algebra operations.
In my testing, only meshes with over 100.000 elements are faster using MKL (`Paradiso Solver`) rather than the default CPU-based `SparseLU` solver.
My guess is that the overhead of using MKL is higher than the performance gain for smaller meshes, because MKL has to copy the data to and from the GPU and/or parallelize the operations, which can be overkill for smaller meshes.

If you want to use MKL, you need to have it installed on your system and add it the environment variables.
```bash
source /path/to/intel/oneapi/setvars.sh  # e.g. source /opt/intel/oneapi/setvars.sh
```

Then, you just run `uv sync` or `uv sync --extra dev` to install the dependencies, and the C++ code will automatically use MKL if it is available. 
The MKL solver is implemented in `fem_cpp_mkl` module, while the default CPU-based solver is implemented in `fem_cpp` module.

> Note: If you had already run `uv sync` before setting up the environment variables for MKL, you need to run `uv sync --force-reinstall` to force reinstall the dependencies as opposed to using the cached versions.
