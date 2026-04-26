# FEM Solver in C++
## 1D
This is a simple implementation of a finite element method (FEM) solver in C++ for 1D problems. 
### build
To build the project, you must have at least a `c++17` compliant compiler and `cmake` installed. 
Also here a list of packages you need to install:
- `eigen3` (for linear algebra)
- `pybind11` (for python bindings)

To build the `main.cpp` file, you can use the following commands:
```bash
cmake -S . -B build
cmake --build build
```
This will create an executable file named `fem_app` in the `build` directory. You can run it using:
```bash
./build/fem_app
```
### python bindings
Usually, you would use the C++ implementation of the FEM_1D class for calculations and visualization in python. To use this, we first need a venv to install the required packages.
```bash
python -m venv venv
source venv/bin/activate
```
Then we can install the required packages using pip:
```bash
pip install -e .
```
This will build the C++ code and install the python packages needed. 
Now you can import the `fem_1d` module in python and use the `FEM_1D` class for your calculations. 
```python
import fem_cpp
```

### usage
To use the `FEM_1D` class, you need to create an instance of the class.
The constructor takes the following parameters:
- `xD` 
    - a vector of doubles representing the coordinates of dirichlet boundaries. This should be a `np.array` or a `Eigen::VectorXd`.
- `xR` 
    - a vector of doubles representing the coordinates of Robin boundaries. This should be a `np.array` or a `Eigen::VectorXd`.
- `plist`
    - a vector of doubles representing the coordinates of the points in the domain. This should be a `np.array` or a `Eigen::VectorXd`.
- `alpha`
    - a function that takes a double and returns a double representing the coefficient of the alpha term in the Partial Differential Equation (PDE). This should be a `std::function<double(double)>`. In python, you just write your normal function and add the `@cfunc(float64(float64))` decorator to it. Then you can pass it to the constructor like normal.
- `beta`
    - a function that takes a double and returns a double representing the coefficient of the beta term in the PDE. Using this is the exact same as `alpha`.
- `f`
    - a function that takes a double and returns a double representing the Inhomogenity (Right hand side) in the PDE. This should be a `std::function<double(double)>`. Using this is the exact same as `alpha`.
- `phi`
    - a function that takes a double and returns a double representing the Dirichlet boundary condition. This should be a `std::function<double(double)>`. Using this is the exact same as `alpha`.
- `gamma`
    - a function that takes a double and returns a double representing the Robin boundary condition. This should be a `std::function<double(double)>`. Using this is the exact same as `alpha`.
- `q`
    - a function that takes a double and returns a double representing the coefficient of the Robin boundary condition. This should be a `std::function<double(double)>`. Using this is the exact same as `alpha`.

Once the object is created, you can call the `full_solve` method to solve the PDE. This method will return a vector of tuple of doubles full of times of each step of the solving process. This could then be used to print the timing (speed of the solver).
```python
timings = fem_solver.full_solve()
```
The printing of the timings are done ideally in a python function. An example can be found in the [`validate_1D_cpp.py`](../validations/validate_1d_cpp.py) file.

Once the `full_solve` method has been called, you can get the solution using the `get_Solution` method. This will return a numpy array of doubles representing the solution corresponding to the points in `plist`.
```python
sol = fem_solver.get_Solution()
```
You can then visualize the solution in `matplotlib`. An example of how to visualize the solution can be found in the [`validate_1D_cpp.py`](../validations/validate_1d_cpp.py) file.

If you want to validate your solution against a solution from the professor, you can use the `validate_sol` method. This method must be called after `full_solve`. It takes a numpy array (or an `Eigen::VectorXd`) of doubles representing the **solution from the professor** and a specific **error tolerance**. It returns a numpy array of doubles representing the error at each point in `plist` AND a list of doubles that represent the maximum, minimum and mean absolute error .
```python
error, error_stats = fem_solver.validate_sol(prof_sol, )
```
Then, you can visualize the error in `matplotlib` and print the error statistics. An example of how to visualize the error and print the error statistics can be found in the [`validate_1D_cpp.py`](../validations/validate_1d_cpp.py) file.
