#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Auto-converts between Eigen types and NumPy arrays
#include <pybind11/functional.h> // Auto-converts Python callables <-> std::function
#include <pybind11/stl.h>

#include "FEM_1D.hpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(fem_cpp, m)
{
    py::class_<FEM_1D>(m, "FEM_1D")
        .def(py::init<
             Vector,
             Vector,
             Vector,
             function<double(double)>,
             function<double(double)>,
             function<double(double)>,
             function<double(double)>,
             function<double(double)>,
             function<double(double)>>())
        .def("gen_tlist", &FEM_1D::gen_tlist, "Generate the tlist based on the plist")
        .def("gen_K11_K12_D1", &FEM_1D::gen_K11_K12_D1, "Generate the K11, K12, and D1 vectors for each element")
        .def("assemble_matrix", &FEM_1D::assemble_matrix, "Assemble the global stiffness matrix K and load vector D from the element contributions", py::arg("K11"), py::arg("K12"), py::arg("D1"))
        .def("solve_LGS", &FEM_1D::solve_LGS, "Solve the linear system K * Sol_noRW = D")
        .def("reconstruct_solution", &FEM_1D::reconstruct_solution, "Reconstruct the full solution vector Sol from the free nodes and Dirichlet boundary conditions")
    
        .def("full_solve", &FEM_1D::full_solve, "Run the full FEM solve process")
        .def("get_Solution", &FEM_1D::get_Solution, "Get the computed solution vector Sol as a NumPy array")
        .def("validate_sol", &FEM_1D::validate_sol, "Validate the computed solution against an analytical solution at the nodes, returning a vector of errors");
}