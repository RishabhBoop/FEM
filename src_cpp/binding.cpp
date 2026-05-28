#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // Auto-converts between Eigen types and NumPy arrays
#include <pybind11/functional.h> // Auto-converts Python callables <-> std::function
#include <pybind11/stl.h>

#include "FEM_1D.hpp"
#include "FEM_2D.hpp"

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
            .def_readwrite("RESOLUTION", &FEM_1D::RESOLUTION)
        .def("gen_tlist", &FEM_1D::gen_tlist, "Generate the tlist based on the plist")
        .def("gen_K11_K12_D1", &FEM_1D::gen_K11_K12_D1, "Generate the K11, K12, and D1 vectors for each element")
        .def("assemble_matrix", &FEM_1D::assemble_matrix, "Assemble the global stiffness matrix K and load vector D from the element contributions", py::arg("K11"), py::arg("K12"), py::arg("D1"))
        .def("solve_LGS", &FEM_1D::solve_LGS, "Solve the linear system K * Sol_noRW = D")
        .def("reconstruct_solution", &FEM_1D::reconstruct_solution, "Reconstruct the full solution vector Sol from the free nodes and Dirichlet boundary conditions")
    
        .def("full_solve", &FEM_1D::full_solve, "Run the full FEM solve process")
        .def("get_Solution", &FEM_1D::get_Solution, "Get the computed solution vector Sol as a NumPy array")
        .def("validate_sol", &FEM_1D::validate_sol, "Validate the computed solution against an analytical solution at the nodes, returning a vector of errors");

    py::class_<FEM_2D>(m, "FEM_2D")
        .def(py::init<
             Vector,
             Vector,
             Matrix,
             MatrixINT,
             function<double(double, double)>,
             function<double(double, double)>,
             function<double(double, double)>,
             function<double(double, double)>,
             function<double(double, double)>,
             function<double(double, double)>,
             function<double(double, double)>>())
        .def_readwrite("RESOLUTION", &FEM_2D::RESOLUTION)
        .def("gen_b", py::overload_cast<const Vector&>(&FEM_2D::gen_b, py::const_), "Generate b from a vector")
        .def("gen_c", py::overload_cast<const Vector&>(&FEM_2D::gen_c, py::const_), "Generate c from a vector")
        .def("gen_delta_E", &FEM_2D::gen_delta_E, "Generate delta_E")
        .def("gen_local_matrices", &FEM_2D::gen_local_matrices, "Generate local matrices")
        .def("assemble_matrix", &FEM_2D::assemble_matrix,
             "Assemble the global stiffness matrix and load vector",
             py::arg("K11"), py::arg("K22"), py::arg("K33"),
             py::arg("K12"), py::arg("K13"), py::arg("K23"), py::arg("D1"))
        .def("solve_LGS", &FEM_2D::solve_LGS, "Solve the linear system")
        .def("reconstruct_solution", &FEM_2D::reconstruct_solution, "Reconstruct the full solution vector")
        .def("full_solve", &FEM_2D::full_solve, "Run the full FEM solve process")
        .def("print_solution", &FEM_2D::print_solution, "Print the computed solution")
        .def("get_Solution", &FEM_2D::get_Solution, "Get the computed solution vector as a NumPy array")
        .def("validate_sol", &FEM_2D::validate_sol, "Validate the computed solution against an analytical solution");
}