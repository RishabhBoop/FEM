#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>  // Eigen dense <-> numpy
#include <nanobind/eigen/sparse.h> // Eigen sparse support
#include <nanobind/stl/function.h> // std::function <-> Python callable
#include <nanobind/stl/vector.h>   // std::vector <-> Python list
#include <nanobind/stl/tuple.h>    // std::tuple <-> Python tuple
#include <nanobind/stl/string.h>   // std::string <-> Python str
#include <nanobind/stl/complex.h>  // std::complex <-> Python complex
#include "FEM_1D.hpp"
#include "FEM_2D.hpp"

#ifndef MODULE_NAME
#define MODULE_NAME fem_cpp
#endif

#define MY_NB_MODULE_IMPL(name, m) NB_MODULE(name, m)
#define MY_NB_MODULE(name, m) MY_NB_MODULE_IMPL(name, m)

namespace nb = nanobind;
using namespace nb::literals; // for "_a" arg literals
using namespace std;
using FEM_2D_real = FEM_2D<double>;
using FEM_2D_complex = FEM_2D<std::complex<double>>;

MY_NB_MODULE(MODULE_NAME, m)
{
     nb::class_<FEM_1D>(m, "FEM_1D")
         .def(nb::init<
              Vector,
              Vector,
              Vector,
              function<double(double)>,
              function<double(double)>,
              function<double(double)>,
              function<double(double)>,
              function<double(double)>,
              function<double(double)>>())
         .def_rw("RESOLUTION", &FEM_1D::RESOLUTION)
         .def("gen_tlist", &FEM_1D::gen_tlist,
              "Generate the tlist based on the plist")
         .def("gen_K11_K12_D1", &FEM_1D::gen_K11_K12_D1,
              "Generate the K11, K12, and D1 vectors for each element")
         .def("assemble_matrix", &FEM_1D::assemble_matrix,
              "Assemble the global stiffness matrix K and load vector D",
              "K11"_a, "K12"_a, "D1"_a)
         .def("solve_LGS", &FEM_1D::solve_LGS,
              "Solve the linear system K * Sol_noRW = D")
         .def("reconstruct_solution", &FEM_1D::reconstruct_solution,
              "Reconstruct the full solution vector Sol")
         .def("full_solve", &FEM_1D::full_solve,
              "Run the full FEM solve process")
         .def("get_Solution", &FEM_1D::get_Solution,
              "Get the solution vector Sol as a NumPy array")
         .def("validate_sol", &FEM_1D::validate_sol,
              "Validate against an analytical solution, returning errors");

     nb::class_<FEM_2D_real>(m, "FEM_2D")
         .def(nb::init<
              VectorINT,
              MatrixINT,
              Matrix,
              MatrixINT,
              function<double(double, double)>,
              function<double(double, double)>,
              function<double(double, double)>,
              function<double(double, double)>,
              function<double(double, double)>,
              function<double(double, double)>,
              function<double(double, double)>>())
         .def_rw("RESOLUTION", &FEM_2D_real::RESOLUTION)
         .def("gen_b", &FEM_2D_real::gen_b,
              "Generate b from a vector")
         .def("gen_c", &FEM_2D_real::gen_c,
              "Generate c from a vector")
         .def("gen_delta_E", &FEM_2D_real::gen_delta_E,
              "Generate delta_E")
         .def("gen_local_matrices", &FEM_2D_real::gen_local_matrices,
              "Generate local matrices")
         .def("assemble_matrix", &FEM_2D_real::assemble_matrix,
              "Assemble the global stiffness matrix and load vector",
              "K11"_a, "K22"_a, "K33"_a, "K12"_a, "K13"_a, "K23"_a, "D1"_a)
         .def("solve_LGS", &FEM_2D_real::solve_LGS,
              "Solve the linear system")
         .def("reconstruct_solution", &FEM_2D_real::reconstruct_solution,
              "Reconstruct the full solution vector")
         .def("full_solve", &FEM_2D_real::full_solve,
              "Run the full FEM solve process")
         .def("print_solution", &FEM_2D_real::print_solution,
              "Print the computed solution")
         .def("get_dr", &FEM_2D_real::get_dr,
              "Get the list of Dirichlet nodes as a NumPy array")
         .def("get_rr", &FEM_2D_real::get_rr,
              "Get the list of Robin elements as a NumPy array")
         .def("get_Randelemente", &FEM_2D_real::get_Randelemente,
              "Get the list of boundary elements as a Python list")
         .def("get_plist", &FEM_2D_real::get_plist,
              "Get the list of node coordinates as a NumPy array")
         .def("get_tlist", &FEM_2D_real::get_tlist,
              "Get the list of element connectivity as a NumPy array")
         .def("get_K", &FEM_2D_real::get_K,
              "Get the global stiffness matrix K as a SciPy sparse matrix")
         .def("get_D", &FEM_2D_real::get_D,
              "Get the load vector D as a NumPy array")
         .def("get_Sol_noRW", &FEM_2D_real::get_Sol_noRW,
              "Get the solution vector before reconstruction")
         .def("get_Solution", &FEM_2D_real::get_Sol,
              "Get the solution vector as a NumPy array")
         .def("validate_sol", &FEM_2D_real::validate_sol,
              "Validate against an analytical solution");

     nb::class_<FEM_2D_complex>(m, "FEM_2D_complex")
         .def(nb::init<
              VectorINT,
              MatrixINT,
              Matrix,
              MatrixINT,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>,
              function<complex<double>(double,double)>>())
         .def_rw("RESOLUTION", &FEM_2D_complex::RESOLUTION)
         .def("gen_b", &FEM_2D_complex::gen_b,
              "Generate b from a vector")
         .def("gen_c", &FEM_2D_complex::gen_c,
              "Generate c from a vector")
         .def("gen_delta_E", &FEM_2D_complex::gen_delta_E,
              "Generate delta_E")
         .def("gen_local_matrices", &FEM_2D_complex::gen_local_matrices,
              "Generate local matrices")
         .def("assemble_matrix", &FEM_2D_complex::assemble_matrix,
              "Assemble the global stiffness matrix and load vector",
              "K11"_a, "K22"_a, "K33"_a, "K12"_a, "K13"_a, "K23"_a, "D1"_a)
         .def("solve_LGS", &FEM_2D_complex::solve_LGS,
              "Solve the linear system")
         .def("reconstruct_solution", &FEM_2D_complex::reconstruct_solution,
              "Reconstruct the full solution vector")
         .def("full_solve", &FEM_2D_complex::full_solve,
              "Run the full FEM solve process")
         .def("print_solution", &FEM_2D_complex::print_solution,
              "Print the computed solution")
         .def("get_dr", &FEM_2D_complex::get_dr,
              "Get the list of Dirichlet nodes as a NumPy array")
         .def("get_rr", &FEM_2D_complex::get_rr,
              "Get the list of Robin elements as a NumPy array")
         .def("get_Randelemente", &FEM_2D_complex::get_Randelemente,
              "Get the list of boundary elements as a Python list")
         .def("get_plist", &FEM_2D_complex::get_plist,
              "Get the list of node coordinates as a NumPy array")
         .def("get_tlist", &FEM_2D_complex::get_tlist,
              "Get the list of element connectivity as a NumPy array")
         .def("get_K", &FEM_2D_complex::get_K,
              "Get the global stiffness matrix K as a SciPy sparse matrix")
         .def("get_D", &FEM_2D_complex::get_D,
              "Get the load vector D as a NumPy array")
         .def("get_Sol_noRW", &FEM_2D_complex::get_Sol_noRW,
              "Get the solution vector before reconstruction")
         .def("get_Solution", &FEM_2D_complex::get_Sol,
              "Get the solution vector as a NumPy array")
         .def("validate_sol", &FEM_2D_complex::validate_sol,
              "Validate against an analytical solution");
}