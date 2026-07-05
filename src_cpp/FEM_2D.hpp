#pragma once

#if USE_MKL
#define EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#endif

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <functional>
#include <vector>
#include <numeric>   // std::iota
#include <algorithm> // std::sort, std::stable_sort
#include <tuple>
#include <cmath>
#include <complex>

typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi VectorINT;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXi MatrixINT;
typedef Eigen::SparseMatrix<double> SparseMatrix;

using namespace std;

template <typename Scalar>
class FEM_2D
{
private:
    VectorINT dr;
    MatrixINT rr;

    vector<int> Randelemente;

    Matrix plist;
    MatrixINT tlist;

    Eigen::SparseMatrix<Scalar> K;
    Eigen::Vector<Scalar, Eigen::Dynamic> D;

    Eigen::Vector<Scalar, Eigen::Dynamic> Sol_noRW;
    Eigen::Vector<Scalar, Eigen::Dynamic> Sol;

    function<Scalar(double, double)> alpha1;
    function<Scalar(double, double)> alpha2;
    function<Scalar(double, double)> beta;
    function<Scalar(double, double)> f;

    function<Scalar(double, double)> phi;
    function<Scalar(double, double)> gamma;
    function<Scalar(double, double)> q;

    vector<bool> is_dirichlet;

public:
    double RESOLUTION = 1e-11;

    FEM_2D(
        VectorINT dr,
        MatrixINT rr,
        Matrix plist,
        MatrixINT tlist,
        function<Scalar(double, double)> alpha1,
        function<Scalar(double, double)> alpha2,
        function<Scalar(double, double)> beta,
        function<Scalar(double, double)> f,
        function<Scalar(double, double)> phi,
        function<Scalar(double, double)> gamma,
        function<Scalar(double, double)> q);

    Vector gen_b(const Vector &) const;
    Vector gen_c(const Vector &) const;
    double gen_delta_E(int, int, int) const;

    tuple<Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>, Eigen::Vector<Scalar, Eigen::Dynamic>> gen_local_matrices();

    void assemble_matrix(Eigen::Vector<Scalar, Eigen::Dynamic> &K11, Eigen::Vector<Scalar, Eigen::Dynamic> &K22, Eigen::Vector<Scalar, Eigen::Dynamic> &K33, Eigen::Vector<Scalar, Eigen::Dynamic> &K12, Eigen::Vector<Scalar, Eigen::Dynamic> &K13, Eigen::Vector<Scalar, Eigen::Dynamic> &K23, Eigen::Vector<Scalar, Eigen::Dynamic> &D1);

    void solve_LGS();

    void reconstruct_solution();

    vector<tuple<string, double>> full_solve();

    void print_solution();

    // getter functions for all member variables
    VectorINT get_dr() { return dr; };
    MatrixINT get_rr() { return rr; };
    vector<int> get_Randelemente() { return Randelemente; };
    Matrix get_plist() { return plist; };
    MatrixINT get_tlist() { return tlist; };
    Eigen::SparseMatrix<Scalar> get_K() { return K; };
    Eigen::Vector<Scalar, Eigen::Dynamic> get_D() { return D; };
    Eigen::Vector<Scalar, Eigen::Dynamic> get_Sol_noRW() { return Sol_noRW; };
    Eigen::Vector<Scalar, Eigen::Dynamic> get_Sol() { return Sol; };

    tuple<Vector, vector<double>> validate_sol(Eigen::Vector<Scalar, Eigen::Dynamic> sol_tst, double max_error);
    ~FEM_2D() = default;
};