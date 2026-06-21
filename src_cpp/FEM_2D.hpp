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

typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi VectorINT;

typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXi MatrixINT;
typedef Eigen::SparseMatrix<double> SparseMatrix;

using namespace std;

class FEM_2D
{
private:
    VectorINT dr;
    MatrixINT rr;

    vector<int> Randelemente;

    Matrix plist;
    MatrixINT tlist;

    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd D;

    Eigen::VectorXd Sol_noRW;
    Eigen::VectorXd Sol;

    function<double(double, double)> alpha1;
    function<double(double, double)> alpha2;
    function<double(double, double)> beta;
    function<double(double, double)> f;

    function<double(double, double)> phi;
    function<double(double, double)> gamma;
    function<double(double, double)> q;

    vector<bool> is_dirichlet;

public:
    double RESOLUTION = 1e-11;

    FEM_2D(
        VectorINT dr,
        MatrixINT rr,
        Matrix plist,
        MatrixINT tlist,
        function<double(double, double)> alpha1,
        function<double(double, double)> alpha2,
        function<double(double, double)> beta,
        function<double(double, double)> f,
        function<double(double, double)> phi,
        function<double(double, double)> gamma,
        function<double(double, double)> q);

    Vector gen_b(const Vector &) const;
    Vector gen_c(const Vector &) const;
    double gen_delta_E(int, int, int) const;

    tuple<Vector, Vector, Vector, Vector, Vector, Vector, Vector> gen_local_matrices();

    void assemble_matrix(Vector &K11, Vector &K22, Vector &K33, Vector &K12, Vector &K13, Vector &K23, Vector &D1);

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
    Eigen::SparseMatrix<double> get_K() { return K; };
    Eigen::VectorXd get_D() { return D; };
    Eigen::VectorXd get_Sol_noRW() { return Sol_noRW; };
    Eigen::VectorXd get_Sol() { return Sol; };

    tuple<Vector, vector<double>> validate_sol(Vector, double);

    ~FEM_2D() = default;
};

vector<int> gen_Randelemente(Vector, Vector, double);
