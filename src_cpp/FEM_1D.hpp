#pragma once

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
typedef Eigen::MatrixXd Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

using namespace std;

class FEM_1D
{
private:
    Vector xD;
    Vector xR;
    vector<int> Randelemente;

    Eigen::VectorXd plist;
    Eigen::MatrixXi tlist;
    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd D;
    Eigen::VectorXd Sol_noRW;
    Eigen::VectorXd Sol;

    function<double(double)> alpha;
    function<double(double)> beta;
    function<double(double)> f;

    function<double(double)> phi;
    function<double(double)> gamma;
    function<double(double)> q;

public:
    FEM_1D(
        Vector xD,
        Vector xR,
        Vector plist,
        function<double(double)> alpha,
        function<double(double)> beta,
        function<double(double)> f,
        function<double(double)> phi,
        function<double(double)> gamma,
        function<double(double)> q);

    void gen_tlist();

    tuple<vector<double>, vector<double>, vector<double>> gen_K11_K12_D1();

    void assemble_matrix(vector<double>, vector<double>, vector<double>);

    void solve_LGS();

    void reconstruct_solution();

    vector<tuple<string, double>> full_solve();

    void print_solution();

    Vector get_Solution();

    tuple<Vector, vector<double>> validate_sol(Vector, double);

    ~FEM_1D() = default;
};

vector<int> gen_Randelemente(Vector xR, Vector plist);
