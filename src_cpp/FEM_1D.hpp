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

#ifndef COLORS
/** @def RESET
 *  @brief Resets terminal text formatting to default.
 */
#define RESET "\033[0m"          /* Reset */
#define BLACK "\033[30m"         /* Black */
#define RED "\033[31m"           /* Red */
#define GREEN "\033[32m"         /* Green */
#define YELLOW "\033[33m"        /* Yellow */
#define BLUE "\033[34m"          /* Blue */
#define MAGENTA "\033[35m"       /* Magenta */
#define CYAN "\033[36m"          /* Cyan */
#define WHITE "\033[37m"         /* White */
#define BOLDBLACK "\033[1;30m"   /* Bold Black */
#define BOLDRED "\033[1;31m"     /* Bold Red */
#define BOLDGREEN "\033[1;32m"   /* Bold Green */
#define BOLDYELLOW "\033[1;33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1;34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1;35m" /* Bold Magenta */
#define BOLDCYAN "\033[1;36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1;37m"   /* Bold White */

#endif

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

    void full_solve();

    void print_solution();

    Vector get_Solution();

    Vector validate_sol(Vector, string, double);

    ~FEM_1D() = default;
};

vector<int> gen_Randelemente(Vector xR, Vector plist);
