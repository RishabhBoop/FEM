#include "FEM_2D.hpp"

using namespace std;

FEM_2D::FEM_2D(
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
    function<double(double, double)> q) : dr(dr),
                                          rr(rr),
                                          plist(plist),
                                          tlist(tlist),
                                          alpha1(alpha1),
                                          alpha2(alpha2),
                                          beta(beta),
                                          f(f),
                                          phi(phi),
                                          gamma(gamma),
                                          q(q)
{
    is_dirichlet.resize(plist.rows(), false);
    for (int i = 0; i < dr.size(); ++i)
    {
        int idx = dr(i);
        if (idx >= 0 && idx < plist.rows())
            is_dirichlet[idx] = true;
    }
}

Vector FEM_2D::gen_b(const Vector &y) const
{
    // y = [y1, y2, y3] - y-coords of 3 nodes of one triangle
    Vector result(3);
    result(0) = y(1) - y(2); // b1
    result(1) = y(2) - y(0); // b2
    result(2) = y(0) - y(1); // b3
    return result;           // Vector of size 3
}

Vector FEM_2D::gen_c(const Vector &x) const
{
    // x = [x1, x2, x3] - x-coords of 3 nodes of one triangle
    Vector result(3);
    result(0) = x(2) - x(1); // c1
    result(1) = x(0) - x(2); // c2
    result(2) = x(1) - x(0); // c3
    return result;           // Vector of size 3
}

double FEM_2D::gen_delta_E(int p0, int p1, int p2) const
{
    double deltaE;

    // 2deltaE = (x1-x3)(y2-y3) - (x2-x3)(y1-y2)
    // twodeltaE = (p[0, 0] - p[2, 0]) * (p[1, 1] - p[2, 1]) - (p[1, 0] - p[2, 0]) * (p[0, 1] - p[2, 1])

    double x1 = plist(p0, 0), y1 = plist(p0, 1);
    double x2 = plist(p1, 0), y2 = plist(p1, 1);
    double x3 = plist(p2, 0), y3 = plist(p2, 1);

    deltaE = (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);

    return deltaE / 2.0;
}

tuple<Vector, Vector, Vector, Vector, Vector, Vector, Vector> FEM_2D::gen_local_matrices()
{
    Vector y_coords(3); // Shape: (3) - y(0, 1, 2)
    Vector x_coords(3); // Shape: (3) - x(0, 1, 2)

    Vector K11(tlist.rows()); // Shape: (N_elements) - K11 for each triangle
    Vector K22(tlist.rows()); // Shape: (N_elements) - K22 for each triangle
    Vector K33(tlist.rows()); // Shape: (N_elements) - K33 for each triangle

    Vector K12(tlist.rows()); // Shape: (N_elements) - K12 for each triangle
    Vector K13(tlist.rows()); // Shape: (N_elements) - K13 for each triangle
    Vector K23(tlist.rows()); // Shape: (N_elements) - K23 for each triangle

    Vector D1(tlist.rows()); // Shape: (N_elements) - D1 for each triangle

    for (int i = 0; i < tlist.rows(); ++i)
    {
        // For triangle i, get the 3 nodes
        y_coords(0) = plist(tlist(i, 0), 1); // y of node 0
        y_coords(1) = plist(tlist(i, 1), 1); // y of node 1
        y_coords(2) = plist(tlist(i, 2), 1); // y of node 2

        x_coords(0) = plist(tlist(i, 0), 0); // x of node 0
        x_coords(1) = plist(tlist(i, 1), 0); // x of node 1
        x_coords(2) = plist(tlist(i, 2), 0); // x of node 2

        Vector bj = gen_b(y_coords);
        Vector cj = gen_c(x_coords);

        double xm = (x_coords(0) + x_coords(1) + x_coords(2)) / 3.0; // centroid x
        double ym = (y_coords(0) + y_coords(1) + y_coords(2)) / 3.0; // centroid y

        double alpha1m = alpha1(xm, ym);
        double alpha2m = alpha2(xm, ym);
        double betam = beta(xm, ym);
        double fm = f(xm, ym);

        double deltaE = gen_delta_E(tlist(i, 0), tlist(i, 1), tlist(i, 2));

        K11(i) = (alpha1m / (4 * deltaE)) * bj(0) * bj(0) +
                 (alpha2m / (4 * deltaE)) * cj(0) * cj(0) +
                 (betam * 2 * deltaE / 12.0);
        K22(i) = (alpha1m / (4 * deltaE)) * bj(1) * bj(1) +
                 (alpha2m / (4 * deltaE)) * cj(1) * cj(1) +
                 (betam * 2 * deltaE / 12.0);
        K33(i) = (alpha1m / (4 * deltaE)) * bj(2) * bj(2) +
                 (alpha2m / (4 * deltaE)) * cj(2) * cj(2) +
                 (betam * 2 * deltaE / 12.0);

        K12(i) = (alpha1m / (4 * deltaE)) * bj(0) * bj(1) +
                 (alpha2m / (4 * deltaE)) * cj(0) * cj(1) +
                 (betam * deltaE / 12.0);
        K13(i) = (alpha1m / (4 * deltaE)) * bj(0) * bj(2) +
                 (alpha2m / (4 * deltaE)) * cj(0) * cj(2) +
                 (betam * deltaE / 12.0);
        K23(i) = (alpha1m / (4 * deltaE)) * bj(1) * bj(2) +
                 (alpha2m / (4 * deltaE)) * cj(1) * cj(2) +
                 (betam * deltaE / 12.0);

        D1(i) = (fm * deltaE / 3.0);
    }
    return {K11, K22, K33, K12, K13, K23, D1};
}

void FEM_2D::assemble_matrix(Vector &K11, Vector &K22, Vector &K33, Vector &K12, Vector &K13, Vector &K23, Vector &D1)
{
    vector<int> node_to_matrix(plist.rows(), -1); // list of size plist, initialized to -1 (indicating Randwert nodes); holds mapping from global node index to matrix index
    int free_count = 0;                           // count of free nodes (unknowns); This will be the size of the matrix and D vector after assembly
    for (int i = 0; i < plist.rows(); ++i)
    {
        // If not in xD, it's a free node (unknown)
        if (!is_dirichlet[i])
            node_to_matrix[i] = free_count++; // assign matrix index and increment free count
    }

    D = Vector::Zero(free_count);            // Initialize D vector with correct size and zeroes
    vector<Eigen::Triplet<double>> triplets; // {row, col, value} for sparse matrix assembly

    for (int i = 0; i < tlist.rows(); ++i)
    {
        int nodes[3] = {tlist(i, 0), tlist(i, 1), tlist(i, 2)}; // nodes of triangle i

        for (int r = 0; r < 3; ++r)
        {
            int global_row = nodes[r];                   // global node index (index of entry in plist)
            int matrix_row = node_to_matrix[global_row]; // corresponding index in the global K Matrix (could be the same if no dirichlet boundaries)
            if (matrix_row != -1)                        // If the node is a free node
            {
                D(matrix_row) += D1(i); // Add contribution to D vector

                for (int c = 0; c < 3; ++c)
                {
                    int global_col = nodes[c];                   // global node index for column
                    int matrix_col = node_to_matrix[global_col]; // corresponding index in the global K Matrix
                    if (matrix_col != -1)                        // If the column node is also a free node, add to K
                    {
                        // Now we know for sure that this entry contributes to K and is not a dirichlet node.
                        double value = 0.0;
                        if (r == c)
                        {
                            if (r == 0)
                                value = K11(i);
                            else if (r == 1)
                                value = K22(i);
                            else
                                value = K33(i);
                        }
                        else if ((r == 0 && c == 1) || (r == 1 && c == 0))
                            value = K12(i);
                        else if ((r == 0 && c == 2) || (r == 2 && c == 0))
                            value = K13(i);
                        else if ((r == 1 && c == 2) || (r == 2 && c == 1))
                            value = K23(i);

                        triplets.emplace_back(matrix_row, matrix_col, value); // Add triplet for K
                    }
                    else // If the column node is a Dirichlet node, move contribution to D
                    {
                        double phi_value = phi(plist(global_col, 0), plist(global_col, 1)); // Dirichlet value at this node
                        double value = 0.0;
                        if (r == c)
                        {
                            if (r == 0)
                                value = K11(i);
                            else if (r == 1)
                                value = K22(i);
                            else
                                value = K33(i);
                        }
                        else if ((r == 0 && c == 1) || (r == 1 && c == 0))
                            value = K12(i);
                        else if ((r == 0 && c == 2) || (r == 2 && c == 0))
                            value = K13(i);
                        else if ((r == 1 && c == 2) || (r == 2 && c == 1))
                            value = K23(i);

                        D(matrix_row) -= value * phi_value; // Move contribution to D
                    }
                }
            }
        }
    }

    // Apply robin boundary conditions
    for (int i = 0; i < rr.rows(); ++i)
    {
        int m0 = node_to_matrix[rr(i, 0)];
        int m1 = node_to_matrix[rr(i, 1)];

        // Skip if both nodes are Dirichlet
        if (m0 == -1 && m1 == -1)
            continue;

        pair<double, double> p0 = {plist(rr(i, 0), 0), plist(rr(i, 0), 1)};
        pair<double, double> p1 = {plist(rr(i, 1), 0), plist(rr(i, 1), 1)};

        double mid_x = (p0.first + p1.first) / 2.0;
        double mid_y = (p0.second + p1.second) / 2.0;
        double edge_length = sqrt(pow(p1.first - p0.first, 2) + pow(p1.second - p0.second, 2));

        double gamma_val = gamma(mid_x, mid_y);
        double q_val = q(mid_x, mid_y);

        double gamma_val_11 = gamma_val * edge_length / 3.0;
        double gamma_val_12 = gamma_val * edge_length / 6.0;
        double q_val_contribution = q_val * edge_length / 2.0;

        // Node 0 contributions
        if (m0 != -1)
        {
            triplets.emplace_back(m0, m0, gamma_val_11);
            D(m0) += q_val_contribution;
            if (m1 != -1)
            {
                triplets.emplace_back(m0, m1, gamma_val_12);
            }
            else
            {
                // m1 is Dirichlet, move its contribution to D
                double phi_value = phi(plist(rr(i, 1), 0), plist(rr(i, 1), 1));
                D(m0) -= gamma_val_12 * phi_value;
            }
        }

        // Node 1 contributions
        if (m1 != -1)
        {
            triplets.emplace_back(m1, m1, gamma_val_11);
            D(m1) += q_val_contribution;
            if (m0 != -1)
            {
                triplets.emplace_back(m1, m0, gamma_val_12);
            }
            else
            {
                // m0 is Dirichlet, move its contribution to D
                double phi_value = phi(plist(rr(i, 0), 0), plist(rr(i, 0), 1));
                D(m1) -= gamma_val_12 * phi_value;
            }
        }
    }

    K.resize(free_count, free_count);
    K.setFromTriplets(triplets.begin(), triplets.end());
}

void FEM_2D::solve_LGS()
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

    solver.compute(K);

    if (solver.info() != Eigen::Success)
    {
        // decomposition failed
        throw runtime_error("SparseLU decomposition failed");
        return;
    }

    Sol_noRW = solver.solve(D);

    if (solver.info() != Eigen::Success)
    {
        // solving failed
        throw runtime_error("SparseLU solving failed");
        return;
    }
}

void FEM_2D::reconstruct_solution()
{
    Sol.resize(plist.rows());

    int free_index = 0;
    for (int i = 0; i < plist.rows(); ++i)
    {
        if (is_dirichlet[i])
        {
            Sol(i) = phi(plist(i, 0), plist(i, 1));
        }
        else
        {
            Sol(i) = Sol_noRW(free_index++);
        }
    }
}

void FEM_2D::print_solution()
{
    // Just print the solution vector
    cout << "Solution at nodes:" << endl;
    cout << " phi = (";
    for (int i = 0; i < plist.rows(); ++i)
    {
        cout << Sol(i) << " ";
    }
    cout << ")" << endl;
}

void FEM_2D::print_D()
{
    cout << "Load vector D:" << endl;
    cout << " D = (";
    for (int i = 0; i < D.size(); ++i)
    {
        cout << D(i) << " ";
    }
    cout << ")" << endl;
}

vector<tuple<string, double>> FEM_2D::full_solve()
{
    auto t1 = chrono::high_resolution_clock::now();

    auto [K11, K22, K33, K12, K13, K23, D1] = gen_local_matrices();
    auto t2 = chrono::high_resolution_clock::now();
    auto t_gen_local_K_D = chrono::duration<double, std::milli>(t2 - t1).count();

    assemble_matrix(K11, K22, K33, K12, K13, K23, D1);
    auto t3 = chrono::high_resolution_clock::now();
    auto t_assemble_matrix = chrono::duration<double, std::milli>(t3 - t2).count();

    solve_LGS();
    auto t4 = chrono::high_resolution_clock::now();
    auto t_solve_LGS = chrono::duration<double, std::milli>(t4 - t3).count();

    reconstruct_solution();
    auto t5 = chrono::high_resolution_clock::now();
    auto t_reconstruct_solution = chrono::duration<double, std::milli>(t5 - t4).count();

    auto t_total = t_gen_local_K_D + t_assemble_matrix + t_solve_LGS + t_reconstruct_solution;

    vector<tuple<string, double>> timings = {
        {"gen_local_K_D", t_gen_local_K_D},
        {"assemble_matrix", t_assemble_matrix},
        {"solve_LGS", t_solve_LGS},
        {"reconstruct_solution", t_reconstruct_solution},
        {"total_time", t_total}};

    return timings;
}

Vector FEM_2D::get_Solution()
{
    // Return Sol for pybind11
    return Sol;
}

Vector FEM_2D::get_D()
{
    // Return D for pybind11
    return D;
}

tuple<Vector, vector<double>> FEM_2D::validate_sol(Vector sol_tst, double max_error)
{
    if (Sol.size() != sol_tst.size())
    {
        string error_msg = "Validation failed: Solution size (" + to_string(Sol.size()) + ") does not match test solution size (" + to_string(sol_tst.size()) + ").";
        throw runtime_error(error_msg);
    }

    Vector error = Sol - sol_tst;
    error = error.cwiseAbs(); // take absolute value of errors
    double max_abs_error = error.maxCoeff();
    double min_abs_error = error.minCoeff();
    double mean_abs_error = error.mean();

    // string suc = format("=> [PASS] Max error is within {:.2e} of actual solution", max_error);
    // string fail = format("=> [FAIL] Max error exceeds threshold of {:.2e} of actual solution", max_error);

    vector<double> error_stats = {max_abs_error, min_abs_error, mean_abs_error};

    return {error, error_stats};
}