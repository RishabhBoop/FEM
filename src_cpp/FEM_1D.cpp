#include "FEM_1D.hpp"

using namespace std;

FEM_1D::FEM_1D(
    Vector xD,
    Vector xR,
    Vector plist,
    function<double(double)> alpha,
    function<double(double)> beta,
    function<double(double)> f,
    function<double(double)> phi,
    function<double(double)> gamma,
    function<double(double)> q) : xD(xD),
                                  xR(xR),
                                  plist(plist),
                                  alpha(alpha),
                                  beta(beta),
                                  f(f),
                                  phi(phi),
                                  gamma(gamma),
                                  q(q)
{
    this->Randelemente = gen_Randelemente(xR, plist);
}

vector<int> gen_Randelemente(Vector xR, Vector plist)
{
    vector<int> randelemente;
    for (int j = 0; j < xR.size(); ++j)
    {
        for (int i = 0; i < plist.size(); ++i)
        {
            if (abs(plist[i] - xR[j]) < 1e-10)
            {
                randelemente.push_back(i); // store node index
                break;
            }
        }
    }
    return randelemente;
}

void FEM_1D::gen_tlist()
{
    int len_plist = plist.size();
    vector<double> tmp_tlist(len_plist);         // create temporary vector to store the tlist values
    iota(tmp_tlist.begin(), tmp_tlist.end(), 0); // fill the temporary vector with values from 0 to len_plist-1

    // sort the temporary vector based on the values in plist
    sort(tmp_tlist.begin(), tmp_tlist.end(), [this](int i, int j)
         { return plist(i) < plist(j); });

    tlist.resize(len_plist - 1, 2); // resize to the correct dimension

    // insert the sorted values into tlist
    for (int i = 0; i < len_plist - 1; ++i)
    {
        tlist(i, 0) = tmp_tlist[i];     // first column gets the sorted indices
        tlist(i, 1) = tmp_tlist[i + 1]; // second column gets the next sorted index
    }
}

tuple<vector<double>, vector<double>, vector<double>> FEM_1D::gen_K11_K12_D1()
{
    vector<double> K11(tlist.rows());
    vector<double> K12(tlist.rows());
    vector<double> D1(tlist.rows());

    for (int i = 0; i < tlist.rows(); ++i)
    {
        int t0 = tlist(i, 0);
        int t1 = tlist(i, 1);
        double x1 = plist(t0);
        double x2 = plist(t1);

        double LE = x2 - x1;
        double xM = (x1 + x2) / 2;

        double alpha_M = alpha(xM);
        double beta_M = beta(xM);
        double f_M = f(xM);

        K11[i] = (alpha_M / LE) + (beta_M * LE / 3);
        K12[i] = (LE * beta_M / 6) - (alpha_M / LE);
        D1[i] = f_M * LE / 2;
    }

    return {K11, K12, D1};
}

void FEM_1D::assemble_matrix(vector<double> K11, vector<double> K12, vector<double> D1)
{
    vector<int> node_to_matrix(plist.size(), -1); // list of size plist, initialized to -1 (indicating Randwert nodes); holds
    int free_count = 0;                           // count of free nodes (unknowns); This will be the size of the matrix and D vector after assembly
    for (int i = 0; i < plist.size(); ++i)
    {
        // If not in xD, it's a free node (unknown)
        if (find(xD.begin(), xD.end(), plist[i]) == xD.end())
        {
            node_to_matrix[i] = free_count++; // assign matrix index and increment free count
        }
    }

    D = Vector::Zero(free_count);            // Initialize D vector with correct size and zeroes
    vector<Eigen::Triplet<double>> triplets; // {row, col, value} for sparse matrix assembly

    for (int i = 0; i < tlist.rows(); ++i)
    {
        int t0 = tlist(i, 0);
        int t1 = tlist(i, 1);
        int nodes[2] = {t0, t1};

        // check both nodes of the element
        for (int r = 0; r < 2; ++r)
        {
            int global_row = nodes[r];                   // global node index (entry in plist)
            int matrix_row = node_to_matrix[global_row]; // corresponding matrix index (could be the same if no dirichlet boundaries)
            if (matrix_row != -1)
            {
                // row is not a dirichlet boundary
                D(matrix_row) += D1[i]; // add contribution to D vector; D[t[0]] += D1[i] in python code
                for (int s = 0; s < 2; ++s)
                {
                    int global_col = nodes[s];                   // global node index (entry in plist)
                    int matrix_col = node_to_matrix[global_col]; // corresponding matrix index (could be the same if no dirichlet boundaries)
                    if (matrix_col != -1)
                    {
                        // col is not a dirichlet boundary
                        double value = (r == s) ? K11[i] : K12[i];            // K11 if r==s else K12
                        triplets.emplace_back(matrix_row, matrix_col, value); // add to triplets for sparse matrix assembly
                    }
                    else
                    {
                        // Node is a dirichlet boundary, add contribution to D vector
                        double value = (r == s) ? K11[i] : K12[i];
                        D(matrix_row) -= value * phi(plist(global_col));
                    }
                }
            }
        }
    }
    // apply robin Randwert
    for (int node_idx : Randelemente)
    {
        int m_row = node_to_matrix[node_idx];

        // Check: Is it a free node AND is it actually in the Robin list?
        if (m_row != -1 && find(xR.begin(), xR.end(), plist[node_idx]) != xR.end())
        {
            double x = plist(node_idx);
            D(m_row) += q(x);
            triplets.emplace_back(m_row, m_row, gamma(x));
        }
    }

    K.resize(free_count, free_count);
    K.setFromTriplets(triplets.begin(), triplets.end());
}

void FEM_1D::solve_LGS()
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

void FEM_1D::reconstruct_solution()
{
    Sol.resize(plist.size());

    for (int i = 0; i < plist.size(); ++i)
    {
        if (find(xD.begin(), xD.end(), plist[i]) != xD.end())
        {
            // Dirichlet boundary node
            Sol(i) = phi(plist(i));
        }
        else
        {
            // Free node, get solution from Sol_noRW
            int free_index = 0;
            for (int j = 0; j < i; ++j)
            {
                if (find(xD.begin(), xD.end(), plist[j]) == xD.end())
                {
                    free_index++;
                }
            }
            Sol(i) = Sol_noRW(free_index);
        }
    }
}

void FEM_1D::print_solution()
{
    // Just print the solution vector
    cout << "Solution at nodes:" << endl;
    cout << " phi = (";
    for (int i = 0; i < plist.size(); ++i)
    {
        cout << Sol(i) << " ";
    }
    cout << ")" << endl;
}

vector<tuple<string, double>> FEM_1D::full_solve(string title)
{
    auto t0 = chrono::high_resolution_clock::now();

    gen_tlist();
    auto t1 = chrono::high_resolution_clock::now();
    auto t_gen_tlist = chrono::duration<double, std::milli>(t1 - t0).count();

    auto [K11, K12, D1] = gen_K11_K12_D1();
    auto t2 = chrono::high_resolution_clock::now();
    auto t_gen_K11_K12_D1 = chrono::duration<double, std::milli>(t2 - t1).count();

    assemble_matrix(K11, K12, D1);
    auto t3 = chrono::high_resolution_clock::now();
    auto t_assemble_matrix = chrono::duration<double, std::milli>(t3 - t2).count();

    solve_LGS();
    auto t4 = chrono::high_resolution_clock::now();
    auto t_solve_LGS = chrono::duration<double, std::milli>(t4 - t3).count();

    reconstruct_solution();
    auto t5 = chrono::high_resolution_clock::now();
    auto t_reconstruct_solution = chrono::duration<double, std::milli>(t5 - t4).count();

    auto t_total = chrono::duration<double, std::milli>(t5 - t0).count();

    vector<tuple<string, double>> timings = {
        {"gen_tlist", t_gen_tlist},
        {"gen_K11_K12_D1", t_gen_K11_K12_D1},
        {"assemble_matrix", t_assemble_matrix},
        {"solve_LGS", t_solve_LGS},
        {"reconstruct_solution", t_reconstruct_solution},
        {"total_time", t_total}};

    return timings;
}

Vector FEM_1D::get_Solution()
{
    // Return Sol for pybind11
    return Sol;
}

tuple<Vector, vector<double>> FEM_1D::validate_sol(Vector sol_tst, string title, double max_error)
{
    if (Sol.size() != sol_tst.size())
    {
        throw runtime_error("Validation failed: Solution size does not match test solution size.");
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