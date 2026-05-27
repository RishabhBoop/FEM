#include "FEM_2D.hpp"

using namespace std;

int main()
{
    auto alpha1 = [](double x, double y)
    { return x * y + 1; };
    auto alpha2 = [](double x, double y)
    { return x + y + 1; };
    auto beta = [](double x, double y)
    { return 2 * pow(x, 2); };
    auto f = [](double x, double y)
    { return x + pow(y, 2); };
    auto phi = [](double x, double y)
    { return pow(x, 2) + y; };
    auto gamma = [](double x, double y)
    { return 0.0; };
    auto q = [](double x, double y)
    { return 0.0; };

    // plist = np.array([[1, 0.7], [0.5, 0.35], [0.5, 0.18], [0, 0], [0, 0.7], [1, 0]])
    Matrix plist(6, 2);
    plist << 1, 0.7,
        0.5, 0.35,
        0.5, 0.18,
        0, 0,
        0, 0.7,
        1, 0;

    // tlist = np.array([[0, 4, 1], [3, 1, 4], [5, 0, 1], [3, 5, 2], [2, 1, 3], [5, 1, 2]])
    MatrixINT tlist(6, 3);
    tlist << 0, 4, 1,
        3, 1, 4,
        5, 0, 1,
        3, 5, 2,
        2, 1, 3,
        5, 1, 2;

    Vector xD(4); // Dirichlet boundary nodes
    VectorINT dr(4);
    dr << 0, 4, 3, 5; // Dirichlet boundary node indices
    for (int i = 0; i < dr.size(); ++i)
    {
        xD(i) = plist(dr(i), 0); // Extract x-coordinate of each Dirichlet node
    }
    printf("Dirichlet boundary nodes xD = [");
    for (int i = 0; i < xD.size(); ++i)    {
        printf("%f ", xD(i));
    }
    printf("]\n");

    Vector xR(0); // No Robin boundary nodes
    // xR << 1.0, 2.0; // Robin boundary nodes

    FEM_2D TST(
        xD,
        xR,
        plist,
        tlist,
        alpha1,
        alpha2,
        beta,
        f,
        phi,
        gamma,
        q);

    TST.full_solve();
    TST.print_solution();

    return 0;
}