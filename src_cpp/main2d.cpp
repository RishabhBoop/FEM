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
    { return 2 + pow(x, 2) + pow(y, 2); };
    auto q = [](double x, double y)
    { return x - y; };

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

    // Vector xD(4); // Dirichlet boundary nodes
    VectorINT dr(0);
    // dr << 0, 4, 3, 5; // Dirichlet boundary node indices
    printf("Dirichlet boundary nodes dR = [");
    for (int i = 0; i < dr.size(); ++i)    {
        printf("%d ", dr(i));
    }
    printf("]\n");

    
    MatrixINT rr(4, 2); // Robin boundary edges (4 edges, each defined by 2 node indices)
    rr << 3, 5,
    0, 4,
    4, 3,
    5, 0;
    
    printf("Robin boundary edges rr = [");
    for (int i = 0; i < 4; ++i)    {
        printf("(%d, %d) ", rr(i, 0), rr(i, 1));
    }    
    printf("]\n");
    

    FEM_2D TST(
        dr,
        rr,
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
    TST.print_D();
    TST.print_solution();

    return 0;
}