#include "FEM_1D.hpp"

using namespace std;

int main()
{
    auto alpha = [](double x)
    { return pow(x, 2); };
    auto beta = [](double x)
    { return x; };
    auto f = [](double x)
    { return -pow(x, 3); };
    auto phi = [](double x)
    { return (x < 1.5) ? 2.0 : 6.0; };
    auto gamma = [](double x)
    { return (abs(x - 1.0) < 1e-10) ? 3.0 : 6.0; };
    auto q = [](double x)
    { return (abs(x - 1.0) < 1e-10) ? 4.0 : 7.0; };

    Vector xD(0);
    Vector xR(2);   // No Robin boundary nodes
    xR << 1.0, 2.0; // Robin boundary nodes
    Vector plist(5);
    plist << 1.75, 2.0, 1.25, 1.0, 1.5; // Nodes

    FEM_1D TST(
        xD,
        xR,
        plist,
        alpha,
        beta,
        f,
        phi,
        gamma,
        q);
    TST.full_solve();
    TST.print_solution();

    return 0;
}