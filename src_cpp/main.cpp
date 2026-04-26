#include "FEM_1D.hpp"

using namespace std;

int main()
{
    auto alpha = [](double x) { return pow(x, 2); };
    auto beta  = [](double x) { return x; };
    auto f     = [](double x) { return -pow(x, 3); };
    auto phi   = [](double x) { return (x < 1.5) ? 2.0 : 6.0; };
    auto gamma = [](double x) { return 0.0; };
    auto q     = [](double x) { return 0.0; };

    Vector xD(2);
    xD << 1.0, 2.0;   // Dirichlet boundary nodes at x=1 and x=2
    Vector xR(0); // No Robin boundary nodes
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
        q
    );
    TST.full_solve();
    TST.print_solution();

    return 0;
}