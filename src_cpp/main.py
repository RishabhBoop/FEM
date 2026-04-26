import sys
import os

# Get the absolute path to the project root's 'bin' directory and prepend it to the path
bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin'))
sys.path.insert(0, bin_dir)

import numpy as np
import matplotlib.pyplot as plt
import fem_cpp
from numba import cfunc, float64


from os import path

current_dir = path.dirname(path.abspath(__file__))

plist = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_p.dat", dtype=float)

@cfunc(float64(float64))
def alpha(x):
    if 1.5 <= x <= 2.7:
        return 3
    else:
        return x**2


@cfunc(float64(float64))

def beta(x):
    if 1 <= x <= 2:
        return x / (1 + x)
    else:
        return x**2

@cfunc(float64(float64))

def f(x):
    if 2 <= x <= 4:
        return x
    else:
        return 1 + x


# Test with Weizi Data - A1/a
def a1_a():
    xD = [1.0, 4.0]  # x-koordinaten der dirichlet boundary conditions
    xR = []  # x-koordinaten der robin boundary conditions

    @cfunc(float64(float64))
    def phi(x):
        if 1 <= x <= 4:
            return np.exp(x)
        else:
            return 0.0

    @cfunc(float64(float64))
    def gamma(x):
        return 0.0

    @cfunc(float64(float64))
    def q(x):
        return 0.0

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungA.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.full_solve(title="Lösung A")
    # fem_solver.validate_sol(sol_tst, title="Lösung A")


def a1_b():
    @cfunc(float64(float64))
    def phi(x):
        return 0.0

    @cfunc(float64(float64))
    def gamma(x):
        return x

    @cfunc(float64(float64))
    def q(x):
        return x**3

    xD = []  # x-koordinaten der dirichlet boundary conditions
    xR = [1.0, 4.0]  # x-koordinaten der robin boundary conditions

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungB.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.full_solve(title="Lösung B")
    try:
        fem_solver.validate_sol(sol_tst, title="Lösung B")
    except ValueError as e:
        print(f"Validierung von Lösung B fehlgeschlagen: {e}")


def a1_c():
    @cfunc(float64(float64))
    def phi(x):
        return 2.0

    @cfunc(float64(float64))
    def gamma(x):
        return 0.0

    @cfunc(float64(float64))
    def q(x):
        return -3.0

    xD = [4.0]  # x-koordinaten der dirichlet boundary conditions
    xR = [1.0]  # x-koordinaten der robin boundary conditions

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungC.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.full_solve(title="Lösung C")
    fem_solver.validate_sol(sol_tst, title="Lösung C")


def main():
    a1_a()
    plt.show()

    a1_b()
    plt.show()

    a1_c()
    plt.show()


if __name__ == "__main__":
    main()
