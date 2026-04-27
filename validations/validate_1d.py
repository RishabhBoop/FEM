import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from numba import float64, vectorize

bin_dir = path.abspath(path.join(path.dirname(__file__), ".."))
sys.path.insert(0, bin_dir)

from src import FEM_1D
from helper_funcs.visualizations import print_timings, print_error_stats, visualize_error, visualize_solution

current_dir = path.dirname(path.abspath(__file__))

ERROR_TOLERANCE = 1e-11

# plist = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_p.dat", dtype=float)
plist = np.linspace(1.0, 4.0, 10000)  # Erstellen eines feineren Plist mit 100 Punkten zwischen 1.0 und 4.0


@vectorize([float64(float64)])
def alpha(x):
    if 1.5 <= x <= 2.7:
        return 3
    else:
        return x**2


@vectorize([float64(float64)])
def beta(x):
    if 1 <= x <= 2:
        return x / (1 + x)
    else:
        return x**2


@vectorize([float64(float64)])
def f(x):
    if 2 <= x <= 4:
        return x
    else:
        return 1 + x


# Test with Weizi Data - A1/a
def a1_a():
    xD = [1.0, 4.0]  # x-koordinaten der dirichlet boundary conditions
    xR = []  # x-koordinaten der robin boundary conditions

    @vectorize([float64(float64)])
    def phi(x):
        if 1 <= x <= 4:
            return np.exp(x)
        else:
            return 0.0

    @vectorize([float64(float64)])
    def gamma(x):
        return 0.0

    @vectorize([float64(float64)])
    def q(x):
        return 0.0

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungA.dat", dtype=float)
    fem_solver = FEM_1D.fem_1d(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    
    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung A", True, "Python", f"LoesungA_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung A", len(plist), len(plist) - 1, True, "Python", f"LoesungA_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung A: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung A", len(plist)-1, True, "Python", f"LoesungA_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung A", True, "Python", f"LoesungA_{len(plist)}points_error.png")
    except RuntimeError as e:
        print(f"Validierung von Lösung A fehlgeschlagen: {e}")
        return


def a1_b():
    @vectorize([float64(float64)])
    def phi(x):
        return 0.0

    @vectorize([float64(float64)])
    def gamma(x):
        return x

    @vectorize([float64(float64)])
    def q(x):
        return x**3

    xD = []  # x-koordinaten der dirichlet boundary conditions
    xR = [1.0, 4.0]  # x-koordinaten der robin boundary conditions

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungB.dat", dtype=float)
    fem_solver = FEM_1D.fem_1d(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    
    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung B", True, "Python", f"LoesungB_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung B", len(plist), len(plist) - 1, True, "Python", f"LoesungB_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung B: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung B", len(plist)-1, True, "Python", f"LoesungB_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung B", True, "Python", f"LoesungB_{len(plist)}points_error.png")
    except RuntimeError as e:
        print(f"Validierung von Lösung B fehlgeschlagen: {e}")
        return


def a1_c():
    @vectorize([float64(float64)])
    def phi(x):
        return 2.0

    @vectorize([float64(float64)])
    def gamma(x):
        return 0.0

    @vectorize([float64(float64)])
    def q(x):
        return -3.0

    xD = [4.0]  # x-koordinaten der dirichlet boundary conditions
    xR = [1.0]  # x-koordinaten der robin boundary conditions

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungC.dat", dtype=float)
    fem_solver = FEM_1D.fem_1d(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    
    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung C", True, "Python", f"LoesungC_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung C", len(plist), len(plist) - 1, True, "Python", f"LoesungC_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung C: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung C", len(plist)-1, True, "Python", f"LoesungC_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung C", True, "Python", f"LoesungC_{len(plist)}points_error.png")
    except RuntimeError as e:
        print(f"Validierung von Lösung C fehlgeschlagen: {e}")
        return


def main():
    a1_a()
    plt.show()

    a1_b()
    plt.show()

    a1_c()
    plt.show()


if __name__ == "__main__":
    main()
