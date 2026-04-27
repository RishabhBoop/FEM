import sys
from os import path

# Get the absolute path to the project root's 'bin' directory and prepend it to the path
bin_dir = path.abspath(path.join(path.dirname(__file__), "../bin"))
sys.path.insert(0, bin_dir)

import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64

import fem_cpp
from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, visualize_error, print_timings, print_error_stats

current_dir = path.dirname(path.abspath(__file__))
current_dir = current_dir + "/../validations"
tst_data_dir = f"{current_dir}/tst_1D"

# ------------------------------------------------------------------------------

ERROR_TOLERANCE = 1e-11

# ------------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# plist = np.loadtxt(f"{tst_data_dir}/Netz1D_p.dat", dtype=float)
plist = np.linspace(1.0, 4.0, 10000)  # Erstellen eines feineren Plist mit 100 Punkten zwischen 1.0 und 4.0


# get smallest distance between points in plist to set as resolution 
RESOLUTION = np.abs(np.min(np.diff(plist)) / 2.0)

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

# -----------------------------------------------------------------------------


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

    sol_tst = np.loadtxt(f"{tst_data_dir}/Netz1D_LoesungA.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.RESOLUTION = RESOLUTION

    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung A", True, "CPP", f"LoesungA_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung A", len(plist), len(plist) - 1, True, "CPP", f"LoesungA_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung A: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung A", len(plist)-1, True, "CPP", f"LoesungA_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung A", True, "CPP", f"LoesungA_{len(plist)}points_error.png")
    except RuntimeError as e:
        print(f"Validierung von Lösung A fehlgeschlagen: {e}")
        return


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

    sol_tst = np.loadtxt(f"{tst_data_dir}/Netz1D_LoesungB.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.RESOLUTION = RESOLUTION

    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung B", True, "CPP", f"LoesungB_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung B", len(plist), len(plist) - 1, True, "CPP", f"LoesungB_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung B: {e}")
        return
    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung B", len(plist)-1, True, "CPP", f"LoesungB_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung B", True, "CPP", f"LoesungB_{len(plist)}points_error.png")
    except RuntimeError as e:
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

    sol_tst = np.loadtxt(f"{tst_data_dir}/Netz1D_LoesungC.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    fem_solver.RESOLUTION = RESOLUTION

    error = None
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(plist, sol, "Lösung C", True, "CPP", f"LoesungC_{len(plist)}points_sol.png")
        print_timings(timings, "TIMING - Lösung C", len(plist), len(plist) - 1, True, "CPP", f"LoesungC_{len(plist)}points")
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung C: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung C", len(plist)-1, True, "CPP", f"LoesungC_{len(plist)}points_stats.txt")
        visualize_error(plist, error, "Fehlerverteilung für Lösung C", True, "CPP", f"LoesungC_{len(plist)}points_error.png")
    except RuntimeError as e:
        print(f"Validierung von Lösung C fehlgeschlagen: {e}")


def main():
    a1_a()
    plt.show()

    a1_b()
    plt.show()

    a1_c()
    plt.show()


if __name__ == "__main__":
    main()
