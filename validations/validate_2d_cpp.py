from time import time

import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64
import gmsh

import sys
from os import path

bin_dir = path.abspath(path.join(path.dirname(__file__), "../bin"))
sys.path.insert(0, bin_dir)
import fem_cpp

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges

current_dir = path.dirname(path.abspath(__file__))
# current_dir = current_dir + "/../validations"
tst_data_dir = f"{current_dir}/tst_2D"
# ------------------------------------------------------------------------------

ERROR_TOLERANCE = 1e-12

# ------------------------------------------------------------------------------


@cfunc(sig=float64(float64, float64))
def alpha1(x, y):
    if 1.25 <= x <= 1.75 and 3 <= y <= 3.5:
        return 0.01
    else:
        return 5 * y * (x**2)


@cfunc(sig=float64(float64, float64))
def alpha2(x, y):
    return y**2


@cfunc(sig=float64(float64, float64))
def beta(x, y):
    cond = (x - 1.5) ** 2 + (y - 1.75) ** 2 <= 0.35**2
    if cond:
        return 500
    else:
        return 5


@cfunc(sig=float64(float64, float64))
def f(x, y):
    if x >= 2:
        return -10 * x * y
    else:
        return 0.0


def a1_a():
    @cfunc(float64(float64, float64))
    def phi(x, y):
        return x**2 - y**2 + 1

    @cfunc(sig=float64(float64, float64))
    def gamma(x, y):
        return 0.0

    @cfunc(sig=float64(float64, float64))
    def q(x, y):
        return 0.0

    plist = np.loadtxt(f"{tst_data_dir}/Netz2D_p.dat", dtype=float)
    tlist = np.loadtxt(f"{tst_data_dir}/Netz2D_t.dat", dtype=int)
    dr = np.loadtxt(f"{tst_data_dir}/Netz2D_dr_a).dat", dtype=int)
    rr = []

    fem_solver = fem_cpp.FEM_2D(dr, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)

    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(
            plist, tlist, sol, False, "Lösung 2D Validierung", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png"
        )
        print_timings(timings, "FEM 2D Validierung Timings", len(plist), len(tlist), False, "CPP")
    except RuntimeError as e:
        print(f"Validierung von Lösung A fehlgeschlagen: {e}")
        return

    sol_tst = np.loadtxt(f"{tst_data_dir}/Netz2D_LoesungA.dat", dtype=float)
    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(
            error_stats, "Lösung A", len(plist) - 1, False, "CPP", f"LoesungA_{len(plist)}points_stats.txt"
        )
        visualize_error(
            plist, error, "Fehlerverteilung für Lösung A", False, "CPP", f"LoesungA_{len(plist)}points_error.png"
        )
    except RuntimeError as e:
        print(f"Validierung von Lösung A fehlgeschlagen: {e}")
        return


def a1_b():
    @cfunc(float64(float64, float64))
    def phi(x, y):
        return x**2 - y**2 + 1

    @cfunc(sig=float64(float64, float64))
    def gamma(x, y):
        return 3 * x * y

    @cfunc(sig=float64(float64, float64))
    def q(x, y):
        return 20.0

    plist = np.loadtxt(f"{tst_data_dir}/Netz2D_p.dat", dtype=float)
    tlist = np.loadtxt(f"{tst_data_dir}/Netz2D_t.dat", dtype=int)
    dr = np.loadtxt(f"{tst_data_dir}/Netz2D_dr_b).dat", dtype=int)
    rr = np.loadtxt(f"{tst_data_dir}/Netz2D_rr_b).dat", dtype=int)

    fem_solver = fem_cpp.FEM_2D(dr, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)

    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(
            plist, tlist, sol, False, "Lösung 2D Validierung", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png"
        )
        print_timings(timings, "FEM 2D Validierung Timings", len(plist), len(tlist), False, "CPP")
    except RuntimeError as e:
        print(f"Validierung von Lösung A fehlgeschlagen: {e}")
        return

    sol_tst = np.loadtxt(f"{tst_data_dir}/Netz2D_LoesungB.dat", dtype=float)
    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, ERROR_TOLERANCE)
        print_error_stats(
            error_stats, "Lösung B", len(plist) - 1, False, "CPP", f"LoesungB_{len(plist)}points_stats.txt"
        )
        visualize_error(
            plist, error, "Fehlerverteilung für Lösung B", False, "CPP", f"LoesungB_{len(plist)}points_error.png"
        )
    except RuntimeError as e:
        print(f"Validierung von Lösung B fehlgeschlagen: {e}")
        return


if __name__ == "__main__":
    a1_a()
    a1_b()
    plt.show()
