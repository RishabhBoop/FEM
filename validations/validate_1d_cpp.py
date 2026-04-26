import sys
import os

# Get the absolute path to the project root's 'bin' directory and prepend it to the path
bin_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin"))
sys.path.insert(0, bin_dir)

import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64
from os import path

import fem_cpp
from src.colors import Colors as colors

current_dir = path.dirname(path.abspath(__file__))
current_dir = current_dir + "/../validations"

# ------------------------------------------------------------------------------
ERROR_TOLERANCE = 1e-11
# ------------------------------------------------------------------------------


def print_timings(timings, title="TIMING - C++ full_solve()", n_points=None, n_elems=None):
    """
    Prints the performance output dynamically, preserving the order from the C++ vector of tuples.
    Expects `timings` to be a list of tuples: e.g., [("gen_tlist", 12.3), ("solve_LGS", 45.6)]
    """
    if n_points is not None and n_elems is not None:
        title_str = f"{title} ({n_points} Punkte, {n_elems} Elemente)"
    else:
        title_str = title

    lines = []
    tot_str = ""

    # Find the longest step name to dynamically align the milliseconds perfectly
    # timings is now a list of lists/tuples, where item[0] is the name
    max_key_len = max([len(str(item[0])) for item in timings] + [10])

    # Iterating through the list (step_name = item[0], t_val = item[1])
    for step_name, t_val in timings:
        formatted_line = f"{step_name}:".ljust(max_key_len + 2) + f"{t_val:.3f} ms"

        # Isolate 'total_time' so we can print it at the very bottom below the separator
        if step_name == "total_time":
            tot_str = formatted_line
        else:
            lines.append(formatted_line)

    if not tot_str:
        tot_str = "Total Time:".ljust(max_key_len + 2) + "N/A"

    # Calculate required box width based on the longest string
    len_sep = max(max((len(l) for l in lines), default=0), len(tot_str), len(title_str)) + 4

    print()
    print("-" * len_sep)
    print(f"{title_str:^{len_sep}}")
    print("-" * len_sep)
    for line in lines:
        print(f" {line}")
    print("-" * len_sep)
    print(f" {tot_str}")
    print("=" * len_sep)


def print_error_stats(error_stats, title="Validierung", n_elems=None):
    """
    Prints the maximum, minimum, and mean absolute error from the C++ validation.
    """
    # Unpack the list sent from C++ (max_abs_error, min_abs_error, mean_abs_error)
    max_err, min_err, mean_err = error_stats

    if n_elems is not None:
        title_str = f"Abweichungen für {title} ({n_elems} Elemente):"
    else:
        title_str = f"Abweichungen für {title}:"

    max_str = f"Maximale Abweichung: {max_err:.6e}"
    min_str = f"Minimale Abweichung: {min_err:.6e}"
    mean_str = f"Mittlere Abweichung: {mean_err:.6e}"

    if max_err > ERROR_TOLERANCE:
        title_str = colors.RED + title_str + " [FAIL]" + colors.RESET
    else:
        title_str = colors.GREEN + title_str + " [PASS]" + colors.RESET

    len_sep = max(len(max_str), len(min_str), len(mean_str), len(title_str)) + 4

    print("\n" + "=" * len_sep)
    print(f"  {title_str}")
    print("-" * len_sep)
    print(f"  {max_str}")
    print(f"  {min_str}")
    print(f"  {mean_str}")
    print("=" * len_sep)


# -----------------------------------------------------------------------------

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

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungA.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    error = None
    try:
        timings = fem_solver.full_solve("Lösung A")
        print_timings(timings, "TIMING - Lösung A", len(plist), len(plist) - 1)
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung A: {e}")
        return

    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, "Lösung A", ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung A")
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

    sol_tst = np.loadtxt(f"{current_dir}/tst_1D/Netz1D_LoesungB.dat", dtype=float)
    fem_solver = fem_cpp.FEM_1D(xD, xR, plist, alpha, beta, f, phi, gamma, q)
    error = None
    try:
        timings = fem_solver.full_solve("Lösung B")
        print_timings(timings, "TIMING - Lösung B", len(plist), len(plist) - 1)
    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung B: {e}")
        return
    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, "Lösung B", ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung B")
    except RuntimeError as e:
        print(f"Validierung von Lösung B fehlgeschlagen: {e}")

    # if error is not None:
    #     plt.figure()
    #     plt.plot(plist, error, label="Fehler")
    #     plt.xlabel("x")
    #     plt.ylabel("Fehler")
    #     plt.title("Fehlerverteilung für Lösung B")
    #     plt.legend()
    #     plt.grid()


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
    error = None

    try:
        timings = fem_solver.full_solve("Lösung C")
        print_timings(timings, "TIMING - Lösung C", len(plist), len(plist) - 1)

    except Exception as e:
        print(f"Fehler bei der Berechnung von Lösung C: {e}")
        return
    try:
        error, error_stats = fem_solver.validate_sol(sol_tst, "Lösung C", ERROR_TOLERANCE)
        print_error_stats(error_stats, "Lösung C")
    except RuntimeError as e:
        print(f"Validierung von Lösung C fehlgeschlagen: {e}")

    # if error is not None:
    #     plt.figure()
    #     plt.plot(plist, error, label="Fehler")
    #     plt.xlabel("x")
    #     plt.ylabel("Fehler")
    #     plt.title("Fehlerverteilung für Lösung C")
    #     plt.legend()
    #     plt.grid()


def main():
    a1_a()
    plt.show()

    a1_b()
    plt.show()

    a1_c()
    plt.show()


if __name__ == "__main__":
    main()
