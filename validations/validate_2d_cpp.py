from time import time

import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64
import gmsh

import fem_cpp

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges

from helper_funcs.gmshtools import ElementMsh, MshHs

import os.path as path

current_dir = path.dirname(path.abspath(__file__))
# current_dir = current_dir + "/../validations"
tst_data_dir = f"{current_dir}/tst_2D"
# ------------------------------------------------------------------------------

ERROR_TOLERANCE = 1e-11

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
    dr = np.loadtxt(f"{tst_data_dir}/Netz2D_dr_a).dat", dtype=int).flatten()
    rr = np.array([], dtype=int).reshape(0, 2)

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

    print("dr =", dr)
    print("rr =", rr)

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


def gen_mesh():
    gmsh.initialize()
    name = "FEM_2D_testing"
    gmsh.model.add(name)

    # =======================
    # create geometry
    #   Add lines, points, surfaces, and holes to define the geometry.
    # =======================
    P1 = (1, 1, 0)
    P1_5 = (2, 1, 0)
    P2 = (3, 1, 0)
    P3 = (1, 4, 0)
    P3_5 = (2, 4, 0)
    P4 = (3, 4, 0)

    i1 = gmsh.model.occ.addPoint(*P1)
    i1_5 = gmsh.model.occ.addPoint(*P1_5)
    i2 = gmsh.model.occ.addPoint(*P2)
    i3 = gmsh.model.occ.addPoint(*P3)
    i3_5 = gmsh.model.occ.addPoint(*P3_5)
    i4 = gmsh.model.occ.addPoint(*P4)

    L1 = gmsh.model.occ.addLine(i1, i1_5)
    L2 = gmsh.model.occ.addLine(i1_5, i2)
    L3 = gmsh.model.occ.addLine(i2, i4)
    L4 = gmsh.model.occ.addLine(i4, i3_5)
    L5 = gmsh.model.occ.addLine(i3_5, i3)
    L6 = gmsh.model.occ.addLine(i3, i1)

    loop1 = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4, L5, L6])

    # Kreisloch
    C2 = (2.5, 2.5, 0)
    r2 = 0.3
    circle2 = gmsh.model.occ.addCircle(*C2, r2)
    loop2 = gmsh.model.occ.addCurveLoop([circle2])

    # Fläche
    surface = gmsh.model.occ.addPlaneSurface([loop1, loop2])  # loop2 wird von loop1 subtrahiert

    # Embed the inner curve (circle)
    C1 = (1.5, 1.75, 0)
    r1 = 0.35
    circle1 = gmsh.model.occ.addCircle(*C1, r1)

    # embed rechteck
    P5 = (1.25, 3, 0)
    P6 = (1.75, 3, 0)
    P7 = (1.25, 3.5, 0)
    P8 = (1.75, 3.5, 0)
    i5 = gmsh.model.occ.addPoint(*P5)
    i6 = gmsh.model.occ.addPoint(*P6)
    i7 = gmsh.model.occ.addPoint(*P7)
    i8 = gmsh.model.occ.addPoint(*P8)

    L5 = gmsh.model.occ.addLine(i5, i6)
    L6 = gmsh.model.occ.addLine(i6, i8)
    L7 = gmsh.model.occ.addLine(i8, i7)
    L8 = gmsh.model.occ.addLine(i7, i5)

    loop3 = gmsh.model.occ.addCurveLoop([L5, L6, L7, L8])

    # embed line

    L9 = gmsh.model.occ.addLine(i1_5, i3_5)

    # =======================
    # Synchronize
    # =======================
    gmsh.model.occ.synchronize()

    # =======================
    # Embed curves into surface
    # =======================
    gmsh.model.mesh.embed(1, [circle1], 2, surface)  # Embed M1
    gmsh.model.mesh.embed(1, [L5, L6, L7, L8], 2, surface)  # Embed Rechteck
    gmsh.model.mesh.embed(1, [L9], 2, surface)  # embed Line

    # =======================
    # Synchronize
    # =======================
    gmsh.model.occ.synchronize()

    # =======================
    # Add physical groups
    # =======================
    pyhsical_surface = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, pyhsical_surface, "MainSurface")

    l0 = gmsh.model.addPhysicalGroup(1, [L1, L4])  # linke und untere Rand vom Rechteck
    gmsh.model.setPhysicalName(1, l0, "DirichletBoundary")

    l1 = gmsh.model.addPhysicalGroup(1, [L2, L3])  # rechte und obere Rand vom Rechteck
    gmsh.model.setPhysicalName(1, l1, "RobinBoundary")

    # =======================
    # Synchronize
    # =======================
    gmsh.model.occ.synchronize()

    dist_tag = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        dist_tag, "EdgesList", [L9]
    )  # zu diesen Linien wird der (minimale) Abstand berechnet
    gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", 100)  # auf den Linien werden dazu 100 pkt verwendet

    # ========================
    # Generate the mesh and save it
    # ========================
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    mesh = gmsh.model.mesh.generate(2)
    gmsh.write(f"{name}.msh")

    try:
        gmsh.fltk.run()
    except:
        print("No FLTK GUI available, skipping visualization.")

    netz = MshHs(gmsh.model)
    gmsh.finalize()
    return netz


def a1_c():
    @cfunc(float64(float64, float64))
    def phi(x, y):
        return x**2 - y**2 + 1

    @cfunc(sig=float64(float64, float64))
    def gamma(x, y):
        return 3 * x * y

    @cfunc(sig=float64(float64, float64))
    def q(x, y):
        return 20.0

    # ----------------------- #

    netz = gen_mesh()
    print("Mesh generated, starting FEM solve...")

    netz.Triangle.plot(color="grey", alpha=0.3)
    netz.DirichletBoundary.plot(color="red")
    netz.RobinBoundary.plot(color="orange")

    plist = netz.points.astype(np.float64)
    tlist = netz.Triangle.elements.astype(np.int32)

    # Dirichlet boundaries (Nodes) -> must shape to 2D column grid [m, 1]
    dr = netz.DirichletBoundary.nodes.astype(np.int32).flatten()

    # Robin boundaries (Line Elements) -> 2D grid matrix [m, 2]
    rr = netz.RobinBoundary.elements.astype(np.int32)

    fem_solver = fem_cpp.FEM_2D(dr, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        visualize_solution(
            plist, tlist, sol, False, "Lösung 2D Validierung", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png"
        )
        print_timings(timings, "FEM 2D Validierung Timings", len(plist), len(tlist), False, "CPP")
    except RuntimeError as e:
        print(f"Fehler: {e}")
        return


if __name__ == "__main__":
    # a1_a()
    # a1_b()
    # a1_c()
    plt.show()
