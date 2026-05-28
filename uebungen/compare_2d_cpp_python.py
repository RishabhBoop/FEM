import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64
import gmsh

# -- CPP -- #
import sys
from os import path

bin_dir = path.abspath(path.join(path.dirname(__file__), "../bin"))
sys.path.insert(0, bin_dir)
import fem_cpp
# --------- #

# -- PY -- #
import src.FEM_2D as py_fem
# -------- #



from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries
from time import time


@cfunc(float64(float64, float64))
def alpha1(x, y):
    return y * x + 1


@cfunc(float64(float64, float64))
def alpha2(x, y):
    return x + y + 1


@cfunc(float64(float64, float64))
def beta(x, y):
    return 2 * x**2


@cfunc(float64(float64, float64))
def f(x, y):
    return x + y**2


@cfunc(float64(float64, float64))
def phi(x, y):
    return x**2 + y


@cfunc(float64(float64, float64))
def gamma(x, y):
    return 0.0


@cfunc(float64(float64, float64))
def g(x, y):
    return 0.0


def gen_mesh(lc=0.1):
    gmsh.initialize()
    # Generate gmsh model here
    name = "fem2d_skript"
    gmsh.model.add(name)

    # net size lc, default 0.1

    # End rectangles
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 0.7, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 0.7, 0, lc, 4)

    # connect to rectangle
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    # add closed loop
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    # create surface
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    gmsh.model.geo.addPoint(0.5, 0.35, 0, lc, 5)
    gmsh.model.geo.addPoint(0.5, 0.18, 0, lc, 6)
    gmsh.model.geo.synchronize()

    ## embed(dimension_of_embedded_entity, [tags_of_entities], dimension_of_target, target_tag)
    # Embed points (dim 0) with tags [5, 6] into the surface (dim 2) with tag 1.
    gmsh.model.mesh.embed(0, [5, 6], 2, 1)

    # add boundary
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 99)

    # Save all elements regardless of physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # sync
    gmsh.model.mesh.generate(2)

    gmsh.write(f"{name}.msh")

    # gmsh.fltk.run()  # Optional: to visualize the mesh
    gmsh.finalize()

# ---------------------------------------------------------------------- #

def prerequisits():
    t0 = time()
    plist, tlist = get_plist_tlist_from_gmsh("fem2d_skript.msh")
    t1 = time()
    print(colors.INFO + "Number of points (plist):", len(plist), colors.RESET)
    print(colors.INFO + "Number of Triangles (tlist):", len(tlist), colors.RESET)
    plist_time = (t1 - t0) * 1000.0

    t0 = time()
    dr = get_boundaries("fem2d_skript.msh")
    t1 = time()
    boundary_time = (t1 - t0) * 1000.0
    print(colors.SUCCESS + "Boundary nodes extracted successfully." + colors.RESET)

    xD = plist[dr, 0]
    xR = []

    return plist, tlist, dr, xD, xR, plist_time, boundary_time

def test_cpp(plist, tlist, xD, xR, plist_time, boundary_time):
    print(colors.INFO + "Running C++ FEM code..." + colors.RESET)

    fem_solver = fem_cpp.FEM_2D(xD, xR, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, g)
    
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        print(colors.SUCCESS + "FEM solve completed successfully." + colors.RESET)
    except Exception as e:
        print(colors.FAIL + "Error during FEM solve: " + str(e) + colors.RESET)
        return
    
    timings.insert(0, ("gen_plist_tlist", plist_time))
    timings.insert(1, ("get_boundaries", boundary_time))

    return timings, sol

def test_py(plist, tlist, xD, xR, plist_time, boundary_time):
    print(colors.INFO + "Running Python FEM code..." + colors.RESET)

    fem_solver_py = py_fem.FEM_2D(xD, xR, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, g)
    
    try:
        timings_py = fem_solver_py.full_solve()
        sol_py = fem_solver_py.get_Solution()
        print(colors.SUCCESS + "Python FEM solve completed successfully." + colors.RESET)
    except Exception as e:
        print(colors.FAIL + "Error during Python FEM solve: " + str(e) + colors.RESET)
        return
    
    timings_py.insert(0, ("gen_plist_tlist", plist_time))
    timings_py.insert(1, ("get_boundaries", boundary_time))

    return timings_py, sol_py

def warm_up():
    print(colors.INFO + "Warming up Python Backend (Numba JIT + SciPy)..." + colors.RESET)
    gen_mesh(1.0)
    warmup_plist, warmup_tlist, _, warmup_xD, warmup_xR, p_time, b_time = prerequisits()
    _ = test_py(warmup_plist, warmup_tlist, warmup_xD, warmup_xR, p_time, b_time)
    print(colors.SUCCESS + "Warm-up complete!" + colors.RESET)

def main():
    lc = 0.1

    warm_up()  # Warm up the Python backend to ensure JIT compilation is done before timing

    gen_mesh(lc)
    print(colors.SUCCESS + f"Mesh generated successfully with lc={lc}." + colors.RESET)
    
    plist, tlist, _, xD, xR, plist_time, boundary_time = prerequisits()
    timings_cpp, sol_cpp = test_cpp(plist, tlist, xD, xR, plist_time, boundary_time)
    timings_py, sol_py = test_py(plist, tlist, xD, xR, plist_time, boundary_time)

    print_timings(timings_cpp, "FEM 2D CPP Timings", len(plist), len(tlist), False, "CPP", "export_cpp.txt", True)
    print_timings(timings_py, "FEM 2D PY Timings", len(plist), len(tlist), False, "PYTHON", "export_py.txt", True)

    visualize_solution(plist, tlist, sol_cpp, True, f"Lösung 2D CPP (lc={lc})", False, "CPP", f"Loesung2D_{len(plist)}points_sol_cpp.png")
    visualize_solution(plist, tlist, sol_cpp, False, f"Lösung 2D CPP (lc={lc})", False, "CPP", f"Loesung2D_{len(plist)}points_sol_cpp.png")
    visualize_solution(plist, tlist, sol_py, False, f"Lösung 2D PYTHON (lc={lc})", False, "PYTHON", f"Loesung2D_{len(plist)}points_sol_py.png")
    visualize_solution(plist, tlist, sol_py, True, f"Lösung 2D PYTHON (lc={lc})", False, "PYTHON", f"Loesung2D_{len(plist)}points_sol_py.png")
    
    plt.show()

if __name__ == "__main__":
    main()