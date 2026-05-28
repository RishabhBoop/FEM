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
from helper_funcs.visualizations import visualize_solution, print_timings
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries

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


def gen_mesh():
    gmsh.initialize()
    # Generate gmsh model here
    name = "fem2d_skript"
    gmsh.model.add(name)

    # net size
    lc = 0.1

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


def main():
    gen_mesh()
    print(colors.SUCCESS + "Mesh generated successfully." + colors.RESET)
    
    t0 = time()
    plist, tlist = get_plist_tlist_from_gmsh("fem2d_skript.msh")
    t1 = time()
    print(colors.INFO + "Number of points (plist):", len(plist), colors.RESET)
    print(colors.INFO + "Number of Triangles (tlist):", len(tlist), colors.RESET)
    plist_time = t1 - t0
    
    t0 = time()
    dr = get_boundaries("fem2d_skript.msh")
    t1 = time()
    boundary_time = t1 - t0
    print(colors.SUCCESS + "Boundary nodes extracted successfully." + colors.RESET)

    xD = plist[dr, 0]
    xR = []

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

    print_timings(timings, "FEM 2D Skript Timings", len(plist), len(tlist), False, "CPP")
    visualize_solution(plist, tlist, sol, True, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png")
    visualize_solution(plist, tlist, sol, False, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png")

    plt.show()


def mesh():
    gen_mesh()


if __name__ == "__main__":
    main()
    # mesh()
