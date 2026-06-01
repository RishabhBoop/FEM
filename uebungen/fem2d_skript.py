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
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges

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
    return 2 + x**2 + y**2


@cfunc(float64(float64, float64))
def g(x, y):
    return x - y


def gen_mesh():
    gmsh.initialize()
    # Generate gmsh model here
    name = "fem2d_skript"
    gmsh.model.add(name)

    # net size
    lc = 1

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

    # add robin boundary
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 20)
    gmsh.model.setPhysicalName(1, 20, "Robin")

    # Save all elements regardless of physical groups
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # sync
    gmsh.model.mesh.generate(2)

    gmsh.write(f"{name}.msh")

    # gmsh.fltk.run()  # Optional: to visualize the mesh
    gmsh.finalize()


def show_mesh(filename = "fem2d_skript.msh"):
    gmsh.initialize()
    gmsh.open(filename)
    gmsh.fltk.run()
    gmsh.finalize()

def main():
    gen_mesh()
    print(colors.SUCCESS + "Mesh generated successfully." + colors.RESET)
    
    t0 = time()
    # plist, tlist = get_plist_tlist_from_gmsh("fem2d_skript.msh")
    plist = np.array([[1, 0.7], [0.5, 0.35], [0.5, 0.18], [0, 0], [0, 0.7], [1, 0]])
    tlist = np.array(
        [[0, 4, 1], [3, 1, 4], [5, 0, 1], [3, 5, 2], [2, 1, 3], [5, 1, 2]]
    )  # tlist will be outputted from gmsh, for now we just hardcode it

    t1 = time()

    print(colors.INFO + "Number of points (plist):", len(plist), colors.RESET)
    print(colors.INFO + "Number of Triangles (tlist):", len(tlist), colors.RESET)
    plist_time = t1 - t0
    
    t0 = time()
    # dr = get_boundary_edges("fem2d_skript.msh", group_tag=20)
    dr = []
    # rr_edges = get_boundary_edges("fem2d_skript.msh", group_tag=20)
    rr_edges = [[3, 5], [0, 4], [4, 3], [5, 0]]
    # rr_edges = []
    t1 = time()
    print(colors.INFO + "Dirichlet boundary nodes (dr):", dr, colors.RESET)
    print(colors.INFO + "Robin boundary edges (rr):", rr_edges, colors.RESET)
    boundary_time = t1 - t0
    print(colors.SUCCESS + "Boundary nodes extracted successfully." + colors.RESET)

    # dirichlet_nodes = np.unique(dr.flatten() if len(dr) else np.array([], dtype=int))
    # xD = plist[dirichlet_nodes, 0]
    # robin_nodes = np.unique(rr_edges.flatten()) if len(rr_edges) else np.array([], dtype=int)
    # xR = plist[robin_nodes, 0]
    print("dr =", dr)
    print("rr =", rr_edges)


    fem_solver = fem_cpp.FEM_2D(dr, rr_edges, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, g)
    
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        print(colors.SUCCESS + "FEM solve completed successfully." + colors.RESET)
        print(colors.INFO + "Solution vector (sol):", sol, colors.RESET)
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
