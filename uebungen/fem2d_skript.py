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



def get_plist_tlist_from_gmsh(meshname):
    gmsh.initialize()

    gmsh.open(meshname)
    nodeTags, coords, parametricCoord = gmsh.model.mesh.getNodes()
    plist = np.array(coords).reshape(-1, 3)[:,:2]
    tp,nm,el = gmsh.model.mesh.getElements(2, -1)
    tlist = np.array(el[0].reshape(-1, 3))-1

    gmsh.finalize()
    return plist, tlist

def get_boundaries(meshname):
    gmsh.initialize()
    gmsh.open(meshname)
    # Get all boundary nodes (those on the edges of the surface)
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, 99)
    # print("Entities in Physical Group 99 (Boundary Edges):", entities)
    # Get nodes from those entities
    boundary_node_tags = set()
    for edge_tag in entities:
        nodes = gmsh.model.mesh.getNodes(1, edge_tag)[0]
        boundary_node_tags.update(nodes)
    
    dr = np.array(sorted(boundary_node_tags)).astype(int) - 1

    gmsh.finalize()
    return dr

def main():
    gen_mesh()
    plist, tlist = get_plist_tlist_from_gmsh("fem2d_skript.msh")
    dr = get_boundaries("fem2d_skript.msh")
    print("dr:", dr)
    # dr = [0, 1, 2, 3]  # dirichlet boundary nodes (all 4 corners)
    xD = plist[dr, 0]  # x-koordinaten der dirichlet boundary conditions
    xR = []

    fem_solver = fem_cpp.FEM_2D(xD, xR, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, g)
    try:
        timings = fem_solver.full_solve()
        sol = fem_solver.get_Solution()
        print(colors.BOLDGREEN + "FEM solve completed successfully." + colors.RESET)
    except Exception as e:
        print(colors.BOLDRED + "Error during FEM solve: " + str(e) + colors.RESET)
        return

    visualize_solution(plist, tlist, sol, True, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png")
    visualize_solution(plist, tlist, sol, False, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points_sol.png")

    plt.show()

def mesh():
    gen_mesh()


if __name__ == "__main__":
    main()
    # mesh()