import gc
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
from numba import cfunc, float64, int32
import gmsh

# import fem_cpp
import fem_cpp_mkl as fem_cpp

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges
import helper_funcs.meshtools as mt

from helper_funcs.gmshtools import ElementMsh, MshHs

import os.path as path

current_dir = path.dirname(path.abspath(__file__))
tst_data_dir = f"{current_dir}/Loesung_FEM_WS1920"
print(f"Tst directory: {tst_data_dir}")

# ------------------------------------------------------------------------------
mesh_name = "Venugopal_Rishabh_mesh"
ERROR_TOLERANCE = 1e-10
LOAD_W_MT = True
# ------------------------------------------------------------------------------

param_Ra = 0.3
param_dd = 0.01  # thickness of the material
param_Hm = 0.08  # Height of the material
param_Bm = 0.08
param_ma = 0.01
param_mb = 0.05
param_rm0 = 0.095
param_zmo = 0.0
param_lls = 0.001
param_sd = 0.005
param_SH = 0.01

param_M0 = 0.0  # Magnetization of the magnet
param_I1 = 1.0  # Current in Spule 1
param_I2 = 0.0  # Current in Spule 2 and 3
param_mu1 = 10.0  # Permeability of material 1
param_mu2 = 10.0  # Permeability of material 2

MU0 = 4 * np.pi * 1e-7


# ------------------------------------------------------------------------------
# @cfunc(sig=float64(float64, float64))
def alpha1(r, z):
    if r == 0:
        return 0.0
    in_outer_box = (r <= param_Bm - param_dd) and (z >= -param_Hm / 2) and (z <= param_Hm / 2)
    in_inner_hole = (
        (r > param_dd)
        and (r < param_Bm - param_dd)
        and (z > -param_Hm / 2 + param_dd)
        and (z < param_Hm / 2 - param_dd)
    )
    in_mat_1 = in_outer_box and not in_inner_hole

    in_mat_2_r_axis = param_Bm - param_dd <= r <= param_Bm
    in_mat_2_z_axis = -param_Hm / 2 <= z <= param_Hm / 2
    in_mat_2 = in_mat_2_r_axis and in_mat_2_z_axis

    if in_mat_1:
        mu_r = param_mu1
    elif in_mat_2:
        mu_r = param_mu2
    else:
        mu_r = 1.0

    mu = MU0 * mu_r
    return 1 / (mu * r)


# @cfunc(sig=float64(float64, float64))
def alpha2(r, z):
    return alpha1(r, z)


# @cfunc(sig=float64(float64, float64))
def beta(r, z):
    if r == 0:
        return 0.0
    return 0.0


# @cfunc(sig=float64(float64, float64))
def f(r, z):
    param_j0_I1 = param_I1 / (2 * param_SH * param_sd)  # CJ = I/A, Stromdichte in der Spule
    param_j0_I2 = param_I2 / (param_SH * param_sd)  # CJ = I/A, Stromdichte in der Spule

    # check if in spule 1
    in_spule_1_r_axis = param_dd + param_lls <= r <= param_dd + param_lls + param_sd
    in_spule_1_z_axis = -param_SH <= z <= param_SH
    in_spule_1 = in_spule_1_r_axis and in_spule_1_z_axis

    # check if in spule 2
    in_spule_2_r_axis = param_Bm - param_dd - param_lls - param_sd <= r <= param_Bm - param_dd - param_lls
    in_spule_2_z_axis = -param_SH / 2 <= z <= param_SH / 2
    in_spule_2 = in_spule_2_r_axis and in_spule_2_z_axis

    # check if in spule 3
    in_spule_3_r_axis = param_Bm + param_lls <= r <= param_Bm + param_lls + param_sd
    in_spule_3_z_axis = -param_SH / 2 <= z <= param_SH / 2
    in_spule_3 = in_spule_3_r_axis and in_spule_3_z_axis

    if in_spule_1:
        param_j0 = param_j0_I1
    elif in_spule_2:
        param_j0 = param_j0_I2
    elif in_spule_3:
        param_j0 = -param_j0_I2
    else:
        param_j0 = 0.0

    val = (
        -param_M0 * 12 / param_ma * (2 * (r - param_rm0) / param_ma) ** 5 * (1 - (2 * (z - param_zmo) / param_mb) ** 6)
    )
    val1 = (
        -param_M0 * 12 / param_mb * (1 - (2 * (r - param_rm0) / param_ma) ** 6) * (2 * (z - param_zmo) / param_mb) ** 5
    )
    return param_j0 + val + val1


# @cfunc(sig=float64(float64, float64))
def gamma(r, z):
    if r == 0:
        return 0.0

    in_outer_box = (r <= param_Bm - param_dd) and (z >= -param_Hm / 2) and (z <= param_Hm / 2)
    in_inner_hole = (
        (r > param_dd)
        and (r < param_Bm - param_dd)
        and (z > -param_Hm / 2 + param_dd)
        and (z < param_Hm / 2 - param_dd)
    )
    in_mat_1 = in_outer_box and not in_inner_hole

    in_mat_2_r_axis = param_Bm - param_dd <= r <= param_Bm
    in_mat_2_z_axis = -param_Hm / 2 <= z <= param_Hm / 2
    in_mat_2 = in_mat_2_r_axis and in_mat_2_z_axis

    if in_mat_1:
        mu = MU0 * param_mu1
    elif in_mat_2:
        mu = MU0 * param_mu2
    else:
        mu = MU0

    val = 1 / (mu * r * np.sqrt(r**2 + z**2))
    return val


# @cfunc(sig=float64(float64, float64))
def phi(r, z):
    return 0.0


# @cfunc(sig=float64(float64, float64))
def q(r, z):
    return 0.0


# ------------------------------------------------------------------------------
def get_boundary_curve_tags(surface_tag):
    """Get 1D boundary curve tags for a 2D surface."""
    boundaries = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
    return [abs(tag) for _, tag in boundaries]


def boundary_curves(surf_tags):
    """Get all unique 1D boundary curve tags for a list of surface tags."""
    curves = set()
    for s in surf_tags:
        for _, ct in gmsh.model.getBoundary([(2, s)], oriented=False, combined=True):
            curves.add(abs(ct))
    return list(curves)


def gen_mesh():
    gmsh.initialize()

    # == Create Geometry ==
    # Create outer semicircle
    C = (0, 0, 0)  # center of semicircle
    S = (0, param_Ra, 0)  # start point of semicircle
    E = (0, -param_Ra, 0)  # end point of semicircle
    PC = gmsh.model.occ.addPoint(C[0], C[1], C[2])
    PS = gmsh.model.occ.addPoint(S[0], S[1], S[2])
    PE = gmsh.model.occ.addPoint(E[0], E[1], E[2])
    semicircle = gmsh.model.occ.addCircleArc(PS, PC, PE)
    symm_boundary = gmsh.model.occ.addLine(PE, PS)
    semicircle_surface = gmsh.model.occ.addCurveLoop([semicircle, symm_boundary])
    gmsh.model.occ.addPlaneSurface([semicircle_surface])

    # create material 1 c shape
    mat1_outer = gmsh.model.occ.addRectangle(0, -param_Hm / 2, 0, param_Bm - param_dd, param_Hm)
    mat1_inner = gmsh.model.occ.addRectangle(
        param_dd, -param_Hm / 2 + param_dd, 0, param_Bm - 2 * param_dd, (param_Hm - 2 * param_dd)
    )
    # mat1_c_shape = [(2, mat1_outer), (2, mat1_inner)]
    mat1_c_shape, _ = gmsh.model.occ.cut([(2, mat1_outer)], [(2, mat1_inner)])

    # create material 2 rectangle
    mat2 = gmsh.model.occ.addRectangle(param_Bm - param_dd, -param_Hm / 2, 0, param_dd, param_Hm)

    # create line 1
    PL1 = (param_dd, 0, 0)
    PL1 = gmsh.model.occ.addPoint(*PL1)
    L1 = gmsh.model.occ.addLine(PC, PL1)

    # create line 2
    PL2_S = (param_Bm - param_dd, 0, 0)
    PL2_S = gmsh.model.occ.addPoint(*PL2_S)
    PL2_E = (param_Bm, 0, 0)
    PL2_E = gmsh.model.occ.addPoint(*PL2_E)
    L2 = gmsh.model.occ.addLine(PL2_S, PL2_E)

    # create Spule 1
    spule1 = gmsh.model.occ.addRectangle(param_dd + param_lls, -param_SH, 0, param_sd, 2 * param_SH)

    # create Spule 2
    spule2 = gmsh.model.occ.addRectangle(
        param_Bm - param_dd - param_lls - param_sd, -param_SH / 2, 0, param_sd, param_SH
    )

    # create spule 3
    spule3 = gmsh.model.occ.addRectangle(param_Bm + param_lls, -param_SH / 2, 0, param_sd, param_SH)

    # create magnet
    magnet_south = gmsh.model.occ.addRectangle(param_rm0 - param_ma / 2, -param_mb / 2, 0, param_ma, param_mb / 2)
    magnet_north = gmsh.model.occ.addRectangle(param_rm0 - param_ma / 2, 0, 0, param_ma, param_mb / 2)

    tools = [
        (1, symm_boundary),
        (2, mat1_c_shape[0][1]),
        (2, mat2),
        (1, L1),
        (1, L2),
        (2, spule1),
        (2, spule2),
        (2, spule3),
        (2, magnet_south),
        (2, magnet_north),
    ]

    gmsh.model.occ.synchronize()

    # fragment the geometry
    out_dimtags, out_map = gmsh.model.occ.fragment([(2, semicircle_surface)], tools)

    gmsh.model.occ.synchronize()

    def new_tags(tool_idx):
        return [t for _, t in out_map[1 + tool_idx]]

    # tags_semicircle = new_tags(0)
    tags_symm_boundary = new_tags(0)
    tags_mat1 = new_tags(1)
    tags_mat2 = new_tags(2)
    tags_L1 = new_tags(3)
    tags_L2 = new_tags(4)
    tags_spule1 = new_tags(5)
    tags_spule2 = new_tags(6)
    tags_spule3 = new_tags(7)
    tags_mag_south = new_tags(8)
    tags_mag_north = new_tags(9)

    all_2d = {t for d, t in out_dimtags if d == 2}
    all_surfs = list(all_2d)
    all_outer = set(boundary_curves(all_surfs))  # only true outer boundary remains
    tags_semicircle = list(all_outer - set(tags_symm_boundary))

    tags_semicircle_surfs = [t for _, t in out_map[0]]

    # === Physical groups — all after fragment+synchronize ===
    gmsh.model.addPhysicalGroup(2, tags_spule1, tag=1, name="Spule1")
    gmsh.model.addPhysicalGroup(2, tags_spule2, tag=2, name="Spule2")
    gmsh.model.addPhysicalGroup(2, tags_spule3, tag=3, name="Spule3")
    gmsh.model.addPhysicalGroup(2, tags_mag_south, tag=4, name="MagnetSouth")
    gmsh.model.addPhysicalGroup(2, tags_mag_north, tag=5, name="MagnetNorth")
    gmsh.model.addPhysicalGroup(2, tags_mat2, tag=6, name="Material2")
    gmsh.model.addPhysicalGroup(2, tags_mat1, tag=7, name="Material1")
    gmsh.model.addPhysicalGroup(2, tags_semicircle_surfs, tag=8, name="Luft")
    gmsh.model.addPhysicalGroup(1, tags_L1, tag=9, name="Linie1")
    gmsh.model.addPhysicalGroup(1, tags_L2, tag=10, name="Linie2")
    gmsh.model.addPhysicalGroup(1, tags_semicircle, tag=11, name="Semicircle")
    gmsh.model.addPhysicalGroup(1, tags_symm_boundary, tag=12, name="SymmetryBoundary")

    # boundary curves
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_mat1), tag=20, name="Material1Boundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_mat2), tag=21, name="Material2Boundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_spule1), tag=22, name="Spule1Boundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_spule2), tag=23, name="Spule2Boundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_spule3), tag=24, name="Spule3Boundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_mag_south), tag=25, name="MagnetSouthBoundary")
    gmsh.model.addPhysicalGroup(1, boundary_curves(tags_mag_north), tag=26, name="MagnetNorthBoundary")

    # === Verfeinerung of the mesh ===
    dist_tag = gmsh.model.mesh.field.add("Distance")
    tags_to_refine = tags_L1 + tags_L2

    gmsh.model.mesh.field.setNumbers(dist_tag, "CurvesList", tags_to_refine)
    gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", 100)
    math_tag = gmsh.model.mesh.field.add("MathEval")
    formula = f"0.0005 + 0.1 * F{dist_tag}"  # 0.0005 is the minimum element size, 0.5 controls how fast the size grows with distance
    gmsh.model.mesh.field.setString(math_tag, "F", formula)
    gmsh.model.mesh.field.setAsBackgroundMesh(math_tag)

    # == Generate Mesh ==
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    mesh = gmsh.model.mesh.generate(2)
    gmsh.write(f"{mesh_name}.msh")

    try:
        gmsh.fltk.run()
    except:
        print("No FLTK GUI available, skipping visualization.")

    netz = MshHs(gmsh.model)
    gmsh.finalize()
    return netz


def a():
    netz = gen_mesh()
    netz.dim = 2
    netz.Triangle.plot(color="gray", alpha=0.2)

    # outlines
    netz.Material1Boundary.plot(color="pink")
    netz.Material2Boundary.plot(color="purple")
    netz.Spule1Boundary.plot(color="blue")
    netz.Spule2Boundary.plot(color="green")
    netz.Spule3Boundary.plot(color="red")
    netz.MagnetSouthBoundary.plot(color="orange")
    netz.MagnetNorthBoundary.plot(color="darkorange")
    netz.SymmetryBoundary.plot(color="black")
    netz.Linie1.plot(color="cyan")
    netz.Linie2.plot(color="magenta")
    netz.Semicircle.plot(color="black")

    plt.axis("equal")


def load_mesh():
    gmsh.initialize()
    gmsh.open(f"{mesh_name}.msh")
    netz = MshHs(gmsh.model)
    gmsh.finalize()
    return netz


def b(part_of_d=False):
    global param_I1, param_I2
    param_I1 = 1.0  # Current in Spule 1
    param_I2 = 0.0  # Current in Spule 2 and 3
    if LOAD_W_MT:
        p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh("Klausur_WS1920_netz.npz", show=False)
        Ps = [(0, param_Ra), (0, -param_Ra)]
        rand1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(0, -param_Ra), (0, param_Ra)]
        rand2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        Ps = [(0, 0), (param_dd, 0)]
        linie1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(param_Bm, 0), (param_Bm - param_dd, 0)]
        linie2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        netz = MshHs(
            None,
            p,
            t,
            {
                "SymmetryBoundary": rand1,
                "Semicircle": rand2,
                "Linie1": linie1,
                "Linie2": linie2,
            },
        )
        netz.dim = 2
        plist = np.asfortranarray(netz.points.astype(np.float64))
        tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
        rr = np.asfortranarray(netz.Semicircle.elements.astype(np.int32).reshape(-1, 2))
        dd = np.ascontiguousarray(netz.SymmetryBoundary.elements.astype(np.int32).flatten())
    else:
        netz = load_mesh()
        netz.dim = 2

        netz.Triangle.plot(color="gray", alpha=0.2)

        plist = netz.points.astype(np.float64)
        tlist = netz.Triangle.elements.astype(np.int32)
        rr = netz.Semicircle.elements.astype(np.int32)
        dd = netz.SymmetryBoundary.elements.astype(np.int32).flatten()

    solver = fem_cpp.FEM_2D(dd, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
    timing = solver.full_solve()
    sol = solver.get_Solution()

    if not part_of_d:
        # ----------------------
        p = plist
        print("Plist shape: ", plist.shape)
        triangulation = tri.Triangulation(p[:, 0], p[:, 1])
        plt.figure(figsize=(8, 6))
        plt.tricontour(triangulation, sol, colors="k", levels=25)
        contour = plt.tricontourf(triangulation, sol, cmap="jet", levels=25)
        plt.colorbar(contour, label="Solution Value ($\\phi$)")
        plt.triplot(triangulation, color="black", alpha=0.3, linewidth=0.5)  # overlay triangulation
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("2D FEA Nodal Solution ($\\phi$)")
        # ----------------------
        print_timings(timing, "FEM 2D Timings", len(plist), len(tlist), False, "CPP")

        if not LOAD_W_MT:
            my_ax = visualize_solution(
                plist, tlist, sol, False, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points.png"
            )
            netz.Triangle.plot(ax=my_ax, color="gray", alpha=0.1)

            # outlines -- only if using gmsh to load and visualize, otherwise they are not available
            netz.Material1Boundary.plot(ax=my_ax, color="pink")
            netz.Material2Boundary.plot(ax=my_ax, color="purple")
            netz.Spule1Boundary.plot(ax=my_ax, color="blue")
            netz.Spule2Boundary.plot(ax=my_ax, color="green")
            netz.Spule3Boundary.plot(ax=my_ax, color="red")
            netz.Linie1.plot(ax=my_ax, color="cyan")
            netz.Linie2.plot(ax=my_ax, color="magenta")
            netz.SymmetryBoundary.plot(ax=my_ax, color="black")
            netz.Semicircle.plot(ax=my_ax, color="black")

        if LOAD_W_MT:
            sol_tst = np.loadtxt(f"{tst_data_dir}/Teil_b_3.dat", dtype=float)
            error, error_stats = solver.validate_sol(sol_tst, ERROR_TOLERANCE)
            print_error_stats(
                error_stats, "Lösung B", len(plist) - 1, False, "CPP", f"LoesungB_{len(plist)}points_stats.txt"
            )
            visualize_error(
                plist, error, "Fehlerverteilung für Lösung B", False, "CPP", f"LoesungB_{len(plist)}points_error.png"
            )

    # get inductivity at line 1 and line 2
    # print("–----------------------")
    current_plist = plist[:, :2]  # only x and y coordinates, ignore z
    # Knotenidizes zu [R0,0], [R1,0], [R2,0]
    Ps = [[0, 0], [param_dd, 0], [param_Bm - param_dd, 0], [param_Bm, 0]]
    fnodes = mt.FindClosestNode(range(len(current_plist)), current_plist, Ps)
    node0 = fnodes[0][0]
    node1 = fnodes[0][1]
    node2 = fnodes[0][2]
    node3 = fnodes[0][3]
    phi1 = 2 * np.pi * (sol[node1] - sol[node0])
    phi2 = 2 * np.pi * (sol[node3] - sol[node2])
    L_1 = phi1 / param_I1
    M = phi2 / param_I1
    # print(f"L1: {L_1:.6e} H")
    # print(f"M: {M:.6e} H")
    # print("–----------------------")

    del solver  # Replace with your actual variable name
    gc.collect()
    # print("Execution finished cleanly.")
    return L_1, M


def c(part_of_d=False):
    global param_I1, param_I2
    param_I1 = 0.0  # Current in Spule 1
    param_I2 = 1.0  # Current in Spule 2 and 3

    if LOAD_W_MT:
        p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh("Klausur_WS1920_netz.npz", show=False)
        Ps = [(0, param_Ra), (0, -param_Ra)]
        rand1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(0, -param_Ra), (0, param_Ra)]
        rand2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        Ps = [(0, 0), (param_dd, 0)]
        linie1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(param_Bm, 0), (param_Bm - param_dd, 0)]
        linie2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        netz = MshHs(
            None,
            p,
            t,
            {
                "SymmetryBoundary": rand1,
                "Semicircle": rand2,
                "Linie1": linie1,
                "Linie2": linie2,
            },
        )
        netz.dim = 2
        plist = np.asfortranarray(netz.points.astype(np.float64))
        tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
        rr = np.asfortranarray(netz.Semicircle.elements.astype(np.int32).reshape(-1, 2))
        dd = np.ascontiguousarray(netz.SymmetryBoundary.elements.astype(np.int32).flatten())
    else:
        netz = load_mesh()
        netz.dim = 2

        netz.Triangle.plot(color="gray", alpha=0.2)

        plist = netz.points.astype(np.float64)
        tlist = netz.Triangle.elements.astype(np.int32)
        rr = netz.Semicircle.elements.astype(np.int32)
        dd = netz.SymmetryBoundary.elements.astype(np.int32).flatten()

    solver = fem_cpp.FEM_2D(dd, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
    timing = solver.full_solve()
    sol = solver.get_Solution()
    if not part_of_d:
        # ----------------------
        p = plist
        print("Plist shape: ", plist.shape)
        triangulation = tri.Triangulation(p[:, 0], p[:, 1])
        plt.figure(figsize=(8, 6))
        plt.tricontour(triangulation, sol, colors="k", levels=25)
        contour = plt.tricontourf(triangulation, sol, cmap="jet", levels=25)
        plt.colorbar(contour, label="Solution Value ($\\phi$)")
        plt.triplot(triangulation, color="black", alpha=0.3, linewidth=0.5)  # overlay triangulation
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("2D FEA Nodal Solution ($\\phi$)")
        # ----------------------
        print_timings(timing, "FEM 2D Timings", len(plist), len(tlist), False, "CPP")

        if not LOAD_W_MT:
            my_ax = visualize_solution(
                plist, tlist, sol, False, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points.png"
            )
            netz.Triangle.plot(ax=my_ax, color="gray", alpha=0.1)

            # outlines -- only if using gmsh to load and visualize, otherwise they are not available
            netz.Material1Boundary.plot(ax=my_ax, color="pink")
            netz.Material2Boundary.plot(ax=my_ax, color="purple")
            netz.Spule1Boundary.plot(ax=my_ax, color="blue")
            netz.Spule2Boundary.plot(ax=my_ax, color="green")
            netz.Spule3Boundary.plot(ax=my_ax, color="red")
            netz.Linie1.plot(ax=my_ax, color="cyan")
            netz.Linie2.plot(ax=my_ax, color="magenta")
            netz.SymmetryBoundary.plot(ax=my_ax, color="black")
            netz.Semicircle.plot(ax=my_ax, color="black")

        if LOAD_W_MT:
            sol_tst = np.loadtxt(f"{tst_data_dir}/Teil_c_3.dat", dtype=float)
            error, error_stats = solver.validate_sol(sol_tst, ERROR_TOLERANCE)
            print_error_stats(
                error_stats, "Lösung C", len(plist) - 1, False, "CPP", f"LoesungC_{len(plist)}points_stats.txt"
            )
            visualize_error(
                plist, error, "Fehlerverteilung für Lösung C", False, "CPP", f"LoesungC_{len(plist)}points_error.png"
            )

    # get inductivity at line 1 and line 2
    # print("–----------------------")
    current_plist = plist[:, :2]  # only x and y coordinates, ignore z
    # Knotenidizes zu [R0,0], [R1,0], [R2,0]
    Ps = [[0, 0], [param_dd, 0], [param_Bm - param_dd, 0], [param_Bm, 0]]
    fnodes = mt.FindClosestNode(range(len(current_plist)), current_plist, Ps)
    node0 = fnodes[0][0]
    node1 = fnodes[0][1]
    node2 = fnodes[0][2]
    node3 = fnodes[0][3]
    phi1 = 2 * np.pi * (sol[node1] - sol[node0])
    phi2 = 2 * np.pi * (sol[node3] - sol[node2])
    L_2 = phi2 / param_I2
    M_other = phi1 / param_I2
    # print(f"L2: {L_2:.6e} H")
    # print(f"M_other: {M_other:.6e} H")
    # print("–----------------------")

    del solver  # Replace with your actual variable name
    gc.collect()
    # print("Execution finished cleanly.")
    return L_2, M_other


def d():
    print("–---------------------- d -----------------------")
    mu_r_list = [1, 10, 50, 120, 300, 700, 1500, 3000, 5000, 10000]
    results = []
    for mu_r in mu_r_list:
        global param_mu1, param_mu2
        param_mu1 = mu_r
        param_mu2 = mu_r
        L_1, M = b(part_of_d=True)
        L_2, M_ = c(part_of_d=True)
        L_1 = abs(L_1)
        M = abs(M)
        L_2 = abs(L_2)
        M_ = abs(M_)
        k = M / np.sqrt(L_1 * L_2)
        results.append({"mu_r": mu_r, "L1 (H)": L_1, "M (H)": M, "L2 (H)": L_2, "M' (H)": M_, "k": k})
    df = pd.DataFrame(results)
    print(
        df.to_string(
            index=False,  # Hides the default row numbers (0, 1, 2...)
            justify="center",  # Centers the column headers
            formatters={  # Applies your specific formatting requirements
                "L1 (H)": lambda x: f"{x:.6e}",
                "M (H)": lambda x: f"{x:.6e}",
                "L2 (H)": lambda x: f"{x:.6e}",
                "M' (H)": lambda x: f"{x:.6e}",
                "k": lambda x: f"{x:.6f}",
            },
        )
    )


def e(part_of_d=False):
    print("–---------------------- e -----------------------")
    global param_I1, param_I2
    param_I1 = 1.0  # Current in Spule 1
    param_I2 = 0.0  # Current in Spule 2 and 3

    length = 0.01
    epsL = length / 100

    if LOAD_W_MT:
        p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh("Klausur_WS1920_netz.npz", show=False)
        Ps = [(0, param_Ra), (0, -param_Ra)]
        rand1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(0, -param_Ra), (0, param_Ra)]
        rand2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        Ps = [(0, 0), (param_dd, 0)]
        linie1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(param_Bm, 0), (param_Bm - param_dd, 0)]
        linie2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        netz = MshHs(
            None,
            p,
            t,
            {
                "SymmetryBoundary": rand1,
                "Semicircle": rand2,
                "Linie1": linie1,
                "Linie2": linie2,
            },
        )
        netz.dim = 2
        plist = np.asfortranarray(netz.points.astype(np.float64))
        tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
        rr = np.asfortranarray(netz.Semicircle.elements.astype(np.int32).reshape(-1, 2))
        dd = np.ascontiguousarray(netz.SymmetryBoundary.elements.astype(np.int32).flatten())
    else:
        netz = load_mesh()
        netz.dim = 2

        netz.Triangle.plot(color="gray", alpha=0.2)

        plist = netz.points.astype(np.float64)
        tlist = netz.Triangle.elements.astype(np.int32)
        rr = netz.Semicircle.elements.astype(np.int32)
        dd = netz.SymmetryBoundary.elements.astype(np.int32).flatten()

    mu_r = [1, 10, 100, 500, 3000]
    B_Z_results = []
    H_Z_results = []

    for mu_r in mu_r:
        global param_mu1, param_mu2
        param_mu1 = mu_r
        param_mu2 = mu_r

        solver = fem_cpp.FEM_2D(dd, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
        timing = solver.full_solve()
        sol = solver.get_Solution()

        if not part_of_d:
            # ----------------------
            p = plist
            print("Plist shape: ", plist.shape)
            triangulation = tri.Triangulation(p[:, 0], p[:, 1])
            plt.figure(figsize=(8, 6))
            plt.tricontour(triangulation, sol, colors="k", levels=25)
            contour = plt.tricontourf(triangulation, sol, cmap="jet", levels=25)
            plt.colorbar(contour, label="Solution Value ($\\phi$)")
            plt.triplot(triangulation, color="black", alpha=0.3, linewidth=0.5)  # overlay triangulation
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("2D FEA Nodal Solution ($\\phi$)")
            # ----------------------
            print_timings(timing, "FEM 2D Timings", len(plist), len(tlist), False, "CPP")

        # get inductivity at line 1 and line 2
        # print("–----------------------")
        current_plist = plist[:, :2]  # only x and y coordinates, ignore z

        # r-Achse
        Ps = [[epsL, 0], [param_Ra - epsL, 0]]
        rAchse = mt.RetrieveSegments(current_plist, CuE, li_CE, Ps, ["Nodes"])
        rAchse = rAchse[0]

        # Extract values on r-Achse
        r_coords = plist[rAchse, 0]
        sol_r = sol[rAchse]
        yp = np.diff(sol_r) / np.diff(r_coords)

        # get midpoints of r-Achse segments for plotting
        r_mid = (r_coords[:-1] + r_coords[1:]) / 2

        B_Z = 1 / r_mid * yp
        H_Z = B_Z / (MU0 * mu_r)
        B_Z_results.append((r_mid, B_Z, mu_r))
        H_Z_results.append((r_mid, H_Z, mu_r))

        del solver  # Replace with your actual variable name
        gc.collect()
        # print("Execution finished cleanly.")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for r_mid, B_Z, mu_rel in B_Z_results:
        plt.plot(r_mid, B_Z, label=f"$\\mu_r$={mu_rel}")

    plt.xlabel("r (m)")
    plt.ylabel("$B_z$ (T)")
    plt.title("Magnetic Flux Density $B_z$ along r-Achse")
    plt.legend()
    plt.subplot(1, 2, 2)
    for r_mid, H_Z, mu_rel in H_Z_results:
        plt.plot(r_mid, H_Z, label=f"$\\mu_r$={mu_rel}")
    plt.xlabel("r (m)")
    plt.ylabel("$H_z$ (A/m)")
    plt.title("Magnetic Field Strength $H_z$ along r-Achse")
    plt.legend()


def f_aufgabe():
    print("–---------------------- f -----------------------")
    global param_I1, param_I2, param_M0, param_mu1, param_mu2
    param_I1 = 1.0
    param_I2 = 0.0
    param_M0 = 1e6
    param_mu1 = 500
    param_mu2 = 1
    length = 0.01
    epsL = length / 100
    # load mesh
    if LOAD_W_MT:
        p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh("Klausur_WS1920_netz.npz", show=False)
        Ps = [(0, param_Ra), (0, -param_Ra)]
        rand1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(0, -param_Ra), (0, param_Ra)]
        rand2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        Ps = [(0, 0), (param_dd, 0)]
        linie1 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])
        Ps = [(param_Bm, 0), (param_Bm - param_dd, 0)]
        linie2 = mt.RetrieveSegments(p, BouE, li_BE, Ps, ["Segments"])

        netz = MshHs(
            None,
            p,
            t,
            {
                "SymmetryBoundary": rand1,
                "Semicircle": rand2,
                "Linie1": linie1,
                "Linie2": linie2,
            },
        )
        netz.dim = 2
        plist = np.asfortranarray(netz.points.astype(np.float64))
        tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
        rr = np.asfortranarray(netz.Semicircle.elements.astype(np.int32).reshape(-1, 2))
        dd = np.ascontiguousarray(netz.SymmetryBoundary.elements.astype(np.int32).flatten())
    else:
        netz = load_mesh()
        netz.dim = 2

        netz.Triangle.plot(color="gray", alpha=0.2)

        plist = netz.points.astype(np.float64)
        tlist = netz.Triangle.elements.astype(np.int32)
        rr = netz.Semicircle.elements.astype(np.int32)
        dd = netz.SymmetryBoundary.elements.astype(np.int32).flatten()

        B_Z_results = []

    B_Z_results = []
    H_Z_results = []

    solver = fem_cpp.FEM_2D(dd, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
    timing = solver.full_solve()
    sol = solver.get_Solution()

    # ----------------------
    p = plist
    print("Plist shape: ", plist.shape)
    triangulation = tri.Triangulation(p[:, 0], p[:, 1])
    plt.figure(figsize=(8, 6))
    plt.tricontour(triangulation, sol, colors="k", levels=25)
    contour = plt.tricontourf(triangulation, sol, cmap="jet", levels=25)
    plt.colorbar(contour, label="Solution Value ($\\phi$)")
    plt.triplot(triangulation, color="black", alpha=0.3, linewidth=0.5)  # overlay triangulation
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D FEA Nodal Solution ($\\phi$)")
    # ----------------------
    print_timings(timing, "FEM 2D Timings", len(plist), len(tlist), False, "CPP")

    current_plist = plist[:, :2]  # only x and y coordinates, ignore z

    # r-Achse
    Ps = [[epsL, 0], [param_Ra - epsL, 0]]
    rAchse = mt.RetrieveSegments(current_plist, CuE, li_CE, Ps, ["Nodes"])
    rAchse = rAchse[0]

    # # Knotenidizes zu linie1 und linie2
    # Ps = [[0, 0], [param_dd, 0], [param_Bm - param_dd, 0], [param_Bm, 0]]
    # fnodes = mt.FindClosestNode(range(len(current_plist)), current_plist, Ps)
    # node0 = fnodes[0][0]
    # node1 = fnodes[0][1]
    # node2 = fnodes[0][2]
    # node3 = fnodes[0][3]

    # Extract values on r-Achse
    r_coords = plist[rAchse, 0]
    sol_r = sol[rAchse]
    yp = np.diff(sol_r) / np.diff(r_coords)

    # CRITICAL: sort before calculating differences
    sort_idx = np.argsort(r_coords)
    r_sorted = r_coords[sort_idx]
    psi_sorted = sol_r[sort_idx]

    # get midpoints of r-Achse segments for plotting
    dr = np.diff(r_sorted)
    yp = np.diff(psi_sorted) / dr
    r_mid = r_sorted[:-1] + dr / 2.0

    # # get midpoints of r-Achse segments for plotting
    # r_mid = (r_coords[:-1] + r_coords[1:]) / 2

    B_Z = 1 / r_mid * yp
    H_Z = np.zeros_like(B_Z)
    mag_min = param_rm0 - (param_ma / 2.0)
    mag_max = param_rm0 + (param_ma / 2.0)

    for i, r_val in enumerate(r_mid):
        # check if inside the permanent magnet?
        if mag_min <= r_val <= mag_max:
            H_Z[i] = (B_Z[i] / MU0) - param_M0
        # check if on other material at z=0
        else:
            # Material 1
            if r_val <= param_dd:
                mu_r = param_mu1
            # Material 2
            elif (param_Bm - param_dd) <= r_val <= param_Bm:
                mu_r = param_mu2
            # Everywhere else is Air
            else:
                mu_r = 1.0

            H_Z[i] = B_Z[i] / (MU0 * mu_r)
    B_Z_results.append((r_mid, B_Z, param_mu1))
    H_Z_results.append((r_mid, H_Z, param_mu1))

    del solver  # Replace with your actual variable name
    gc.collect()
    # print("Execution finished cleanly.")

    plt.figure(figsize=(18, 5))

    # 1. Plot: Potential Psi (Verwende hier die ungekürzten r_sorted und psi_sorted!)
    plt.subplot(1, 3, 1)
    plt.plot(r_sorted, psi_sorted, 'b-', label="$\\Psi(r)$", linewidth=1.5)
    plt.xlabel("r (m)")
    plt.ylabel("$\\Psi$ (Wb/m)")
    plt.title("Vektorpotential $\\Psi$ entlang der r-Achse")
    plt.grid(True)
    plt.legend()

    # 2. Plot: Flussdichte Bz
    plt.subplot(1, 3, 2)
    for r_m, b_val, mu_rel in B_Z_results:
        plt.plot(r_m, b_val, 'r-', label=f"$B_z$ ($\\mu_1$={mu_rel})", linewidth=1.5)
    plt.xlabel("r (m)")
    plt.ylabel("$B_z$ (T)")
    plt.title("Magnetische Flussdichte $B_z$ entlang der r-Achse")
    plt.grid(True)
    plt.legend()

    # 3. Plot: Feldstärke Hz
    plt.subplot(1, 3, 3)
    for r_m, h_val, mu_rel in H_Z_results:
        plt.plot(r_m, h_val, 'g-', label=f"$H_z$ ($\\mu_1$={mu_rel})", linewidth=1.5)
    plt.xlabel("r (m)")
    plt.ylabel("$H_z$ (A/m)")
    plt.title("Magnetische Feldstärke $H_z$ entlang der r-Achse")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


if __name__ == "__main__":
    # a()

    # L_1, M = b()
    # L_2, M_ = c()
    # L_1 = abs(L_1)
    # M = abs(M)
    # L_2 = abs(L_2)
    # M_ = abs(M_)
    # print("–---------------------- b und c -----------------------")
    # print(f"L1: {L_1:.6e} H")
    # print(f"M: {M:.6e} H")
    # print(f"L2: {L_2:.6e} H")
    # print(f"M': {M_:.6e} H")
    # k = M / np.sqrt(L_1 * L_2)
    # print(f"Kopplungsfaktor k: {k:.6f}")
    # # print("–---------------------------------------------------")
    # d()
    # e()
    f_aufgabe()
    # plt.axis("equal")
    plt.show()
