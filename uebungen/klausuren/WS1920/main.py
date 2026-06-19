from time import time

import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64, int32
import gmsh

# bin_dir = path.abspath(path.join(path.dirname(__file__), "../../bin"))
# sys.path.insert(0, bin_dir)
import fem_cpp

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges
import helper_funcs.meshtools as mt

from helper_funcs.gmshtools import ElementMsh, MshHs

# ------------------------------------------------------------------------------
mesh_name = "Venugopal_Rishabh_mesh"


# ------------------------------------------------------------------------------

param_Ra = 0.3
param_dd = 0.01  # thickness of the material
param_Hm = 0.08  # Height of the material
param_Bm = 0.08
param_ma = 0.01
param_mb = 0.05
param_rm0 = 0.095  # Radius of circle
param_zmo = 0.0
param_lls = 0.001
param_sd = 0.005
param_SH = 0.01

param_M0 = 1.0  # Magnetization of the magnet
param_j0 = 100.0  # Current density in the spules


# ------------------------------------------------------------------------------
@cfunc(sig=float64(float64, float64))
def alpha1(r, z):
    if r == 0:
        return 0.0
    mu = 10
    return 1 / (10 * r)


@cfunc(sig=float64(float64, float64))
def alpha2(r, z):
    if r == 0:
        return 0.0
    mu = 10
    return 1 / (10 * r)


@cfunc(sig=float64(float64, float64))
def beta(r, z):
    if r == 0:
        return 0.0
    return 0.0


@cfunc(sig=float64(float64, float64))
def f(r, z):
    val = -param_M0 * 12 / param_ma * (2 * (-param_rm0) / param_ma) ** 5 * (1 - (2 * (z - param_zmo) / param_mb) ** 6)
    val1 = -param_M0 * 12 / param_mb * (1 - (2 * (z - param_rm0) / param_ma) ** 6) * (2 * (-param_zmo) / param_mb) ** 5
    return param_j0 + val + val1


@cfunc(sig=float64(float64, float64))
def gamma(r, z):
    if r == 0:
        return 0.0
    mu = 10
    return 1 / (mu * r)


@cfunc(sig=float64(float64, float64))
def phi(r, z):
    if r == 0:
        return 0.0
    mu = 10
    return 1 / (mu * r)


@cfunc(sig=float64(float64, float64))
def q(r, z):
    if r == 0:
        return 0.0
    mu = 10
    val = -1 / (mu * r * np.sqrt(r**2 + z**2))
    return val


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
    formula = f"0.0005 + 0.5 * F{dist_tag}"  # 0.0005 is the minimum element size, 0.5 controls how fast the size grows with distance
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


def a1():
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


if __name__ == "__main__":
    # a1()

    # p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh("Klausur_WS1920_netz.npz", show=True)
    netz = load_mesh()
    netz.dim = 2
    # netz.Triangle.plot(color="gray", alpha=0.2)

    plist = netz.points.astype(np.float64)
    tlist = netz.Triangle.elements.astype(np.int32)
    dd = netz.Semicircle.elements.astype(np.int32).reshape(-1, 1)
    rr = np.vstack(
        (netz.Material2Boundary.elements.astype(np.int32), netz.SymmetryBoundary.elements.astype(np.int32))
    )
    solver = fem_cpp.FEM_2D(dd, rr, plist, tlist, alpha1, alpha2, beta, f, phi, gamma, q)
    timing = solver.full_solve()
    sol = solver.get_Solution()
    visualize_solution(plist, tlist, sol, False, "Lösung 2D", False, "CPP", f"Loesung2D_{len(plist)}points.png")
    print_timings(timing, "FEM 2D Timings", len(plist), len(tlist), False, "CPP")
    # plt.axis("equal")
    plt.show()
