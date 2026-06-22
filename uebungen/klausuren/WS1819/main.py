import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, float64, int32, complex128
import gmsh

import fem_cpp

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges

import helper_funcs.meshtools as mt
from helper_funcs.gmshtools import ElementMsh, MshHs

# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

mesh_name = "Rishabh_Venugopal_WS1819"

SIGMA = 10e-4
EPSILON_0 = 8.854187817e-12
EPSILON_R = 11.0
OMEGA = 0.0

Ra = 25e-2  # 25cm
HH = 10e-2  # 10cm -- höhe platte
Dx = 3e-3  # 3mm
V0 = 1.0  # 1v
hh = 5e-2  # 5cm -- höhe dielektrikums

num_platten = 4


# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----


def alpha_func(x, y):
    return -complex(SIGMA, OMEGA * EPSILON_0 * EPSILON_R)


def beta_func(x, y):
    return 0.0


def f_func(x, y):
    return 0.0


def gamma_func(x, y):
    return 1.0


def phi_func(x, y):
    return 0.0


def q_func(x, y):
    return 0.0


# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----


def boundary_curves(surface_tags):
    """Returns a clean, flat list of unique integer line tags for the given surfaces."""
    dim_tags = [(2, t) for t in surface_tags]
    # getBoundary returns a list of (dim, curve_tag)
    boundary = gmsh.model.getBoundary(dim_tags, oriented=False)

    # Extract only the absolute integer ID of the curve
    clean_tags = list(set(tag for dim, tag in boundary if dim == 1))
    return clean_tags


def gen_mesh():
    gmsh.initialize()

    # --- create geometry ---
    # create circle
    C = (0, 0, 0)
    r = Ra
    PC = gmsh.model.occ.addPoint(C[0], C[1], C[2])
    Circ = gmsh.model.occ.addCircle(C[0], C[1], C[2], r)
    circle_loop = gmsh.model.occ.addCurveLoop([Circ])
    circle = gmsh.model.occ.addPlaneSurface([circle_loop])

    # create capacitor plates
    # calculate starting position so the alignment is centered
    breite = num_platten * Dx + (num_platten - 1) * Dx  # total width of all plates and dielektrikums
    start_pos_x = -breite / 2
    plate_orange_tags = []
    plate_blue_tags = []
    dielek_tags = []
    is_plate_1 = True
    # First plate (orange, left)
    p1 = (start_pos_x, -HH / 2, 0)
    p2 = (start_pos_x + Dx, -HH / 2, 0)
    p3 = (start_pos_x + Dx, HH / 2, 0)
    p4 = (start_pos_x, HH / 2, 0)
    P1 = gmsh.model.occ.addPoint(*p1)
    P2 = gmsh.model.occ.addPoint(*p2)
    P3 = gmsh.model.occ.addPoint(*p3)
    P4 = gmsh.model.occ.addPoint(*p4)
    L1 = gmsh.model.occ.addLine(P1, P2)
    L2 = gmsh.model.occ.addLine(P2, P3)
    L3 = gmsh.model.occ.addLine(P3, P4)
    L4 = gmsh.model.occ.addLine(P4, P1)
    plate_1_loop = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4])
    plate_1 = gmsh.model.occ.addPlaneSurface([plate_1_loop])

    plate_orange_tags.append(plate_1)
    is_plate_1 = False

    for i in range(1, num_platten):
        start_pos_x += Dx

        # dielektrikum
        p5 = (start_pos_x, -HH / 2, 0)
        p6 = (start_pos_x + Dx, -HH / 2, 0)
        p7 = (start_pos_x + Dx, (-HH / 2) + hh, 0)
        p8 = (start_pos_x, (-HH / 2) + hh, 0)
        P5 = gmsh.model.occ.addPoint(*p5)
        P6 = gmsh.model.occ.addPoint(*p6)
        P7 = gmsh.model.occ.addPoint(*p7)
        P8 = gmsh.model.occ.addPoint(*p8)
        L5 = gmsh.model.occ.addLine(P5, P6)
        L6 = gmsh.model.occ.addLine(P6, P7)
        L7 = gmsh.model.occ.addLine(P7, P8)
        L8 = gmsh.model.occ.addLine(P8, P5)
        dielek_loop = gmsh.model.occ.addCurveLoop([L5, L6, L7, L8])
        dielek = gmsh.model.occ.addPlaneSurface([dielek_loop])

        start_pos_x += Dx

        # plate
        p9 = (start_pos_x, -HH / 2, 0)
        p10 = (start_pos_x + Dx, -HH / 2, 0)
        p11 = (start_pos_x + Dx, HH / 2, 0)
        p12 = (start_pos_x, HH / 2, 0)
        P9 = gmsh.model.occ.addPoint(*p9)
        P10 = gmsh.model.occ.addPoint(*p10)
        P11 = gmsh.model.occ.addPoint(*p11)
        P12 = gmsh.model.occ.addPoint(*p12)
        L9 = gmsh.model.occ.addLine(P9, P10)
        L10 = gmsh.model.occ.addLine(P10, P11)
        L11 = gmsh.model.occ.addLine(P11, P12)
        L12 = gmsh.model.occ.addLine(P12, P9)
        plate_loop = gmsh.model.occ.addCurveLoop([L9, L10, L11, L12])
        plate = gmsh.model.occ.addPlaneSurface([plate_loop])

        # add to fragment list
        dielek_tags.append(dielek)

        if is_plate_1:
            plate_orange_tags.append(plate)
            is_plate_1 = False
        else:
            plate_blue_tags.append(plate)
            is_plate_1 = True

    stuff_to_fragment = [(2, t) for t in plate_orange_tags + plate_blue_tags + dielek_tags]

    # --- Sync ---
    gmsh.model.occ.synchronize()

    # --- fragment the geometry ---
    out_dimtags, out_map = gmsh.model.occ.fragment([(2, circle)], stuff_to_fragment)

    # --- Sync ---
    gmsh.model.occ.synchronize()

    # --- Remap physical groups to the new geometry ---
    def new_tags(tool_idx):
        return [t for _, t in out_map[1 + tool_idx]]

    # --- extract new tags and create physical groups ---
    # orange plates
    all_orange_new_tags = []
    for idx, tags in enumerate(plate_orange_tags):
        print(f"--- Processing orange plate {idx + 1} with original tag {tags}...")
        new_tags_plate = new_tags(idx)
        gmsh.model.addPhysicalGroup(2, new_tags_plate, tag=idx + 1, name=f"Plate_{idx + 1}")
        all_orange_new_tags += new_tags_plate

    # blue plates
    all_blue_new_tags = []
    for idx, tags in enumerate(plate_blue_tags):
        print(f"--- Processing blue plate {idx + 1} with original tag {tags}...")
        new_tags_plate = new_tags(idx + len(plate_orange_tags))
        gmsh.model.addPhysicalGroup(
            2, new_tags_plate, tag=idx + 1 + len(plate_orange_tags), name=f"Plate_{idx + 1 + len(plate_orange_tags)}"
        )
        all_blue_new_tags += new_tags_plate

    # dielektrikum
    all_dielek_new_tags = []
    for idx, tags in enumerate(dielek_tags):
        print(f"--- Processing dielektrikum {idx + 1} with original tag {tags}...")
        new_tags_dielek = new_tags(idx + len(plate_orange_tags) + len(plate_blue_tags))
        gmsh.model.addPhysicalGroup(
            2,
            new_tags_dielek,
            tag=idx + 1 + len(plate_orange_tags) + len(plate_blue_tags),
            name=f"Dielektrikum_{idx + 1}",
        )
        all_dielek_new_tags += new_tags_dielek

    gmsh.model.addPhysicalGroup(2, all_orange_new_tags, name="OrangePlates")
    gmsh.model.addPhysicalGroup(2, all_blue_new_tags, name="BluePlates")
    gmsh.model.addPhysicalGroup(2, all_dielek_new_tags, name="Dieleks")

    # orange boundary curves
    orange_boundary_curves = boundary_curves([t for t in plate_orange_tags])
    gmsh.model.addPhysicalGroup(1, orange_boundary_curves, tag=100, name="Boundary_Orange")

    # blue boundary curves
    blue_boundary_curves = boundary_curves([t for t in plate_blue_tags])
    gmsh.model.addPhysicalGroup(1, blue_boundary_curves, tag=200, name="Boundary_Blue")

    # dielektrikum boundary curves
    dielek_boundary_curves = boundary_curves([t for t in dielek_tags])
    gmsh.model.addPhysicalGroup(1, dielek_boundary_curves, tag=300, name="Boundary_Dielek")

    # circle boundary curves
    circle_boundary_curves = boundary_curves([t for _, t in out_map[0]])
    gmsh.model.addPhysicalGroup(1, circle_boundary_curves, tag=400, name="Boundary_Circle")

    # --- Verfeinerung ---
    dist_tag = gmsh.model.mesh.field.add("Distance")
    tags_to_refine = orange_boundary_curves + blue_boundary_curves + dielek_boundary_curves

    gmsh.model.mesh.field.setNumbers(dist_tag, "CurvesList", tags_to_refine)
    gmsh.model.mesh.field.setNumber(dist_tag, "Sampling", 100)
    math_tag = gmsh.model.mesh.field.add("MathEval")
    formula = f"0.01 + 0.1 * F{dist_tag}"  # 0.0005 is the minimum element size, 0.5 controls how fast the size grows with distance
    gmsh.model.mesh.field.setString(math_tag, "F", formula)
    gmsh.model.mesh.field.setAsBackgroundMesh(math_tag)

    # --- Sync ---
    gmsh.model.occ.synchronize()

    # --- Generate Mesh ---
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


# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----


def a1():
    netz = gen_mesh()
    netz.dim = 2

    netz.Triangle.plot(color="gray", alpha=0.1)
    netz.OrangePlates.plot(color="orange")
    netz.BluePlates.plot(color="blue")
    netz.Dieleks.plot(color="green")

    netz.Boundary_Circle.plot(color="red", direction=True)
    plt.legend()

    plt.axis("equal")


if __name__ == "__main__":
    a1()
    plt.show()
