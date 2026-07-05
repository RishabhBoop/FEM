import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from numba import cfunc, float64, int32, complex128
import gmsh

import fem_cpp
from scipy import integrate
from uebungen.klausuren.WS1819.KlausurWS1819_netz import HH, RU, RO, HolPoi

from helper_funcs.colors import Colors as colors
from helper_funcs.visualizations import visualize_solution, print_timings, visualize_error, print_error_stats
from helper_funcs.mesh import get_plist_tlist_from_gmsh, get_boundaries, get_boundary_edges
import scipy.integrate as integrate

import helper_funcs.meshtools as mt
from helper_funcs.gmshtools import ElementMsh, MshHs
import helper_funcs.gmshtools as gm
from uebungen.klausuren.WS1819.KlausurWS1819_netz import RU

# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

mesh_name = "Rishabh_Venugopal_WS1819"

SIGMA_0 = 1e-12
SIGMA = 1e-4
EPSILON_0 = 8.854187817e-12
EPSILON_R = 11.0
OMEGA = 0.0

Ra = 25e-2  # 25cm
# HH = 10e-2  # 10cm -- höhe platte
Dx = 3e-3  # 3mm
V0 = 1.0  # 1v
# hh = 5e-2  # 5cm -- höhe dielektrikums

num_platten = 4
breite = num_platten * Dx + (num_platten - 1) * Dx  # total width of all plates and dielektrikums

# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----


def alpha_func(x, y):
    # check if in dielektrikum region
    for i in range(num_platten - 1):
        dielek_x_start = -breite / 2 + (i * Dx * 2) + Dx
        dielek_x_end = dielek_x_start + Dx
        dielek_y_start = -HH / 2
        dielek_y_end = dielek_y_start + hh

        if dielek_x_start <= x <= dielek_x_end and dielek_y_start <= y <= dielek_y_end:
            return EPSILON_0 * EPSILON_R

    return EPSILON_0  # Outside dielektrikum, return e0

    # return EPSILON_0 * EPSILON_R


def beta_func(x, y):
    return 0.0


def f_func(x, y):
    return 0.0


def gamma_func(x, y):
    zahler = EPSILON_0
    nenner = np.sqrt(x**2 + y**2) * np.log(np.sqrt(x**2 + y**2))
    return -zahler / nenner


def phi_func(x, y):
    # check if platte is orange or blue
    for i in range(num_platten):
        plate_x_start = -breite / 2 + (i * Dx * 2)
        plate_x_end = plate_x_start + Dx
        plate_y_start = -HH / 2
        plate_y_end = HH / 2

        if plate_x_start <= x <= plate_x_end and plate_y_start <= y <= plate_y_end:
            if i % 2 == 0:  # orange plate
                return V0
            else:  # blue plate
                return -V0

    return 0.0  # Outside plates, return 0


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


def load_mt_mesh(mesh_file):
    p, t, BouE, li_BE, bou_elem, CuE, li_CE = mt.LoadTriMesh(mesh_file, show=False)

    # Robin Rand (Kreis Außen)
    Ps_robin = [[Ra, 0.0], [Ra, 0.0]]
    bseg_robin = mt.RetrieveSegments(p, BouE, li_BE, Ps_robin, ["Segments"])
    robinRand = bseg_robin[0]

    # Segmente für die Platten abrufen
    Ps_dirichlet = []
    typ_dirichlet = []
    for k in range(num_platten):
        Ps_dirichlet += [RO[k], RO[k]]
        # Ps_dirichlet += [HolPoi[k], HolPoi[k]]
        typ_dirichlet += ["Segments"]

    bseg_dirichlet = mt.RetrieveSegments(p, BouE, li_BE, Ps_dirichlet, typ_dirichlet)

    # Dictionary für das Netz-Objekt vorbereiten
    mesh_dict = {"robinRand": robinRand, "plattenRand": [seg for sublist in bseg_dirichlet for seg in sublist]}

    # Jede Platte einzeln abspeichern
    for k in range(num_platten):
        mesh_dict[f"plate_{k}"] = list(bseg_dirichlet[k])

    netz = MshHs(None, p, t, mesh_dict)
    return netz


# –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----


def aufg_a():
    netz = gen_mesh()
    netz.dim = 2

    netz.Triangle.plot(color="gray", alpha=0.1)
    netz.OrangePlates.plot(color="orange")
    netz.BluePlates.plot(color="blue")
    netz.Dieleks.plot(color="green")

    netz.Boundary_Circle.plot(color="red", direction=True)
    plt.legend()

    plt.axis("equal")


def aufg_b():
    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----
    HH = 10e-2  # 10cm -- höhe platte
    hh = 5e-2  # 5cm -- höhe dielektrikums

    def alpha_func(x, y):
        # check if in dielektrikum region
        for i in range(num_platten - 1):
            dielek_x_start = -breite / 2 + (i * Dx * 2) + Dx
            dielek_x_end = dielek_x_start + Dx
            dielek_y_start = -HH / 2
            dielek_y_end = dielek_y_start + hh

            if dielek_x_start <= x <= dielek_x_end and dielek_y_start <= y <= dielek_y_end:
                return EPSILON_0 * EPSILON_R

        return EPSILON_0  # Outside dielektrikum, return e0

    def beta_func(x, y):
        return 0.0

    def f_func(x, y):
        return 0.0

    def gamma_func(x, y):
        zahler = EPSILON_0
        nenner = np.sqrt(x**2 + y**2) * np.log(np.sqrt(x**2 + y**2))
        return -zahler / nenner

    def phi_func(x, y):
        # check if platte is orange or blue
        for i in range(num_platten):
            plate_x_start = -breite / 2 + (i * Dx * 2)
            plate_x_end = plate_x_start + Dx
            plate_y_start = -HH / 2
            plate_y_end = HH / 2

            if plate_x_start <= x <= plate_x_end and plate_y_start <= y <= plate_y_end:
                if i % 2 == 0:  # orange plate
                    return V0
                else:  # blue plate
                    return -V0

        return 0.0  # Outside plates, return 0

    def q_func(x, y):
        return 0.0

    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

    def calc_epsilon_r(netz_obj, group_name, segment_indices):
        epsilon_r = []

        # Dynamischer Abruf der korrekten Platte (z.B. "plate_0")
        seg = getattr(netz_obj, group_name).elements[segment_indices]

        # Look up the actual coordinates of these nodes and calculate their midpoints
        p1 = netz_obj.points[seg[:, 0]]
        p2 = netz_obj.points[seg[:, 1]]
        midpoints = (p1 + p2) / 2.0

        # Pass scalar floats into alpha_func
        for i in range(len(midpoints)):
            x, y = midpoints[i, 0], midpoints[i, 1]
            alpha = alpha_func(x, y)
            epsilon_r.append(alpha / EPSILON_0)

        return np.array(epsilon_r)

    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

    netz = load_mt_mesh("mesh_WS1819.npz")

    plist = np.asfortranarray(netz.points.astype(np.float64))
    tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
    rr = np.asfortranarray(netz.robinRand.elements.astype(np.int32).reshape(-1, 2))
    dd = np.ascontiguousarray(netz.plattenRand.elements.astype(np.int32).flatten())

    hhs = [0, HH / 4, HH / 2, 3 * HH / 4, HH]
    kap_vals = []
    hh = HH / 2
    solver = fem_cpp.FEM_2D(
        dd, rr, plist, tlist, alpha_func, alpha_func, beta_func, f_func, phi_func, gamma_func, q_func
    )
    solver.full_solve()
    sol = solver.get_Solution()
    plt.figure(figsize=(20, 10))

    for k in range(num_platten):
        group_name = f"plate_{k}"
        rl, n_der, idx, _, _ = gm.NormalDerivative(group_name, netz, sol, "left")

        eps_r = calc_epsilon_r(netz, group_name, idx)

        ladungsdichte = -EPSILON_0 * eps_r * n_der

        qq = integrate.simpson(ladungsdichte, x=rl)

        plt.subplot(2, 2, k + 1)
        plt.plot(rl, ladungsdichte, label=f"Plattennummer={k}")

        plt.title(f"hh={hh} --> Gesamtladung {qq:.5e} C/m")

        plt.xlabel("Länge / m")
        plt.ylabel(r"Ladungsdichte = $\epsilon \frac{\partial \Phi}{\partial n}$ / $C/m^2$")
        plt.legend()

    plt.tight_layout()
    plt.show()

    del solver
    gc.collect()

    for hh_val in hhs:
        # global hh
        hh = hh_val
        print(f"\n--- Solving for Dielectric Height HH = {hh} ---")

        solver = fem_cpp.FEM_2D(
            dd, rr, plist, tlist, alpha_func, alpha_func, beta_func, f_func, phi_func, gamma_func, q_func
        )
        solver.full_solve()
        sol = solver.get_Solution()

        Q0 = 0.0
        Q1 = 0.0

        # Auswertung PRO PLATTE
        for k in range(num_platten):
            group_name = f"plate_{k}"
            rl, n_der, idx, _, _ = gm.NormalDerivative(group_name, netz, sol, "left")

            # Übergebe den korrekten Gruppennamen an den Helper!
            eps_r = calc_epsilon_r(netz, group_name, idx)
            ladungsdichte = -EPSILON_0 * eps_r * n_der

            qq = integrate.simpson(ladungsdichte, x=rl)

            if k % 2 == 0:
                Q0 += qq
            else:
                Q1 += qq

        C = 0.5 * (Q0 - Q1) / (2 * V0)
        kap_vals.append(C)

        print(f"Q0 (Positive Platten): {Q0:.5e} C")
        print(f"Q1 (Negative Platten): {Q1:.5e} C")
        print(f"Capacitance C:         {C:.5e} F")

        del solver
        gc.collect()

    # ==============================================================
    # PLOTTING
    # ==============================================================

    # Plot 2: Kapazität als Funktion der Höhe (laut Klausur Teil b, letzter Punkt)
    plt.figure(figsize=(8, 6))
    plt.plot(hhs, kap_vals, marker="o", linestyle="-", color="red", linewidth=2)
    plt.title("Kapazität als Funktion der Füllstandshöhe hh")
    plt.xlabel("Füllstandshöhe hh [m]")
    plt.ylabel("Kapazität C [F]")

    plt.show()


def aufg_d():
    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----
    HH = 10e-2  # 10cm -- höhe platte
    hh = HH/2  # 5cm -- höhe dielektrikums
    # f0 = 250e3  # 250kHz

    def alpha_func(x, y):
        # check if in dielektrikum region
        loc_sigma = SIGMA_0
        loc_eps = EPSILON_0
        for i in range(num_platten - 1):
            dielek_x_start = -breite / 2 + (i * Dx * 2) + Dx
            dielek_x_end = dielek_x_start + Dx
            dielek_y_start = -HH / 2
            dielek_y_end = dielek_y_start + hh

            if dielek_x_start <= x <= dielek_x_end and dielek_y_start <= y <= dielek_y_end:
                loc_sigma = SIGMA_0 * SIGMA
                loc_eps = EPSILON_0 * EPSILON_R

        return complex(loc_sigma, 2 * np.pi * f0 * loc_eps)

    def beta_func(x, y):
        return complex(0.0, 0.0)

    def f_func(x, y):
        return complex(0.0, 0.0)

    def gamma_func(x, y):
        zahler = EPSILON_0
        nenner = np.sqrt(x**2 + y**2) * np.log(np.sqrt(x**2 + y**2))
        return complex(-zahler / nenner, 0.0)

    def phi_func(x, y):
        # check if platte is orange or blue
        for i in range(num_platten):
            plate_x_start = -breite / 2 + (i * Dx * 2)
            plate_x_end = plate_x_start + Dx
            plate_y_start = -HH / 2
            plate_y_end = HH / 2

            if plate_x_start <= x <= plate_x_end and plate_y_start <= y <= plate_y_end:
                if i % 2 == 0:  # orange plate
                    return complex(V0, 0.0)
                else:  # blue plate
                    return complex(-V0, 0.0)

        return complex(0.0, 0.0)  # Outside plates, return 0

    def q_func(x, y):
        return complex(0.0, 0.0)

    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

    def calc_sigma(netz_obj, group_name, segment_indices):
        sigma = []

        # Dynamischer Abruf der korrekten Platte (z.B. "plate_0")
        seg = getattr(netz_obj, group_name).elements[segment_indices]

        # Look up the actual coordinates of these nodes and calculate their midpoints
        p1 = netz_obj.points[seg[:, 0]]
        p2 = netz_obj.points[seg[:, 1]]
        midpoints = (p1 + p2) / 2.0

        # Pass scalar floats into alpha_func
        for i in range(len(midpoints)):
            x, y = midpoints[i, 0], midpoints[i, 1]
            alpha = alpha_func(x, y)
            sigma.append(alpha / EPSILON_0)

        return np.array(sigma)

    # –-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----–-–----

    netz = load_mt_mesh("mesh_WS1819.npz")

    plist = np.asfortranarray(netz.points.astype(np.float64))
    tlist = np.asfortranarray(netz.Triangle.elements.astype(np.int32))
    rr = np.asfortranarray(netz.robinRand.elements.astype(np.int32).reshape(-1, 2))
    dd = np.ascontiguousarray(netz.plattenRand.elements.astype(np.int32).flatten())

    frequencies = np.logspace(1,8,15)

    f0 = 10 # 10Hz

    solver = fem_cpp.FEM_2D_complex(
        dd, rr, plist, tlist, alpha_func, alpha_func, beta_func, f_func, phi_func, gamma_func, q_func
    )
    solver.full_solve()
    sol = solver.get_Solution()

    rl, n_der, idx, _, _ = gm.NormalDerivative("plattenRand", netz, sol, "left")

    sigma = calc_sigma(netz, "plattenRand", idx)

    ladungsdichte = sigma * n_der

    qq = integrate.simpson(ladungsdichte, x=rl)

    print(qq)
    del solver
    gc.collect()

    plt.show()


if __name__ == "__main__":
    # aufg_a()
    print("------------------------- Aufgabe b) -------------------------")
    # aufg_b()
    aufg_d()
