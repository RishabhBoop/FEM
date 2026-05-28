import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from numba import float64, vectorize
import time
import gmsh

# ------------------------- Randbedingungen --------------------------
# xD = [1.0, 4.0]  # x-koordinaten der dirichlet boundary conditions
xD = [1.0, 0.0, 0.0, 1.0]  # x-koordinaten der dirichlet boundary conditions
xR = []  # x-koordinaten der robin boundary conditions


# ------------------------- Funktionen --------------------------
# @vectorize([float64(float64)])
# def alpha(x):
# if 1.5 <= x <= 2.7:
# return 3
# else:
# return x**2
def alpha1(x, y):
    return y * x + 1


def alpha2(x, y):
    return x + y + 1


def beta(x, y):
    return 2 * x**2


def f(x, y):
    return x + y**2


global_xa = xR[0] if xR else xD[0]
global_xb = xR[1] if xR else xD[1]

# global_x_M = (global_xa + global_xb) / 2


def phi(x, y):
    return x**2 + y


# def gamma(x):
#     pass


# def q(x):
#     pass


# ------------------------- Berechnung -------------------------


# def gen_tlist(plist: np.ndarray) -> np.ndarray:
#     tlist_tmp = np.argsort(plist)
#     tmp1 = tlist_tmp[0:-1]
#     tmp2 = tlist_tmp[1:]
#     tlist = np.array(list(zip(tmp1, tmp2)))
#     return tlist

# def gen_2d_tlist(plist: np.ndarray) -> np.ndarray:
#     triangles = []
#     for i in range(N0 - 1):
#         for j in range(N0 - 1):
#             p1 = i * N0 + j
#             p2 = p1 + 1
#             p3 = p1 + N0
#             p4 = p3 + 1
#             triangles += [[p1, p2, p3], [p2, p4, p3]]
#             plt.triplot(plist[:, 0], plist[:, 1], [[p1, p2, p3]], marker="o", color="red")
#             plt.pause(interval=0.2)
#             plt.triplot(plist[:, 0], plist[:, 1], [[p2, p4, p3]], marker="o", color="red")
#             plt.pause(interval=0.2)

#     triangles = np.array(triangles)


# def gen_randwerte_liste(plist: np.ndarray) -> np.ndarray:
#     randelemente = []
#     for val in xD + xR:
#         # Check for matching coordinates in plist
#         idx = np.where(np.isclose(plist, val))[0]
#         if len(idx) > 0:
#             randelemente.append(idx[0])
#     return np.array(randelemente)


# def gen_table(tlist: np.ndarray, plist: np.ndarray) -> pd.DataFrame:
#     table = []
#     for t in tlist:
#         x1 = plist[t[0]]
#         x2 = plist[t[1]]
#         L_E = x2 - x1
#         x_M = (x1 + x2) / 2
#         alpha_M = alpha(x_M)
#         beta_M = beta(x_M)
#         f_M = f(x_M)
#         K11 = (alpha_M / L_E) + (L_E * beta_M / 3)
#         # K22 = K11

#         K12 = (L_E * beta_M / 6) - (alpha_M / L_E)
#         # K21 = K12

#         D1 = L_E * f_M / 2
#         # D2 = D1

#         table.append(
#             {
#                 "t": t,
#                 "x1 ; x2": (x1, x2),
#                 "L_E": L_E,
#                 "x_M": x_M,
#                 "alpha(x_M)": alpha_M,
#                 "beta(x_M)": beta_M,
#                 "f(x_M)": f_M,
#                 "K11 = K22": K11,  # 11 and 22 are local indexes, the global indexes are dependent on the corresponding t-element
#                 "K12 = K21": K12,  # 12 and 21 are local indexes, the global indexes are dependent on the corresponding t-element
#                 "D1 = D2": D1,
#             }
#         )
#     table = pd.DataFrame(table)
#     return table


def gen_b(y):
    # Slice the nodes (columns)
    b1 = y[:, 1] - y[:, 2]
    b2 = y[:, 2] - y[:, 0]
    b3 = y[:, 0] - y[:, 1]
    return np.stack([b1, b2, b3], axis=1)  # stack into rows


def gen_c(x):
    # Slice the nodes (columns)
    c1 = x[:, 2] - x[:, 1]
    c2 = x[:, 0] - x[:, 2]
    c3 = x[:, 1] - x[:, 0]
    return np.stack([c1, c2, c3], axis=1)  # stack into rows


def gen_area(p):
    twodeltaE = (p[:, 0, 0] - p[:, 2, 0]) * (p[:, 1, 1] - p[:, 2, 1]) - (p[:, 1, 0] - p[:, 2, 0]) * (
        p[:, 0, 1] - p[:, 2, 1]
    )
    return twodeltaE / 2


def gen_2area(p):
    # 2deltaE = (x1-x3)(y2-y3) - (x2-x3)(y1-y2)
    twodeltaE = (p[:, 0, 0] - p[:, 2, 0]) * (p[:, 1, 1] - p[:, 2, 1]) - (p[:, 1, 0] - p[:, 2, 0]) * (
        p[:, 0, 1] - p[:, 2, 1]
    )
    return twodeltaE


def gen_necessary_data(tlist: np.ndarray, plist: np.ndarray) -> tuple:
    p = plist[tlist]
    bj = gen_b(p[:, :, 1])
    cj = gen_c(p[:, :, 0])
    twodeltae = gen_2area(p)
    
    xm = np.mean(p[:, :, 0], axis=1)
    ym = np.mean(p[:, :, 1], axis=1)
    a1m = alpha1(xm, ym)
    a2m = alpha2(xm, ym)
    betam = beta(xm, ym)
    fm = f(xm, ym)

    # Calculate the scalar coefficients for each triangle
    coeff_b = a1m / (2 * twodeltae)
    coeff_c = a2m / (2 * twodeltae)
    coeff_beta = (betam * twodeltae) / 24

    # Extract the individual columns
    # Shape of each of these is (N,)
    b0, b1, b2 = bj[:, 0], bj[:, 1], bj[:, 2]
    c0, c1, c2 = cj[:, 0], cj[:, 1], cj[:, 2]

    # tmpmat = np.array([2, 1, 1, 1, 2, 1, 1, 1, 2]).reshape(3, 3) 
    # Diagonals (tmpmat value = 2)
    K11 = coeff_b * (b0 * b0) + coeff_c * (c0 * c0) + coeff_beta * 2
    K22 = coeff_b * (b1 * b1) + coeff_c * (c1 * c1) + coeff_beta * 2
    K33 = coeff_b * (b2 * b2) + coeff_c * (c2 * c2) + coeff_beta * 2

    # Non-diagonals (tmpmat value = 1)
    K12 = coeff_b * (b0 * b1) + coeff_c * (c0 * c1) + coeff_beta * 1
    K13 = coeff_b * (b0 * b2) + coeff_c * (c0 * c2) + coeff_beta * 1
    K23 = coeff_b * (b1 * b2) + coeff_c * (c1 * c2) + coeff_beta * 1

    Dloc_val = (fm * gen_area(p) / 3) 
    Dloc = np.column_stack((Dloc_val, Dloc_val, Dloc_val)) # Shape (N, 3)

    return K11, K22, K33, K12, K13, K23, Dloc


def sort_into_matrix(
    plist: np.ndarray,
    tlist: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    K33: np.ndarray,
    K12: np.ndarray,
    K13: np.ndarray,
    K23: np.ndarray,
    D1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    K = np.zeros((len(plist), len(plist)))
    D = np.zeros(len(plist))
    # K11 = K_local[:, 0, 0]
    # K12 = K_local[:, 0, 1]
    # K13 = K_local[:, 0, 2]
    # K22 = K_local[:, 1, 1]
    # K23 = K_local[:, 1, 2]
    # K33 = K_local[:, 2, 2]
    D1 = D1[:, 0, 0]
    for i in range(len(tlist)):
        t = tlist[i]

        K[t[0], t[0]] += K11[i]
        K[t[1], t[1]] += K22[i]
        K[t[2], t[2]] += K33[i]

        K[t[0], t[1]] += K12[i]
        K[t[0], t[2]] += K13[i]
        K[t[1], t[2]] += K23[i]

        K[t[1], t[0]] += K12[i]
        K[t[2], t[0]] += K13[i]
        K[t[2], t[1]] += K23[i]

        D[t[0]] += D1[i]
        D[t[1]] += D1[i]
        D[t[2]] += D1[i]

    # print("K Matrix ohne Randbedingung:\n", K)
    # print("D Vector ohne Randbedingung:\n", D)

    return K, D


def apply_robin_boundary_conditions(
    K: np.ndarray, D: np.ndarray, randelemente: np.ndarray, plist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    # sort out randlelemente for Robin RW (check if they are in xR)
    actualRE = [re for re in randelemente if plist[re] in xR]

    for re in actualRE:
        gamma_val = gamma(plist[re])
        q_val = q(plist[re])

        # adjust K and D for Robin BCs
        K[re, re] += gamma_val
        D[re] += q_val

    return K, D


def apply_dirichlet_boundary_conditions(
    K: np.ndarray, D: np.ndarray, randelemente: np.ndarray, plist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    # print(K)
    # print(randelemente)
    # auf rechte seite bringen
    for re in randelemente:
        phi_val = phi(plist[re][0], plist[re][1])
        D -= K[:, re] * phi_val

    # wegstreichen
    newK = np.delete(K, [re for re in randelemente], axis=1)  # Spalte von Rand a und b in der K-Matrix wegstreichen
    newK = np.delete(newK, [re for re in randelemente], axis=0)  # Zeile von Rand a in der K-Matrix wegstreichen

    newD = np.delete(D, [re for re in randelemente])  # Rand a in D Vector wegstreichen
    return newK, newD


def alt_apply_dirichlet_boundary_conditions(
    K: np.ndarray, D: np.ndarray, randelemente: np.ndarray, plist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    # determine actual randelemente by matching x-coordinate (plist entries are [x,y])
    actualRE = [re for re in randelemente if plist[re, 0] in xD]

    # adjust D vector since Dirichlet BCs will be imposed in K
    for re in actualRE:
        phi_val = phi(plist[re, 0], plist[re, 1])
        D -= K[:, re] * phi_val

    # set rows/cols to enforce Dirichlet: zero row/col, set diagonal to 1 and D to prescribed value
    for re in actualRE:
        K[:, re] = 0
        K[re, :] = 0
        K[re, re] = 1
        D[re] = phi(plist[re, 0], plist[re, 1])

    return K, D


def reconstruct_sol(sol: np.ndarray, plist: np.ndarray, randelemente: np.ndarray) -> np.ndarray:
    if len(sol) == len(plist):
        return sol  # no reconstruction needed

    # Identify Dirichlet boundary nodes based on x-coordinate
    actualRE = [re for re in randelemente if plist[re, 0] in xD]

    # Allocate full solution array
    sol_new = np.zeros(len(plist))
    free_indices = np.delete(np.arange(len(plist)), actualRE)

    # Fill free nodes with the solution from LGS
    sol_new[free_indices] = sol

    # Fill Dirichlet boundary nodes using 2D coordinates
    for re in actualRE:
        x_coord = plist[re, 0]
        y_coord = plist[re, 1]
        sol_new[re] = phi(x_coord, y_coord)

    return sol_new


# ------------------------- Printing and Visualization functions -------------------------
def print_RB():
    if xD:
        print("Dirichlet-Randbedingungen:")
        for x in xD:
            print(f"\tphi({x}) = {phi(x)}")

    if xR:
        print("Robin-Randbedingungen:\n")
        for x in xR:
            print(f"\tgamma({x}) = {gamma(x)}, q({x}) = {q(x)}")


def plot_sol(plist: np.ndarray, sol_phi: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.scatter(plist, sol_phi, color="blue", label="Lösung (phi)")
    plt.title("Lösung der DGL an den Punkten")
    plt.xlabel("Punkte (plist)")
    plt.ylabel("Lösung (phi)")
    plt.grid()
    plt.legend()
    # plt.show()


# ------------------------- Validierung mit Weizi Data -------------------------
def validate_with_weizi_data(K, D, sol, plist):
    # Load Weizi data
    # wK = np.loadtxt("tst_1D/Netz1D_Matrix_K.dat", dtype=float)
    # wD = np.loadtxt("tst_1D/Netz1D_D.dat", dtype=float)
    wSol = np.loadtxt("tst_1D/Netz1D_LoesungA.dat", dtype=float)

    error = wSol - sol
    error = np.abs(error)

    # print errors
    print(f"Maximale Abweichung in K: {np.max(error):.6e}")
    print(f"Minimale Abweichung in K: {np.min(error):.6e}")
    print(f"Mittlere Abweichung in K: {np.mean(error):.6e}")

    # Compare K
    plt.figure(figsize=(12, 5))
    plt.plot(plist, error, marker="o", linestyle="", label="Difference in Solution")
    plt.xlabel("Punkte")
    plt.ylabel("Differenz")
    plt.title("Validierung mit Weizi Data")
    plt.legend()
    plt.grid()


# -------------------------------------------------------------------------------- Main Code --------------------------------------------------------------------------------


def main():
    # generate mesh
    ## To this later with gmsh
    plist = np.array([[1, 0.7], [0.5, 0.35], [0.5, 0.18], [0, 0], [0, 0.7], [1, 0]])
    tlist = np.array(
        [[0, 4, 1], [3, 1, 4], [5, 0, 1], [3, 5, 2], [2, 1, 3], [5, 1, 2]]
    )  # tlist will be outputted from gmsh, for now we just hardcode it
    dr = [0, 4, 3, 5]
    xD = [plist[i, 0] for i in dr]
    print("xD =", xD)
    K11, K22, K33, K12, K13, K23, Dloc = gen_necessary_data(tlist, plist)
    K, D = sort_into_matrix(plist, tlist, K11, K22, K33, K12, K13, K23, Dloc)
    K, D = apply_dirichlet_boundary_conditions(K, D, dr, plist)
    print("K Matrix mit Randbedingung:\n", K)
    print("D Vector mit Randbedingung:\n", D)

    K = sp.csr_matrix(K)  # convert to sparse format for efficient solving
    sol = sp.linalg.spsolve(K, D)  # solve the linear system
    sol_full = reconstruct_sol(sol, plist, dr)
    print(sol_full)


# ---------------------


if __name__ == "__main__":
    main()
