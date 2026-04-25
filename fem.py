import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from numba import jit
import time


# ------------------------- Randbedingungen --------------------------
xD = [1.0, 4.0]  # x-koordinaten der dirichlet boundary conditions
xR = []  # x-koordinaten der robin boundary conditions


# ------------------------- Funktionen --------------------------
@jit(nopython=True)
def alpha(x) -> float:
    if 1.5 <= x <= 2.7:
        return 3
    else:
        return x**2


@jit(nopython=True)
def beta(x) -> float:
    if 1 <= x <= 2:
        return x / (1 + x)
    else:
        return x**2


@jit(nopython=True)
def f(x) -> float:
    if 2 <= x <= 4:
        return x
    else:
        return 1 + x


global_xa = xR[0] if xR else xD[0]
global_xb = xR[1] if xR else xD[1]

global_x_M = (global_xa + global_xb) / 2


def phi(x):
    if 1 <= x <= 4:
        print(f"phi({x}) called")
        return np.exp(x)
    else:
        print(f"phi({x}) called but x is out of bounds")
        return None


def gamma(x):
    pass


def q(x):
    pass


# ------------------------- Berechnung -------------------------


def gen_tlist(plist: np.ndarray) -> np.ndarray:
    tlist_tmp = np.argsort(plist)
    tmp1 = tlist_tmp[0:-1]
    tmp2 = tlist_tmp[1:]
    tlist = np.array(list(zip(tmp1, tmp2)))
    return tlist


def gen_randwerte_liste(plist: np.ndarray) -> np.ndarray:
    randelemente = []
    for val in xD + xR:
        # Check for matching coordinates in plist
        idx = np.where(np.isclose(plist, val))[0]
        if len(idx) > 0:
            randelemente.append(idx[0])
    return np.array(randelemente)


def gen_table(tlist: np.ndarray, plist: np.ndarray) -> pd.DataFrame:
    table = []
    for t in tlist:
        x1 = plist[t[0]]
        x2 = plist[t[1]]
        L_E = x2 - x1
        x_M = (x1 + x2) / 2
        alpha_M = alpha(x_M)
        beta_M = beta(x_M)
        f_M = f(x_M)
        K11 = (alpha_M / L_E) + (L_E * beta_M / 3)
        # K22 = K11

        K12 = (L_E * beta_M / 6) - (alpha_M / L_E)
        # K21 = K12

        D1 = L_E * f_M / 2
        # D2 = D1

        table.append(
            {
                "t": t,
                "x1 ; x2": (x1, x2),
                "L_E": L_E,
                "x_M": x_M,
                "alpha(x_M)": alpha_M,
                "beta(x_M)": beta_M,
                "f(x_M)": f_M,
                "K11 = K22": K11,  # 11 and 22 are local indexes, the global indexes are dependent on the corresponding t-element
                "K12 = K21": K12,  # 12 and 21 are local indexes, the global indexes are dependent on the corresponding t-element
                "D1 = D2": D1,
            }
        )
    table = pd.DataFrame(table)
    return table


@jit(nopython=True)
def gen_necessary_data(
    tlist: np.ndarray, plist: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # only gen K11, K12 and D1
    K11 = []
    K12 = []
    D1 = []
    for t in tlist:
        x1 = plist[t[0]]
        x2 = plist[t[1]]
        L_E = x2 - x1
        x_M = (x1 + x2) / 2
        alpha_M = alpha(x_M)
        beta_M = beta(x_M)
        f_M = f(x_M)
        K11.append((alpha_M / L_E) + (L_E * beta_M / 3))
        # K22 = K11

        K12.append((L_E * beta_M / 6) - (alpha_M / L_E))
        # K21 = K12

        D1.append(L_E * f_M / 2)
        # D2 = D1

    return K11, K12, D1


def sort_into_matrix(
    plist: np.ndarray,
    tlist: np.ndarray,
    K11: np.ndarray,
    K12: np.ndarray,
    D1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    K = np.zeros((len(plist), len(plist)))
    D = np.zeros(len(plist))

    for i in range(len(tlist)):
        t = tlist[i]
        K[t[0], t[0]] += K11[i]  # K11 (local index 1)
        K[t[1], t[1]] += K11[i]  # K22 (local index 2)
        K[t[0], t[1]] += K12[i]  # K12 (local index 1,2)
        K[t[1], t[0]] += K12[i]  # K21 (local index 2,1)
        D[t[0]] += D1[i]
        D[t[1]] += D1[i]

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
    actualRE = [re for re in randelemente if plist[re] in xD]
    rand_re = []

    for re in actualRE:
        phi_re = phi(plist[re])
        rand_re.append(K[:, re] * phi_re)  # spalte von Rand re in der K-Matrix kopieren

    D -= np.sum(
        rand_re, axis=0
    )  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

    newK = np.delete(
        K, [re for re in actualRE], axis=1
    )  # Spalte von Rand a und b in der K-Matrix wegstreichen
    newK = np.delete(
        newK, [re for re in actualRE], axis=0
    )  # Zeile von Rand a in der K-Matrix wegstreichen

    newD = np.delete(D, [re for re in actualRE])  # Rand a in D Vector wegstreichen
    return newK, newD


def alt_apply_dirichlet_boundary_conditions(
    K: np.ndarray, D: np.ndarray, randelemente: np.ndarray, plist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actualRE = [re for re in randelemente if plist[re] in xD]

    # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden
    for re in actualRE:
        phi_val = phi(plist[re])
        D -= K[:, re] * phi_val

    # spalte und Zeile von Rand zu 0 setzen, diagonale zu 1 und D auf korrekten Wert
    for re in actualRE:
        K[:, re] = 0
        K[re, :] = 0
        K[re, re] = 1
        D[re] = phi(plist[re])

    return K, D


def reconstruct_sol(
    sol: np.ndarray, plist: np.ndarray, randelemente: np.ndarray
) -> np.ndarray:
    if len(sol) == len(plist):
        return sol  # no reconstruction needed if the solution already has the same length as plist

    actualRE = [re for re in randelemente if plist[re] in xD]
    sol_new = np.zeros(len(plist))
    free_indices = np.delete(np.arange(len(plist)), actualRE)

    sol_new[free_indices] = sol  # fill with solution from LGS
    for re in actualRE:
        sol_new[re] = phi(plist[re])

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


def show_table(tlist: np.ndarray, plist: np.ndarray):
    table = gen_table(tlist, plist)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 6)  # Controls decimal places
    print("-" * 100)
    print(table.to_string(index=False))
    print("-" * 100)


# ------------------------- Validierung mit Weizi Data -------------------------
def validate_with_weizi_data(K, D, sol, plist):
    # Load Weizi data
    # wK = np.loadtxt("tst_1D/Netz1D_Matrix_K.dat", dtype=float)
    # wD = np.loadtxt("tst_1D/Netz1D_D.dat", dtype=float)
    wSol = np.loadtxt("tst_1D/Netz1D_LoesungA.dat", dtype=float)

    # Compare K
    plt.figure(figsize=(12, 5))
    plt.plot(
        plist, np.abs(wSol - sol), marker="o", linestyle="", label="Difference in Solution"
    )
    plt.xlabel("Punkte")
    plt.ylabel("Differenz")
    plt.title("Validierung mit Weizi Data")
    plt.legend()
    plt.grid()


# -------------------------------------------------------------------------------- Main Code --------------------------------------------------------------------------------


def main():
    # ----------- Print Randbedingungen -----------
    print_RB()

    # ----------- Erstellung P-Liste, T-Liste, Randelemente -----------
    # plist = np.array([1.75, 2, 1.25, 1.0, 1.5])
    plist = np.loadtxt("tst_1D/Netz1D_p.dat", dtype=float)
    start_time = time.time()
    tlist = gen_tlist(plist)
    end_time = time.time()
    print(f"Time taken to generate tlist: {end_time - start_time:.6f} seconds")
    randelemente = gen_randwerte_liste(plist)
    # print("Punkte (plist):\n", plist)
    # print("T-Liste (tlist):\n", tlist)
    print("randelemente:", randelemente)

    # ----------- Berechnung der notwendigen Daten für die Matrix -----------
    start_time = time.time()
    K11, K12, D1 = gen_necessary_data(tlist, plist)
    end_time = time.time()
    print(f"Time taken to generate necessary data: {end_time - start_time:.6f} seconds")
    K11 = np.array(K11)
    K12 = np.array(K12)
    D1 = np.array(D1)

    # ----------- Sortieren der Daten in die Matrix -----------
    start_time = time.time()
    K, D = sort_into_matrix(plist, tlist, K11, K12, D1)

    # ----------- Anwendung der Randbedingungen -----------
    K, D = apply_robin_boundary_conditions(K, D, randelemente, plist)
    K, D = apply_dirichlet_boundary_conditions(K, D, randelemente, plist)
    end_time = time.time()
    print(
        f"Time taken to sort into matrix and apply boundary conditions: {end_time - start_time:.6f} seconds"
    )
    # K, D = alt_apply_dirichlet_boundary_conditions(K, D, randelemente, plist)

    # print("K Matrix:\n", K)
    # print("D Vector:\n", D)

    # ----------- Lösen des LGS -----------
    # convert K to sparse matrix
    start_time = time.time()
    K = sp.csr_matrix(K)
    # solve LGS
    sol = sp.linalg.spsolve(K, D)  # Note D_sparse here is just D now
    end_time = time.time()
    print(f"Time taken to solve the LGS: {end_time - start_time:.6f} seconds")

    # ----------- Rekonstruktion der Lösung für alle Punkte in plist -----------
    sol = reconstruct_sol(sol, plist, randelemente)

    # ----------- Plotten der Lösung -----------
    # print("Lösung des LGS (phi):\n", sol)
    plot_sol(plist, sol)

    # ----------- Validierung mit Weizi Data -----------
    validate_with_weizi_data(K, D, sol, plist)

    plt.show()


# ---------------------


if __name__ == "__main__":
    main()
