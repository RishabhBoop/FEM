import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from numba import jit

# ------------------------- Randbedingungen --------------------------
xD = []  # x-koordinaten der dirichlet boundary conditions
xR = [1, 2]  # x-koordinaten der robin boundary conditions
dD = []  # dirichlet boundary conditions
dR = [(3, 4), (6, 7)]  # Robin boundary conditions (tuple (gamma, q))


# ------------------------- Funktionen --------------------------
@jit(nopython=True)
def alpha(x):
    return x**2


@jit(nopython=True)
def beta(x):
    return x


@jit(nopython=True)
def f(x):
    return -1 * x**3


global_xa = xR[0] if xR else xD[0]
global_xb = xR[1] if xR else xD[1]

global_x_M = (global_xa + global_xb) / 2


def phi(x):
    return dD[0] if x < global_x_M else dD[1]


def gamma(x):
    if x < global_x_M:
        return dR[0][0]
    else:
        return dR[0][1]


def q(x):
    if x < global_x_M:
        return dR[1][0]
    else:
        return dR[1][1]


# ------------------------- Berechnung -------------------------


def gen_tlist(plist: np.ndarray) -> np.ndarray:
    tlist_tmp = np.argsort(plist)
    tmp1 = tlist_tmp[0:-1]
    tmp2 = tlist_tmp[1:]
    tlist = np.array(list(zip(tmp1, tmp2)))
    return tlist


def gen_randwerte_liste(plist: np.ndarray) -> np.ndarray:
    tmp_max_val_idx = np.where(plist == np.max(plist))
    tmp_min_val_idx = np.where(plist == np.min(plist))
    randelemente = np.array([tmp_min_val_idx[0][0], tmp_max_val_idx[0][0]])
    return randelemente


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
    
    D -= (np.sum(rand_re, axis=0))  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

    newK = np.delete(
        K, [re for re in actualRE], axis=1
    )  # Spalte von Rand a und b in der K-Matrix wegstreichen
    newK = np.delete(
        newK, [re for re in actualRE], axis=0
    )  # Zeile von Rand a in der K-Matrix wegstreichen

    newD = np.delete(
        D, [re for re in actualRE]
    )  # Rand a in D Vector wegstreichen
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


# -------------------------------------------------------------------------------- Main Code --------------------------------------------------------------------------------


def main():
    # ----------- Print Randbedingungen -----------
    if dD:
        print("Dirichlet-Randbedingungen:")
        for i in range(len(xD)):
            print(f"\tphi({xD[i]}) = {dD[i]}")

    if dR:
        print("Robin-Randbedingungen:\n")
        for i in range(len(xR)):
            print(f"\tgamma({xR[i]}) = {dR[i][0]}, q({xR[i]}) = {dR[i][1]}")

    # ----------- Erstellung P-Liste, T-Liste, Randelemente -----------
    plist = np.array([1.75, 2, 1.25, 1.0, 1.5])
    tlist = gen_tlist(plist)
    randelemente = gen_randwerte_liste(plist)
    # print("Punkte (plist):\n", plist)
    # print("T-Liste (tlist):\n", tlist)
    # print("randwerte:", randwerte)
    # table = gen_table(tlist, plist)

    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", 1000)
    # pd.set_option("display.precision", 6)  # Controls decimal places
    # print('-' * 100)
    # print(table.to_string(index=False))
    # print('-' * 100)

    # ----------- Berechnung der notwendigen Daten für die Matrix -----------
    K11, K12, D1 = gen_necessary_data(tlist, plist)
    K11 = np.array(K11)
    K12 = np.array(K12)
    D1 = np.array(D1)

    # ----------- Sortieren der Daten in die Matrix -----------
    K, D = sort_into_matrix(plist, tlist, K11, K12, D1)

    # ----------- Anwendung der Randbedingungen -----------
    K, D = apply_robin_boundary_conditions(K, D, randelemente, plist)
    K, D = apply_dirichlet_boundary_conditions(K, D, randelemente, plist)
    # K, D = alt_apply_dirichlet_boundary_conditions(K, D, randelemente, plist)

    # print("K Matrix:\n", K)
    # print("D Vector:\n", D)

    # ----------- Lösen des LGS -----------
    # convert K to sparse matrix
    K = sp.csr_matrix(K)
    # solve LGS
    sol = sp.linalg.spsolve(K, D)  # Note D_sparse here is just D now

    # ----------- Rekonstruktion der Lösung für alle Punkte in plist -----------
    sol_phi = reconstruct_sol(sol, plist, randelemente)

    print("Lösung des LGS (phi):\n", sol_phi)

    # show in scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(plist, sol_phi, color="blue", label="Lösung (phi)")
    plt.title("Lösung des LGS (phi) an den Punkten in plist")
    plt.xlabel("Punkte (plist)")
    plt.ylabel("Lösung (phi)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
