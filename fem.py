import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numba


alpha = lambda x: x**2
beta = lambda x: x
f = lambda x: -1 * x**3
phi_a = 2
phi_b = 6
global_x_M = (phi_a + phi_b) / 2
phi = lambda x: phi_a if x < global_x_M else phi_b


def gen_tlist(plist):
    tlist_tmp = np.argsort(plist)
    tmp1 = tlist_tmp[0:-1]
    tmp2 = tlist_tmp[1:]
    tlist = np.array(list(zip(tmp1, tmp2)))
    return tlist


def gen_dR_phiR(plist):
    tmp_max_val_idx = np.where(plist == np.max(plist))
    tmp_min_val_idx = np.where(plist == np.min(plist))
    dR = np.array([tmp_min_val_idx[0][0], tmp_max_val_idx[0][0]])
    phiR = np.array([phi_a, phi_b])
    return dR, phiR


def gen_table(tlist, plist, dR, phiR):
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


def apply_boundary_conditions(K, D, dR, phiR):
    rand_a = K[:, dR[0]] * phiR[0]  # spalte von Rand a in der K-Matrix kopieren
    rand_b = K[:, dR[1]] * phiR[1]  # spalte von Rand b in der K-Matrix kopieren
    D -= (rand_a + rand_b)  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

    newK = np.delete(K, [dR[0], dR[1]], axis=1)  # spalte und Zeile von Rand a und b in der K-Matrix wegstreichen
    newK = np.delete(newK, [dR[0], dR[1]], axis=0)  # Zeile von Rand a in der K-Matrix wegstreichen

    newD = np.delete(D, [dR[0], dR[1]])  # Rand a in D Vector wegstreichen
    return newK, newD


def alt_apply_boundary_conditions(K, D, dR, phiR):
    rand_a = K[:, dR[0]] * phiR[0]  # spalte von Rand a in der K-Matrix kopieren
    rand_b = K[:, dR[1]] * phiR[1]  # spalte von Rand b in der K-Matrix kopieren
    D -= (
        rand_a + rand_b
    )  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

    K[:, dR[0]] = 0  # spalte von Rand a in der K-Matrix zu 0 setzen
    K[dR[0], :] = 0  # Zeile von Rand a in der K-Matrix zu 0 setzen
    K[:, dR[1]] = 0  # spalte von Rand b in der K-Matrix zu 0 setzen
    K[dR[1], :] = 0  # spalte von Rand b in der K-Matrix zu 0 setzen
    K[dR[0], dR[1]] = 1  # diagonale zu 1 setzen
    K[dR[1], dR[0]] = 1  # diagonale zu 1 setzen

    return K, D


def sort_into_matrix(
    plist: np.ndarray,
    tlist: np.ndarray,
    K11: np.ndarray,
    K12: np.ndarray,
    D1: np.ndarray,
):
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


# -------------------------------------------------------------------------------- Main Code --------------------------------------------------------------------------------


def main():
    # Tabellenberechnung
    plist = np.array([1.75, 2, 1.25, 1.0, 1.5])
    tlist = gen_tlist(plist)
    dR, phiR = gen_dR_phiR(plist)
    print("Punkte (plist):\n", plist)
    print("T-Liste (tlist):\n", tlist)
    print("dR:", dR)
    print("phiR:", phiR)
    table = gen_table(tlist, plist, dR, phiR)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.precision", 6)  # Controls decimal places
    print(table.to_string(index=False))

    K, D = sort_into_matrix(
        plist,
        tlist,
        table["K11 = K22"].to_numpy(),
        table["K12 = K21"].to_numpy(),
        table["D1 = D2"].to_numpy(),
    )
    K, D = apply_boundary_conditions(K, D, dR, phiR)

    # print("K Matrix:\n", K)
    # print("D Vector:\n", D)

    K = sp.csr_matrix(K)

    # solve LGS
    sol = sp.linalg.spsolve(K, D)  # Note D_sparse here is just D now
    # reconstruct the solution vector
    sol_phi = np.zeros(len(plist))
    # Hole die Indizes, an denen wir die Lösung berechnet haben (alles außer dR)
    free_indices = np.delete(np.arange(len(plist)), dR)
    # Fülle die Originalplätze mit der berechnenden Lösung auf
    sol_phi[free_indices] = sol
    # Fülle die Randwerte auf
    sol_phi[dR[0]] = phiR[0]
    sol_phi[dR[1]] = phiR[1]

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
