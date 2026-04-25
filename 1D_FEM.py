import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from numba import float64, jit, vectorize
import time

# -------------------------------- Anpassen je nach Aufgabe --------------------------------
# ------------------------- Randbedingungen --------------------------
xD = [1.0, 4.0]  # x-koordinaten der dirichlet boundary conditions
xR = []  # x-koordinaten der robin boundary conditions


@vectorize([float64(float64)])
def alpha(x):
    if 1.5 <= x <= 2.7:
        return 3
    else:
        return x**2


@vectorize([float64(float64)])
def beta(x):
    if 1 <= x <= 2:
        return x / (1 + x)
    else:
        return x**2


@vectorize([float64(float64)])
def f(x):
    if 2 <= x <= 4:
        return x
    else:
        return 1 + x


def phi(x):
    if 1 <= x <= 4:
        return np.exp(x)
    else:
        return None


def gamma(x):
    pass


def q(x):
    pass


class fem_1d:
    def __init__(self, xD, xR, plist):
        self.dR = xR
        self.dD = xD
        self.plist = plist
        self.tlist = None
        self.K = None
        self.D = None
        self.sol = None

    def gen_tlist(self):
        tlist_tmp = np.argsort(self.plist)
        tmp1 = tlist_tmp[0:-1]
        tmp2 = tlist_tmp[1:]
        self.tlist = np.array(list(zip(tmp1, tmp2)))

    def gen_randwerte_list(self) -> np.ndarray:
        randelemente = []
        for val in xD + xR:
            # Check for matching coordinates in plist
            idx = np.where(np.isclose(plist, val))[0]
            if len(idx) > 0:
                randelemente.append(idx[0])
        return np.array(randelemente)

    def gen_necessary_data(self
    ):

        x1 = self.plist[self.tlist[:, 0]]
        x2 = self.plist[self.tlist[:, 1]]
        L_E = x2 - x1
        x_M = (x1 + x2) / 2

        alpha_M = alpha(x_M)
        beta_M = beta(x_M)
        f_M = f(x_M)

        K11 = (alpha_M / L_E) + (L_E * beta_M / 3)

        K12 = (L_E * beta_M / 6) - (alpha_M / L_E)

        D1 = L_E * f_M / 2

        return K11, K12, D1

    def sort_into_matrix(
        self,
        K11: np.ndarray,
        K12: np.ndarray,
        D1: np.ndarray,
    ):
        len_plist = len(self.plist)
        K = np.zeros((len_plist, len_plist))
        D = np.zeros(len_plist)

        for i in range(len(self.tlist)):
            t = self.tlist[i]
            K[t[0], t[0]] += K11[i]  # K11 (local index 1)
            K[t[1], t[1]] += K11[i]  # K22 (local index 2)
            K[t[0], t[1]] += K12[i]  # K12 (local index 1,2)
            K[t[1], t[0]] += K12[i]  # K21 (local index 2,1)
            D[t[0]] += D1[i]
            D[t[1]] += D1[i]

        self.K = K
        self.D = D
        # print("K Matrix ohne Randbedingung:\n", K)
        # print("D Vector ohne Randbedingung:\n", D)

        return K, D

    def apply_robin_boundary_conditions(self, randelemente: np.ndarray):

        # sort out randlelemente for Robin RW (check if they are in xR)
        actualRE = [re for re in randelemente if self.plist[re] in xR]

        for re in actualRE:
            gamma_val = gamma(self.plist[re])
            q_val = q(self.plist[re])

            # adjust K and D for Robin BCs
            self.K[re, re] += gamma_val
            self.D[re] += q_val

        # self.K = self.K
        # self.D = self.D

        # return K, D

    def apply_dirichlet_boundary_conditions(self, randelemente: np.ndarray):
        actualRE = [re for re in randelemente if self.plist[re] in xD]
        rand_re = []

        for re in actualRE:
            phi_re = phi(self.plist[re])
            rand_re.append(
                self.K[:, re] * phi_re
            )  # spalte von Rand re in der K-Matrix kopieren

        self.D -= np.sum(
            rand_re, axis=0
        )  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

        newK = np.delete(
            self.K, [re for re in actualRE], axis=1
        )  # Spalte von Rand a und b in der K-Matrix wegstreichen
        newK = np.delete(
            newK, [re for re in actualRE], axis=0
        )  # Zeile von Rand a in der K-Matrix wegstreichen

        newD = np.delete(self.D, [re for re in actualRE])  # Rand a in D Vector wegstreichen

        self.K = newK
        self.D = newD

        # return newK, newD

    def solve_LGS(self):
        K_Sparse = sp.csr_matrix(self.K)
        self.sol = sp.linalg.spsolve(K_Sparse, self.D)
        # return sp.linalg.spsolve(K_Sparse, self.D)

    def reconstruct_solution(self, randelemente: np.ndarray):
        # numpy insert
        if len(self.sol) == len(self.plist):
            return

        actualRE = [re for re in randelemente if self.plist[re] in xD]
        sol_new = np.zeros(len(self.plist))
        free_indices = np.delete(np.arange(len(self.plist)), actualRE)

        sol_new[free_indices] = self.sol  # fill with solution from LGS

        sol_new[actualRE] = [phi(x) for x in self.plist[actualRE]]  # fill with Dirichlet RW values

        self.sol = sol_new
        # return sol_new
    
    def visualize_solution(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.plist, self.sol, color="blue", label="Lösung (phi)")
        plt.title("Lösung der DGL mit 1D FEM")
        plt.xlabel("Punkte (plist)")
        plt.ylabel("Lösung (phi)")
        plt.grid()
        plt.legend()

    def full_solve(self):
        t0 = time.time()
        
        t1 = time.time()
        self.gen_tlist()
        t_gen_tlist = time.time() - t1
        
        t1 = time.time()
        K11, K12, D1 = self.gen_necessary_data()
        t_gen_data = time.time() - t1
        
        t1 = time.time()
        self.sort_into_matrix(K11, K12, D1)
        t_sort = time.time() - t1
        
        t1 = time.time()
        randelemente = self.gen_randwerte_list()
        t_rand = time.time() - t1
        
        t1 = time.time()
        self.apply_robin_boundary_conditions(randelemente)
        t_robin = time.time() - t1
        
        t1 = time.time()
        self.apply_dirichlet_boundary_conditions(randelemente)
        t_dirich = time.time() - t1
        
        t1 = time.time()
        self.solve_LGS()
        t_solve = time.time() - t1
        
        t1 = time.time()
        self.reconstruct_solution(randelemente)
        t_recon = time.time() - t1
        
        t_total = time.time() - t0
        
        print(f"------------- Timings für {len(self.plist)} Elemente -------------")
        print(f"gen_tlist:                  {t_gen_tlist:.6f} s")
        print(f"gen_necessary_data:         {t_gen_data:.6f} s")
        print(f"sort_into_matrix:           {t_sort:.6f} s")
        print(f"gen_randwerte_list:         {t_rand:.6f} s")
        print(f"apply_robin_bc:             {t_robin:.6f} s")
        print(f"apply_dirichlet_bc:         {t_dirich:.6f} s")
        print(f"solve_LGS:                  {t_solve:.6f} s")
        print(f"reconstruct_solution:       {t_recon:.6f} s")
        print(f"TOTAL (excl. visualization): {t_total:.6f} s")
        print("-----------------------------------")
        
        self.visualize_solution()
        plt.show()



plist = np.loadtxt("tst_1D/Netz1D_p.dat", dtype=float)
fem_solver = fem_1d(xD, xR, plist)
fem_solver.full_solve()
