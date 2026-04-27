import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numba import jit
import time
from typing import Callable

# -------------------------------- Global Functions --------------------------------


def gen_randwerte_list(xD, xR, plist):
    randelemente = []
    for val in xD + xR:
        # Check for matching coordinates in plist
        idx = np.where(np.isclose(plist, val))[0]
        if len(idx) > 0:
            randelemente.append(idx[0])
    return np.array(randelemente)


@jit(nopython=True)
def vec_sort_into_matrix(K11: np.ndarray, K12: np.ndarray, D1: np.ndarray, tlist: np.ndarray, plist_len: int):
    K = np.zeros((plist_len, plist_len))
    D = np.zeros(plist_len)

    for i in range(len(tlist)):
        t = tlist[i]
        K[t[0], t[0]] += K11[i]  # K11 (local index 1)
        K[t[1], t[1]] += K11[i]  # K22 (local index 2)
        K[t[0], t[1]] += K12[i]  # K12 (local index 1,2)
        K[t[1], t[0]] += K12[i]  # K21 (local index 2,1)
        D[t[0]] += D1[i]
        D[t[1]] += D1[i]

    return K, D


# ------------------------------------------------------------------------------------------------


class fem_1d:
    def __init__(
        self,
        xD: list,
        xR: list,
        plist: np.ndarray,
        alpha: Callable,
        beta: Callable,
        f: Callable,
        phi: Callable,
        gamma: Callable,
        q: Callable,
    ):
        """
        Class for a 1D FEM Solution.
        Takes all necessary data and functions as input and provides methods to solve the FEM problem, apply boundary conditions, and visualize the solution.

        Args:
            xD (list): list of x-coordinates for Dirichlet boundary conditions
            xR (list): list of x-coordinates for Robin boundary conditions
            plist (np.ndarray): numpy array of points in the domain
            alpha (Callable): alpha-part of the partial differential equation
            beta (Callable): beta-part of the partial differential equation
            f (Callable): f-part of the partial differential equation
            phi (Callable): phi function for Dirichlet boundary conditions
            gamma (Callable): gamma function for Robin boundary conditions
            q (Callable): q function for Robin boundary conditions
        """
        # --- Data needed to apply Randbedingungen ---
        self.xR = xR
        self.xD = xD
        self.randelemente = gen_randwerte_list(xD, xR, plist)
        # --- Data needed to solve ---
        self.plist = plist
        self.tlist = None
        self.K = None
        self.D = None
        self.sol = None
        # --- Functions ---
        self.alpha = alpha
        self.beta = beta
        self.f = f
        self.phi = phi
        self.gamma = gamma
        self.q = q

    def gen_tlist(self):
        """
        generates a t-list from the object's p-list
        """
        tlist_tmp = np.argsort(self.plist)
        tmp1 = tlist_tmp[0:-1]
        tmp2 = tlist_tmp[1:]
        self.tlist = np.array(list(zip(tmp1, tmp2)))

    def gen_necessary_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the necessary data for the FEM solution: K11, K12 and D1 for each element in the t-list.


        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 3 numpy arrays containing K11, K12 and D1 for each element in the t-list
        """
        x1 = self.plist[self.tlist[:, 0]]
        x2 = self.plist[self.tlist[:, 1]]
        L_E = x2 - x1
        x_M = (x1 + x2) / 2

        alpha_M = self.alpha(x_M)
        beta_M = self.beta(x_M)
        f_M = self.f(x_M)

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
        """
        Sorts the K11, K12 and D1 arrays into the global K matrix and D vector according to the t-list and p-list.

        Args:
            K11 (np.ndarray): numpy array containing K11 values for each element in the t-list
            K12 (np.ndarray): numpy array containing K12 values for each element in the t-list
            D1 (np.ndarray): numpy array containing D1 values for each element in the t-list
        """
        self.K, self.D = vec_sort_into_matrix(K11, K12, D1, self.tlist, len(self.plist))
        # print("K Matrix ohne Randbedingung:\n", K)
        # print("D Vector ohne Randbedingung:\n", D)

    def apply_robin_boundary_conditions(self):
        """
        Applies the Robin boundary conditions to the K matrix and D vector according to the randelemente list.
        """
        # sort out randlelemente for Robin RW (check if they are in xR)
        actualRE = [re for re in self.randelemente if self.plist[re] in self.xR]

        for re in actualRE:
            gamma_val = self.gamma(self.plist[re])
            q_val = self.q(self.plist[re])

            # adjust K and D for Robin BCs
            self.K[re, re] += gamma_val
            self.D[re] += q_val

        # self.K = self.K
        # self.D = self.D

        # return K, D

    def apply_dirichlet_boundary_conditions(self):
        """
        Applies the Dirichlet boundary conditions to the K matrix and D vector according to the randelemente list.
        """
        actualRE = [re for re in self.randelemente if self.plist[re] in self.xD]

        rand_re = np.array(
            [self.K[:, re] * self.phi(self.plist[re]) for re in actualRE]
        )  # spalte von Rand re in der K-Matrix kopieren

        self.D -= np.sum(
            rand_re, axis=0
        )  # D Vector anpassen, da die Randbedingungen nicht mehr in der K-Matrix berücksichtigt werden

        newK = np.delete(
            self.K, [re for re in actualRE], axis=1
        )  # Spalte von Rand a und b in der K-Matrix wegstreichen
        newK = np.delete(newK, [re for re in actualRE], axis=0)  # Zeile von Rand a in der K-Matrix wegstreichen

        newD = np.delete(self.D, [re for re in actualRE])  # Rand a in D Vector wegstreichen

        self.K = newK
        self.D = newD

    def solve_LGS(self):
        """
        Converts the K matrix to a sparse matrix and solves the LGS K * sol = D for sol using scipy's sparse linear solver.
        """
        K_Sparse = sp.csr_matrix(self.K)
        self.sol = sp.linalg.spsolve(K_Sparse, self.D)

    def reconstruct_solution(self):
        """
        Reconstructs the full solution vector by inserting the Dirichlet boundary values back into the solution vector at the correct positions according to the randelemente list.
        """
        # numpy insert
        if len(self.sol) == len(self.plist):
            return

        actualRE = [re for re in self.randelemente if self.plist[re] in self.xD]
        sol_new = np.zeros(len(self.plist))
        free_indices = np.delete(np.arange(len(self.plist)), actualRE)

        sol_new[free_indices] = self.sol  # fill with solution from LGS

        sol_new[actualRE] = self.phi(self.plist[actualRE])  # fill with Dirichlet RW values

        self.sol = sol_new

    def visualize_solution(self):
        """
        Visualizes the solution using matplotlib. Plots the solution values at the points in the plist.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.plist, self.sol, color="blue", label="Lösung (phi)")
        plt.title("Lösung der DGL mit 1D FEM")
        plt.xlabel("Punkte (plist)")
        plt.ylabel("Lösung (phi)")
        plt.grid()
        plt.legend()

    def full_solve(self):
        """
        Fully solve the FEM Problem and returns the timings of each step.
        """
        import time
        t0 = time.time()
        timings = []

        t1 = time.time()
        self.gen_tlist()
        t_gen_tlist = (time.time() - t1) * 1000.0
        timings.append(("gen_tlist", t_gen_tlist))

        t1 = time.time()
        K11, K12, D1 = self.gen_necessary_data()
        t_gen_data = (time.time() - t1) * 1000.0
        timings.append(("gen_K11_K12_D1", t_gen_data))

        t1 = time.time()
        self.sort_into_matrix(K11, K12, D1)
        self.apply_robin_boundary_conditions()
        self.apply_dirichlet_boundary_conditions()
        t_assemble = (time.time() - t1) * 1000.0
        timings.append(("assemble_matrix", t_assemble))

        t1 = time.time()
        self.solve_LGS()
        t_solve = (time.time() - t1) * 1000.0
        timings.append(("solve_LGS", t_solve))

        t1 = time.time()
        self.reconstruct_solution()
        t_recon = (time.time() - t1) * 1000.0
        timings.append(("reconstruct_solution", t_recon))

        t_total = (time.time() - t0) * 1000.0
        timings.append(("total_time", t_total))

        return timings

    def get_Solution(self):
        """Returns the computed solution."""
        return self.sol

    def validate_sol(self, sol_test: np.ndarray, error_tolerance: float = 1e-11):
        """
        Validates the computed solution against a provided test solution.
        """
        if len(sol_test) != len(self.sol):
            raise RuntimeError("Validation failed: Solution size does not match test solution size.")

        error = np.abs(sol_test - self.sol)
        error_stats = (float(np.max(error)), float(np.min(error)), float(np.mean(error)))

        return error, error_stats
        print("=" * len_sep)

        plt.figure(figsize=(12, 5))
        plt.plot(self.plist, sol_test, marker="o", linestyle="", label="Weizis Testlösung")
        plt.plot(self.plist, self.sol, marker="x", linestyle="", label="Meine Lösung")
        plt.xlabel("Punkte")
        plt.ylabel("Lösung")
        plt.title(title)
        plt.legend()

        plt.figure(figsize=(12, 5))
        plt.plot(self.plist, error, marker="o", linestyle="", label="Difference in Solution")
        plt.xlabel("Punkte")
        plt.ylabel("abs(Differenz)")
        plt.title(f"{title} - Abweichung zu Weizis Data")
        plt.legend()
