import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numba import jit
import time
from typing import Callable

# -------------------------------- Global Functions --------------------------------


# def gen_randwerte_list(xD, xR, plist):
#     randelemente = []
#     for val in xD + xR:
#         # Check for matching coordinates in plist
#         idx = np.where(np.isclose(plist, val))[0]
#         if len(idx) > 0:
#             randelemente.append(idx[0])
#     return np.array(randelemente)


@jit(nopython=True)
def vec_sort_into_matrix(
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


# ------------------------------------------------------------------------------------------------


class fem_2d:
    def __init__(
        self,
        xD: list,
        xR: list,
        plist: np.ndarray,
        tlist: np.ndarray,
        randelemente: list,
        alpha1: Callable,
        alpha2: Callable,
        beta: Callable,
        f: Callable,
        # phi: Callable,
        # gamma: Callable,
        # q: Callable,
    ):
        """
        Class for a 2D FEM Solution.
        Takes all necessary data and functions as input and provides methods to solve the FEM problem, apply boundary conditions, and visualize the solution.

        Args:
            xD (list): list of x-coordinates for Dirichlet boundary conditions
            xR (list): list of x-coordinates for Robin boundary conditions
            plist (np.ndarray): numpy array of points in the domain
            alpha1 (Callable): alpha1-part of the partial differential equation
            alpha2 (Callable): alpha2-part of the partial differential equation
            beta (Callable): beta-part of the partial differential equation
            f (Callable): f-part of the partial differential equation
        """
        # --- Data needed to apply Randbedingungen ---
        self.xR = xR
        self.xD = xD
        self.randelemente = randelemente
        # --- Data needed to solve ---
        self.plist = plist
        self.tlist = tlist
        self.K = None
        self.D = None
        self.sol = None
        # --- Functions ---
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.f = f
        # self.phi = phi
        # self.gamma = gamma
        # self.q = q

    def gen_necessary_data(self) -> tuple:
        p = self.plist[self.tlist]
        bj = gen_b(p[:, :, 1])
        cj = gen_c(p[:, :, 0])
        twodeltae = gen_2area(p)

        xm = np.mean(p[:, :, 0], axis=1)
        ym = np.mean(p[:, :, 1], axis=1)
        a1m = self.alpha1(xm, ym)
        a2m = self.alpha2(xm, ym)
        betam = self.beta(xm, ym)
        fm = self.f(xm, ym)

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

        Dloc_val = fm * gen_area(p) / 3
        Dloc = np.column_stack((Dloc_val, Dloc_val, Dloc_val))  # Shape (N, 3)

        return K11, K22, K33, K12, K13, K23, Dloc

    def sort_into_matrix(
        self,
        K11: np.ndarray,
        K22: np.ndarray,
        K33: np.ndarray,
        K12: np.ndarray,
        K13: np.ndarray,
        K23: np.ndarray,
        D1: np.ndarray,
    ):
        """
        Sorts the K11, K12 and D1 arrays into the global K matrix and D vector according to the t-list and p-list.

        Args:
            K11 (np.ndarray): numpy array containing K11 values for each element in the t-list
            K12 (np.ndarray): numpy array containing K12 values for each element in the t-list
            D1 (np.ndarray): numpy array containing D1 values for each element in the t-list
        """
        self.K, self.D = vec_sort_into_matrix(self.plist, self.tlist, K11, K22, K33, K12, K13, K23, D1)
        # print("K Matrix ohne Randbedingung:\n", K)
        # print("D Vector ohne Randbedingung:\n", D)

    def apply_robin_boundary_conditions(self):
        """
        Applies the Robin boundary conditions to the K matrix and D vector according to the randelemente list.
        RIGHT NOW: NOT IMPLEMENTED FOR 2D
        """
        pass


    def apply_dirichlet_boundary_conditions(self):
        """
        Applies the Dirichlet boundary conditions to the K matrix and D vector according to the randelemente list.
        """
        for re in self.randelemente:
            phi_val = self.phi(self.plist[re][0], self.plist[re][1])
            self.D -= self.K[:, re] * phi_val

        # wegstreichen
        newK = np.delete(self.K, [re for re in self.randelemente], axis=1)  # Spalte von Rand a und b in der K-Matrix wegstreichen
        newK = np.delete(newK, [re for re in self.randelemente], axis=0)  # Zeile von Rand a in der K-Matrix wegstreichen

        newD = np.delete(self.D, [re for re in self.randelemente])  # Rand a in D Vector wegstreichen
        return newK, newD

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
        if len(self.sol) == len(self.plist):
            return  # no reconstruction needed

        # Identify Dirichlet boundary nodes based on x-coordinate
        actualRE = [re for re in self.randelemente if self.plist[re, 0] in self.xD]

        # Allocate full solution array
        sol_new = np.zeros(len(self.plist))
        free_indices = np.delete(np.arange(len(self.plist)), actualRE)

        # Fill free nodes with the solution from LGS
        sol_new[free_indices] = self.sol

        # Fill Dirichlet boundary nodes using 2D coordinates
        for re in actualRE:
            x_coord = self.plist[re, 0]
            y_coord = self.plist[re, 1]
            sol_new[re] = self.phi(x_coord, y_coord)

        return sol_new

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

        t_assemble_start = time.time()

        t1 = time.time()
        self.sort_into_matrix(K11, K12, D1)
        t_sort = (time.time() - t1) * 1000.0

        t1 = time.time()
        self.apply_robin_boundary_conditions()
        t_robin = (time.time() - t1) * 1000.0

        t1 = time.time()
        self.apply_dirichlet_boundary_conditions()
        t_dirich = (time.time() - t1) * 1000.0

        t_assemble = (time.time() - t_assemble_start) * 1000.0
        timings.append(("assemble_matrix", t_assemble))
        timings.append(("  |- sort_into_matrix", t_sort))
        timings.append(("  |- apply_robin_BCs", t_robin))
        timings.append(("  L apply_dirichlet_BCs", t_dirich))

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
