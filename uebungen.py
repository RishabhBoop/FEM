import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import scipy.linalg as spla
import json
import os
import pickle
# --------------------------------------------------

TO_MB = 1024 * 1024

# 1 d)
def generate_t(pList: np.ndarray) -> np.ndarray:
    t = []
    pList = np.array(pList)
    pList = pList[pList[:, 0].argsort()] # Sortiere pList nach den x-Werten
    plist_tmp = pList[:, 1][:-1]
    plist_1 = pList[:, 1][1:]
    t = np.array(list(zip(plist_tmp, plist_1)))
    return t

def get_teil_Abstand(plist: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    plist = np.array(plist)
    plist = plist[plist[:, 0].argsort()]  # Sortiere plist nach den x-Werten
    abstaende = np.diff(plist[:, 0])
    return abstaende

def gen_mittelpunkte(pList: np.ndarray) -> np.ndarray:
    # pList = np.array(pList)
    pList = pList[pList[:, 0].argsort()]
    mittelpunkte = (pList[:-1, 0] + pList[1:, 0]) / 2
    return mittelpunkte

def area_under_kurve(x, y, abstaende):
    # rechteck + dreiecksfläche
    rechteck = np.sum(abstaende * y[:-1])
    dreiecksfläche = np.sum(abstaende * (y[1:] - y[:-1]) / 2)
    return rechteck + dreiecksfläche

def exaact_area_under_kurve(lim_a, lim_b):
    part1 = 2 * lim_a * (np.log(lim_a) - 1)
    part2 = 2 * lim_b * (np.log(lim_b) - 1)
    return part2 - part1

# --------------------------------------------------

class Aufgabe1:
    def __init__(self, N=5):
        self.a = math.sqrt(2)
        self.b = 5 * math.e
        self.N = N

        # State Variables
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.newP2 = None
        
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.newT2 = None

    def func(self, x):
        # f(x) = ln(x^2)
        return 2 * np.log(x)

    def aufgabe_a(self):
        print("------------ a) ------------")
        p0_0 = np.linspace(self.a, self.b, num=self.N)
        p0_1 = np.arange(self.N)
        self.p0 = np.column_stack((p0_0, p0_1))
        print("p0:\n", self.p0)

    def aufgabe_b(self):
        print("------------ b) ------------")
        p1_0 = np.random.uniform(self.a, self.b, size=self.N)
        p1_1 = np.arange(self.N)
        np.random.shuffle(p1_1)
        self.p1 = np.column_stack((p1_0, p1_1))
        print("p1:\n", self.p1)

    def aufgabe_c(self):
        print("------------ c) ------------")
        # Ensure dependencies exist
        if self.p0 is None: self.aufgabe_a()
        if self.p1 is None: self.aufgabe_b()

        # p0[:, 0] enthält sortierte x_k Werte, p1[:, 1] enthält zufällige Nummern
        self.p2 = np.column_stack((self.p0[:, 0], self.p1[:, 1]))
        print("p2:\n", self.p2)

    def aufgabe_d(self):
        print("------------ d) ------------")
        if self.p2 is None: self.aufgabe_c()

        # --- p0 ---
        print("--- p0 ---")
        self.t0 = generate_t(self.p0)
        abstaende_p0 = get_teil_Abstand(self.p0, self.t0)
        print(f"Maximaler Abstand p0: {np.max(abstaende_p0):.4f}")
        print(f"Minimaler Abstand p0: {np.min(abstaende_p0):.4f}")
        print(f"Mittlerer Abstand p0: {np.mean(abstaende_p0):.4f}")

        # --- p1 ---
        print("--- p1 ---")
        self.t1 = generate_t(self.p1)
        abstaende_p1 = get_teil_Abstand(self.p1, self.t1)
        print(f"Maximaler Abstand p1: {np.max(abstaende_p1):.4f}")
        print(f"Minimaler Abstand p1: {np.min(abstaende_p1):.4f}")
        print(f"Mittlerer Abstand p1: {np.mean(abstaende_p1):.4f}")

        # --- p2 ---
        print("--- p2 ---")
        self.t2 = generate_t(self.p2) 
        abstaende_p2 = get_teil_Abstand(self.p2, self.t2)
        print(f"Maximaler Abstand p2: {np.max(abstaende_p2):.4f}")
        print(f"Minimaler Abstand p2: {np.min(abstaende_p2):.4f}")
        print(f"Mittlerer Abstand p2: {np.mean(abstaende_p2):.4f}")

        # --- Erstellung mit N=100000 ---
        print("--- Erstellung mit N=100000 ---")
        N_large = 100000
        Np0_0 = np.random.uniform(self.a, self.b, size=N_large)
        Np0_1 = np.arange(N_large)
        Np0 = np.column_stack((Np0_0, Np0_1))
        
        Nt0 = generate_t(Np0)
        abstaende_large = get_teil_Abstand(Np0, Nt0) # Bug fixed: Used Nt0 instead of t0
        
        print(f"Maximaler Abstand p0 (100k): {np.max(abstaende_large):.4f}")
        print(f"Minimaler Abstand p0 (100k): {np.min(abstaende_large):.4f}")
        print(f"Mittlerer Abstand p0 (100k): {np.mean(abstaende_large):.4f}")
        
        plt.figure(figsize=(10, 5))
        plt.hist(abstaende_large, bins=500)
        plt.title("Häufigkeitsverteilung der Abstände (N=100000)")
        plt.xlabel("Abstand")
        plt.ylabel("Häufigkeit")

    def aufgabe_e(self):
        print("------------ e) ------------")
        if self.p2 is None: self.aufgabe_c()
        
        cond_to_remove = (self.p2[:, 1] % 3 == 0) & (self.p2[:, 1] != 0)
        self.newP2 = self.p2[~cond_to_remove]
        print("new p2:\n", self.newP2)
        
        self.newT2 = generate_t(self.newP2)
        print("new t2:\n", self.newT2)

    def aufgabe_f(self):
        print("------------ f) ------------")
        if self.newP2 is None: self.aufgabe_e()

        self.y_p0 = self.func(self.p0[:, 0])
        self.y_p1 = self.func(self.p1[:, 0])
        self.y_newP2 = self.func(self.newP2[:, 0])

        plt.figure(figsize=(10, 6))
        plt.scatter(self.p0[:, 0], self.y_p0, label='p0', marker='o', linewidths=0.3)
        plt.scatter(self.p1[:, 0], self.y_p1, label='p1', marker='o', linewidths=0.3)
        plt.scatter(self.newP2[:, 0], self.y_newP2, label='new p2', marker='o', linewidths=0.3)
        
        plt.title("Auswertung der Funktion an verschiedenen Knotenpunkten")
        plt.xlabel("x")
        plt.ylabel("f(x) = ln(x^2)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
    
    def aufgabe_g(self):
        print("------------ g) ------------")
        if not hasattr(self, 'y_p0'): self.aufgabe_f()

        abstaende_p0 = get_teil_Abstand(self.p0, self.t0)
        abstaende_p1 = get_teil_Abstand(self.p1, self.t1)
        abstaende_p2 = get_teil_Abstand(self.newP2, self.newT2)

        area_p0 = area_under_kurve(self.p0[:, 0], self.y_p0, abstaende_p0)
        area_p1 = area_under_kurve(self.p1[:, 0], self.y_p1, abstaende_p1)
        area_p2 = area_under_kurve(self.newP2[:, 0], self.y_newP2, abstaende_p2)

        print(f"Fläche unter der Kurve für p0 (N={self.N}): {area_p0:.4f}")
        print(f"Fläche unter der Kurve für p1 (N={self.N}): {area_p1:.4f}")
        print(f"Fläche unter der Kurve für p2 (N={self.N}): {area_p2:.4f}")

        exact_area = exaact_area_under_kurve(self.a, self.b)
        print(f"Exakte Fläche unter der Kurve: {exact_area:.4f}")


    def execute_aufgabe(self, aufgabe = []):

        aufgaben = {
            'a': self.aufgabe_a,
            'b': self.aufgabe_b,
            'c': self.aufgabe_c,
            'd': self.aufgabe_d,
            'e': self.aufgabe_e,
            'f': self.aufgabe_f,
            'g': self.aufgabe_g
        }

        to_execute = aufgabe if aufgabe else aufgaben.keys()

        for e in to_execute:
            if e in aufgaben:
                aufgaben[e]()
            else:
                print(f"Ungültige Aufgabe: {e}")

        plt.show()


class Aufgabe2:
    def __init__(self):
        self.times_dense = None   # b)
        self.times_banded = None  # c)
        self.times_banded_special = None  # c) with scipy.solve_banded
        self.times_sparse = None  # d)
    
        self.memory_dense = None  # b)
        self.memory_banded = None # c)
        self.memory_banded_special = None # c) with scipy.solve_banded
        self.memory_sparse = None # d)


    def messe_zeit(self, matrix, vector):
        start_time = time.time()
        x = np.linalg.solve(matrix, vector)
        end_time = time.time()
        return end_time - start_time


    def aufgabe_a(self):
        print("------------ a) ------------")
        dimMatrix = 10

        matrA = np.random.rand(dimMatrix, dimMatrix)
        vecB = np.random.rand(dimMatrix)

        x = np.linalg.solve(matrA, vecB)
        print("A =\n", matrA)
        print("B =\n", vecB)
        print("x =\n", x)


    def aufgabe_b(self):
        print("------------ b) ------------")
        # Dense Matrix
        self.times_dense = {}
        self.memory_dense = {}
        for i in range(500, 15000, 1000):
            matrA = np.random.rand(i, i)
            vecB = np.random.rand(i)
            time_taken = self.messe_zeit(matrA, vecB)
            arr_Size = matrA.nbytes / TO_MB
            print(f"Zeit für N = {i}: {time_taken:.4f} Sekunden")
            print(f"Memory für N = {i}: {arr_Size:.4f} MB")
            self.times_dense[i] = time_taken
            self.memory_dense[i] = arr_Size

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.times_dense.keys()), list(self.times_dense.values()), marker='x', label='Dense')
        plt.title("Zeit um NxN Dense Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()


    def aufgabe_c(self):
        print("------------ c) ------------")
        # Banded Matrix        
        self.times_banded = {}
        self.memory_banded = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i-1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i-1)
            matrA = np.diag(unten, -1) + np.diag(mitte, 0) + np.diag(oben, 1)
            vecB = np.random.rand(i)
            time_taken = self.messe_zeit(matrA, vecB)
            arr_Size = matrA.nbytes / TO_MB
            print(f"Zeit für N = {i}: {time_taken:.4f} Sekunden")
            print(f"Memory für N = {i}: {arr_Size:.4f} MB")
            self.times_banded[i] = time_taken
            self.memory_banded[i] = arr_Size

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.times_banded.keys()), list(self.times_banded.values()), marker='x', label='Banded')
        plt.title("Zeit um NxN Banded Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()


    def aufgabe_c_special_solver(self):
        print("------------ c) ------------")
        # Banded Matrix with scipy.solve_banded       
        self.times_banded_special = {}
        self.memory_banded_special = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i-1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i-1)

            
            ab = np.zeros((3, i))
            ab[0, 1:] = oben  # obere Diagonale
            ab[1, :] = mitte   # Hauptdiagonale
            ab[2, :-1] = unten # untere Diagonale
            vecB = np.random.rand(i)

            start_time = time.time()
            x = spla.solve_banded((1, 1), ab, vecB)
            end_time = time.time()

            time_taken = end_time - start_time
            arr_Size = ab.nbytes / TO_MB
            print(f"Zeit für N = {i}: {time_taken:.4f} Sekunden")
            print(f"Memory für N = {i}: {arr_Size:.4f} MB")
            self.times_banded_special[i] = time_taken
            self.memory_banded_special[i] = arr_Size

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.times_banded_special.keys()), list(self.times_banded_special.values()), marker='x', label='Banded (with scipy.solve_banded)')
        plt.title("Zeit um NxN Banded Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()


    def aufgabe_d(self):
        if self.times_dense is None: self.aufgabe_b()
        if self.times_banded is None: self.aufgabe_c()

        print("------------ d) ------------")
        # Sparse Matrix
        self.times_sparse = {}
        self.memory_sparse = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i-1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i-1)

            matrA = sp.lil_matrix((i, i))
            matrA.setdiag(mitte, 0)
            matrA.setdiag(oben, 1)
            matrA.setdiag(unten, -1)
            matrA = matrA.tocsr()  

            vecB = np.random.rand(i)

            start_time = time.time()
            x = sp.linalg.spsolve(matrA, vecB)
            end_time = time.time()

            time_taken = end_time - start_time
            arr_Size = (matrA.data.nbytes + matrA.indices.nbytes + matrA.indptr.nbytes) / TO_MB
            print(f"Zeit für N = {i}: {time_taken:.4f} Sekunden")
            print(f"Memory für N = {i}: {arr_Size:.4f} MB")
            self.times_sparse[i] = time_taken
            self.memory_sparse[i] = arr_Size

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.times_sparse.keys()), list(self.times_sparse.values()), marker='x', label='Sparse')
        plt.title("Zeit um NxN Sparse Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.scatter(list(self.times_sparse.keys()), list(self.times_sparse.values()), marker='x', label='Sparse')
        plt.scatter(list(self.times_dense.keys()), list(self.times_dense.values()), marker='x', label='Dense')
        plt.scatter(list(self.times_banded.keys()), list(self.times_banded.values()), marker='x', label='Banded')
        plt.title("Zeit um NxN Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()


    def aufgabe_e(self):
        if self.times_sparse is None: self.aufgabe_d()


        print("------------ e) ------------")
        dense_times = np.array(list(self.times_dense.values()))
        banded_times = np.array(list(self.times_banded.values()))
        sparse_times = np.array(list(self.times_sparse.values()))

        nds = np.array(list(self.times_dense.keys()))

        # fit to polynom (ax² + bx + c) in log-log space
        dense = np.polyfit(np.log(nds), np.log(dense_times), 1)
        banded = np.polyfit(np.log(nds), np.log(banded_times), 1)
        sparse = np.polyfit(np.log(nds), np.log(sparse_times), 1)

        y_fit_dense = np.exp(dense[1]) * nds**dense[0]
        y_fit_banded = np.exp(banded[1]) * nds**banded[0]
        y_fit_sparse = np.exp(sparse[1]) * nds**sparse[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(list(self.times_sparse.keys()), list(self.times_sparse.values()), marker='x', label='Sparse', color='orange')
        plt.scatter(list(self.times_dense.keys()), list(self.times_dense.values()), marker='x', label='Dense', color='blue')
        plt.scatter(list(self.times_banded.keys()), list(self.times_banded.values()), marker='x', label='Banded', color='green')
        plt.plot(nds, y_fit_dense, label=f'Dense Fit: O(N^{dense[0]:.2f})', linestyle='--', color='blue')
        plt.plot(nds, y_fit_banded, label=f'Banded Fit: O(N^{banded[0]:.2f})', linestyle='--', color='green')
        plt.plot(nds, y_fit_sparse, label=f'Sparse Fit: O(N^{sparse[0]:.2f})', linestyle='--', color='orange')    
        plt.title("Zeit um NxN Matrix zu lösen mit Fit")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()

        print(f"Dense: O(N^{dense[0]:.2f})")
        print(f"Banded: O(N^{banded[0]:.2f})")
        print(f"Sparse: O(N^{sparse[0]:.2f})")


    def aufgabe_f(self):
        if self.times_sparse is None: self.aufgabe_d()

        print("------------ f) ------------")

        plt.figure(figsize=(10, 6))
        plt.plot(list(self.memory_dense.keys()), list(self.memory_dense.values()), marker='x', label='Dense')
        plt.plot(list(self.memory_banded.keys()), list(self.memory_banded.values()), marker='x', label='Banded')
        plt.plot(list(self.memory_sparse.keys()), list(self.memory_sparse.values()), marker='x', label='Sparse')
        plt.title("Speicherverbrauch für NxN Matrix")
        plt.xlabel("Dimension N")
        plt.ylabel("Speicherverbrauch (MB)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()


    def load_data(self):
        filename = "messungen_uebungen_a2.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.times_dense = data.get("times_dense", {})
                self.times_banded = data.get("times_banded", {})
                self.times_banded_special = data.get("times_banded_special", {})
                self.times_sparse = data.get("times_sparse", {})
                self.memory_dense = data.get("memory_dense", {})
                self.memory_banded = data.get("memory_banded", {})
                self.memory_banded_special = data.get("memory_banded_special", {})
                self.memory_sparse = data.get("memory_sparse", {})
            print("Data successfully loaded from file!")
        else:
            print("No existing file found, recalculating data...")

    def save_data(self):
        filename = "messungen_uebungen_a2.pkl"
        data = {
            "times_dense": self.times_dense,
            "times_banded": self.times_banded,
            "times_banded_special": self.times_banded_special,
            "times_sparse": self.times_sparse,
            "memory_dense": self.memory_dense,
            "memory_banded": self.memory_banded,
            "memory_banded_special": self.memory_banded_special,
            "memory_sparse": self.memory_sparse
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {filename}!")


    def execute_aufgabe(self, aufgabe = []):

        aufgaben = {
            'a': self.aufgabe_a,
            'b': self.aufgabe_b,
            'c': self.aufgabe_c,
            'z': self.aufgabe_c_special_solver,
            'd': self.aufgabe_d,
            'e': self.aufgabe_e,
            'f': self.aufgabe_f,
        }
        to_execute = aufgabe if aufgabe else aufgaben.keys()

        self.load_data()

        for e in to_execute:
            if e in aufgaben:
                aufgaben[e]()
            else:
                print(f"Ungültige Aufgabe: {e}")

        self.save_data()

        plt.show()

        




if __name__ == "__main__":
    # a1 = Aufgabe1()
    # a1.execute_aufgabe()

    a2 = Aufgabe2()
    a2.execute_aufgabe(['e'])
