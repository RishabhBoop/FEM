import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import scipy.linalg as spla
import os
import pickle
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation


# --------------------------------------------------

TO_MB = 1024 * 1024

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

    @staticmethod
    def generate_t(plist: np.ndarray) -> np.ndarray:
        tlist_tmp = np.argsort(plist)
        tmp1 = tlist_tmp[0:-1]
        tmp2 = tlist_tmp[1:]
        tlist = np.array(list(zip(tmp1, tmp2)))
        return tlist

    @staticmethod
    def get_teil_Abstand(plist: np.ndarray, tlist: np.ndarray) -> np.ndarray:
        plist = np.array(plist)
        plist = plist[plist[:, 0].argsort()]  # Sortiere plist nach den x-Werten
        abstaende = np.diff(plist[:, 0])
        return abstaende

    @staticmethod
    def gen_mittelpunkte(pList: np.ndarray) -> np.ndarray:
        # pList = np.array(pList)
        pList = pList[pList[:, 0].argsort()]
        mittelpunkte = (pList[:-1, 0] + pList[1:, 0]) / 2
        return mittelpunkte

    @staticmethod
    def area_under_kurve(x, y, abstaende):
        # rechteck + dreiecksfläche
        rechteck = np.sum(abstaende * y[:-1])
        dreiecksfläche = np.sum(abstaende * (y[1:] - y[:-1]) / 2)
        return rechteck + dreiecksfläche

    @staticmethod
    def exaact_area_under_kurve(lim_a, lim_b):
        part1 = 2 * lim_a * (np.log(lim_a) - 1)
        part2 = 2 * lim_b * (np.log(lim_b) - 1)
        return part2 - part1

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
        if self.p0 is None:
            self.aufgabe_a()
        if self.p1 is None:
            self.aufgabe_b()

        print("------------ c) ------------")
        self.p2 = np.column_stack((self.p0[:, 0], self.p1[:, 1]))
        print("p2:\n", self.p2)

    def aufgabe_d(self):
        if self.p2 is None:
            self.aufgabe_c()

        print("------------ d) ------------")

        # --- p0 ---
        print("--- p0 ---")
        self.t0 = self.generate_t(self.p0)
        abstaende_p0 = self.get_teil_Abstand(self.p0, self.t0)
        print(f"Maximaler Abstand p0: {np.max(abstaende_p0):.4f}")
        print(f"Minimaler Abstand p0: {np.min(abstaende_p0):.4f}")
        print(f"Mittlerer Abstand p0: {np.mean(abstaende_p0):.4f}")

        # --- p1 ---
        print("--- p1 ---")
        self.t1 = self.generate_t(self.p1)
        abstaende_p1 = self.get_teil_Abstand(self.p1, self.t1)
        print(f"Maximaler Abstand p1: {np.max(abstaende_p1):.4f}")
        print(f"Minimaler Abstand p1: {np.min(abstaende_p1):.4f}")
        print(f"Mittlerer Abstand p1: {np.mean(abstaende_p1):.4f}")

        # --- p2 ---
        print("--- p2 ---")
        self.t2 = self.generate_t(self.p2)
        abstaende_p2 = self.get_teil_Abstand(self.p2, self.t2)
        print(f"Maximaler Abstand p2: {np.max(abstaende_p2):.4f}")
        print(f"Minimaler Abstand p2: {np.min(abstaende_p2):.4f}")
        print(f"Mittlerer Abstand p2: {np.mean(abstaende_p2):.4f}")

        # --- Erstellung mit N=100000 ---
        print("--- Erstellung mit N=100000 ---")
        N_large = 100000
        Np0_0 = np.random.uniform(self.a, self.b, size=N_large)
        Np0_1 = np.arange(N_large)
        Np0 = np.column_stack((Np0_0, Np0_1))

        Nt0 = self.generate_t(Np0)
        abstaende_large = self.get_teil_Abstand(Np0, Nt0)  # Bug fixed: Used Nt0 instead of t0

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
        if self.p2 is None:
            self.aufgabe_c()

        cond_to_remove = (self.p2[:, 1] % 3 == 0) & (self.p2[:, 1] != 0)
        self.newP2 = self.p2[~cond_to_remove]
        print("new p2:\n", self.newP2)

        self.newT2 = self.generate_t(self.newP2)
        print("new t2:\n", self.newT2)

    def aufgabe_f(self):
        print("------------ f) ------------")
        if self.newP2 is None:
            self.aufgabe_e()

        self.y_p0 = self.func(self.p0[:, 0])
        self.y_p1 = self.func(self.p1[:, 0])
        self.y_newP2 = self.func(self.newP2[:, 0])

        plt.figure(figsize=(10, 6))
        plt.scatter(self.p0[:, 0], self.y_p0, label="p0", marker="o", linewidths=0.3)
        plt.scatter(self.p1[:, 0], self.y_p1, label="p1", marker="o", linewidths=0.3)
        plt.scatter(self.newP2[:, 0], self.y_newP2, label="new p2", marker="o", linewidths=0.3)

        plt.title("Auswertung der Funktion an verschiedenen Knotenpunkten")
        plt.xlabel("x")
        plt.ylabel("f(x) = ln(x^2)")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7)

    def aufgabe_g(self):
        print("------------ g) ------------")
        if not hasattr(self, "y_p0"):
            self.aufgabe_f()

        abstaende_p0 = self.get_teil_Abstand(self.p0, self.t0)
        abstaende_p1 = self.get_teil_Abstand(self.p1, self.t1)
        abstaende_p2 = self.get_teil_Abstand(self.newP2, self.newT2)

        area_p0 = self.area_under_kurve(self.p0[:, 0], self.y_p0, abstaende_p0)
        area_p1 = self.area_under_kurve(self.p1[:, 0], self.y_p1, abstaende_p1)
        area_p2 = self.area_under_kurve(self.newP2[:, 0], self.y_newP2, abstaende_p2)

        print(f"Fläche unter der Kurve für p0 (N={self.N}): {area_p0:.4f}")
        print(f"Fläche unter der Kurve für p1 (N={self.N}): {area_p1:.4f}")
        print(f"Fläche unter der Kurve für p2 (N={self.N}): {area_p2:.4f}")

        exact_area = self.exaact_area_under_kurve(self.a, self.b)
        print(f"Exakte Fläche unter der Kurve: {exact_area:.4f}")

    def execute_aufgabe(self, aufgabe=[]):

        aufgaben = {
            "a": self.aufgabe_a,
            "b": self.aufgabe_b,
            "c": self.aufgabe_c,
            "d": self.aufgabe_d,
            "e": self.aufgabe_e,
            "f": self.aufgabe_f,
            "g": self.aufgabe_g,
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
        self.times_dense = None  # b)
        self.times_banded = None  # c)
        self.times_banded_special = None  # c) with scipy.solve_banded
        self.times_sparse = None  # d)

        self.memory_dense = None  # b)
        self.memory_banded = None  # c)
        self.memory_banded_special = None  # c) with scipy.solve_banded
        self.memory_sparse = None  # d)

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
        plt.plot(
            list(self.times_dense.keys()),
            list(self.times_dense.values()),
            marker="x",
            label="Dense",
        )
        plt.title("Zeit um NxN Dense Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

    def aufgabe_c(self):
        print("------------ c) ------------")
        # Banded Matrix
        self.times_banded = {}
        self.memory_banded = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i - 1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i - 1)
            matrA = np.diag(unten, -1) + np.diag(mitte, 0) + np.diag(oben, 1)
            vecB = np.random.rand(i)
            time_taken = self.messe_zeit(matrA, vecB)
            arr_Size = matrA.nbytes / TO_MB
            print(f"Zeit für N = {i}: {time_taken:.4f} Sekunden")
            print(f"Memory für N = {i}: {arr_Size:.4f} MB")
            self.times_banded[i] = time_taken
            self.memory_banded[i] = arr_Size

        plt.figure(figsize=(10, 6))
        plt.plot(
            list(self.times_banded.keys()),
            list(self.times_banded.values()),
            marker="x",
            label="Banded",
        )
        plt.title("Zeit um NxN Banded Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

    def aufgabe_c_special_solver(self):
        print("------------ c) ------------")
        # Banded Matrix with scipy.solve_banded
        self.times_banded_special = {}
        self.memory_banded_special = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i - 1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i - 1)

            ab = np.zeros((3, i))
            ab[0, 1:] = oben  # obere Diagonale
            ab[1, :] = mitte  # Hauptdiagonale
            ab[2, :-1] = unten  # untere Diagonale
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
        plt.plot(
            list(self.times_banded_special.keys()),
            list(self.times_banded_special.values()),
            marker="x",
            label="Banded (with scipy.solve_banded)",
        )
        plt.title("Zeit um NxN Banded Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

    def aufgabe_d(self):
        if self.times_dense is None:
            self.aufgabe_b()
        if self.times_banded is None:
            self.aufgabe_c()

        print("------------ d) ------------")
        # Sparse Matrix
        self.times_sparse = {}
        self.memory_sparse = {}
        for i in range(500, 15000, 1000):
            oben = np.random.rand(i - 1)
            mitte = np.random.rand(i)
            unten = np.random.rand(i - 1)

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
        plt.plot(
            list(self.times_sparse.keys()),
            list(self.times_sparse.values()),
            marker="x",
            label="Sparse",
        )
        plt.title("Zeit um NxN Sparse Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            list(self.times_sparse.keys()),
            list(self.times_sparse.values()),
            marker="x",
            label="Sparse",
        )
        plt.scatter(
            list(self.times_dense.keys()),
            list(self.times_dense.values()),
            marker="x",
            label="Dense",
        )
        plt.scatter(
            list(self.times_banded.keys()),
            list(self.times_banded.values()),
            marker="x",
            label="Banded",
        )
        plt.title("Zeit um NxN Matrix zu lösen")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

    def aufgabe_e(self):
        if self.times_sparse is None:
            self.aufgabe_d()

        print("------------ e) ------------")
        dense_times = np.array(list(self.times_dense.values()))
        banded_times = np.array(list(self.times_banded.values()))
        sparse_times = np.array(list(self.times_sparse.values()))

        nds = np.array(list(self.times_dense.keys()))

        # fit to polynom (ax² + bx + c) in log-log space
        dense = np.polyfit(np.log(nds), np.log(dense_times), 1)
        banded = np.polyfit(np.log(nds), np.log(banded_times), 1)
        sparse = np.polyfit(np.log(nds), np.log(sparse_times), 1)

        y_fit_dense = np.exp(dense[1]) * nds ** dense[0]
        y_fit_banded = np.exp(banded[1]) * nds ** banded[0]
        y_fit_sparse = np.exp(sparse[1]) * nds ** sparse[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(
            list(self.times_sparse.keys()),
            list(self.times_sparse.values()),
            marker="x",
            label="Sparse",
            color="orange",
        )
        plt.scatter(
            list(self.times_dense.keys()),
            list(self.times_dense.values()),
            marker="x",
            label="Dense",
            color="blue",
        )
        plt.scatter(
            list(self.times_banded.keys()),
            list(self.times_banded.values()),
            marker="x",
            label="Banded",
            color="green",
        )
        plt.plot(
            nds,
            y_fit_dense,
            label=f"Dense Fit: O(N^{dense[0]:.2f})",
            linestyle="--",
            color="blue",
        )
        plt.plot(
            nds,
            y_fit_banded,
            label=f"Banded Fit: O(N^{banded[0]:.2f})",
            linestyle="--",
            color="green",
        )
        plt.plot(
            nds,
            y_fit_sparse,
            label=f"Sparse Fit: O(N^{sparse[0]:.2f})",
            linestyle="--",
            color="orange",
        )
        plt.title("Zeit um NxN Matrix zu lösen mit Fit")
        plt.xlabel("Dimension N")
        plt.ylabel("Zeit (Sekunden)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

        print(f"Dense: O(N^{dense[0]:.2f})")
        print(f"Banded: O(N^{banded[0]:.2f})")
        print(f"Sparse: O(N^{sparse[0]:.2f})")

    def aufgabe_f(self):
        if self.times_sparse is None:
            self.aufgabe_d()

        print("------------ f) ------------")

        plt.figure(figsize=(10, 6))
        plt.plot(
            list(self.memory_dense.keys()),
            list(self.memory_dense.values()),
            marker="x",
            label="Dense",
        )
        plt.plot(
            list(self.memory_banded.keys()),
            list(self.memory_banded.values()),
            marker="x",
            label="Banded",
        )
        plt.plot(
            list(self.memory_sparse.keys()),
            list(self.memory_sparse.values()),
            marker="x",
            label="Sparse",
        )
        plt.title("Speicherverbrauch für NxN Matrix")
        plt.xlabel("Dimension N")
        plt.ylabel("Speicherverbrauch (MB)")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

    def load_data(self):
        filename = "messungen_uebungen_a2.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
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
            "memory_sparse": self.memory_sparse,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {filename}!")

    def execute_aufgabe(self, aufgabe=[]):

        aufgaben = {
            "a": self.aufgabe_a,
            "b": self.aufgabe_b,
            "c": self.aufgabe_c,
            "z": self.aufgabe_c_special_solver,
            "d": self.aufgabe_d,
            "e": self.aufgabe_e,
            "f": self.aufgabe_f,
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


class Aufgabe3:
    def __init__(self):
        self.plist = None
        self.tlist = None
        self.e = None
        self.r = None
        self.rt = None

    def aufgabe_a(self):
        print("------------ a) ------------")
        l = 3
        h = 4
        x0, y0 = 0, 0
        xN, yN = l, h
        N0 = 4
        N = N0**2
        anzahl_dreiecke = 2 * (N0 - 1) ** 2
        print(f"Anzahl Dreiecke: {anzahl_dreiecke}, Anzahl Punkte: {N}, N0: {N0}")

        # Lege Punkte fest
        px = np.random.uniform(x0, xN, N0)
        px.sort()
        py = np.random.uniform(y0, yN, N0)
        py.sort()
        points = np.array([[x, y] for y in py for x in px])
        # print("Punkte:\n", points)

        plt.plot(points[:, 0], points[:, 1], "ko")
        plt.title(f"T-Liste generieren mit {anzahl_dreiecke} Dreiecken")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.axis("equal")
        plt.tight_layout()

        # T-Liste erstellen
        triangles = []
        for i in range(N0 - 1):
            for j in range(N0 - 1):
                p1 = i * N0 + j
                p2 = p1 + 1
                p3 = p1 + N0
                p4 = p3 + 1
                triangles += [[p1, p2, p3], [p2, p4, p3]]
                plt.triplot(points[:, 0], points[:, 1], [[p1, p2, p3]], marker="o", color="red")
                plt.pause(interval=0.2)
                plt.triplot(points[:, 0], points[:, 1], [[p2, p4, p3]], marker="o", color="red")
                plt.pause(interval=0.2)

        triangles = np.array(triangles)

    def aufgabe_b(self):
        print("------------ b) ------------")
        print("Keine Aufgabe zu machen, nur code aus pdf")
        # Anfangs und Endpunkt
        y0 = 1
        yN = 5
        x0 = -1
        xN = 2

        # Anzahl der Positionen N=N0*N0 Anzahl Dreiecke etwa N0*N0/2
        N0 = 7

        # Lege Punkte fest
        points = []
        for i in range(N0):
            for j in range(N0):
                x = x0 + i * (xN - x0) / (N0 - 1)
                y = y0 + j * (yN - y0) / (N0 - 1)
                points += [[x, y]]
        points = np.array(points)

        # Nummeriere und lege Elemente fest
        tri = Delaunay(points)

        # entnehme p und t-Liste
        t = tri.simplices
        p = 1.0 * points

        print("p-Liste: ", p)
        print("t-Liste: ", t)

        plt.figure(figsize=(10, 6))

        # Male die Dreiecke
        plt.triplot(p[:, 0], p[:, 1], t)

        # male Kreise an den Positionen
        plt.plot(p[:, 0], p[:, 1], "o")

        self.tlist = t
        self.plist = p

    def aufgabe_c(self):
        if self.tlist is None or self.plist is None:
            self.aufgabe_b()
        print("------------ c) ------------")
        # liste mit allen Kanten e
        # create each edge from tlist
        all_edges = np.vstack([self.tlist[:, [0, 1]], self.tlist[:, [1, 2]], self.tlist[:, [2, 0]]])

        # sort each edge from smaller to bigger index ([20, 32] and [32, 20] should be the same edge)
        all_edges = np.sort(all_edges, axis=1)

        # filter out duplicate edges
        e, inverse, count = np.unique(all_edges, axis=0, return_inverse=True, return_counts=True)
        r = []
        rt = []

        boundary_indices_in_e = np.where(count == 1)[0]
        r = e[boundary_indices_in_e]
        for idx in boundary_indices_in_e:
            raw_idx = np.where(inverse == idx)[0][0]  # index in all_edges that corresponds to the edge e[idx]

            tri_idx = raw_idx % len(self.tlist)
            rt.append(self.tlist[tri_idx])

            # print(f"Randkante: {e[idx]}, zugehöriges Dreieck: {self.tlist[tri_idx]}")

        self.e = e
        self.r = r
        rt = np.unique(rt, axis=0)
        self.rt = rt

        plt.triplot(self.plist[:, 0], self.plist[:, 1], self.tlist)

        for i, (x, y) in enumerate(self.plist):
            plt.text(x + 0.05, y + 0.05, str(i), color="red", fontsize=12, ha="center", va="center")

        print("Alle Kanten e:\n", self.e)
        print("Randkanten r:\n", self.r)
        print("Dreiecke an den Randkanten rt:\n", self.rt)

    @staticmethod
    def f(x, y):
        return x / (x**2 + y**2)

    def aufgabe_d(self):
        if self.tlist is None or self.plist is None:
            self.aufgabe_b()
        if self.rt is None or self.r is None or self.e is None:
            self.aufgabe_c()

        print("------------ d) ------------")
        func_val = np.array([self.f(x, y) for x, y in self.plist])
        print(func_val)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(self.plist[:, 0], self.plist[:, 1], func_val, triangles=self.tlist, cmap='viridis', edgecolor='none')

        plt.figure(figsize=(10, 6))
        tri = Triangulation(self.plist[:, 0], self.plist[:, 1], self.tlist)
        plt.tricontourf(tri, func_val, cmap="viridis")
        plt.colorbar(label="f(x, y)")
        # plt.triplot(tri, color="gray", alpha=0.5)
        # plt.scatter(self.plist[:, 0], self.plist[:, 1], c=func_val)

    def schwerpunkt(self, triangle):
        # Berechnet den Schwerpunkt eines Dreiecks gegeben durch die Indizes der Punkte
        p1, p2, p3 = self.plist[triangle]
        return (p1 + p2 + p3) / 3

    def aufgabe_e(self):
        print("------------ e) ------------")
        if self.tlist is None or self.plist is None:
            self.aufgabe_b()

        # all triangles in upper right corner
        x_min, x_max = np.min(self.plist[:, 0]), np.max(self.plist[:, 0])
        y_min, y_max = np.min(self.plist[:, 1]), np.max(self.plist[:, 1])
        x_diff = x_max - x_min
        y_diff = y_max - y_min
        x_mid = x_min + x_diff / 2
        y_mid = y_min + y_diff / 2

        print(f"x_diff: {x_diff}, y_diff: {y_diff}")
        print(f"Upper right corner starts at x > {x_mid} and y > {y_mid}")

        # plt.figure(figsize=(10, 6))
        plt.triplot(self.plist[:, 0], self.plist[:, 1], self.tlist, color="gray", alpha=0.5)

        for triangle in self.tlist:
            # calculate the schwerpunkt of all triangles
            sp = self.schwerpunkt(triangle)

            # check if the schwerpunkt is in the upper right corner
            if sp[0] > x_mid and sp[1] > y_mid:
                plt.plot(sp[0], sp[1], marker="x", color="red")

    def aufgabe_f(self):
        print("------------ f) ------------")
        # ------------------------------------- ohne Kreis -------------------------------------
        # Anfangs und Endpunkt
        y0 = 1
        yN = 5
        x0 = -1
        xN = 2

        # Anzahl der Positionen N=N0*N0 Anzahl Dreiecke etwa N0*N0/2
        N0 = 200

        # Lege Punkte fest
        points = []
        for i in range(N0):
            for j in range(N0):
                x = x0 + i * (xN - x0) / (N0 - 1)
                y = y0 + j * (yN - y0) / (N0 - 1)
                points += [[x, y]]
        points = np.array(points)

        # Nummeriere und lege Elemente fest
        tri = Delaunay(points)

        t = tri.simplices
        p = 1.0 * points

        plt.figure(figsize=(10, 6))
        plt.triplot(p[:, 0], p[:, 1], t)
        plt.plot(p[:, 0], p[:, 1], "x")

        self.tlist = t
        self.plist = p


        # ------------------------------------- mit Kreis -------------------------------------
        x0 = -1
        xN = 2
        y0 = 1
        yN = 5
        N0 = 200

        # cut circle out
        radius = 0.5
        center = np.array([0.5, 3])
        # (x - center_x)² + (y - center_y)² = r²
        mask = (p[:, 0] - center[0]) ** 2 + (p[:, 1] - center[1]) ** 2 > radius**2
        self.plist =  self.plist[mask]
        tri = Delaunay(self.plist)
        self.tlist = tri.simplices

        plt.figure(figsize=(10, 6))
        plt.triplot(self.plist[:, 0], self.plist[:, 1], self.tlist)
        plt.plot(self.plist[:, 0], self.plist[:, 1], "x")

    def execute_aufgabe(self, aufgabe=[]):
        aufgaben = {
            # 'a': self.aufgabe_a,
            # "b": self.aufgabe_b,
            # "c": self.aufgabe_c,
            # 'd': self.aufgabe_d,
            # 'e': self.aufgabe_e,
            "f": self.aufgabe_f,
        }
        to_execute = aufgabe if aufgabe else aufgaben.keys()

        for e in to_execute:
            if e in aufgaben:
                aufgaben[e]()
            else:
                print(f"Ungültige Aufgabe: {e}")

        plt.show()


if __name__ == "__main__":
    # a1 = Aufgabe1()
    # a1.execute_aufgabe()

    # a2 = Aufgabe2()
    # a2.execute_aufgabe()

    a3 = Aufgabe3()
    a3.execute_aufgabe()
