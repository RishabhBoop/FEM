import numpy as np
import math
import matplotlib.pyplot as plt
import time
# --------------------------------------------------

# d)
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

        # Show all plots at the end
        plt.show()


    def all_aufgaben(self):
        self.aufgabe_a()
        self.aufgabe_b()
        self.aufgabe_c()
        self.aufgabe_d()
        self.aufgabe_e()
        self.aufgabe_f()
        self.aufgabe_g()


def aufgabe_2():
    # a)
    dimMatrix = 10

    matrA = np.random.rand(dimMatrix, dimMatrix)
    vecB = np.random.rand(dimMatrix)

    x = np.linalg.solve(matrA, vecB)
    print("A =\n", matrA)
    print("B =\n", vecB)
    print("x =\n", x)

    # b)
    n_t = {}
    for i in range(500, 15000, 1000):
        matrA = np.random.rand(i, i)
        vecB = np.random.rand(i)
        start_time = time.time()
        x = np.linalg.solve(matrA, vecB)
        end_time = time.time()
        print(f"Zeit für Dimension {i}: {end_time - start_time:.4f} Sekunden")
        n_t[i] = end_time - start_time
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(n_t.keys()), list(n_t.values()), marker='o')
    plt.title("Zeit um NxN Matrix zu lösen")
    plt.xlabel("Dimension N")
    plt.ylabel("Zeit (Sekunden)")
    plt.grid(True, linestyle=':', alpha=0.7)



    plt.show()




if __name__ == "__main__":
    # a1 = Aufgabe1()
    # a1.all_aufgaben()
    aufgabe_1()
