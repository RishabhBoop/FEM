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
def aufgabe_1():
    # consts
    a = math.sqrt(2)
    b = 5*math.e
    N = 5


    # ---------------------------- a) ----------------------------
    print("------------ a) ------------")
    p0_0 = np.linspace(a, b, num=N)
    p0_1 = np.array(range(N))
    p0 = np.array(list(zip(p0_0, p0_1)))
    print("p0:", p0)

    # ---------------------------- b) ----------------------------
    print("------------ b) ------------")
    p1_0 = np.random.uniform(a, b, size=N)
    p1_1 = np.array(range(N))
    np.random.shuffle(p1_1)
    p1 = np.array(list(zip(p1_0, p1_1)))
    print("p1:", p1)

    # ---------------------------- c) ----------------------------
    # p0_0 enthält sortierte x_k Werte,
    # p1_1 enthält zufällige positionswerte von 0 bis N-1
    print("------------ c) ------------")
    p2 = np.array(list(zip(p0_0, p1_1)))
    print("p2:", p2)

    # ---------------------------- d) ----------------------------
    print("------------ d) ------------")
    print("--- p0 ---")
    t0 = generate_t(p0)
    #print("t0:", t0)

    abstaende = get_teil_Abstand(p0, t0)
    maxAbstand = np.max(abstaende)
    minAbstand = np.min(abstaende)
    meanAbstand = np.mean(abstaende)
    print(f"Maximaler Abstand p0: {maxAbstand}")
    print(f"Minimaler Abstand p0: {minAbstand}")
    print(f"Mittlerer Abstand p0: {meanAbstand}")

    print("--- p1 ---")
    t1 = generate_t(p1)
    #print("t1:", t1)
    abstaende = get_teil_Abstand(p1, t1)
    maxAbstand = np.max(abstaende)
    minAbstand = np.min(abstaende)
    meanAbstand = np.mean(abstaende)
    print(f"Maximaler Abstand p1: {maxAbstand}")
    print(f"Minimaler Abstand p1: {minAbstand}")
    print(f"Mittlerer Abstand p1: {meanAbstand}")


    print("--- p2 ---")
    t2 = generate_t(p2) 
    #print("t2:", t2)
    abstaende = get_teil_Abstand(p2, t2)
    maxAbstand = np.max(abstaende)
    minAbstand = np.min(abstaende)
    meanAbstand = np.mean(abstaende)
    print(f"Maximaler Abstand p2: {maxAbstand}")
    print(f"Minimaler Abstand p2: {minAbstand}")
    print(f"Mittlerer Abstand p2: {meanAbstand}")

    print("--- Erstellung mit N=10000 ---")
    N = 100000
    Np0_0 = np.random.uniform(a, b, size=N)
    Np0_1 = np.array(range(N))
    Np0 = np.array(list(zip(Np0_0, Np0_1)))
    Nt0 = generate_t(Np0)
    abstaende = get_teil_Abstand(Np0, t0)
    maxAbstand = np.max(abstaende)
    minAbstand = np.min(abstaende)
    meanAbstand = np.mean(abstaende)
    print(f"Maximaler Abstand p0: {maxAbstand}")
    print(f"Minimaler Abstand p0: {minAbstand}")
    print(f"Mittlerer Abstand p0: {meanAbstand}")
    plt.hist(abstaende, bins=500)

    # ---------------------------- e) ----------------------------
    print("------------ e) ------------")
    # p2 = np.array(p2)
    cond_to_remove = (p2[:, 1] % 3 == 0) & (p2[:, 1] != 0)
    newP2 = p2[~cond_to_remove]
    print("new p2:", newP2)
    newT2 = generate_t(newP2)
    print("new t2:", newT2)

    # ---------------------------- f) ----------------------------
    print("------------ f) ------------")
    def func(x):
        # f(x) = ln(x^2)
        return 2 * np.log(x)

    # p0 = np.array(p0)
    # p1 = np.array(p1)
    # p2 = np.array(p2)
    y_p0 = func(p0[:,0])
    y_p1 = func(p1[:,0])
    y_newP2 = func(newP2[:,0])
    print(y_p0)
    print(y_p1)
    print(y_newP2)

    plt.figure(figsize=(10, 6))
    plt.scatter(p0[:,0], y_p0, label='p0', marker='o', linewidths=0.3)
    plt.scatter(p1[:,0], y_p1, label='p1', marker='o', linewidths=0.3)
    plt.scatter(newP2[:,0], y_newP2, label='p2', marker='o', linewidths=0.3)
    plt.title("Auswertung der Funktion an verschiedenen Knotenpunkten (scatter plot)")
    plt.xlabel("x (P-Liste)")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.autoscale()

    # ---------------------------- g) ----------------------------
    print("------------ g) ------------")
    abstaende_p0 = get_teil_Abstand(p0, t0)
    abstaende_p1 = get_teil_Abstand(p1, t1)
    abstaende_p2 = get_teil_Abstand(newP2, newT2)
    abstaende_p2_1000 = get_teil_Abstand(Np0, Nt0)
    area_p0 = area_under_kurve(p0[:,0], y_p0, abstaende_p0)
    area_p1 = area_under_kurve(p1[:,0], y_p1, abstaende_p1)
    area_p2 = area_under_kurve(newP2[:,0], y_newP2, abstaende_p2)
    area_p2_1000 = area_under_kurve(Np0[:,0], func(Np0[:,0]), abstaende_p2_1000)
    print(f"Fläche unter der Kurve für p0 (N=05): {area_p0}")
    print(f"Fläche unter der Kurve für p1 (N=05): {area_p1}")
    print(f"Fläche unter der Kurve für p2 (N=05): {area_p2}")
    print(f"Fläche unter der Kurve für p2 (N=1k): {area_p2_1000}")
    exact_area = exaact_area_under_kurve(a, b)
    print(f"Exakte Fläche unter der Kurve: {exact_area}")

    plt.show()



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




aufgabe_2()