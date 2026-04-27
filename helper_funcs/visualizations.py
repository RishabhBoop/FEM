import matplotlib.pyplot as plt
import numpy as np
from os import path

def print_timings(timings, title="TIMING - C++ full_solve()", n_points=None, n_elems=None, save_to_file=False, backend=None, export_filename=None):
    """
    Prints the performance output dynamically, preserving the order from the C++ vector of tuples.
    Expects `timings` to be a list of tuples: e.g., [("gen_tlist", 12.3), ("solve_LGS", 45.6)]
    """
    if n_points is not None and n_elems is not None:
        title_str = f"{title} ({n_points} Punkte, {n_elems} Elemente)"
    else:
        title_str = title

    if not n_points:
        n_points = "xx"

    if not n_elems:
        n_elems = "xx"
    

    lines = []
    tot_str = ""

    # Find the longest step name to dynamically align the milliseconds perfectly
    # timings is now a list of lists/tuples, where item[0] is the name
    max_key_len = max([len(str(item[0])) for item in timings] + [10])

    # Iterating through the list (step_name = item[0], t_val = item[1])
    for step_name, t_val in timings:
        formatted_line = f"{step_name}:".ljust(max_key_len + 2) + f"{t_val:.3f} ms"

        # Isolate 'total_time' so we can print it at the very bottom below the separator
        if step_name == "total_time":
            tot_str = formatted_line
        else:
            lines.append(formatted_line)

    if not tot_str:
        tot_str = "Total Time:".ljust(max_key_len + 2) + "N/A"

    # Calculate required box width based on the longest string
    len_sep = max(max((len(l) for l in lines), default=0), len(tot_str), len(title_str)) + 4

    print()
    print("=" * len_sep)
    print(f"{title_str:^{len_sep}}")
    print("-" * len_sep)
    for line in lines:
        print(f" {line}")
    print("-" * len_sep)
    print(f" {tot_str}")
    print("=" * len_sep)

    if not export_filename:
        export_filename = f"{title}_{n_points}points_{n_elems}elements.txt".replace(" ", "_").replace("_-_", "_")

    export_filename = "Times__" +  backend + "__" + export_filename + ".txt"

    if save_to_file:
        filename = export_filename
        with open(filename, "w") as f:
            f.write(f"{title_str}\n")
            f.write("=" * len_sep + "\n")
            f.write("-" * len_sep + "\n")
            for line in lines:
                f.write(f" {line}\n")
            f.write("-" * len_sep + "\n")
            f.write(f" {tot_str}\n")


def print_error_stats(error_stats, title="Validierung", n_elems=None, save_to_file=False, backend=None, export_filename=None):
    """
    Prints the maximum, minimum, and mean absolute error from the C++ validation.
    """
    # Unpack the list sent from C++ (max_abs_error, min_abs_error, mean_abs_error)
    max_err, min_err, mean_err = error_stats

    if n_elems is not None:
        title_raw = f"Abweichungen für {title} ({n_elems} Elemente):"
    else:
        title_raw = f"Abweichungen für {title}:"

    max_str = f"Maximale Abweichung: {max_err:.6e}"
    min_str = f"Minimale Abweichung: {min_err:.6e}"
    mean_str = f"Mittlere Abweichung: {mean_err:.6e}"

    if max_err > ERROR_TOLERANCE:
        title_str = colors.RED + title_raw + " [FAIL]" + colors.RESET
        file_title = title_raw + " [FAIL]"
    else:
        title_str = colors.GREEN + title_raw + " [PASS]" + colors.RESET
        file_title = title_raw + " [PASS]"

    len_sep = max(len(max_str), len(min_str), len(mean_str), len(title_raw) + 7) + 4

    print()
    print("=" * len_sep)
    print(f"{title_str:^{len_sep + len(colors.RED) + len(colors.RESET)}}")
    print("-" * len_sep)
    print(f" {max_str}")
    print(f" {min_str}")
    print(f" {mean_str}")
    print("=" * len_sep)

    if save_to_file:
        if not export_filename:
            export_filename = f"ErrorStats_{title}.txt".replace(" ", "_")
        if backend:
            export_filename = f"ErrorStats__{backend}__{export_filename}"
        with open(export_filename, "w") as f:
            f.write(f"{file_title}\n")
            f.write("=" * len_sep + "\n")
            f.write("-" * len_sep + "\n")
            f.write(f" {max_str}\n")
            f.write(f" {min_str}\n")
            f.write(f" {mean_str}\n")
            f.write("=" * len_sep + "\n")


def visualize_error(plist, error, title="Fehlerverteilung", save_to_file=False, backend=None, export_filename=None):
    plt.figure(figsize=(12, 5))
    plt.plot(plist, error, marker="o", linestyle="", label="Fehler")
    plt.xlabel("x")
    plt.ylabel("Absoluter Fehler")
    plt.title(title)
    plt.legend()
    
    if save_to_file:
        if not export_filename:
            export_filename = f"{title}.png".replace(" ", "_")
        if backend:
            export_filename = f"{backend}__{export_filename}"
        plt.savefig(export_filename)


def visualize_solution(plist, solution, title="Lösung", save_to_file=False, backend=None, export_filename=None):
    plt.figure(figsize=(12, 5))
    plt.scatter(plist, solution, color="blue", label="Lösung (phi)")
    plt.xlabel("x")
    plt.ylabel("PHI(x)")
    plt.title(title)
    plt.legend()
    
    if save_to_file:
        if not export_filename:
            export_filename = f"{title}.png".replace(" ", "_")
        if backend:
            export_filename = f"{backend}__{export_filename}"
        plt.savefig(export_filename)
