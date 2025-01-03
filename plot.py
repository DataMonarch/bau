import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def read_results_from_csv(filename="sorting_results.csv"):
    """
    Reads benchmark results from a CSV file.
    Each row is expected to have: 'algorithm', 'array_size', 'average_time_s'.

    Returns:
        A list of dicts:
            [
              {'algorithm': str, 'array_size': int, 'average_time_s': float},
              ...
            ]
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file '{filename}' does not exist.")

    results = []
    with open(filename, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["array_size"] = int(row["array_size"])
            row["average_time_s"] = float(row["average_time_s"])
            results.append(row)
    return results


def plot_logarithmic_data(results, output_image="sorting_log_plot.png"):
    """
    Given results with keys:
      - 'algorithm'
      - 'array_size'
      - 'average_time_s'
    Produces a log-log plot of array_size vs. average_time_s for each algorithm.
    """
    # Group data by algorithm
    data_by_algo = {}
    for r in results:
        algo = r["algorithm"]
        n = r["array_size"]
        t = r["average_time_s"]
        data_by_algo.setdefault(algo, []).append((n, t))

    # Sort each algorithm's data by array size for consistent plotting
    for algo in data_by_algo:
        data_by_algo[algo].sort(key=lambda x: x[0])

    # Initialize the plot
    plt.figure(figsize=(8, 5))

    # Plot each algorithm
    for algo, vals in data_by_algo.items():
        sizes = [v[0] for v in vals]
        times = [v[1] for v in vals]
        plt.plot(sizes, times, marker="o", label=algo)

    # Use logarithmic scales on both axes
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Array Size (N) [Log Scale]")
    plt.ylabel("Time (Seconds) [Log Scale]")
    plt.title("Sorting Algorithms: Log-Log Plot of Empirical Times")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.savefig(output_image, dpi=300)
    print(f"Logarithmic plot saved to {output_image}")
    plt.close()


def main():
    # 1. Read the CSV
    csv_filename = "sorting_results.csv"
    results = read_results_from_csv(csv_filename)

    # 2. Plot the data on a log-log scale
    plot_logarithmic_data(results, output_image="sorting_log_plot.png")


if __name__ == "__main__":
    main()
