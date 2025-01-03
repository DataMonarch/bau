import logging
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt

###############################################################################
# Logging Setup
###############################################################################
logging.basicConfig(
    filename="sorting_benchmarks.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

###############################################################################
# Sorting Algorithm Implementations
###############################################################################


def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr


def quick_sort(arr):
    _quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_helper(arr, low, high):
    if low < high:
        pivot_index = _partition(arr, low, high)
        _quick_sort_helper(arr, low, pivot_index - 1)
        _quick_sort_helper(arr, pivot_index + 1, high)


def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


###############################################################################
# Dictionary of available algorithms + time complexities
###############################################################################
# We’ll map each algorithm to its approximate worst-case Big O, for plotting:
#   'selection', 'bubble', 'insertion' -> O(n^2)
#   'merge', 'quick' -> O(n log n)
#
ALGORITHMS = {
    "selection": {"function": selection_sort, "complexity": "n^2"},  # Theoretical Big O
    "bubble": {"function": bubble_sort, "complexity": "n^2"},
    "insertion": {"function": insertion_sort, "complexity": "n^2"},
    "merge": {"function": merge_sort, "complexity": "n log n"},
    "quick": {"function": quick_sort, "complexity": "n log n"},
}


###############################################################################
# Benchmark a Single Algorithm
###############################################################################
def benchmark_algorithm(algorithm_name, sizes, num_runs=3):
    """
    Benchmarks the specified sorting algorithm for each size in 'sizes'.
    Returns a list of dicts: [{'algorithm', 'array_size', 'average_time_s'}, ...]
    """
    algo_info = ALGORITHMS[algorithm_name]
    sort_func = algo_info["function"]

    results = []

    for n in sizes:
        logging.info(f"Benchmarking {algorithm_name} with array size {n}")

        total_time = 0.0
        for _ in range(num_runs):
            # Create a random array of size n
            arr = np.random.randint(1, 1001, size=n)

            # Time the sort
            start_time = time.perf_counter()
            sort_func(arr)  # Not logging inside the loop
            end_time = time.perf_counter()

            total_time += end_time - start_time

        avg_time = total_time / num_runs
        logging.info(
            f"{algorithm_name} with n={n} took {avg_time:.6f}s (avg over {num_runs} runs)"
        )

        results.append(
            {"algorithm": algorithm_name, "array_size": n, "average_time_s": avg_time}
        )

    return results


###############################################################################
# Benchmark All Algorithms
###############################################################################
def benchmark_all_algorithms(sizes, num_runs=3):
    """
    Benchmarks all algorithms in ALGORITHMS over given sizes.
    Returns a list of all results combined.
    """
    all_results = []
    for algo_name in ALGORITHMS.keys():
        algo_results = benchmark_algorithm(algo_name, sizes, num_runs)
        all_results.extend(algo_results)
    return all_results


###############################################################################
# Save Results to CSV
###############################################################################
def save_results_to_csv(results, filename="sorting_results.csv"):
    """
    Saves the benchmark results (list of dictionaries) to a CSV file.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["algorithm", "array_size", "average_time_s"]
        )
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    logging.info(f"Results saved/appended to {filename}")


###############################################################################
# Plotting
###############################################################################
def plot_results(results, output_image="sorting_comparison.png"):
    """
    Given results (list of dicts with: algorithm, array_size, average_time_s),
    plots empirical times vs. theoretical complexity for each sorting method.
    """
    # Group results by algorithm
    data_by_algo = {}
    for r in results:
        algo = r["algorithm"]
        size = r["array_size"]
        time_s = r["average_time_s"]
        data_by_algo.setdefault(algo, []).append((size, time_s))

    # Sort data by N for each algorithm
    for algo in data_by_algo:
        data_by_algo[algo].sort(key=lambda x: x[0])

    # Prepare figure
    plt.figure(figsize=(10, 6))

    # Define simple theoretical models (for worst-case):
    #   n^2 (selection, bubble, insertion)
    #   n log n (merge, quick)
    def theoretical_n2(n):
        return n**2

    def theoretical_nlogn(n):
        return n * np.log2(n) if n > 1 else n

    colors = {
        "selection": "blue",
        "bubble": "orange",
        "insertion": "green",
        "merge": "red",
        "quick": "purple",
    }

    for algo, vals in data_by_algo.items():
        # Extract real data
        sizes = [v[0] for v in vals]
        times = [v[1] for v in vals]

        plt.plot(
            sizes,
            times,
            marker="o",
            color=colors.get(algo, "black"),
            label=f"{algo.capitalize()} (empirical)",
        )

        # Determine the theoretical curve
        if ALGORITHMS[algo]["complexity"] == "n^2":
            theo_vals = [theoretical_n2(s) for s in sizes]
        else:
            theo_vals = [theoretical_nlogn(s) for s in sizes]

        # Scale the theoretical curve for visual comparison
        max_empirical = max(times)
        max_theoretical = max(theo_vals)
        scale = max_empirical / max_theoretical if max_theoretical != 0 else 1
        theo_scaled = [v * scale for v in theo_vals]

        plt.plot(
            sizes,
            theo_scaled,
            linestyle="--",
            color=colors.get(algo, "black"),
            label=f"{algo.capitalize()} (theoretical scaled)",
        )

    plt.xlabel("Array Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("Empirical vs. Theoretical Complexity (All Algorithms)")
    plt.legend()
    plt.grid(True)

    # Save figure
    plt.savefig(output_image, dpi=300)
    logging.info(f"Comparison plot saved to {output_image}")
    plt.close()


###############################################################################
# Main
###############################################################################
def main():
    # We will test these sizes
    sizes = [10**2, 10**3, 10**4]

    logging.info("Starting benchmark for all sorting algorithms")

    # Benchmark all algorithms
    all_results = benchmark_all_algorithms(sizes=sizes, num_runs=3)

    # Save the results to CSV
    save_results_to_csv(all_results, filename="sorting_results.csv")

    # Plot the combined results
    # (We could read from CSV, but since we already have 'all_results', we can use that)
    plot_results(all_results, output_image="sorting_comparison.png")

    logging.info("Benchmark completed for all sorting algorithms")


if __name__ == "__main__":
    main()
