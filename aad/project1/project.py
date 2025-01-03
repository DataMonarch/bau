import logging
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd  # <-- NEW: Use pandas for CSV reading/writing

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


###############################################################################
# Logging Setup
###############################################################################
logging.basicConfig(
    filename=os.path.join(PARENT_DIR, "sorting_benchmark.log"),
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
ALGORITHMS = {
    "selection": {"function": selection_sort, "complexity": "n^2"},
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
    Returns a list of dicts: [{'algorithm', 'array_size', 'average_time_ns'}, ...]
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
            start_time = time.perf_counter_ns()
            sort_func(arr)  # Not logging inside the loop
            end_time = time.perf_counter_ns()

            total_time += end_time - start_time

        avg_time = total_time / num_runs
        logging.info(
            f"{algorithm_name} with n={n} took {avg_time:.6f}s (avg over {num_runs} runs)"
        )

        results.append(
            {"algorithm": algorithm_name, "array_size": n, "average_time_ns": avg_time}
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
# Save/Read Results with pandas
###############################################################################
def save_results_to_csv(results, filename="sorting_results.csv"):
    """
    Saves the benchmark results (list of dictionaries) to a CSV file using pandas.
    Appends if the file exists, otherwise writes a new file.
    """
    df = pd.DataFrame(results)
    file_exists = os.path.isfile(filename)
    # If file exists, append; otherwise write new
    df.to_csv(
        filename, mode="a" if file_exists else "w", header=not file_exists, index=False
    )
    logging.info(f"Results saved/appended to {filename}")


def read_results_from_csv(filename="sorting_results.csv"):
    """
    Reads the benchmark results from a CSV file into a pandas DataFrame.
    Returns the DataFrame or None if the file doesn't exist.
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        logging.info(f"Results read from {filename}")
        return df
    else:
        logging.warning(f"No existing file {filename} to read from.")
        return None


###############################################################################
# Plotting
###############################################################################
def plot_results(
    df,
    output_image_linear="sorting_comparison_linear.png",
    output_image_log="sorting_comparison_log.png",
):
    """
    Given a pandas DataFrame with columns [algorithm, array_size, average_time_ns],
    plots empirical times vs. theoretical complexity for each sorting method.
    Produces two plots:
      1) Linear scale (x- and y-axes)
      2) Log scale (x- and y-axes)
    """
    # Group results by algorithm
    data_by_algo = {}
    for row in df.itertuples(index=False):
        algo = row.algorithm
        size = row.array_size
        time_s = row.average_time_ns
        data_by_algo.setdefault(algo, []).append((size, time_s))

    # Sort data by N for each algorithm
    for algo in data_by_algo:
        data_by_algo[algo].sort(key=lambda x: x[0])

    # Theoretical models
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

    ###################################################################
    # 1) Linear Scale Plot
    ###################################################################
    plt.figure(figsize=(10, 6))

    for algo, vals in data_by_algo.items():
        sizes = [v[0] for v in vals]
        times = [v[1] for v in vals]

        # Empirical
        plt.plot(
            sizes,
            times,
            marker="o",
            color=colors.get(algo, "black"),
            label=f"{algo.capitalize()} (empirical)",
        )

        # Theoretical
        if ALGORITHMS[algo]["complexity"] == "n^2":
            theo_vals = [theoretical_n2(s) for s in sizes]
        else:
            theo_vals = [theoretical_nlogn(s) for s in sizes]

        # Scale for visual comparison
        max_empirical = max(times)
        max_theoretical = max(theo_vals)
        scale = max_empirical / max_theoretical if max_theoretical != 0 else 1
        theo_scaled = [v * scale for v in theo_vals]

        plt.plot(
            sizes,
            theo_scaled,
            linestyle="--",
            color=colors.get(algo, "black"),
            label=f"{algo.capitalize()} - {ALGORITHMS[algo]['complexity']} (theoretical scaled)",
        )

    plt.xlabel("Array Size (N)")
    plt.ylabel("Time (Nanoseconds)")
    plt.title("Empirical vs. Theoretical Complexity (Linear Scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image_linear, dpi=300)
    logging.info(f"Linear scale comparison plot saved to {output_image_linear}")
    plt.close()

    ###################################################################
    # 2) Log-Log Scale Plot
    ###################################################################
    plt.figure(figsize=(10, 6))

    for algo, vals in data_by_algo.items():
        sizes = [v[0] for v in vals]
        times = [v[1] for v in vals]

        # Empirical
        plt.plot(
            sizes,
            times,
            marker="o",
            color=colors.get(algo, "black"),
            label=f"{algo.capitalize()} (empirical)",
        )

        # Theoretical
        if ALGORITHMS[algo]["complexity"] == "n^2":
            theo_vals = [theoretical_n2(s) for s in sizes]
        else:
            theo_vals = [theoretical_nlogn(s) for s in sizes]

        # Scale for visual comparison
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

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Array Size (N) [Log Scale]")
    plt.ylabel("Time (Nanoseconds) [Log Scale]")
    plt.title("Empirical vs. Theoretical Complexity (Log Scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image_log, dpi=300)
    logging.info(f"Log scale comparison plot saved to {output_image_log}")
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

    # Save the results to CSV (append if exists, otherwise create new)
    res_filepath = os.path.join(PARENT_DIR, "sorting_results.csv")
    save_results_to_csv(all_results, filename=res_filepath)

    # Read the combined results from CSV into a pandas DataFrame
    df = read_results_from_csv(filename=res_filepath)

    # If we have valid data, plot
    if df is not None and not df.empty:
        plot_results(
            df,
            output_image_linear=os.path.join(
                PARENT_DIR, "sorting_comparison_linear.png"
            ),
            output_image_log=os.path.join(PARENT_DIR, "sorting_comparison_log.png"),
        )

    logging.info("Benchmark completed for all sorting algorithms")


if __name__ == "__main__":
    main()
