import math
import time

import matplotlib.pyplot as plt
import numpy as np

def pi_estimate(N: int) -> float:
    """
    Estimate pi by Monte Carlo in the *quarter* unit circle.

    We sample uniformly in the unit square: x,y ~ U(0,1).
    A point is inside the quarter circle of radius 1 (center at origin) if:
        x^2 + y^2 <= 1
    The unit square area is 1.
    The quarter circle area is (pi * 1^2) / 4 = pi/4.
    So fraction_inside ~ (pi/4) / 1  =>  pi ≈ 4 * fraction_inside.
    """
    xs = np.random.random(N)  # uniform on [0,1)
    ys = np.random.random(N)  # uniform on [0,1)

    inside = (xs * xs + ys * ys) <= 1.0
    count_inside = np.sum(inside)

    pi_est = 4 * (count_inside / N)
    return float(pi_est)


def part1(MC_steps, repeats:int = 10):
    """
    Calculates error range and estimates of pi for a given rectangle area and number of steps
    x_max: max value of x
    y_max: max value of y
    MC_steps: number of monte carlo steps to run 
    returns: results tuple of N and the mean of pi estimate (and SE), MAE, and time (and SE)
    """
    # The area of our square
    A = 1

    results = []
    for N in MC_steps:
        pis = []
        errors = []
        elaps = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            pi_est = pi_estimate(N)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            error = abs(pi_est-math.pi) / math.pi
            pis.append(pi_est)
            errors.append(error)
            elaps.append(elapsed)
        # get averages
        pis_avg = np.mean(pis)
        error_avg = np.mean(errors)
        elaps_avg = np.mean(elaps)

        # get standard errors
        pis_se = np.std(pis, ddof=1) / np.sqrt(repeats)
        elaps_se = np.std(elaps, ddof=1) / np.sqrt(repeats)

        results.append((N, pis_avg, error_avg, elaps_avg, pis_se, elaps_se))
    return results


def save_table_png(results, filename="pi_mc_results.png"):
    """
    Saving our tabulated results as a nice png
    results: list of results
    returns: nothing
    """
    headers = ["N", "Pi Estimate", "MAPE", "Avg Time (s)"]

    table_data = [
        [f"{N:.0e}", f"{pi:.6f} ± {pi_se:.6f}", f"{100*error:.6f}%", 
        f"{elapsed:.6f} ± {elapsed_se:.6f}"]
        for N, pi, error, elapsed, pi_se, elapsed_se in results
    ]

    fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(table_data)))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_results(results, filename = "plot_pi_mc.png"):
    """
    Plotting time, pi est, and MAPE vs. N, all error bars are in Std. Error
    """
    N = np.array([r[0] for r in results], dtype = float)
    pis = np.array([r[1] for r in results], dtype = float)
    mape_mean = np.array([r[2] for r in results], dtype=float)
    t_mean = np.array([r[3] for r in results], dtype=float)

    pi_se = np.array([r[4] for r in results], dtype=float)
    t_se = np.array([r[5] for r in results], dtype=float)

    fig, axes = plt.subplots(3,1,figsize=(7,10), sharex = True)
    # pi vs. N
    ax = axes[0]
    ax.errorbar(N, pis, yerr=pi_se, fmt="o-", capsize=2)
    ax.axhline(math.pi, linestyle="--", label=r"Value of $\pi$", color="red")
    ax.legend()
    ax.set_ylabel(r"$\pi$ estimate")
    ax.set_xscale("log")

    # time vs. N
    ax = axes[1]
    ax.errorbar(N, t_mean, yerr=t_se, fmt="o-", capsize=2)
    ax.set_ylabel("Simulation time (s)")
    ax.set_xscale("log")

    # MaPE vs. N
    ax = axes[2]
    ax.errorbar(N, 100*mape_mean, fmt="o-", capsize=2)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("N samples")
    ax.set_xscale("log")
    ax.set_yscale("log")  # log log plot to get the nice linear decrease in error

    labels = ["(a)", "(b)", "(c)"]
    for ax, label in zip(axes, labels):
        ax.text(0.02, 0.95, label,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top")


    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    MC_steps = 10 ** np.arange(1, 9, 1) # exp of 10 are 2, 3, 4...8, added 10 for "warm up for timing"
    results = part1(MC_steps)

    # Tabulated Results 
    print("N\tPi Estimate\tError\t\tTime Elapsed")
    for N, pi, error, elapsed, pi_se, time_se in results:
        print(f"{N:.0e}\t{pi:.6f}\t{error:.6f}\t{elapsed:.6f}")

    save_table_png(results)
    plot_results(results)
