# === RFFG Simulation & Plotting Toolkit (clean, arXiv-friendly) ===
# - Pure numpy + matplotlib (no seaborn, no styles, no custom colors)
# - Synchronous two-phase update (recommended), optional asynchronous in-place
# - Conflict detection via largest-gap split with persistence
# - Presets: T1, T2, T3, T4 (+ stricter, async)
# - Improved plots: Histogram (μ/σ + Gaussian-Fit), Temporal (consensus + lock band), Phase (sorted option)
# - Uniform axes/bins across runs
# - Exports figures and CSV logs to /mnt/data

import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import os

# ---------------------- GLOBAL PLOT SETTINGS ----------------------
HIST_BINS = np.arange(3010, 3036, 1)   # common histogram bin edges
HIST_XLIM = (HIST_BINS[0], HIST_BINS[-1])
TEMPORAL_YLIM = (3010, 3036)           # common y-limits for temporal plots
PHASE_YLIM = (-12, 12)                 # common y-limits for phase plots (deviation from mean)

# ---------------------- PARAMETERS ----------------------
@dataclass
class RFFGParams:
    N: int = 10
    W: int = 5
    step: int = 1
    T: int = 500
    init_low: int = 3018
    init_high: int = 3029
    mode: str = "sync"                  # "sync" or "async"
    consensus_method: str = "window_mean"  # "window_mean" | "mean" | "median"
    conflict_delta: int = 1
    conflict_persist: int = 3
    seed: Optional[int] = 42
    enable_oscillation: bool = True
    oscillate_smaller_group_only: bool = False

# ---------------------- CORE LOGIC ----------------------
def compute_consensus(x: np.ndarray, W: int, method: str) -> float:
    xmean = np.mean(x)
    if method == "window_mean":
        in_win = np.abs(x - xmean) <= W
        if not np.any(in_win):
            return float(xmean)
        return float(np.mean(x[in_win]))
    elif method == "mean":
        return float(xmean)
    elif method == "median":
        return float(np.median(x))
    else:
        raise ValueError("Unknown consensus_method")

def largest_gap_split(x: np.ndarray) -> Tuple[int, float]:
    xs = np.sort(x)
    gaps = xs[1:] - xs[:-1]
    if len(gaps) == 0:
        return -1, 0.0
    k = int(np.argmax(gaps))
    return k, float(gaps[k])

def detect_bimodal_conflict(x: np.ndarray, W: int, delta: int) -> Tuple[bool, np.ndarray, np.ndarray]:
    xs = np.sort(x)
    idx, _ = largest_gap_split(xs)
    if idx < 0:
        return False, np.array([], dtype=int), np.array([], dtype=int)
    left = xs[:idx+1]
    right = xs[idx+1:]
    if len(left) < 2 or len(right) < 2:
        return False, np.array([], dtype=int), np.array([], dtype=int)
    sep = float(np.mean(right) - np.mean(left))
    if sep > (2*W + delta):
        left_idx = np.where(np.isin(x, left))[0]
        right_idx = np.where(np.isin(x, right))[0]
        return True, left_idx, right_idx
    return False, np.array([], dtype=int), np.array([], dtype=int)

def rffg_update_sync(x: np.ndarray, y: np.ndarray, params: RFFGParams,
                     conflict_state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float, int]:
    # Two-phase: consensus from x(t); then produce x(t+1), y(t+1)
    consensus = compute_consensus(x, params.W, params.consensus_method)
    in_win = np.abs(x - consensus) <= params.W
    x_new = x.copy()
    y_new = y.copy()

    # Rule 1: Stability
    y_new[in_win] = consensus

    # Rule 2: Adjustment
    gt = x > (consensus + params.W)
    lt = x < (consensus - params.W)
    x_new[gt] = x[gt] - params.step
    x_new[lt] = x[lt] + params.step

    # Rule 3: Oscillation (via persistent conflict)
    conflict, L, R = detect_bimodal_conflict(x, params.W, params.conflict_delta)
    if conflict:
        conflict_state["persist"] += 1
        conflict_state["last_groups"] = (L, R)
    else:
        conflict_state["persist"] = 0
        conflict_state["last_groups"] = (np.array([], dtype=int), np.array([], dtype=int))

    if params.enable_oscillation and conflict_state["persist"] >= params.conflict_persist:
        l_idx, r_idx = conflict_state["last_groups"]
        if params.oscillate_smaller_group_only:
            if len(l_idx) <= len(r_idx):
                y_new[l_idx] = 1 - y[l_idx]
            else:
                y_new[r_idx] = 1 - y[r_idx]
        else:
            y_new[l_idx] = 1 - y[l_idx]
            y_new[r_idx] = 1 - y[r_idx]

    return x_new, y_new, conflict_state, float(consensus), int(np.sum(in_win))

def rffg_update_async(x: np.ndarray, y: np.ndarray, params: RFFGParams,
                      conflict_state: Dict[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float, int]:
    consensus = compute_consensus(x, params.W, params.consensus_method)
    order = rng.permutation(len(x))
    in_win_count = 0
    for i in order:
        if abs(x[i] - consensus) <= params.W:
            y[i] = consensus
            in_win_count += 1
        elif x[i] > (consensus + params.W):
            x[i] -= params.step
        elif x[i] < (consensus - params.W):
            x[i] += params.step

    conflict, L, R = detect_bimodal_conflict(x, params.W, params.conflict_delta)
    if conflict:
        conflict_state["persist"] += 1
        conflict_state["last_groups"] = (L, R)
    else:
        conflict_state["persist"] = 0
        conflict_state["last_groups"] = (np.array([], dtype=int), np.array([], dtype=int))

    if params.enable_oscillation and conflict_state["persist"] >= params.conflict_persist:
        l_idx, r_idx = conflict_state["last_groups"]
        if params.oscillate_smaller_group_only:
            if len(l_idx) <= len(r_idx):
                y[l_idx] = 1 - y[l_idx]
            else:
                y[r_idx] = 1 - y[r_idx]
        else:
            y[l_idx] = 1 - y[l_idx]
            y[r_idx] = 1 - y[r_idx]

    return x, y, conflict_state, float(consensus), int(in_win_count)

# ---------------------- RUNNER & METRICS ----------------------
def analyze_oscillation_flags(inwin_hist: np.ndarray, conflict_persist_hist: np.ndarray, params: RFFGParams) -> Dict[str, Any]:
    oscillation_detected = bool(np.any(conflict_persist_hist >= params.conflict_persist))
    total_conflict_steps = int(np.sum(conflict_persist_hist > 0))
    final_all_locked = bool(inwin_hist[-1] == params.N)
    return {
        "oscillation_detected": oscillation_detected,
        "total_conflict_steps": total_conflict_steps,
        "final_all_locked": final_all_locked
    }

def run_experiment(params: RFFGParams,
                   init_pattern: str = "random",
                   fixed_values: Optional[List[int]] = None,
                   export_prefix: str = "rffg_run") -> Dict[str, Any]:
    rng = np.random.default_rng(params.seed)

    # x(0)
    if init_pattern == "random":
        x0 = rng.integers(low=params.init_low, high=params.init_high + 1, size=params.N).astype(float)
    elif init_pattern == "two_peaks":
        assert fixed_values is not None and len(fixed_values) == 2, "two_peaks requires two fixed values"
        half = params.N // 2
        x0 = np.array([fixed_values[0]] * half + [fixed_values[1]] * (params.N - half), dtype=float)
    elif init_pattern == "all_outside":
        x0 = np.empty(params.N, dtype=float)
        low_val = params.init_low - 2 * params.W
        high_val = params.init_high + 2 * params.W
        for i in range(params.N):
            x0[i] = low_val if (i % 2 == 0) else high_val
    else:
        raise ValueError("Unknown init_pattern")

    y0 = np.zeros(params.N, dtype=float)

    X = np.zeros((params.T + 1, params.N), dtype=float)
    Y = np.zeros((params.T + 1, params.N), dtype=float)
    consensus_hist = np.zeros((params.T + 1,), dtype=float)
    inwin_hist = np.zeros((params.T + 1,), dtype=int)
    conflict_persist_hist = np.zeros((params.T + 1,), dtype=int)

    X[0], Y[0] = x0, y0
    conflict_state = {"persist": 0, "last_groups": (np.array([], dtype=int), np.array([], dtype=int))}

    x, y = x0.copy(), y0.copy()
    for t in range(params.T):
        if params.mode == "sync":
            x, y, conflict_state, c_t, iw_t = rffg_update_sync(x, y, params, conflict_state)
        elif params.mode == "async":
            x, y, conflict_state, c_t, iw_t = rffg_update_async(x, y, params, conflict_state, rng)
        else:
            raise ValueError("Unknown mode")

        X[t+1] = x
        Y[t+1] = y
        consensus_hist[t+1] = c_t
        inwin_hist[t+1] = iw_t
        conflict_persist_hist[t+1] = conflict_state["persist"]

    # export CSV log
    os.makedirs("/mnt/data", exist_ok=True)
    csv_path = f"/mnt/data/{export_prefix}_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "consensus", "in_window", "conflict_persist"] + [f"x{i}" for i in range(params.N)])
        for t in range(params.T + 1):
            row = [t, consensus_hist[t], inwin_hist[t], conflict_persist_hist[t]] + list(X[t])
            writer.writerow(row)

    final_mean = float(np.mean(X[-1]))
    final_std = float(np.std(X[-1], ddof=1)) if params.N > 1 else 0.0
    osc = analyze_oscillation_flags(inwin_hist, conflict_persist_hist, params)

    return {
        "params": asdict(params),
        "X": X, "Y": Y,
        "consensus": consensus_hist,
        "in_window": inwin_hist,
        "conflict_persist": conflict_persist_hist,
        "final_mean": final_mean,
        "final_std": final_std,
        "csv_path": csv_path,
        **osc
    }

# ---------------------- PLOTTING (uniform axes/bins) ----------------------
def plot_histogram_final(X: np.ndarray,
                         out_path: str,
                         W: int = None,
                         fixed_bin_edges: np.ndarray = None,
                         xlim: Tuple[float,float] = None):
    """
    Final-Histogramm mit:
      - vertikalen Linien für μ und ±σ,
      - schattiertem Lock-Fenster (μ±W),
      - Gaussian-Fit, skaliert auf Histogrammzählungen.
    """
    final = X[-1]
    mu = float(np.mean(final))
    sigma = float(np.std(final, ddof=1)) if len(final) > 1 else 0.0

    if fixed_bin_edges is None:
        fixed_bin_edges = HIST_BINS

    counts, edges = np.histogram(final, bins=fixed_bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    plt.figure()
    plt.hist(final, bins=fixed_bin_edges)

    if W is not None:
        plt.axvspan(mu - W, mu + W, alpha=0.12)

    plt.axvline(mu)
    if sigma > 0:
        plt.axvline(mu - sigma, linestyle='--')
        plt.axvline(mu + sigma, linestyle='--')

    if sigma > 0:
        norm_pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
        curve = len(final) * binw * norm_pdf
        plt.plot(centers, curve)

    plt.text(0.98, 0.95, f"$\\mu$={mu:.1f}, $\\sigma$={sigma:.1f}",
             transform=plt.gca().transAxes, ha="right", va="top")

    if xlim is None:
        xlim = HIST_XLIM
    plt.xlim(*xlim)

    plt.xlabel("Lock-time (bits)")
    plt.ylabel("Frequency")
    plt.title("Final Lock-time Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_temporal_evolution(X: np.ndarray, out_path: str,
                            consensus_hist: np.ndarray = None, W: int = None,
                            ylim: Tuple[float,float] = None):
    plt.figure()
    T, N = X.shape
    for i in range(N):
        plt.plot(np.arange(T), X[:, i], linewidth=0.9, marker='.', markevery=max(T//25, 1), markersize=2.5)
    if consensus_hist is not None and len(consensus_hist) == T:
        plt.plot(np.arange(T), consensus_hist, linewidth=1.2)
    if W is not None:
        mu_final = float(np.mean(X[-1]))
        plt.axhspan(mu_final - W, mu_final + W, alpha=0.10)
    if ylim is None:
        ylim = TEMPORAL_YLIM
    plt.ylim(*ylim)
    plt.xlabel("Time (cycles)")
    plt.ylabel("Event timing (bits)")
    plt.title("Temporal Evolution of Channels")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_phase_alignment(X: np.ndarray, out_path: str,
                         sort_by_deviation: bool = False,
                         ylim: Tuple[float,float] = None):
    final = X[-1]
    mu = float(np.mean(final))
    dev = final - mu
    if sort_by_deviation:
        order = np.argsort(dev)
        dev = dev[order]
        idx = np.arange(len(dev))
    else:
        idx = np.arange(len(final))

    plt.figure()
    # Removed 'use_line_collection' argument as it's deprecated/removed in recent matplotlib versions
    plt.stem(idx, dev)
    plt.axhline(0.0)
    dmin, dmax = float(np.min(dev)), float(np.max(dev))
    plt.text(0.02, 0.95, f"min={dmin:.1f}, max={dmax:.1f}",
             transform=plt.gca().transAxes, ha="left", va="top")
    if ylim is None:
        ylim = PHASE_YLIM
    plt.ylim(*ylim)
    plt.xlabel("Channel index" + (" (sorted by deviation)" if sort_by_deviation else ""))
    plt.ylabel("Final deviation from mean (bits)")
    plt.title("Phase Alignment at Final Time")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_diagnostics(in_window: np.ndarray, conflict_persist: np.ndarray, out_prefix: str, N: int):
    # in_window(t)
    plt.figure()
    plt.plot(np.arange(len(in_window)), in_window, linewidth=1.2)
    plt.axhline(N)
    plt.xlabel("Time (cycles)")
    plt.ylabel("Channels in window")
    plt.title("In-window Count Over Time")
    plt.tight_layout()
    plt.savefig(out_prefix + "_inwindow.pdf")
    plt.close()

    # conflict_persist(t)
    plt.figure()
    plt.plot(np.arange(len(conflict_persist)), conflict_persist, linewidth=1.2)
    plt.xlabel("Time (cycles)")
    plt.ylabel("Conflict persist (steps)")
    plt.title("Conflict Persistence Over Time")
    plt.tight_layout()
    plt.savefig(out_prefix + "_conflict.pdf")
    plt.close()

# ---------------------- PRESETS ----------------------
def run_preset(test_name: str = "T1_random_sync") -> Dict[str, Any]:
    if test_name == "T1_random_sync":
        params = RFFGParams(mode="sync", seed=42)
        result = run_experiment(params, init_pattern="random", export_prefix="T1_random_sync")
    elif test_name == "T2_twopeaks_sync":
        params = RFFGParams(mode="sync", seed=7)
        result = run_experiment(params, init_pattern="two_peaks", fixed_values=[3020, 3028], export_prefix="T2_twopeaks_sync")
    elif test_name == "T3_alloutside_sync":
        params = RFFGParams(mode="sync", seed=21)
        result = run_experiment(params, init_pattern="all_outside", export_prefix="T3_alloutside_sync")
    elif test_name == "T1_random_async":
        params = RFFGParams(mode="async", seed=42)
        result = run_experiment(params, init_pattern="random", export_prefix="T1_random_async")
    # --- T4 variants (oscillation) ---
    elif test_name == "T4_oscillation_demo":
        params = RFFGParams(
            N=10, W=4, step=1, T=600, mode="sync",
            consensus_method="mean", conflict_delta=0, conflict_persist=5,
            seed=13, enable_oscillation=True, oscillate_smaller_group_only=False
        )
        result = run_experiment(params, init_pattern="two_peaks", fixed_values=[3010, 3032], export_prefix="T4_oscillation_demo")
    elif test_name == "T4A_more_strict":
        params = RFFGParams(
            N=10, W=3, step=1, T=700, mode="sync",
            consensus_method="mean", conflict_delta=0, conflict_persist=7,
            seed=17, enable_oscillation=True
        )
        result = run_experiment(params, init_pattern="two_peaks", fixed_values=[3008, 3034], export_prefix="T4A_more_strict")
    elif test_name == "T4B_async":
        params = RFFGParams(
            N=10, W=4, step=1, T=700, mode="async",
            consensus_method="mean", conflict_delta=0, conflict_persist=5,
            seed=23, enable_oscillation=True
        )
        result = run_experiment(params, init_pattern="two_peaks", fixed_values=[3010, 3032], export_prefix="T4B_async")
    else:
        raise ValueError("Unknown test preset")

    # export plots (uniform axes/bins)
    base = f"/mnt/data/{test_name}"
    plot_histogram_final(result["X"], base + "_histogram.pdf",
                         W=RFFGParams().W, fixed_bin_edges=HIST_BINS, xlim=HIST_XLIM)
    plot_temporal_evolution(result["X"], base + "_temporal.pdf",
                            consensus_hist=result.get("consensus"), W=RFFGParams().W,
                            ylim=TEMPORAL_YLIM)
    plot_phase_alignment(result["X"], base + "_phase.pdf",
                         sort_by_deviation=True, ylim=PHASE_YLIM)
    plot_diagnostics(result["in_window"], result["conflict_persist"], base + "_diag",
                     N=RFFGParams().N)

    # summary file
    save_path = base + "_summary.txt"
    with open(save_path, "w") as f:
        f.write(f"params: {result['params']}\n")
        f.write(f"final_mean: {result['final_mean']}\n")
        f.write(f"final_std: {result['final_std']}\n")
        f.write(f"csv_log: {result['csv_path']}\n")
        f.write(f"oscillation_detected: {result['oscillation_detected']}\n")
        f.write(f"total_conflict_steps: {result['total_conflict_steps']}\n")
        f.write(f"final_all_locked: {result['final_all_locked']}\n")

    return {
        "result": result,
        "histogram_path": base + "_histogram.pdf",
        "temporal_path": base + "_temporal.pdf",
        "phase_path": base + "_phase.pdf",
        "diag_inwindow_path": base + "_diag_inwindow.pdf",
        "diag_conflict_path": base + "_diag_conflict.pdf",
        "summary_path": save_path,
        "csv_path": result["csv_path"]
    }

# ---------------------- OPTIONAL: batch re-render histograms ----------------------
def rerender_histograms_for_all(outputs_list: List[Dict[str, Any]]):
    """Re-apply the uniform histogram style to a list of outputs."""
    for out in outputs_list:
        res = out["result"]
        base = out["histogram_path"].replace("_histogram.pdf", "")
        plot_histogram_final(res["X"], base + "_histogram.pdf",
                             W=RFFGParams().W, fixed_bin_edges=HIST_BINS, xlim=HIST_XLIM)

# ---------------------- DEMO (comment/uncomment as needed) ----------------------
# outs = [
#     run_preset("T1_random_sync"),
#     run_preset("T2_twopeaks_sync"),
#     run_preset("T3_alloutside_sync"),
#     run_preset("T4_oscillation_demo"),
# ]
# rerender_histograms_for_all(outs)
