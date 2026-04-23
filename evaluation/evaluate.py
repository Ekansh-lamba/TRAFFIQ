"""
Step 9 — evaluation/evaluate.py
Reads results from Conditions A, B, C and generates:
  - 3 bar charts: waiting time / queue length / throughput
  - 1 reward convergence line plot from TensorBoard logs
  - Console summary table with % improvements
  - Saves all plots to results/plots/
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

A_PATH = os.path.join(RESULTS_DIR, "fixed_time_results.json")
B_PATH = os.path.join(RESULTS_DIR, "ppo_groundtruth_results.json")
C_PATH = os.path.join(RESULTS_DIR, "ppo_classical_results.json")

# ── Load JSON ──────────────────────────────────────────────────────────────
def load(path, label):
    if not os.path.exists(path):
        print(f"[WARN] {label} result file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)

res_a = load(A_PATH, "Condition A")
res_b = load(B_PATH, "Condition B")
res_c = load(C_PATH, "Condition C")

# ── Extract metric helper ──────────────────────────────────────────────────
def get(res, metric, stat="mean"):
    try:
        return res["summary"][metric][stat]
    except (KeyError, TypeError):
        return 0.0

# ── Collect values ─────────────────────────────────────────────────────────
labels = ["A: Fixed-Time", "B: PPO+TraCI", "C: PPO+CV"]
colors = ["#E74C3C", "#3498DB", "#2ECC71"]

metrics = {
    "avg_waiting_time": {
        "title": "Average Waiting Time",
        "ylabel": "Waiting Time (seconds)",
        "filename": "waiting_time.png",
        "lower_is_better": True,
    },
    "avg_queue_length": {
        "title": "Average Queue Length",
        "ylabel": "Queue Length (vehicles)",
        "filename": "queue_length.png",
        "lower_is_better": True,
    },
    "throughput": {
        "title": "Throughput",
        "ylabel": "Vehicles Completed per Episode",
        "filename": "throughput.png",
        "lower_is_better": False,
    },
}

results = [res_a, res_b, res_c]

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Plot 1–3: Bar Charts ───────────────────────────────────────────────────
for metric_key, cfg in metrics.items():
    means, stds = [], []
    for res in results:
        means.append(get(res, metric_key, "mean") if res else 0.0)
        stds.append(get(res, metric_key, "std")  if res else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, edgecolor="white", linewidth=1.2,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=15)

    note = "Lower is better ↓" if cfg["lower_is_better"] else "Higher is better ↑"
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#888")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, cfg["filename"])
    plt.savefig(out)
    plt.close()
    print(f"✓ Saved → {out}")

# ── Plot 4: TensorBoard Reward Convergence ─────────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = os.path.join(RESULTS_DIR, "tensorboard_logs")
    found  = False

    if os.path.exists(tb_dir):
        for root, dirs, files in os.walk(tb_dir):
            event_files = [f for f in files if f.startswith("events.out")]
            if not event_files:
                continue

            ea = EventAccumulator(root)
            ea.Reload()

            tag = None
            for candidate in ["rollout/ep_rew_mean", "train/ep_rew_mean"]:
                if candidate in ea.Tags().get("scalars", []):
                    tag = candidate
                    break

            if tag is None:
                continue

            events     = ea.Scalars(tag)
            steps      = [e.step  for e in events]
            rew_values = [e.value for e in events]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(steps, rew_values, color="#3498DB", linewidth=1.8, label="Episode Reward Mean")
            ax.fill_between(steps, rew_values, alpha=0.1, color="#3498DB")

            ax.set_xlabel("Training Timesteps", fontsize=11)
            ax.set_ylabel("Mean Episode Reward",  fontsize=11)
            ax.set_title("PPO Reward Convergence (Condition C — PPO + Classical CV)",
                         fontsize=13, fontweight="bold", pad=15)
            ax.legend(fontsize=10)
            plt.tight_layout()

            out = os.path.join(PLOTS_DIR, "reward_convergence.png")
            plt.savefig(out)
            plt.close()
            print(f"✓ Saved → {out}")
            found = True
            break

    if not found:
        print("[INFO] No TensorBoard event files found — skipping convergence plot.")

except ImportError:
    print("[INFO] tensorboard not installed — skipping convergence plot.")
    print("       Install with: pip install tensorboard")

# ── Console Summary Table ──────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)

header = f"{'Metric':<28} {'A: Fixed':>10} {'B: PPO+TraCI':>12} {'C: PPO+CV':>10}  {'ΔC vs A':>8}"
print(header)
print("-" * 65)

display_metrics = [
    ("avg_waiting_time", "Avg Waiting Time (s)"),
    ("avg_queue_length", "Avg Queue Length"),
    ("throughput",       "Throughput"),
]

for key, label in display_metrics:
    va = get(res_a, key) if res_a else 0
    vb = get(res_b, key) if res_b else 0
    vc = get(res_c, key) if res_c else 0

    if va != 0:
        delta = ((vc - va) / va) * 100
        delta_str = f"{delta:+.1f}%"
    else:
        delta_str = "N/A"

    print(f"  {label:<26} {va:>10.3f} {vb:>12.3f} {vc:>10.3f}  {delta_str:>8}")

print("=" * 65)

# Highlight key result
if res_a and res_c:
    wt_a = get(res_a, "avg_waiting_time")
    wt_c = get(res_c, "avg_waiting_time")
    if wt_a > 0:
        improvement = ((wt_a - wt_c) / wt_a) * 100
        print(f"\n  ★  Waiting time improvement (C vs A): {improvement:.1f}%")
        print(f"     From {wt_a:.3f}s  →  {wt_c:.3f}s")

tp_c = get(res_c, "throughput") if res_c else 0
tp_a = get(res_a, "throughput") if res_a else 0
print(f"  ★  Throughput (C): {tp_c:.0f} vehicles  (baseline A: {tp_a:.0f})")
print()
print(f"  Plots saved to → {PLOTS_DIR}")
print()