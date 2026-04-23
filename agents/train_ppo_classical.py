"""
agents/train_ppo_classical.py
------------------------------
Condition C — PPO trained with ClassicalDetector (CV-perceived) state.

The agent observes lane densities estimated from SUMO-GUI screenshots via
HSV + contour detection, but reward is always computed from TraCI ground
truth so evaluation is fair.

⚠️  SPEED WARNING
Each environment step takes one PIL screen-grab (~0.05–0.15 s on most
machines).  200,000 steps will take roughly 3–8 hours depending on your
hardware.  Use --timesteps 10000 for a quick smoke-test first.

Prerequisites
-------------
1. ROI config must exist:
       D:\\Minor Project\\results\\roi_config.json
   If missing, run:
       python perception/classical_detector.py <screenshot.png> --calibrate

2. Python packages:
       pip install stable-baselines3 tensorboard Pillow pygetwindow

Usage
-----
    # Quick smoke-test (≈ 5–10 min)
    python agents/train_ppo_classical.py --timesteps 10000

    # Full training run
    python agents/train_ppo_classical.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

# ── Imports ───────────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from env.traffic_env import ClassicalTrafficEnv, SB3EnvWrapper
from perception.classical_detector import ClassicalDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
SUMO_CONFIG    = os.path.join(_PROJECT_ROOT, "sumo_files", "intersection.sumocfg")
ROI_CONFIG     = os.path.join(_PROJECT_ROOT, "results", "roi_config.json")
RESULTS_DIR    = os.path.join(_PROJECT_ROOT, "results")
TB_LOG_DIR     = os.path.join(RESULTS_DIR, "tensorboard_logs")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints_ppo_classical")
BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_ppo_classical")
MODEL_SAVE     = os.path.join(RESULTS_DIR, "ppo_classical_model.zip")
RESULTS_SAVE   = os.path.join(RESULTS_DIR, "ppo_classical_results.json")

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_HYPERPARAMS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    verbose       = 1,
    tensorboard_log = TB_LOG_DIR,
)

TOTAL_TIMESTEPS  = 200_000
EVAL_EPISODES    = 20
CHECKPOINT_FREQ  = 50_000   # save a checkpoint every N timesteps
TRACI_PORT       = 8813


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_prerequisites():
    """Abort early with a clear message if ROI config is missing."""
    if not os.path.isfile(ROI_CONFIG):
        print("=" * 60)
        print("ERROR: ROI config not found!")
        print(f"  Expected: {ROI_CONFIG}")
        print()
        print("Run calibration first:")
        print("  python perception/classical_detector.py <screenshot.png> --calibrate")
        print("=" * 60)
        sys.exit(1)

    if not os.path.isfile(SUMO_CONFIG):
        print(f"ERROR: SUMO config not found: {SUMO_CONFIG}")
        sys.exit(1)

    print(f"✓ ROI config found:  {ROI_CONFIG}")
    print(f"✓ SUMO config found: {SUMO_CONFIG}")


def load_rois() -> dict:
    with open(ROI_CONFIG) as f:
        rois = json.load(f)
    print(f"✓ Loaded {len(rois)} ROIs: {list(rois.keys())}")
    return rois


def make_env(rois: dict, port: int) -> SB3EnvWrapper:
    """Create and wrap a ClassicalTrafficEnv for SB3."""
    detector = ClassicalDetector()
    raw_env  = ClassicalTrafficEnv(
        sumo_config = SUMO_CONFIG,
        detector    = detector,
        rois        = rois,
        port        = port,
        max_steps   = 300,   # 300 decisions = 3000 sim seconds
    )
    env = SB3EnvWrapper(raw_env)
    env = Monitor(env)   # records episode rewards/lengths for SB3
    return env


def make_dirs():
    for d in [RESULTS_DIR, TB_LOG_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(total_timesteps: int, rois: dict):
    print()
    print("=" * 60)
    print("CONDITION C — PPO + Classical CV Perception")
    print("=" * 60)
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Checkpoint freq : {CHECKPOINT_FREQ:,} steps")
    print(f"  TensorBoard     : {TB_LOG_DIR}")
    print()
    print("⚠️  Each step takes a screen-grab.")
    print(f"   Estimated time : {total_timesteps * 0.1 / 3600:.1f}–"
          f"{total_timesteps * 0.15 / 3600:.1f} hours")
    print()
    print("Keep the SUMO-GUI window VISIBLE and un-minimised throughout.")
    print("=" * 60)

    env = make_env(rois, port=TRACI_PORT)

    # Checkpoint callback — saves model every CHECKPOINT_FREQ steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_classical",
        verbose     = 1,
    )

    model = PPO(
        env = env,
        **PPO_HYPERPARAMS,
    )

    print("\nStarting training...\n")
    t0 = time.time()

    model.learn(
        total_timesteps = total_timesteps,
        callback        = checkpoint_cb,
        progress_bar    = True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save final model
    model.save(MODEL_SAVE)
    print(f"✓ Model saved → {MODEL_SAVE}")

    env.close()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, rois: dict, n_episodes: int = EVAL_EPISODES):
    """
    Run n_episodes with the trained model and collect metrics.
    Uses ClassicalTrafficEnv (Condition C — CV perception) throughout.
    """
    print()
    print("=" * 60)
    print(f"EVALUATION — {n_episodes} episodes (Condition C)")
    print("=" * 60)

    all_waiting   = []
    all_queue     = []
    all_throughput = []
    all_rewards   = []

    detector = ClassicalDetector()
    raw_env  = ClassicalTrafficEnv(
        sumo_config = SUMO_CONFIG,
        detector    = detector,
        rois        = rois,
        port        = TRACI_PORT,
        max_steps   = 300,   # must match training
    )
    env = SB3EnvWrapper(raw_env)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        metrics = env.get_metrics()
        all_waiting.append(metrics["avg_waiting_time"])
        all_queue.append(metrics["avg_queue_length"])
        all_throughput.append(metrics["throughput"])
        all_rewards.append(ep_reward)

        print(f"  Episode {ep+1:2d}/{n_episodes} | "
              f"reward={ep_reward:8.1f} | "
              f"wait={metrics['avg_waiting_time']:6.2f}s | "
              f"queue={metrics['avg_queue_length']:5.2f} | "
              f"throughput={metrics['throughput']}")

    env.close()

    # Compute summary statistics
    # Build episode_results list — same format as train_ppo.py
    episode_results = []
    for i in range(n_episodes):
        episode_results.append({
            "avg_waiting_time": float(all_waiting[i]),
            "avg_queue_length": float(all_queue[i]),
            "throughput":       int(all_throughput[i]),
            "total_reward":     float(all_rewards[i]),
        })

    # Build summary — same format as train_ppo.py
    results = {
        "configuration": {
            "algorithm":       "PPO",
            "condition":       "C_ppo_classical_cv",
            "total_timesteps": 200000,
            "eval_episodes":   n_episodes,
            "hyperparameters": {
                "policy":         "MlpPolicy",
                "learning_rate":  3e-4,
                "n_steps":        2048,
                "batch_size":     64,
                "n_epochs":       10,
                "gamma":          0.99,
                "clip_range":     0.2,
            },
        },
        "episode_results": episode_results,
        "summary": {
            "avg_waiting_time": {
                "mean": float(np.mean(all_waiting)),
                "std":  float(np.std(all_waiting)),
                "min":  float(np.min(all_waiting)),
                "max":  float(np.max(all_waiting)),
            },
            "avg_queue_length": {
                "mean": float(np.mean(all_queue)),
                "std":  float(np.std(all_queue)),
                "min":  float(np.min(all_queue)),
                "max":  float(np.max(all_queue)),
            },
            "throughput": {
                "mean": float(np.mean(all_throughput)),
                "std":  float(np.std(all_throughput)),
                "min":  int(np.min(all_throughput)),
                "max":  int(np.max(all_throughput)),
            },
            "total_reward": {
                "mean": float(np.mean(all_rewards)),
                "std":  float(np.std(all_rewards)),
                "min":  float(np.min(all_rewards)),
                "max":  float(np.max(all_rewards)),
            },
        },
    }

    # Print summary
    s = results["summary"]
    print()
    print("─" * 60)
    print("CONDITION C RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Avg waiting time : {s['avg_waiting_time']['mean']:7.3f} "
          f"± {s['avg_waiting_time']['std']:.3f} s")
    print(f"  Avg queue length : {s['avg_queue_length']['mean']:7.3f} "
          f"± {s['avg_queue_length']['std']:.3f} vehicles")
    print(f"  Throughput       : {s['throughput']['mean']:7.1f} "
          f"± {s['throughput']['std']:.1f} vehicles")
    print(f"  Episode reward   : {s['total_reward']['mean']:7.1f} "
          f"± {s['total_reward']['std']:.1f}")
    print("─" * 60)

    # Save results
    with open(RESULTS_SAVE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved → {RESULTS_SAVE}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison against Condition B (if results exist)
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_groundtruth(results_c: dict):
    """Print a quick comparison against Condition B if its results exist."""
    b_path = os.path.join(RESULTS_DIR, "ppo_groundtruth_results.json")
    a_path = os.path.join(RESULTS_DIR, "fixed_time_results.json")

    if not os.path.isfile(b_path):
        print(f"\n[INFO] Condition B results not found at {b_path} — skipping comparison.")
        return

    with open(b_path) as f:
        results_b = json.load(f)

    def get_mean(r, key):
        """Read mean from either top-level or nested summary key."""
        # train_ppo.py format: r["summary"][key]["mean"]
        if "summary" in r:
            return float(r["summary"].get(key, {}).get("mean", 0.0))
        # train_ppo_classical.py legacy format: r[key]["mean"]
        v = r.get(key, {})
        if isinstance(v, dict):
            return float(v.get("mean", 0.0))
        return float(v)

    wait_a  = get_mean(results_b, "avg_waiting_time")
    wait_c  = results_c["summary"]["avg_waiting_time"]["mean"]
    queue_a = get_mean(results_b, "avg_queue_length")
    queue_c = results_c["summary"]["avg_queue_length"]["mean"]
    thru_a  = get_mean(results_b, "throughput")
    thru_c  = results_c["summary"]["throughput"]["mean"]

    def pct_change(new, old):
        if old == 0:
            return 0.0
        return (new - old) / old * 100

    print()
    print("=" * 60)
    print("CONDITION B vs CONDITION C COMPARISON")
    print("  B = PPO + TraCI ground truth")
    print("  C = PPO + Classical CV perception  (this run)")
    print("=" * 60)
    print(f"  {'Metric':<22} {'Cond B':>10} {'Cond C':>10} {'Δ %':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Avg waiting time (s)':<22} {wait_a:>10.3f} {wait_c:>10.3f} "
          f"{pct_change(wait_c, wait_a):>+9.1f}%")
    print(f"  {'Avg queue length':<22} {queue_a:>10.3f} {queue_c:>10.3f} "
          f"{pct_change(queue_c, queue_a):>+9.1f}%")
    print(f"  {'Throughput':<22} {thru_a:>10.1f} {thru_c:>10.1f} "
          f"{pct_change(thru_c, thru_a):>+9.1f}%")
    print("=" * 60)
    print("Positive Δ for waiting/queue = degradation vs B (expected due to CV noise)")
    print("Positive Δ for throughput    = improvement vs B")

    if os.path.isfile(a_path):
        with open(a_path) as f:
            results_a = json.load(f)
        wait_ft  = get_mean(results_a, "avg_waiting_time")
        queue_ft = get_mean(results_a, "avg_queue_length")
        thru_ft  = get_mean(results_a, "throughput")
        # Note: fixed_time_results.json may also use "summary" key
        print()
        print(f"  vs Fixed-time (A): wait {pct_change(wait_c, wait_ft):+.1f}% | "
              f"queue {pct_change(queue_c, queue_ft):+.1f}% | "
              f"throughput {pct_change(thru_c, thru_ft):+.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with Classical CV perception (Condition C)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and only run evaluation on saved model"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {EVAL_EPISODES})"
    )
    parser.add_argument(
        "--n-steps", type=int, default=PPO_HYPERPARAMS["n_steps"],
        help="PPO n_steps — set to 128 for quick smoke-tests"
    )
    args = parser.parse_args()

    make_dirs()
    check_prerequisites()
    rois = load_rois()

    # Override n_steps if specified (useful for quick smoke-tests)
    if args.n_steps != PPO_HYPERPARAMS["n_steps"]:
        PPO_HYPERPARAMS["n_steps"] = args.n_steps
        print(f"[INFO] n_steps overridden to {args.n_steps}")

    if args.eval_only:
        # Load existing model and evaluate only
        if not os.path.isfile(MODEL_SAVE):
            print(f"ERROR: No saved model found at {MODEL_SAVE}")
            print("Run training first (without --eval-only)")
            sys.exit(1)
        print(f"Loading model from {MODEL_SAVE}...")
        env = make_env(rois, port=TRACI_PORT)
        model = PPO.load(MODEL_SAVE, env=env)
        env.close()
        print("✓ Model loaded")
    else:
        model = train(args.timesteps, rois)

    results = evaluate(model, rois, n_episodes=args.eval_episodes)
    compare_with_groundtruth(results)

    print()
    print("✓ Step 8 complete.")
    print(f"  Model   → {MODEL_SAVE}")
    print(f"  Results → {RESULTS_SAVE}")
    print()
    print("Next: run evaluate.py (Step 9) to generate comparison plots.")


if __name__ == "__main__":
    main()