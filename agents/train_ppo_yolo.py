"""
agents/train_ppo_yolo.py
------------------------
Condition C — PPO agent trained with Classical CV perceived state.

Identical PPO hyperparameters and timesteps as train_ppo.py (Condition B).
Uses YOLOTrafficEnv with ClassicalDetector so lane density features come 
from HSV color detection on SUMO-GUI screenshots rather than TraCI ground truth.

IMPORTANT: Calibrate ROIs first using perception/classical_detector.py --calibrate
The calibrated config must exist at: results/roi_config.json
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.traffic_env import YOLOTrafficEnv
from perception.classical_detector import ClassicalDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT          = os.path.join(os.path.dirname(__file__), "..")
SUMO_CONFIG    = os.path.abspath(os.path.join(_ROOT, "sumo_files", "intersection.sumocfg"))
ROI_CONFIG     = os.path.abspath(os.path.join(_ROOT, "results", "roi_config.json"))
TB_LOG_DIR     = os.path.abspath(os.path.join(_ROOT, "results", "tensorboard_logs"))
BEST_MODEL_DIR = os.path.abspath(os.path.join(_ROOT, "results", "best_ppo_yolo"))
FINAL_MODEL    = os.path.abspath(os.path.join(_ROOT, "results", "ppo_yolo_model"))
RESULTS_JSON   = os.path.abspath(os.path.join(_ROOT, "results", "ppo_yolo_results.json"))

# ── Hyperparameters (identical to Condition B) ────────────────────────────────
TOTAL_TIMESTEPS = 200_000
NUM_EVAL_EPS    = 20
PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    verbose       = 1,
)

TRAIN_PORT = 8815
EVAL_PORT  = 8816
# ─────────────────────────────────────────────────────────────────────────────


def _check_rois() -> dict:
    """Exit with a clear message if ROI config is missing."""
    if not os.path.isfile(ROI_CONFIG):
        print("=" * 60)
        print("ERROR: ROI configuration file not found.")
        print(f"Expected path: {ROI_CONFIG}")
        print()
        print("You must calibrate your ROIs before running CV training.")
        print("Steps:")
        print("  1. Start SUMO GUI and let it run for a few seconds.")
        print("  2. Take a screenshot using TraCI or Snipping Tool")
        print("  3. Run calibration:")
        print("       python perception/classical_detector.py test_screenshot.png --calibrate")
        print("  4. Re-run this script.")
        print("=" * 60)
        sys.exit(1)

    with open(ROI_CONFIG) as f:
        rois = json.load(f)
    print(f"ROI config loaded from {ROI_CONFIG}")
    return rois


def make_yolo_env(rois: dict, detector: ClassicalDetector, port: int) -> gym.Env:
    env = YOLOTrafficEnv(
        sumo_config = SUMO_CONFIG,
        rois        = rois,
        detector    = detector,
        port        = port,
    )
    return Monitor(env)


def train():
    for d in [TB_LOG_DIR, BEST_MODEL_DIR, os.path.dirname(RESULTS_JSON)]:
        os.makedirs(d, exist_ok=True)

    rois     = _check_rois()
    detector = ClassicalDetector()

    print("=" * 60)
    print("Condition C — PPO with Classical CV Perceived State")
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"Detector        : HSV Color Detection (Classical CV)")
    print("=" * 60)

    train_env = make_yolo_env(rois, detector, TRAIN_PORT)
    eval_env  = make_yolo_env(rois, detector, EVAL_PORT)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_DIR,
        log_path             = BEST_MODEL_DIR,
        eval_freq            = 10_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
    )

    model = PPO(
        env             = train_env,
        tensorboard_log = TB_LOG_DIR,
        **PPO_KWARGS
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = eval_callback,
        tb_log_name     = "PPO_yolo",
    )

    model.save(FINAL_MODEL)
    print(f"\nFinal model saved → {FINAL_MODEL}.zip")

    train_env.close()
    eval_env.close()

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\nEvaluating over {NUM_EVAL_EPS} episodes …")
    eval_results = []

    for ep in range(NUM_EVAL_EPS):
        env = YOLOTrafficEnv(
            sumo_config = SUMO_CONFIG,
            rois        = rois,
            detector    = detector,
            port        = EVAL_PORT,
        )
        obs, _ = env.reset()
        done   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        metrics = env.get_metrics()
        eval_results.append(metrics)
        env.close()

        print(
            f"  Ep {ep+1:02d}/{NUM_EVAL_EPS}  |  "
            f"avg_wait={metrics['avg_waiting_time']:7.2f}s  |  "
            f"avg_queue={metrics['avg_queue_length']:5.2f}  |  "
            f"throughput={metrics['throughput']:4d}"
        )

    wt   = [m["avg_waiting_time"] for m in eval_results]
    q    = [m["avg_queue_length"]  for m in eval_results]
    tput = [m["throughput"]        for m in eval_results]

    summary = {
        "condition":         "C_ppo_yolo",
        "total_timesteps":   TOTAL_TIMESTEPS,
        "num_eval_episodes": NUM_EVAL_EPS,
        "avg_waiting_time":  {"mean": float(np.mean(wt)),   "std": float(np.std(wt))},
        "avg_queue_length":  {"mean": float(np.mean(q)),    "std": float(np.std(q))},
        "throughput":        {"mean": float(np.mean(tput)), "std": float(np.std(tput))},
        "per_episode":       eval_results,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("─" * 60)
    print("FINAL RESULTS — Condition C (PPO + Classical CV)")
    print("─" * 60)
    print(f"  avg_waiting_time : {np.mean(wt):.2f} ± {np.std(wt):.2f} s")
    print(f"  avg_queue_length : {np.mean(q):.2f} ± {np.std(q):.2f} veh")
    print(f"  throughput       : {np.mean(tput):.1f} ± {np.std(tput):.1f} veh")
    print(f"\nResults saved → {RESULTS_JSON}")


if __name__ == "__main__":
    train()
