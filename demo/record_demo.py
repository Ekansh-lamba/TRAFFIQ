"""
Step 10 — demo/record_demo.py
Runs a visual side-by-side demo of:
  - Condition A: Fixed-time signal control
  - Condition C: PPO + Classical CV signal control

Usage:
  python demo/record_demo.py              ← runs both conditions
  python demo/record_demo.py --mode a     ← fixed-time only
  python demo/record_demo.py --mode c     ← PPO+CV only
"""

import os
import sys
import time
import argparse
import json

# ── Path setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import traci
from stable_baselines3 import PPO
from env.traffic_env import ClassicalTrafficEnv, SB3EnvWrapper

# ── Paths ───────────────────────────────────────────────────────────────────
SUMO_CFG    = os.path.join(BASE_DIR, "sumo_files", "intersection.sumocfg")
MODEL_PATH  = os.path.join(BASE_DIR, "results", "ppo_classical_model.zip")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Config ──────────────────────────────────────────────────────────────────
STEP_DELAY      = 0.05   # seconds between sim steps — makes it visually watchable
PRINT_EVERY     = 50     # print stats every N steps
MAX_STEPS       = 300    # one full episode
PHASE_DURATION  = 30     # fixed-time phase duration (steps)

# ── Helpers ─────────────────────────────────────────────────────────────────
def print_header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_stats(step, waiting_time, queue_length, throughput, phase):
    phase_names = {0: "NS-Straight", 1: "NS-Turn", 2: "EW-Straight", 3: "EW-Turn"}
    name = phase_names.get(phase, f"Phase {phase}")
    print(f"  Step {step:>4} | Phase: {name:<12} | "
          f"Wait: {waiting_time:>6.2f}s | "
          f"Queue: {queue_length:>5.3f} | "
          f"Thru: {throughput:>4}")

def print_episode_summary(label, avg_wait, avg_queue, throughput, total_reward):
    print()
    print(f"  ── {label} Episode Summary ──")
    print(f"     Avg Waiting Time : {avg_wait:.3f} s")
    print(f"     Avg Queue Length : {avg_queue:.3f}")
    print(f"     Throughput       : {throughput} vehicles")
    print(f"     Total Reward     : {total_reward:.1f}")
    print()

# ════════════════════════════════════════════════════════════════════════════
# CONDITION A — Fixed-Time Demo
# ════════════════════════════════════════════════════════════════════════════
def run_fixed_time_demo():
    print_header("CONDITION A — Fixed-Time Signal Control")
    print("  Signals cycle every 30 steps regardless of traffic.")
    print("  Watch how cars pile up even when lanes are empty.")
    print()
    print("  Starting SUMO-GUI... (keep window visible)")
    time.sleep(1)

    sumo_cmd = [
        "sumo-gui",
        "-c", SUMO_CFG,
        "--start",
        "--quit-on-end", "true",
        "--no-step-log",
        "--end", "999999"
    ]

    traci.start(sumo_cmd)

    # Set nice view
    try:
        traci.gui.setZoom("View #0", 500)
        traci.gui.setOffset("View #0", 200.0, 200.0)
    except Exception:
        pass

    phase         = 0
    step          = 0
    total_wait    = 0.0
    total_queue   = 0.0
    throughput    = 0
    total_reward  = 0.0
    wait_list     = []
    queue_list    = []

    print(f"  {'Step':>5}  {'Phase':<14}  {'Wait':>8}  {'Queue':>7}  {'Thru':>5}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*8}  {'-'*7}  {'-'*5}")

    while step < MAX_STEPS:
        # Fixed-time phase switching
        if step % PHASE_DURATION == 0:
            traci.trafficlight.setPhase("center", phase % 4)
            phase += 1

        traci.simulationStep()
        time.sleep(STEP_DELAY)

        # Read metrics
        vehicles = traci.vehicle.getIDList()
        wait  = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / max(len(vehicles), 1)
        queue = sum(traci.vehicle.getSpeed(v) < 0.1 for v in vehicles) / max(len(vehicles), 1)
        thru  = traci.simulation.getArrivedNumber()

        throughput   += thru
        total_wait   += wait
        total_queue  += queue
        total_reward -= (wait + queue)
        wait_list.append(wait)
        queue_list.append(queue)

        if step % PRINT_EVERY == 0:
            current_phase = traci.trafficlight.getPhase("center")
            print_stats(step, wait, queue, throughput, current_phase)

        step += 1

    traci.close()

    avg_wait  = total_wait  / MAX_STEPS
    avg_queue = total_queue / MAX_STEPS
    print_episode_summary("Condition A (Fixed-Time)", avg_wait, avg_queue, throughput, total_reward)

    return {
        "avg_waiting_time": avg_wait,
        "avg_queue_length": avg_queue,
        "throughput":       throughput,
        "total_reward":     total_reward
    }


# ════════════════════════════════════════════════════════════════════════════
# CONDITION C — PPO + Classical CV Demo
# ════════════════════════════════════════════════════════════════════════════
def run_ppo_cv_demo():
    print_header("CONDITION C — PPO + Classical CV Signal Control")
    print("  Smart agent reads lane counts via CV and decides signal phases.")
    print("  Watch how it prioritises busy lanes.")
    print()

    if not os.path.exists(MODEL_PATH):
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        print("  Run full training first: python agents/train_ppo_classical.py")
        return None

    # Load ROI config
    roi_path = os.path.join(RESULTS_DIR, "roi_config.json")
    if not os.path.exists(roi_path):
        print(f"  [ERROR] ROI config not found: {roi_path}")
        print("  Run calibration first: python perception/classical_detector.py --calibrate")
        return None

    with open(roi_path) as f:
        rois = json.load(f)
    print(f"  ROI config loaded ✓")

    # Load detector
    from perception.classical_detector import ClassicalDetector
    detector = ClassicalDetector()
    print(f"  Detector loaded ✓")

    print(f"  Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    print("  Model loaded ✓")
    print()
    print("  Starting SUMO-GUI... (keep window visible)")
    time.sleep(1)

    # Build env with correct parameter names
    raw_env = ClassicalTrafficEnv(
        sumo_config=SUMO_CFG,
        detector=detector,
        rois=rois,
        max_steps=MAX_STEPS
    )
    env = SB3EnvWrapper(raw_env)

    obs, _ = env.reset()

    step          = 0
    throughput    = 0
    total_reward  = 0.0
    total_wait    = 0.0
    total_queue   = 0.0

    phase_names = {0: "NS-Straight", 1: "NS-Turn", 2: "EW-Straight", 3: "EW-Turn"}

    print(f"  {'Step':>5}  {'Phase':<14}  {'Wait':>8}  {'Queue':>7}  {'Thru':>5}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*8}  {'-'*7}  {'-'*5}")

    while step < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(STEP_DELAY)

        # Read live metrics
        try:
            vehicles = traci.vehicle.getIDList()
            wait  = sum(traci.vehicle.getWaitingTime(v) for v in vehicles) / max(len(vehicles), 1)
            queue = sum(traci.vehicle.getSpeed(v) < 0.1 for v in vehicles) / max(len(vehicles), 1)
            thru  = traci.simulation.getArrivedNumber()
            current_phase = traci.trafficlight.getPhase("center")
        except Exception:
            wait, queue, thru, current_phase = 0, 0, 0, int(action)

        throughput   += thru
        total_reward += reward
        total_wait   += wait
        total_queue  += queue

        if step % PRINT_EVERY == 0:
            print_stats(step, wait, queue, throughput, current_phase)

        if terminated or truncated:
            break

        step += 1

    env.close()

    avg_wait  = total_wait  / max(step, 1)
    avg_queue = total_queue / max(step, 1)
    print_episode_summary("Condition C (PPO+CV)", avg_wait, avg_queue, throughput, total_reward)

    return {
        "avg_waiting_time": avg_wait,
        "avg_queue_length": avg_queue,
        "throughput":       throughput,
        "total_reward":     total_reward
    }


# ════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ════════════════════════════════════════════════════════════════════════════
def print_comparison(res_a, res_c):
    if not res_a or not res_c:
        return

    print()
    print("=" * 60)
    print("  DEMO COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<24} {'Fixed-Time':>12} {'PPO+CV':>10}  {'Δ %':>8}")
    print(f"  {'-'*24} {'-'*12} {'-'*10}  {'-'*8}")

    metrics = [
        ("avg_waiting_time", "Avg Waiting Time (s)", True),
        ("avg_queue_length", "Avg Queue Length",     True),
        ("throughput",       "Throughput",           False),
    ]

    for key, label, lower_better in metrics:
        va = res_a.get(key, 0)
        vc = res_c.get(key, 0)
        if va != 0:
            delta = ((vc - va) / va) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"
        print(f"  {label:<24} {va:>12.3f} {vc:>10.3f}  {delta_str:>8}")

    print("=" * 60)
    wt_a = res_a.get("avg_waiting_time", 1)
    wt_c = res_c.get("avg_waiting_time", 0)
    if wt_a > 0:
        imp = ((wt_a - wt_c) / wt_a) * 100
        print(f"\n  ★  Waiting time improved by {imp:.1f}% with PPO+CV")
    tp = res_c.get("throughput", 0)
    print(f"  ★  Throughput: {tp} vehicles completed in PPO+CV episode")
    print()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Signal Demo")
    parser.add_argument("--mode", choices=["a", "c", "both"], default="both",
                        help="a = fixed-time only | c = PPO+CV only | both = run both (default)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between steps in seconds (default 0.05)")
    args = parser.parse_args()

    STEP_DELAY = args.delay

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Vision-Based Dynamic Traffic Signal Optimization       ║")
    print("║   Live Demo — Step 10                                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Mode  : {args.mode.upper()}")
    print(f"  Delay : {STEP_DELAY}s per step")
    print(f"  Steps : {MAX_STEPS} per episode")

    res_a, res_c = None, None

    if args.mode in ("a", "both"):
        res_a = run_fixed_time_demo()
        if args.mode == "both":
            print("\n  Fixed-time demo complete.")
            print("  Starting PPO+CV demo in 3 seconds...")
            time.sleep(3)

    if args.mode in ("c", "both"):
        res_c = run_ppo_cv_demo()

    if args.mode == "both":
        print_comparison(res_a, res_c)

    print("  Demo complete. ✓")