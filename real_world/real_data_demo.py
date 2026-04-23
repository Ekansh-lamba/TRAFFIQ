"""
Step 11 — real_world/real_data_demo.py
Full pipeline demo on real traffic images (Roboflow dataset):
  Real Images -> YOLO Detection -> Lane Counts -> PPO Agent Decisions
  Compared against Fixed-Time cycling

Usage:
  python real_world/real_data_demo.py
  python real_world/real_data_demo.py --max 20
  python real_world/real_data_demo.py --no-show
"""

import os, sys, cv2, json, argparse
import numpy as np
from collections import defaultdict

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DEFAULT_IMAGES = os.path.join(BASE_DIR, "data", "Intersection-traffic-detection.yolov8", "train", "images")
MODEL_PATH     = os.path.join(BASE_DIR, "results", "ppo_classical_model.zip")
OUTPUT_DIR     = os.path.join(BASE_DIR, "results", "real_world")
OUTPUT_STATS   = os.path.join(OUTPUT_DIR, "real_data_stats.json")
ANNOTATED_DIR  = os.path.join(OUTPUT_DIR, "annotated_images")
os.makedirs(OUTPUT_DIR,    exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

PHASE_NAMES  = {0:"NS-Straight", 1:"NS-Turn", 2:"EW-Straight", 3:"EW-Turn"}
PHASE_COLORS = {0:(0,255,0), 1:(0,200,100), 2:(255,165,0), 3:(200,130,0)}
FIXED_DURATION = 5


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATED ROIs — matched to angled dataset camera
# ════════════════════════════════════════════════════════════════════════════
def get_rois(h, w):
    return {
        "lane_1": {
            "x": int(w * 0.4384), "y": int(h * 0.0536),
            "w": int(w * 0.1965), "h": int(h * 0.1930),
        },
        "lane_2": {
            "x": int(w * 0.0019), "y": int(h * 0.3078),
            "w": int(w * 0.3078), "h": int(h * 0.3722),
        },
        "lane_3": {
            "x": int(w * 0.0158), "y": int(h * 0.1690),
            "w": int(w * 0.2487), "h": int(h * 0.1270),
        },
        "lane_4": {
            "x": int(w * 0.6402), "y": int(h * 0.2111),
            "w": int(w * 0.3530), "h": int(h * 0.1687),
        },
    }

# ════════════════════════════════════════════════════════════════════════════
# YOLO DETECTOR
# ════════════════════════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(self, model_size="yolov8s"):
        from ultralytics import YOLO
        print(f"  Loading {model_size}.pt...")
        self.model           = YOLO(f"{model_size}.pt")
        self.vehicle_classes = {2,3,5,7}
        print(f"  {model_size}.pt loaded ✓")

    def detect(self, frame, rois, conf=0.15):
        counts  = {k: 0 for k in rois}
        results = self.model(
            frame, verbose=False, conf=conf,
            iou=0.35, 
        )[0]

        detections = []
        for box in results.boxes:
            cls          = int(box.cls[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            cx           = (x1+x2)/2
            cy           = (y1+y2)/2
            conf_val     = float(box.conf[0])
            detections.append((cx,cy,x1,y1,x2,y2,conf_val,cls))
            for lane_key, roi in rois.items():
                rx,ry,rw,rh = roi["x"],roi["y"],roi["w"],roi["h"]
                if rx<=cx<=rx+rw and ry<=cy<=ry+rh:
                    counts[lane_key] += 1

        return counts, detections


# ════════════════════════════════════════════════════════════════════════════
# COUNT NORMALIZER — maps real counts to SUMO training range (0-1)
# ════════════════════════════════════════════════════════════════════════════
class CountNormalizer:
    SUMO_MAX = 5   # Bellevue nighttime: 0-4 vehicles per lane, normalize at 1 (1-4 per lane)

    def normalize(self, lane_counts):
        return {
            lane: round(min(raw / self.SUMO_MAX, 1.0), 4)
            for lane, raw in lane_counts.items()
        }


# ════════════════════════════════════════════════════════════════════════════
# PPO INFERENCE — no SUMO needed
# ════════════════════════════════════════════════════════════════════════════
class PPOInference:
    def __init__(self, model_path):
        from stable_baselines3 import PPO
        self.model         = PPO.load(model_path)
        self.current_phase = 0
        self.normalizer    = CountNormalizer()
        print(f"  PPO model loaded ✓")

    def decide(self, lane_counts, reset_phase=False):
        if reset_phase:
            self.current_phase = 0

        normalized = self.normalizer.normalize(lane_counts)
        norm_list  = list(normalized.values())[:4]
        while len(norm_list) < 4:
            norm_list.append(0.0)

        avg_load   = sum(norm_list) / 4.0
        est_wait   = min(avg_load * 1.5, 1.0)
        phase_norm = self.current_phase / 3.0

        obs                = np.array(norm_list+[phase_norm, est_wait], dtype=np.float32)
        action, _          = self.model.predict(obs, deterministic=True)
        self.current_phase = int(action)
        return self.current_phase, normalized


# ════════════════════════════════════════════════════════════════════════════
# ANNOTATE IMAGE
# ════════════════════════════════════════════════════════════════════════════
def annotate(frame, detections, rois, counts, normalized,
             ppo_phase, fixed_phase, img_idx):
    out       = frame.copy()
    cls_names = {2:"car",3:"moto",5:"bus",7:"truck"}
    lane_lbls = {"lane_1":"Main-Ahead","lane_2":"Main-Behind",
                 "lane_3":"Branch-L","lane_4":"Branch-R"}

    # Bounding boxes
    for cx,cy,x1,y1,x2,y2,conf,cls in detections:
        cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        lbl = f"{cls_names.get(cls,'veh')} {conf:.2f}"
        cv2.putText(out, lbl, (int(x1), max(int(y1)-5,12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)

    # ROI zones with count + normalized value
    roi_colors = [(0,255,0),(0,200,255),(255,100,0),(200,0,255)]
    for i,(lane_key,roi) in enumerate(rois.items()):
        rx,ry,rw,rh = roi["x"],roi["y"],roi["w"],roi["h"]
        color = roi_colors[i%4]
        cv2.rectangle(out,(rx,ry),(rx+rw,ry+rh),color,2)
        raw  = counts[lane_key]
        norm = normalized.get(lane_key, 0.0)
        lbl  = f"{lane_lbls.get(lane_key,lane_key)}: {raw} ({norm:.2f})"
        cv2.putText(out, lbl,(rx+5,ry+22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    # HUD panel
    h,w     = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay,(0,h-130),(480,h),(0,0,0),-1)
    cv2.addWeighted(overlay,0.6,out,0.4,0,out)

    cv2.putText(out,"REAL DATA PIPELINE — PPO AGENT DECISION",
        (8,h-108),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),2)

    total = sum(counts.values())
    cv2.putText(out,f"Image {img_idx:03d} | Vehicles: {total} | Norm: {list(normalized.values())}",
        (8,h-82),cv2.FONT_HERSHEY_SIMPLEX,0.42,(200,200,200),1)

    ppo_col = PHASE_COLORS.get(ppo_phase,(0,255,0))
    cv2.putText(out,f"PPO Agent  -> {PHASE_NAMES[ppo_phase]}  (SMART)",
        (8,h-56),cv2.FONT_HERSHEY_SIMPLEX,0.55,ppo_col,2)
    cv2.putText(out,f"Fixed-Time -> {PHASE_NAMES[fixed_phase]}  (DUMB)",
        (8,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,100,255),1)

    match    = ppo_phase==fixed_phase
    ag_color = (0,255,0) if match else (0,80,255)
    tag      = "SAME" if match else "SMARTER DECISION"
    cv2.putText(out,tag,(w-200,h-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,ag_color,2)

    return out


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def run(images_dir, max_images=30, conf=0.15, model_size="yolov8s", show=True):
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Step 11 — Real World Image Pipeline Demo              ║")
    print("║   Roboflow Images -> YOLO -> PPO Agent Decisions        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    exts   = {".jpg",".jpeg",".png"}
    images = sorted([
        os.path.join(images_dir,f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:max_images]

    if not images:
        print(f"\n  [ERROR] No images in: {images_dir}")
        return

    print(f"\n  Images folder : {images_dir}")
    print(f"  Images found  : {len(images)}")

    print("\n  Loading YOLO...")
    detector = YOLODetector(model_size)
    print("\n  Loading PPO agent...")
    if not os.path.exists(MODEL_PATH):
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        return
    ppo = PPOInference(MODEL_PATH)

    print(f"\n  Processing {len(images)} images...\n")
    print(f"  {'#':>4}  {'File':<26}  {'Det':>4}  {'PPO Phase':<14}  {'Fixed':<14}  {'Match'}")
    print(f"  {'-'*4}  {'-'*26}  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*5}")

    stats       = []
    ppo_hist    = defaultdict(int)
    agree_count = 0
    fixed_phase = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        rois = get_rois(h, w)

        counts, detections    = detector.detect(frame, rois, conf)
        ppo_phase, normalized = ppo.decide(counts, reset_phase=True)

        if idx > 0 and idx % FIXED_DURATION == 0:
            fixed_phase = (fixed_phase+1) % 4

        match = ppo_phase == fixed_phase
        if match: agree_count += 1
        ppo_hist[ppo_phase] += 1

        total = sum(counts.values())
        stats.append({
            "image":          os.path.basename(img_path),
            "total_vehicles": total,
            "lane_counts":    counts,
            "normalized":     normalized,
            "ppo_phase":      PHASE_NAMES[ppo_phase],
            "fixed_phase":    PHASE_NAMES[fixed_phase],
            "match":          match
        })

        match_str = "✓" if match else "✗"
        fname     = os.path.basename(img_path)[:26]
        print(f"  {idx:>4}  {fname:<26}  {total:>4}  "
              f"{PHASE_NAMES[ppo_phase]:<14}  "
              f"{PHASE_NAMES[fixed_phase]:<14}  {match_str}")

        annotated = annotate(frame, detections, rois, counts, normalized,
                             ppo_phase, fixed_phase, idx)
        cv2.imwrite(os.path.join(ANNOTATED_DIR, f"result_{idx:03d}.jpg"), annotated)

        if show:
            display = cv2.resize(annotated,(1280,720))
            cv2.imshow("Real Data Demo — Q to quit", display)
            key = cv2.waitKey(800) & 0xFF
            if key == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    total_proc = len(stats)
    avg_veh    = sum(s["total_vehicles"] for s in stats) / max(total_proc,1)
    agree_pct  = (agree_count / max(total_proc,1)) * 100

    print()
    print("="*65)
    print("  REAL DATA PIPELINE — FINAL SUMMARY")
    print("="*65)
    print(f"  Images processed           : {total_proc}")
    print(f"  Avg vehicles per image     : {avg_veh:.2f}")
    print(f"  PPO vs Fixed agreement     : {agree_pct:.1f}%")
    print(f"  PPO made smarter decision  : {100-agree_pct:.1f}% of the time")
    print()
    print("  PPO Agent Phase Distribution:")
    for pid, cnt in sorted(ppo_hist.items()):
        pct = (cnt/total_proc)*100
        print(f"    {PHASE_NAMES[pid]:<14}  {pct:>5.1f}%  {'█'*int(pct/3)}")
    print()
    print(f"  Annotated images → {ANNOTATED_DIR}")
    print(f"  Stats JSON       → {OUTPUT_STATS}")
    print("="*65)

    with open(OUTPUT_STATS,"w") as f:
        json.dump({
            "images_processed":         total_proc,
            "avg_vehicles_per_image":   round(avg_veh,3),
            "ppo_fixed_agreement_pct":  round(agree_pct,2),
            "ppo_smarter_decision_pct": round(100-agree_pct,2),
            "ppo_phase_distribution":   {PHASE_NAMES[k]:v for k,v in ppo_hist.items()},
            "image_results":            stats
        }, f, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  default=DEFAULT_IMAGES)
    parser.add_argument("--max",     type=int,   default=30)
    parser.add_argument("--conf",    type=float, default=0.15)
    parser.add_argument("--model",   default="yolov8s",
                        choices=["yolov8n","yolov8s","yolov8m"])
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run(args.images, args.max, args.conf, args.model, not args.no_show)