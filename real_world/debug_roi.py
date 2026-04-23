"""
ROI Debug Tool — Updated
Shows new custom ROI layout calibrated to the dataset camera angle.
Run this to verify zones before running real_data_demo.py

Usage:
  python real_world/debug_roi.py
"""

import os, sys, cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "Intersection-traffic-detection.yolov8", "train", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "real_world")
os.makedirs(OUTPUT_DIR, exist_ok=True)

images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg",".jpeg",".png"))
])
if not images:
    print(f"No images found in {IMAGES_DIR}")
    sys.exit(1)

img_path = os.path.join(IMAGES_DIR, images[0])
frame    = cv2.imread(img_path)
h, w     = frame.shape[:2]
print(f"Image : {images[0]}")
print(f"Size  : {w} x {h}")

# ── Custom ROIs calibrated to angled intersection camera ─────────────────────
# Based on the debug image:
#   - Main road (ahead)  : upper center  — buses and cars visible
#   - Main road (behind) : lower left    — road coming toward camera
#   - Branch left        : left center   — side road / junction arm
#   - Branch right       : upper right   — opposite junction arm
rois = {
    "lane_1 (Main-Ahead)":  {
        "x": int(w*0.28), "y": int(h*0.00),
        "w": int(w*0.35), "h": int(h*0.38),
        "color": (0,255,0)
    },
    "lane_2 (Main-Behind)": {
        "x": int(w*0.00), "y": int(h*0.50),
        "w": int(w*0.40), "h": int(h*0.40),
        "color": (0,200,255)
    },
    "lane_3 (Branch-Left)": {
        "x": int(w*0.00), "y": int(h*0.28),
        "w": int(w*0.28), "h": int(h*0.30),
        "color": (255,100,0)
    },
    "lane_4 (Branch-Right)":{
        "x": int(w*0.55), "y": int(h*0.00),
        "w": int(w*0.35), "h": int(h*0.35),
        "color": (200,0,255)
    },
}

out = frame.copy()
for label, roi in rois.items():
    rx,ry,rw,rh = roi["x"],roi["y"],roi["w"],roi["h"]
    color       = roi["color"]
    overlay = out.copy()
    cv2.rectangle(overlay,(rx,ry),(rx+rw,ry+rh),color,-1)
    cv2.addWeighted(overlay,0.2,out,0.8,0,out)
    cv2.rectangle(out,(rx,ry),(rx+rw,ry+rh),color,3)
    cv2.putText(out, label,          (rx+6, ry+26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(out, f"({rx},{ry})", (rx+6, ry+48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

cv2.putText(out, "ROI DEBUG v2 — Calibrated to dataset camera angle",
    (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

out_path = os.path.join(OUTPUT_DIR, "roi_debug_v2.jpg")
cv2.imwrite(out_path, out)
print(f"\n  Saved → {out_path}")
print("  Open in File Explorer and check if zones cover road lanes correctly.")