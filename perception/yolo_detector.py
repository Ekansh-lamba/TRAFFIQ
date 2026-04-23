"""
perception/yolo_detector.py — FIXED VERSION
--------------------------------------------
YOLOv8n vehicle detector with SUMO-specific preprocessing.

KEY FIX: SUMO vehicles are 5-10px sprites, too small for vanilla YOLO.
This version:
  1. Upscales the screenshot 2x before inference
  2. Uses confidence threshold = 0.10 (very permissive)
  3. Uses IoU threshold = 0.3 (aggressive NMS)
  4. Enhances contrast for better detection

If you still get 0 detections, the vehicles might be too small even
with these fixes. Alternative: use TraCI state (Condition B only).
"""

import sys
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# ── COCO class IDs for vehicles ───────────────────────────────────────────────
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# ── Default placeholder ROIs ──────────────────────────────────────────────────
DEFAULT_ROIS = {
    "lane_1": {"x1":  430, "y1":   50, "x2":  600, "y2":  300},
    "lane_2": {"x1":  430, "y1":  470, "x2":  600, "y2":  720},
    "lane_3": {"x1":  680, "y1":  320, "x2":  970, "y2":  450},
    "lane_4": {"x1":   50, "y1":  320, "x2":  340, "y2":  450},
}

ROI_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "roi_config.json"
)


# ─────────────────────────────────────────────────────────────────────────────
class YOLODetector:
    """
    YOLOv8n detector optimized for tiny SUMO vehicle sprites.

    Parameters
    ----------
    model_path : str
        YOLOv8 weights (default: yolov8n.pt, auto-downloaded).
    conf_threshold : float
        Detection confidence (default 0.10 — very low for small objects).
    iou_threshold : float
        NMS IoU threshold (default 0.3 — aggressive suppression).
    upscale_factor : float
        Image upscaling before inference (default 2.0 for tiny vehicles).
    """

    def __init__(self,
                 model_path:      str   = "yolov8n.pt",
                 conf_threshold:  float = 0.10,
                 iou_threshold:   float = 0.30,
                 upscale_factor:  float = 2.0):
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model           = YOLO(model_path)
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.upscale_factor  = upscale_factor
        print("✓ YOLOv8 model loaded successfully!")

    # ─────────────────────────────────────────── preprocess
    def _preprocess(self, image_path: str) -> tuple:
        """
        Load and preprocess screenshot for better small-object detection.

        Returns
        -------
        (preprocessed_image_array, original_image_for_viz, scale_factor)
        """
        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Screenshot not found: {image_path}")

        # Upscale for better detection
        h, w = img_orig.shape[:2]
        new_w = int(w * self.upscale_factor)
        new_h = int(h * self.upscale_factor)
        img_scaled = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast (helps with low-contrast vehicle sprites)
        lab = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        return img_enhanced, img_orig, self.upscale_factor

    # ---------------------------------------------------------------- detect
    def detect(self, image_path: str):
        """
        Run YOLOv8 inference with SUMO-specific preprocessing.

        Returns
        -------
        (results, scale_factor) — results are in upscaled coordinate space
        """
        img_processed, _, scale = self._preprocess(image_path)

        results = self.model(
            img_processed,
            conf = self.conf_threshold,
            iou  = self.iou_threshold,
            verbose = False
        )
        return results, scale

    # ─────────────────────────────────────────── count_per_lane
    def count_per_lane(self, image_path: str, rois: dict) -> dict:
        """
        Count vehicles in each ROI.

        Parameters
        ----------
        image_path : str  — SUMO screenshot path
        rois       : dict — {"lane_1": {"x1":…,"y1":…,"x2":…,"y2":…}, …}
                             Coordinates in ORIGINAL image space.

        Returns
        -------
        dict {"lane_1": int, "lane_2": int, "lane_3": int, "lane_4": int}
        """
        results, scale = self.detect(image_path)
        counts = {k: 0 for k in rois}

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return counts

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in VEHICLE_CLASSES:
                continue

            # Bounding box in UPSCALED space → convert back to original
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx_scaled = (x1 + x2) / 2
            cy_scaled = (y1 + y2) / 2

            cx_orig = cx_scaled / scale
            cy_orig = cy_scaled / scale

            # Check which ROI contains this vehicle
            for lane_key, roi in rois.items():
                if (roi["x1"] <= cx_orig <= roi["x2"] and
                        roi["y1"] <= cy_orig <= roi["y2"]):
                    counts[lane_key] += 1

        return counts

    # ─────────────────────────────────────────── visualize
    def visualize(self, image_path: str, rois: dict,
                  output_path: str = None) -> str:
        """
        Draw detections and ROIs on the ORIGINAL screenshot (not upscaled).
        Saves to results/yolo_debug.png.
        """
        output_path = output_path or os.path.join(
            os.path.dirname(__file__), "..", "results", "yolo_debug.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results, scale = self.detect(image_path)
        img_orig = cv2.imread(image_path)

        # Draw detections (convert coords back to original space)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(v / scale) for v in boxes.xyxy[i].tolist()]
                conf  = float(boxes.conf[i].item())
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"

                cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_orig, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw ROIs
        roi_colours = [(255, 0, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
        for idx, (lane_key, roi) in enumerate(rois.items()):
            colour = roi_colours[idx % len(roi_colours)]
            cv2.rectangle(img_orig,
                          (roi["x1"], roi["y1"]),
                          (roi["x2"], roi["y2"]),
                          colour, 2)
            cv2.putText(img_orig, lane_key, (roi["x1"], roi["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imwrite(output_path, img_orig)
        print(f"[YOLODetector] Debug image saved → {output_path}")
        return output_path

    # ─────────────────────────────────────────── calibrate_rois (CLI version)
    @staticmethod
    def calibrate_rois_cli(screenshot_path: str) -> dict:
        """
        Interactive ROI calibration via OpenCV GUI.
        Click 2 corners per lane × 4 lanes.
        Saves to results/roi_config.json.
        """
        img = cv2.imread(screenshot_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open: {screenshot_path}")

        display    = img.copy()
        points     = []
        lane_names = ["lane_1", "lane_2", "lane_3", "lane_4"]
        rois_out   = {}

        print("=" * 60)
        print("ROI CALIBRATION TOOL")
        print("=" * 60)
        print("Instructions:")
        print("1. A window will open showing the SUMO screenshot")
        print("2. For each lane (lane_1, lane_2, lane_3, lane_4):")
        print("   - Click TOP-LEFT corner of the lane area")
        print("   - Click BOTTOM-RIGHT corner of the lane area")
        print("3. Press any key to move to the next lane")
        print("4. Press ESC to cancel at any time")
        print("Tip: Draw boxes where vehicles queue at the intersection")
        print("=" * 60)

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
                cv2.imshow("Calibrate ROIs", display)

        cv2.namedWindow("Calibrate ROIs")
        cv2.setMouseCallback("Calibrate ROIs", _mouse_callback)

        for i, lane in enumerate(lane_names):
            print(f"Click 2 corners for {lane} …")

            while len(points) < 2 * (i + 1):
                cv2.imshow("Calibrate ROIs", display)
                key = cv2.waitKey(20)
                if key == 27:  # ESC
                    print("Calibration aborted.")
                    cv2.destroyAllWindows()
                    return {}

            p1 = points[2 * i]
            p2 = points[2 * i + 1]
            roi = {
                "x1": min(p1[0], p2[0]),
                "y1": min(p1[1], p2[1]),
                "x2": max(p1[0], p2[0]),
                "y2": max(p1[1], p2[1]),
            }
            rois_out[lane] = roi
            print(f"✓ {lane} ROI set: {roi}")

            # Draw on display
            cv2.rectangle(display,
                          (roi["x1"], roi["y1"]),
                          (roi["x2"], roi["y2"]),
                          (0, 255, 0), 2)
            cv2.putText(display, lane, (roi["x1"], roi["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.destroyAllWindows()

        # Save config
        print("=" * 60)
        print("CALIBRATION COMPLETE!")
        print("=" * 60)
        print("Copy this ROI configuration into your code:")
        print(f"rois = {rois_out}")
        print("=" * 60)

        os.makedirs(os.path.dirname(ROI_CONFIG_PATH), exist_ok=True)
        with open(ROI_CONFIG_PATH, "w") as f:
            json.dump(rois_out, f, indent=2)
        print(f"Saved to: {ROI_CONFIG_PATH}")

        return rois_out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 vehicle detector for SUMO")
    parser.add_argument("screenshot", help="Path to SUMO screenshot")
    parser.add_argument("--calibrate", action="store_true", help="Run ROI calibration")
    parser.add_argument("--rois", default=ROI_CONFIG_PATH, help="ROI config JSON path")
    args = parser.parse_args()

    # Calibration mode
    if args.calibrate:
        rois = YOLODetector.calibrate_rois_cli(args.screenshot)
        if not rois:
            sys.exit(1)
    else:
        # Load ROIs
        if os.path.isfile(args.rois):
            with open(args.rois) as f:
                rois = json.load(f)
            print(f"Loaded ROI config from {args.rois}")
        else:
            print(f"No ROI config at {args.rois}; using DEFAULT_ROIS.")
            rois = DEFAULT_ROIS

    # Test detection
    print("Testing with calibrated ROIs...")
    detector = YOLODetector()
    counts   = detector.count_per_lane(args.screenshot, rois)

    print(f"Vehicle counts per lane: {counts}")
    print(f"Total vehicles detected: {sum(counts.values())}")

    debug_img = detector.visualize(args.screenshot, rois)
    print(f"\nOpen {debug_img} to verify detections visually.")
