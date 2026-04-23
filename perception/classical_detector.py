"""
perception/classical_detector.py
---------------------------------
Classical computer vision vehicle detector for SUMO.

Uses HSV color space + contour detection to find yellow vehicle sprites.
This works MUCH better than YOLO for SUMO's simplified graphics.

Usage:
    from perception.classical_detector import ClassicalDetector
    detector = ClassicalDetector()
    counts = detector.count_per_lane("screenshot.png", rois)
"""

import os
import json
import cv2
import numpy as np

# ── ROI config path ────────────────────────────────────────────────────────
ROI_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "roi_config.json"
)

# ── Default ROIs ───────────────────────────────────────────────────────────
DEFAULT_ROIS = {
    "lane_1": {"x1": 430, "y1": 50, "x2": 600, "y2": 300},
    "lane_2": {"x1": 430, "y1": 470, "x2": 600, "y2": 720},
    "lane_3": {"x1": 680, "y1": 320, "x2": 970, "y2": 450},
    "lane_4": {"x1": 50, "y1": 320, "x2": 340, "y2": 450},
}


class ClassicalDetector:
    """
    Detects SUMO vehicles using HSV color thresholding + contour detection.
    
    Parameters
    ----------
    yellow_lower : tuple
        Lower HSV bound for yellow (default: (20, 100, 100))
    yellow_upper : tuple
        Upper HSV bound for yellow (default: (30, 255, 255))
    min_area : int
        Minimum contour area in pixels (default: 50)
    """
    
    def __init__(self,
                 yellow_lower: tuple = (20, 100, 100),
                 yellow_upper: tuple = (30, 255, 255),
                 min_area: int = 50):
        self.yellow_lower = np.array(yellow_lower)
        self.yellow_upper = np.array(yellow_upper)
        self.min_area = min_area
        print("✓ Classical CV detector initialized")
    
    def detect(self, image_path: str):
        """
        Detect yellow vehicle sprites in the image.
        
        Returns
        -------
        list of (x, y, w, h) bounding boxes
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Threshold for yellow
        mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and get bounding boxes
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, w, h))
        
        return detections, mask
    
    def count_per_lane(self, image_path: str, rois: dict) -> dict:
        """
        Count vehicles in each ROI.
        
        Parameters
        ----------
        image_path : str
            Path to SUMO screenshot
        rois : dict
            {"lane_1": {"x1":...,"y1":...,"x2":...,"y2":...}, ...}
        
        Returns
        -------
        dict {"lane_1": int, "lane_2": int, "lane_3": int, "lane_4": int}
        """
        detections, _ = self.detect(image_path)
        counts = {k: 0 for k in rois}
        
        for (x, y, w, h) in detections:
            # Center of bounding box
            cx = x + w / 2
            cy = y + h / 2
            
            # Check which ROI contains this vehicle
            for lane_key, roi in rois.items():
                if (roi["x1"] <= cx <= roi["x2"] and
                    roi["y1"] <= cy <= roi["y2"]):
                    counts[lane_key] += 1
                    break  # Each vehicle only counts once
        
        return counts
    
    def visualize(self, image_path: str, rois: dict,
                  output_path: str = None) -> str:
        """
        Draw detections and ROIs on the image.
        
        Returns
        -------
        str: path to saved debug image
        """
        output_path = output_path or os.path.join(
            os.path.dirname(__file__), "..", "results", "cv_debug.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        img = cv2.imread(image_path)
        detections, mask = self.detect(image_path)
        
        # Draw detections
        for (x, y, w, h) in detections:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "vehicle", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw ROIs
        roi_colours = [(255, 0, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
        for idx, (lane_key, roi) in enumerate(rois.items()):
            colour = roi_colours[idx % len(roi_colours)]
            cv2.rectangle(img,
                         (roi["x1"], roi["y1"]),
                         (roi["x2"], roi["y2"]),
                         colour, 2)
            cv2.putText(img, lane_key, (roi["x1"], roi["y1"] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        
        cv2.imwrite(output_path, img)
        print(f"[ClassicalDetector] Debug image saved → {output_path}")
        return output_path
    
    @staticmethod
    def calibrate_rois_cli(screenshot_path: str) -> dict:
        """
        Interactive ROI calibration.
        Click top-left then bottom-right corner for each of the 4 lanes.
        """
        import time as _time

        img = cv2.imread(screenshot_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open: {screenshot_path}")

        # Resize if image is larger than screen so window fits
        h, w = img.shape[:2]
        max_dim = 1400
        scale = 1.0
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        display = img.copy()
        points = []
        lane_names = ["lane_1", "lane_2", "lane_3", "lane_4"]
        rois_out = {}

        print("=" * 60)
        print("ROI CALIBRATION TOOL")
        print("=" * 60)
        print("1. CLICK inside the OpenCV window first to give it focus")
        print("2. Click TOP-LEFT corner of the lane approach area")
        print("3. Click BOTTOM-RIGHT corner of the lane approach area")
        print("4. Repeat for all 4 lanes")
        print("5. Press ESC ONLY if you want to cancel")
        print("=" * 60)

        def _mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Scale back to original image coords
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                points.append((orig_x, orig_y))
                cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(display, f"({orig_x},{orig_y})",
                            (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow("Calibrate ROIs", display)

        cv2.namedWindow("Calibrate ROIs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibrate ROIs", img.shape[1], img.shape[0])
        cv2.setMouseCallback("Calibrate ROIs", _mouse_callback)
        cv2.imshow("Calibrate ROIs", display)

        # Flush stale keypresses — wait 1.5 s before enabling ESC abort
        # This prevents terminal Enter/ESC from immediately aborting
        flush_deadline = _time.time() + 1.5
        while _time.time() < flush_deadline:
            cv2.waitKey(20)

        for i, lane in enumerate(lane_names):
            print(f"\nClick 2 corners for {lane} "
                  f"({2*i+1}/8 and {2*i+2}/8 clicks)...")

            while len(points) < 2 * (i + 1):
                cv2.imshow("Calibrate ROIs", display)
                key = cv2.waitKey(20) & 0xFF
                if key == 27:  # ESC — only active after flush period
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

            # Draw on DISPLAY (scaled) image — convert coords back to display space
            dx1 = int(roi["x1"] * scale)
            dy1 = int(roi["y1"] * scale)
            dx2 = int(roi["x2"] * scale)
            dy2 = int(roi["y2"] * scale)
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            cv2.putText(display, lane, (dx1, dy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.destroyAllWindows()
        
        print("=" * 60)
        print("CALIBRATION COMPLETE!")
        print("=" * 60)
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
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Classical CV vehicle detector for SUMO")
    parser.add_argument("screenshot", help="Path to SUMO screenshot")
    parser.add_argument("--calibrate", action="store_true", help="Run ROI calibration")
    parser.add_argument("--rois", default=ROI_CONFIG_PATH, help="ROI config JSON path")
    args = parser.parse_args()
    
    if args.calibrate:
        rois = ClassicalDetector.calibrate_rois_cli(args.screenshot)
        if not rois:
            sys.exit(1)
    else:
        if os.path.isfile(args.rois):
            with open(args.rois) as f:
                rois = json.load(f)
            print(f"Loaded ROI config from {args.rois}")
        else:
            print(f"No ROI config at {args.rois}; using DEFAULT_ROIS.")
            rois = DEFAULT_ROIS
    
    print("\nTesting detection...")
    detector = ClassicalDetector()
    counts = detector.count_per_lane(args.screenshot, rois)
    
    print(f"Vehicle counts per lane: {counts}")
    print(f"Total vehicles detected: {sum(counts.values())}")
    
    debug_img = detector.visualize(args.screenshot, rois)
    print(f"\nOpen {debug_img} to verify detections visually.")