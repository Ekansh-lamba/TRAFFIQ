# download_intersection.py
# Run from: D:\Minor Project\
# Usage: python download_intersection.py

from roboflow import Roboflow
import os

API_KEY = "8T5Xj5XAOGBidiYSN2Fw"   # paste your Roboflow API key

rf      = Roboflow(api_key=API_KEY)
project = rf.workspace("ai-lf0x6").project("traffic-intersection-ypnvf")
dataset = project.version(1)

dataset.download(
    "yolov8",
    location=os.path.join("data", "intersection_images"),
    overwrite=True
)

print("\nDone. Images saved to data/intersection_images/")
print("Next: python real_world/real_data_demo.py --images data/intersection_images/test/images")