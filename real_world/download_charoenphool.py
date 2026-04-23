# download_charoenphool.py
# Run from: D:\Minor Project\
# pip install roboflow  (if not already installed)

from roboflow import Roboflow
import os

rf      = Roboflow(api_key="8T5Xj5XAOGBidiYSN2Fw")
project = rf.workspace("subrata-roy-ejpwc").project("m2_charoenphool-intersection")
dataset = project.version(1)          # change number if version is different

dataset.download(
    "yolov8",
    location=os.path.join("data", "charoenphool_images"),
    overwrite=True
)

print("\nDone. Images saved to data/charoenphool_images/")
print("Use train/, valid/, test/ subfolders as before.")