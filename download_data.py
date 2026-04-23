from roboflow import Roboflow

rf = Roboflow(api_key="8T5Xj5XAOGBidiYSN2Fw")
project = rf.workspace("vai").project("traffic-intersection-vehicle-detection")
dataset = project.version(1).download("yolov8", location="data/real_images")
print("Done! Images saved to data/real_images")