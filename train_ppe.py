from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch

# Use the model
ppe_results = model.train(data="config_ppe.yaml",epochs=1) # train the ppe model
