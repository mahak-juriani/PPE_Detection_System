from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml") # build a new model from scratch

# Use the model
person_results = model.train(data="config_person.yaml",epochs=1) # train the person model