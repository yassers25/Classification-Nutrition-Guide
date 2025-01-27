from ultralytics import YOLO

#LOAD A MODEL
# model = YOLO("yolov8n.yaml")
model = YOLO("runs/detect/train13/weights/epoc15.pt")


# use the model
results = model.train(data="config.yaml", epochs =1,save_period =3) # train the model
