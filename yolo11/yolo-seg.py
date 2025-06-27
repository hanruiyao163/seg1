from ultralytics import YOLO


model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)



if __name__ == "__main__":
    results = model.train(data=r"C:\Users\hanma\Programming\seg1\yolo11\dataset.yaml", epochs=100, imgsz=640)
