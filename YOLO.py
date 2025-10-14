from ultralytics import YOLO

model = YOLO(r"D:\yolo_car_plate\yolo11s.pt")

if __name__ == "__main__":
    model.train(
        data = "D:\\yolo_car_plate\\License-Plate-Data\\data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 32,
        device = 0,
        workers = 6,
        name = "yolo_car_plate"
    )