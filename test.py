from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    model = YOLO(r"runs/detect/yolo_car_plate10/weights/best.pt")
    # Evaluate on validation set using correct data.yaml
    metrics = model.val(data="License-Plate-Data/data.yaml")
    print("\n===== Evaluation Results on Validation Set =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
