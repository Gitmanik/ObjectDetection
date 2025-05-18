import random
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

# Add ClearML import
from clearml import Task, Logger

# === Configuration ===
DATA_ROOT = Path(".")
IMG_DIR = DATA_ROOT / "generated_images"
LABEL_DIR = DATA_ROOT / "yolo_labels"
TRAIN_DIR = IMG_DIR / "train"
VAL_DIR = IMG_DIR / "val"
TEST_DIR = IMG_DIR / "test"
DATA_YAML = DATA_ROOT / "data.yaml"

TRAINED_NAME = "tool_detector3"

def train():
    # Create ClearML task
    task = Task.init(
        project_name="Tool Detection",
        task_name="YOLO Tool Detection",
        tags=["object detection", "yolov8"]
    )

    hyp_params = {
        'lr0': 0.001,
        'weight_decay': 1e-4,
        'dropout': 0.3,
    }
    # Log hyperparameters to ClearML
    task.connect(hyp_params)

    # ---- Train YOLOv8 Model ----
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=-1,
        project="runs",
        name=TRAINED_NAME,
        exist_ok=False,
        **hyp_params  # -> lr0, weight_decay, dropout
    )

    # Validate and check mAP
    metrics = model.val()  # returns metrics dict
    mAP50 = metrics.box.map50 * 100

    # Log metrics to ClearML
    Logger.current_logger().report_scalar(
        title='Model Metrics',
        series='mAP50',
        value=mAP50,
        iteration=1
    )

    # Add assertion with more informative logging
    assert mAP50 > 60, f"Validation mAP50 too low: {mAP50:.2f}%"
    print(f"Validation mAP50: {mAP50:.2f}%")

    test(task)

    # Close the ClearML task
    task.close()
    print("Task closed.")

def test(task = None):
    # ---- Inference on 4 random test images ----
    best_model = YOLO(f'runs/{TRAINED_NAME}/weights/best.pt')
    test_images = list(TEST_DIR.glob('*.jpg')) + list(TEST_DIR.glob('*.png'))
    assert len(test_images) >= 4
    sample_imgs = random.sample(test_images, 4)

    print(sample_imgs)

    # Log inference results to ClearML
    for img_path in sample_imgs:
        print(img_path)
        results = best_model.predict(source=str(img_path), conf=0.25, imgsz=640)
        res = results[0]

        # plot & save
        vis = res.plot()
        out_path = DATA_ROOT / f"inference_{img_path.stem}.jpg"
        Image.fromarray(vis).save(out_path)

        if task is not None:
            # Log inference images to ClearML
            Logger.current_logger().report_image(
                title='Inference Results',
                series=img_path.stem,
                local_path=str(out_path)
            )

        print(f"Saved inference visualization: {out_path}")

# train()
test()