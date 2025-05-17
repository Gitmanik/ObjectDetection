import random
from pathlib import Path

import yaml
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

# Create ClearML task
task = Task.init(
    project_name="Tool Detection",
    task_name="YOLO Tool Detection",
    tags=["object detection", "yolov8"]
)

# Create splits if not exist (80/20 for train/val, reserve separate folder for test images)
all_images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
if not TRAIN_DIR.exists():
    TRAIN_DIR.mkdir(parents=True)
    VAL_DIR.mkdir(parents=True)
    TEST_DIR.mkdir(parents=True)
    random.shuffle(all_images)

    n_total = len(all_images)
    n_val = int(0.2 * n_total)
    # Use last 4 images as test
    test_images = all_images[-4:]
    train_val = all_images[:-4]
    val_images = train_val[:n_val]
    train_images = train_val[n_val:]

    # Move images and labels
    for img_path in train_images:
        img_path.rename(TRAIN_DIR / img_path.name)
        lbl = LABEL_DIR / (img_path.stem + ".txt")
        if lbl.exists(): lbl.rename(TRAIN_DIR / lbl.name)
    for img_path in val_images:
        img_path.rename(VAL_DIR / img_path.name)
        lbl = LABEL_DIR / (img_path.stem + ".txt")
        if lbl.exists(): lbl.rename(VAL_DIR / lbl.name)
    for img_path in test_images:
        img_path.rename(TEST_DIR / img_path.name)
        lbl = LABEL_DIR / (img_path.stem + ".txt")
        if lbl.exists(): lbl.rename(TEST_DIR / lbl.name)


with open('classes.txt') as f:
    names = [line.rstrip() for line in f]

# Create data.yaml for YOLO
data_yaml = DATA_ROOT / "data.yaml"
with open(data_yaml, 'w') as f:
    yaml.dump({
        'train': str(TRAIN_DIR),
        'val': str(VAL_DIR),
        'names': names,
    }, f)

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
    data=str(data_yaml),
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs",
    name="tool_detector",
    exist_ok=True,
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

# ---- Inference on 4 random test images ----
best_model = YOLO('runs/tool_detector/weights/best.pt')
test_images = list(TEST_DIR.glob('*.*'))
assert len(test_images) >= 4
sample_imgs = random.sample(test_images, 4)

# Log inference results to ClearML
for img_path in sample_imgs:
    results = best_model.predict(source=str(img_path), conf=0.25, imgsz=640)
    res = results[0]

    # plot & save
    vis = res.plot()
    out_path = DATA_ROOT / f"inference_{img_path.stem}.jpg"
    Image.fromarray(vis).save(out_path)

    # Log inference images to ClearML
    Logger.current_logger().report_image(
        title='Inference Results',
        series=img_path.stem,
        local_path=str(out_path)
    )

    print(f"Saved inference visualization: {out_path}")

# Close the ClearML task
task.close()