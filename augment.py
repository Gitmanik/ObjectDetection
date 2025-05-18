import random
import uuid
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance

# === Configuration ===
DATA_ROOT        = Path(".")
BACKGROUND_FOLDER= DATA_ROOT / "resized_backgrounds"
TOOLS_FOLDER     = DATA_ROOT / "raw_tools"
IMG_DIR          = DATA_ROOT / "generated_images"
LABEL_DIR        = DATA_ROOT / "yolo_labels"
TRAIN_DIR        = IMG_DIR / "train"
VAL_DIR          = IMG_DIR / "val"
TEST_DIR         = IMG_DIR / "test"
DATA_YAML        = DATA_ROOT / "data.yaml"

NUM_IMAGES = 400
SCALE_MIN  = 0.2
SCALE_MAX  = 0.3

# === Reset folderów output/label ===
shutil.rmtree(IMG_DIR,   ignore_errors=True)
shutil.rmtree(LABEL_DIR, ignore_errors=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
print("✅ Folders reset")

# === Wczytaj ścieżki do obrazów ===
background_paths = list(BACKGROUND_FOLDER.glob("*.*"))
tool_paths       = list(TOOLS_FOLDER.glob("*.png"))

# === Przygotuj mapę klas (tylko tekst przed "_") ===
tool_names = sorted({p.stem.split("_")[0] for p in tool_paths})
class_map  = {name: idx for idx, name in enumerate(tool_names)}

def paste_tool_on_background(bg_img: Image.Image, tool_img: Image.Image):
    """Wkleja pojedyncze narzędzie na tło, zwraca dict z danymi do adnotacji."""
    bg_w, bg_h = bg_img.size

    # Skalowanie
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    new_w = int(tool_img.width  * scale)
    new_h = int(tool_img.height * scale)
    tool_resized = tool_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Rotacja losowa
    angle = random.randint(0, 359)
    tool_rotated = tool_resized.rotate(angle, expand=True)

    # Pozycja
    max_x = max(0, bg_w - tool_rotated.width)
    max_y = max(0, bg_h - tool_rotated.height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Wklej z alfa
    bg_img.paste(tool_rotated, (x, y), tool_rotated)

    # Nazwa klasy = to, co przed "_"
    tool_name = Path(tool_img.filename).stem.split("_")[0]

    return {
        "tool":   tool_name,
        "x":      x,
        "y":      y,
        "width":  tool_rotated.width,
        "height": tool_rotated.height,
        "angle":  angle
    }

def convert_to_yolo(ann: dict, img_w: int, img_h: int) -> str:
    """Konwertuje adnotację do formatu YOLO: cls x_center y_center w h (wszystko znormalizowane)."""
    x_c = (ann["x"] + ann["width"]/2)  / img_w
    y_c = (ann["y"] + ann["height"]/2) / img_h
    w_n = ann["width"]  / img_w
    h_n = ann["height"] / img_h
    cls = class_map[ann["tool"]]
    return f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

def augment_image(img: Image.Image) -> Image.Image:
    img = img.rotate(random.choice([0, 180]))

    if random.choice([True, False]):
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img

def augment_colors(img: Image.Image) -> Image.Image:
    choice = random.choice(["color", "contrast", "brightness", "sharpness"])
    if choice == "color":
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 2.0))
    elif choice == "contrast":
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 2.0))
    elif choice == "brightness":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 2.0))
    else:  # sharpness
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.0))
    return img

# === Generacja obrazów i etykiet ===
for i in range(NUM_IMAGES):
    bg_path = random.choice(background_paths)
    bg = Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg.size

    bg = augment_image(bg)

    tools_on_img = random.randint(1, 3)
    chosen_tools = random.sample(tool_paths, tools_on_img)

    yolo_lines = []
    for tp in chosen_tools:
        tool = Image.open(tp).convert("RGBA")
        tool.filename = str(tp)  # by paste_tool mogło odczytać ścieżkę
        ann = paste_tool_on_background(bg, tool)
        yolo_lines.append(convert_to_yolo(ann, bg_w, bg_h))

    bg = augment_colors(bg)

    # Zapis pliku PNG
    img_id     = uuid.uuid4().hex
    out_png    = IMG_DIR / f"{img_id}.png"
    bg.convert("RGB").save(out_png)

    # Zapis etykiety
    out_txt    = LABEL_DIR / f"{img_id}.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"[{i+1}/{NUM_IMAGES}] Zapisano {img_id}")

print("✅ All images generated and labeled.")

# === Split na train/val/test ===
all_imgs = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
if not TRAIN_DIR.exists():
    TRAIN_DIR.mkdir(parents=True)
    VAL_DIR.mkdir(parents=True)
    TEST_DIR.mkdir(parents=True)

    random.shuffle(all_imgs)
    n_total = len(all_imgs)
    n_val   = int(0.2 * n_total)
    test_imgs = all_imgs[-4:]
    tv_imgs   = all_imgs[:-4]
    val_imgs  = tv_imgs[:n_val]
    train_imgs= tv_imgs[n_val:]

    for imgp in train_imgs:
        lbl = LABEL_DIR / f"{imgp.stem}.txt"
        imgp.rename(TRAIN_DIR / imgp.name)
        if lbl.exists(): lbl.rename(TRAIN_DIR / lbl.name)

    for imgp in val_imgs:
        lbl = LABEL_DIR / f"{imgp.stem}.txt"
        imgp.rename(VAL_DIR / imgp.name)
        if lbl.exists(): lbl.rename(VAL_DIR / lbl.name)

    for imgp in test_imgs:
        lbl = LABEL_DIR / f"{imgp.stem}.txt"
        imgp.rename(TEST_DIR / imgp.name)
        if lbl.exists(): lbl.rename(TEST_DIR / lbl.name)

    print("✅ Dataset split into train/val/test.")

# === Zapis data.yaml dla YOLO ===
with open(DATA_YAML, "w") as f:
    yaml.dump({
        "train": str(TRAIN_DIR),
        "val":   str(VAL_DIR),
        "names": tool_names
    }, f)

print("✅ data.yaml saved.")

# === Cleanup ===
shutil.rmtree(LABEL_DIR, ignore_errors=True)
print("✅ Cleaned up temporary files.")
