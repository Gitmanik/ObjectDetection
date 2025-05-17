import os
import random
import uuid
import json
from PIL import Image, ImageEnhance
import shutil

# Ścieżki
BACKGROUND_FOLDER = "resized_backgrounds"
TOOLS_FOLDER = "raw_tools"
OUTPUT_FOLDER = "generated_images"
ANNOTATION_FOLDER = "annotations"
YOLO_LABEL_FOLDER = "yolo_labels"

# Parametry
NUM_IMAGES = 20
SCALE_MIN = 0.2
SCALE_MAX = 0.3

# Resetuj foldery
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
shutil.rmtree(ANNOTATION_FOLDER, ignore_errors=True)
shutil.rmtree(YOLO_LABEL_FOLDER, ignore_errors=True)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)
os.makedirs(YOLO_LABEL_FOLDER, exist_ok=True)

# Wczytaj obrazy
background_paths = [os.path.join(BACKGROUND_FOLDER, f) for f in os.listdir(BACKGROUND_FOLDER)]
tool_paths = [os.path.join(TOOLS_FOLDER, f) for f in os.listdir(TOOLS_FOLDER)]

# Mapa klas: narzędzie -> ID
tool_names = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in tool_paths))
class_map = {name: idx for idx, name in enumerate(tool_names)}

def paste_tool_on_background(bg_img, tool_img):
    bg_w, bg_h = bg_img.size

    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    new_w = int(tool_img.width * scale)
    new_h = int(tool_img.height * scale)
    tool_resized = tool_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    angle = random.randint(0, 359)
    tool_rotated = tool_resized.rotate(angle, expand=True)

    max_x = max(0, bg_w - tool_rotated.width)
    max_y = max(0, bg_h - tool_rotated.height)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    bg_img.paste(tool_rotated, (x, y), tool_rotated)

    return {
        "tool": os.path.basename(tool_img.filename),
        "x": x,
        "y": y,
        "width": tool_rotated.width,
        "height": tool_rotated.height,
        "angle": angle
    }

def convert_to_yolo(annotation, img_w, img_h):
    x_center = (annotation["x"] + annotation["width"] / 2) / img_w
    y_center = (annotation["y"] + annotation["height"] / 2) / img_h
    width = annotation["width"] / img_w
    height = annotation["height"] / img_h
    class_id = class_map[os.path.splitext(annotation["tool"])[0]]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def augment_image(image):
    # Losowy obrót
    angle = random.choice([0, 180])
    image = image.rotate(angle)

    # Odwrócenie poziome lub pionowe
    if random.choice([True, False]):
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image

def augment_colors(image):

    # Zmiana koloru
    enhancement_type = random.choice(["color", "contrast", "brightness", "sharpness"])
    if enhancement_type == "color":
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.5, 2))  # Losowy poziom nasycenia
    elif enhancement_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.5, 2))  # Losowy kontrast
    elif enhancement_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.5, 2))  # Jasność
    elif enhancement_type == "sharpness":
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(random.uniform(0.5, 2.0))  # Ostrość
    
    return image

# Generuj obrazy
for i in range(NUM_IMAGES):
    bg_path = random.choice(background_paths)
    bg = Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg.size
    bg = augment_image(bg)

    tools_on_image = random.randint(1, 3)
    used_tools = random.sample(tool_paths, tools_on_image)

    annotations = []
    yolo_lines = []

    for tool_path in used_tools:
        tool = Image.open(tool_path).convert("RGBA")
        tool.filename = tool_path
        annotation = paste_tool_on_background(bg, tool)
        annotations.append(annotation)
        yolo_line = convert_to_yolo(annotation, bg_w, bg_h)
        yolo_lines.append(yolo_line)

    bg = augment_colors(bg)

    image_uuid = str(uuid.uuid4())
    output_path = os.path.join(OUTPUT_FOLDER, f"{image_uuid}.png")
    bg.convert("RGB").save(output_path)

    # JSON
    json_path = os.path.join(ANNOTATION_FOLDER, f"{image_uuid}.json")
    with open(json_path, "w") as f:
        json.dump({
            "image": f"{image_uuid}.png",
            "annotations": annotations
        }, f, indent=4)

    # YOLO TXT
    yolo_path = os.path.join(YOLO_LABEL_FOLDER, f"{image_uuid}.txt")
    with open(yolo_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"[{i+1}/{NUM_IMAGES}] Obraz i adnotacje zapisane jako {image_uuid}")

# Zapisz mapę klas (YOLO wymaga class listy np. dla Roboflow)
with open("classes.txt", "w") as f:
    for name in tool_names:
        f.write(name + "\n")

print("✅ Wszystkie obrazy i adnotacje zapisane.")