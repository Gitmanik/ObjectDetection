import os
import cv2

def crop_to_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Obraz jest zbyt szeroki — przycinamy boki
        new_width = int(h * target_ratio)
        x_start = (w - new_width) // 2
        cropped = image[:, x_start:x_start + new_width]
    else:
        # Obraz jest zbyt wysoki — przycinamy górę i dół
        new_height = int(w / target_ratio)
        y_start = (h - new_height) // 2
        cropped = image[y_start:y_start + new_height, :]

    return cropped

def resize_images_in_folder(folder_path, target_width, target_height, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(supported_exts):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Błąd przy wczytywaniu {filename}")
            continue

        cropped = crop_to_aspect_ratio(image, target_width, target_height)
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized)
        print(f"✅ Zapisano: {output_path}")

# 🔧 Użycie
resize_images_in_folder("raw_backgrounds", 1280, 720, "resized_backgrounds")