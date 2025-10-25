import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags


def is_valid_jpeg(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, "Файл не існує."
    if not path.is_file():
        return False, "Це не файл."
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return False, "Файл не має розширення .jpg або .jpeg."

    try:
        with Image.open(path) as im:
            im.verify()  # швидка перевірка цілісності
    except Exception as e:
        return False, f"Помилка перевірки JPEG: {e!r}"

    try:
        with Image.open(path) as im2:
            fmt = im2.format
        if fmt != "JPEG":
            return False, f"Невірний формат: {fmt}. Очікується JPEG."
    except Exception as e:
        return False, f"Помилка відкриття файлу як JPEG: {e!r}"

    return True, "OK"


def load_image_rgb_corrected(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with Image.open(path) as im:
        exif_raw = {}
        if hasattr(im, "getexif"):
            exif_raw = im.getexif()
        elif hasattr(im, "_getexif"):
            exif_raw = im._getexif() or {}

        try:
            orientation_tag = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_tag and exif_raw and orientation_tag in exif_raw:
                orientation = exif_raw[orientation_tag]
                if orientation == 2:  # Mirror horizontal
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:  # Rotate 180
                    im = im.rotate(180, expand=True)
                elif orientation == 4:  # Mirror vertical
                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:  # Mirror horizontal and rotate 270 CW
                    im = im.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
                elif orientation == 6:  # Rotate 270 CW
                    im = im.rotate(270, expand=True)
                elif orientation == 7:  # Mirror horizontal and rotate 90 CW
                    im = im.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                elif orientation == 8:  # Rotate 90 CW
                    im = im.rotate(90, expand=True)
        except Exception:
            pass

        rgb = np.array(im.convert("RGB"))
    return rgb, dict(exif_raw) if exif_raw else {}


def exif_to_human_readable(exif: Dict[int, Any]) -> Dict[str, Any]:
    if not exif:
        return {}

    tagmap = {k: v for k, v in ExifTags.TAGS.items()}
    readable: Dict[str, Any] = {}
    for tag_id, value in exif.items():
        tag_name = tagmap.get(tag_id, str(tag_id))
        try:
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = str(value)
            elif hasattr(value, "numerator") and hasattr(value, "denominator"):
                value = float(value.numerator) / float(value.denominator) if value.denominator else float(value.numerator)
            elif isinstance(value, (list, tuple)):
                value = [str(v) for v in value]
            else:
                if not isinstance(value, (int, float, str, bool)):
                    value = str(value)
        except Exception:
            value = str(value)

        readable[tag_name] = value

    return readable


def detect_faces_bboxes_bgr(bgr_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Не вдалося завантажити каскад Хаара: {cascade_path}")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return list(map(tuple, faces))


def main() -> None:
    parser = argparse.ArgumentParser(description="Перевірка JPEG, EXIF -> JSON, детекція облич (OpenCV).")
    parser.add_argument("image", type=Path, help="Шлях до JPEG-файлу (наприклад, new_york.jpeg)")
    args = parser.parse_args()

    img_path: Path = args.image
    ok, reason = is_valid_jpeg(img_path)
    if not ok:
        print(f"[Помилка] {reason}", file=sys.stderr)
        sys.exit(1)

    rgb, exif_raw = load_image_rgb_corrected(img_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    faces = detect_faces_bboxes_bgr(bgr)

    for (x, y, w, h) in faces:
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # BGR: червоний

    out_img = img_path.with_name(f"{img_path.stem}_faces.jpg")
    cv2.imwrite(str(out_img), bgr)

    exif_readable = exif_to_human_readable(exif_raw)

    meta: Dict[str, Any] = {
        "source_file": str(img_path),
        "output_image": str(out_img),
        "image_size": {"width": int(bgr.shape[1]), "height": int(bgr.shape[0])},
        "faces_count": len(faces),
        "faces_bboxes": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces],
        "exif": exif_readable,
    }

    out_json = img_path.with_name(f"{img_path.stem}_metadata.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Оброблено: {img_path.name}")
    print(f" → Збережено зображення з рамками: {out_img.name}")
    print(f" → Збережено метадані: {out_json.name}")
    print(f"Знайдено облич: {len(faces)}")


if __name__ == "__main__":
    main()
