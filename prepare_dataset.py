"""
Convert the NEU-DET dataset (Pascal VOC XML annotations) into YOLO format
and create an 80/20 train/val split.

Usage:
    python prepare_dataset.py --src path/to/NEU-DET --dst dataset

Output structure:
    dataset/
      images/train, images/val
      labels/train, labels/val
      data.yaml
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_to_yolo_lines(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASSES:
            print(f"  [warn] unknown class '{name}' in {xml_path.name}, skipped")
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing NEU-DET images + XML files")
    ap.add_argument("--dst", default="dataset", help="Output folder")
    ap.add_argument("--val", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    xmls = sorted(src.rglob("*.xml"))
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
    print(f"Found {len(xmls)} XML files and {len(images)} images")

    pairs = [(x, images[x.stem]) for x in xmls if x.stem in images]
    if not pairs:
        raise SystemExit("No image/XML pairs found. Check --src path.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for split, items in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml_path, img_path in items:
            lines = voc_to_yolo_lines(xml_path)
            shutil.copy(img_path, dst / "images" / split / img_path.name)
            (dst / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
        print(f"{split}: {len(items)} images")

    yaml = (
        f"path: {dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (dst / "data.yaml").write_text(yaml)
    print(f"Done. Wrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
