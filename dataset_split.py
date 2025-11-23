"""
Dataset split and quick visual analysis
--------------------------------------
Splits `mask_dataset/` into train/val/test and saves:
- `data_split/_analysis/split_counts.png`: bar chart of counts per class/split
- `data_split/_analysis/samples_grid.png`: small sample grid from the train split

Run: `python dataset_split.py`
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt


CLASS_DIRS = [
    "with_mask",
    "without_mask",
    "mask_weared_incorrect",
]
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(dir_path: Path) -> List[Path]:
    """Return all image files in a directory with supported extensions.

    Args:
        dir_path: Directory to scan.

    Returns:
        List of file paths for images.
    """
    files = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def split_list(items: List[Path], train_ratio=0.8, val_ratio=0.1, seed=42) -> Tuple[List[Path], List[Path], List[Path]]:
    """Shuffle and split a list of paths into train/val/test.

    Args:
        items: List of file paths.
        train_ratio: Fraction for the training set.
        val_ratio: Fraction for the validation set.
        seed: Random seed for reproducibility.

    Returns:
        (train, val, test) lists of paths.
    """
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def copy_split(split_map: Dict[str, Dict[str, List[Path]]], out_dir: Path) -> None:
    """Copy split files into `out_dir/<split>/<class>/`.

    Args:
        split_map: Mapping of split name -> class -> list of paths.
        out_dir: Root output directory for the copied structure.
    """
    for split_name, class_map in split_map.items():
        for cls, paths in class_map.items():
            target_dir = out_dir / split_name / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                dst = target_dir / src.name
                shutil.copy2(src, dst)


def count_split(split_map: Dict[str, Dict[str, List[Path]]]) -> Dict[str, Counter]:
    """Count items per split and per class.

    Returns:
        Dict mapping split name -> Counter(class -> count).
    """
    counts = {}
    for split_name, class_map in split_map.items():
        counts[split_name] = Counter({cls: len(paths) for cls, paths in class_map.items()})
    return counts


def plot_counts(counts: Dict[str, Counter], out_png: Path) -> None:
    """Plot a grouped bar chart of counts by class and split.

    Args:
        counts: Split/class counts from `count_split`.
        out_png: Output PNG path for the chart.
    """
    splits = sorted(counts.keys())
    classes = CLASS_DIRS
    # Collect counts in consistent order
    data = {s: [counts[s].get(c, 0) for c in classes] for s in splits}

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(classes))
    width = 0.25

    for i, s in enumerate(splits):
        offset = (i - (len(splits) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], data[s], width=width, label=s)

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel("images")
    ax.set_title("Counts per split/class")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def make_samples_grid(split_dir: Path, out_png: Path, per_class: int = 3) -> None:
    """Create a small image grid from the train split.

    Produces a grid with rows for classes and `per_class` columns of samples.

    Args:
        split_dir: Directory of the split (e.g., `data_split/train`).
        out_png: Output PNG path for the grid image.
        per_class: Number of samples per class to include.
    """
    # Grid geometry
    rows = len(CLASS_DIRS)
    cols = per_class
    thumb_size = (128, 128)
    grid_w = thumb_size[0] * cols
    grid_h = thumb_size[1] * rows
    grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))

    for r, cls in enumerate(CLASS_DIRS):
        cls_dir = split_dir / cls
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(images)
        images = images[:per_class]
        for c, img_path in enumerate(images):
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail(thumb_size)
                x = c * thumb_size[0]
                y = r * thumb_size[1]
                grid.paste(im, (x, y))
            except Exception:
                continue

    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)


def main(
    src_root: Path = Path("mask_dataset"),
    out_root: Path = Path("data_split"),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Orchestrate dataset split, copy, counts, and minimal visuals.

    Args:
        src_root: Source dataset root with class subdirectories.
        out_root: Destination root for split folders and analysis outputs.
        train_ratio: Fraction for the training set.
        val_ratio: Fraction for the validation set.
        seed: Random seed for reproducible shuffling.
    """
    # Gather per-class files
    per_class_files: Dict[str, List[Path]] = {}
    for cls in CLASS_DIRS:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            print(f"WARN: missing class directory: {cls_dir}")
            per_class_files[cls] = []
        else:
            per_class_files[cls] = list_images(cls_dir)

    # Split
    split_map: Dict[str, Dict[str, List[Path]]] = defaultdict(dict)
    for cls, files in per_class_files.items():
        train, val, test = split_list(files, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        split_map["train"][cls] = train
        split_map["val"][cls] = val
        split_map["test"][cls] = test

    # Copy into structure
    copy_split(split_map, out_root)

    # Counts
    counts = count_split(split_map)
    for split_name in ("train", "val", "test"):
        print(f"{split_name}:")
        for cls in CLASS_DIRS:
            print(f"  {cls}: {counts[split_name].get(cls, 0)}")

    # Visuals
    plot_counts(counts, out_root / "_analysis" / "split_counts.png")
    make_samples_grid(out_root / "train", out_root / "_analysis" / "samples_grid.png", per_class=3)
    print(f"Saved charts to: {out_root / '_analysis'}")


if __name__ == "__main__":
    # For quick usage, run: `python dataset_split.py`
    main()