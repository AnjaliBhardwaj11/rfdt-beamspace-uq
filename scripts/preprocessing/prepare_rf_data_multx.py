#!/usr/bin/env python3
import argparse
import copy
import glob
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path


DEFAULT_CFG = {
    "rf_root": "dataset_custom_scene_ideal_mpc_with_object_txmulti",
    "images_per_tx": 200,
    "views_per_rx": 4,
    "train_tx_max": 2,
    "train_rx_max": 40,
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a multi-TX RF dataset for RF-3DGS training."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to uq_config.json")
    parser.add_argument("--rf-root", type=str, default=None, help="Root dataset directory")
    parser.add_argument("--images-per-tx", type=int, default=None)
    parser.add_argument("--views-per-rx", type=int, default=None)
    parser.add_argument("--train-tx-max", type=int, default=None)
    parser.add_argument("--train-rx-max", type=int, default=None)
    return parser.parse_args()


def load_multitx_config(args):
    cfg = copy.deepcopy(DEFAULT_CFG)
    base_dir = Path.cwd()

    if args.config:
        merged_cfg, base_dir = load_config(args.config)
        multitx_cfg = merged_cfg.get("preprocess", {}).get("rf_multitx", {})
        deep_update(cfg, multitx_cfg)

    if args.rf_root:
        cfg["rf_root"] = args.rf_root
    if args.images_per_tx is not None:
        cfg["images_per_tx"] = int(args.images_per_tx)
    if args.views_per_rx is not None:
        cfg["views_per_rx"] = int(args.views_per_rx)
    if args.train_tx_max is not None:
        cfg["train_tx_max"] = int(args.train_tx_max)
    if args.train_rx_max is not None:
        cfg["train_rx_max"] = int(args.train_rx_max)

    cfg["rf_root"] = resolve_path(cfg["rf_root"], base_dir)
    return cfg


def main():
    args = parse_args()
    cfg = load_multitx_config(args)

    rf_root = cfg["rf_root"]
    sparse_dir = os.path.join(rf_root, "sparse", "0")
    images_dir = os.path.join(rf_root, "images")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"Created {sparse_dir}")
    print(f"Created {images_dir}")

    for fname in ["cameras.txt", "images.txt"]:
        src = os.path.join(rf_root, fname)
        dst = os.path.join(sparse_dir, fname)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {fname}")
        else:
            print(f"Warning: {fname} missing")

    points3d_path = os.path.join(sparse_dir, "points3D.txt")
    if not os.path.exists(points3d_path):
        with open(points3d_path, "w") as f:
            f.write("# dummy point\n")
            f.write("1 0 0 0 255 255 255 0\n")
        print("Created dummy points3D.txt")

    spectrum_dirs = sorted(glob.glob(os.path.join(rf_root, "spectrum_tx*")))
    if not spectrum_dirs:
        print("No spectrum_tx* directories found.")
        return

    all_images = []
    for spec_dir in spectrum_dirs:
        imgs = sorted(glob.glob(os.path.join(spec_dir, "*.png")))
        all_images.extend(imgs)

    print(f"Found {len(all_images)} images")

    image_names = []
    for idx, img in enumerate(all_images, start=1):
        new_name = f"{idx:05d}.png"
        dst = os.path.join(images_dir, new_name)
        shutil.copy2(img, dst)
        image_names.append(new_name)

    print(f"Copied {len(image_names)} images")

    train_indices = []
    test_indices = []

    images_per_tx = int(cfg["images_per_tx"])
    views_per_rx = int(cfg["views_per_rx"])
    train_tx_max = int(cfg["train_tx_max"])
    train_rx_max = int(cfg["train_rx_max"])

    for i, name in enumerate(image_names):
        tx_id = i // images_per_tx
        rx_id = (i % images_per_tx) // views_per_rx

        if tx_id < train_tx_max and rx_id < train_rx_max:
            train_indices.append(name)
        else:
            test_indices.append(name)

    print("Train images:", len(train_indices))
    print("Test images :", len(test_indices))

    with open(os.path.join(rf_root, "train_index.txt"), "w") as f:
        for name in train_indices:
            f.write(name + "\n")

    with open(os.path.join(rf_root, "test_index.txt"), "w") as f:
        for name in test_indices:
            f.write(name + "\n")

    print("Created train_index.txt and test_index.txt")


if __name__ == "__main__":
    main()