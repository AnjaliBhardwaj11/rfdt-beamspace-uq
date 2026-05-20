#!/usr/bin/env python3
"""Lightweight validation of configuration, dependencies, and outputs."""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run basic smoke tests for the RFDT UQ pipeline.")
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory that should contain pipeline outputs",
    )
    parser.add_argument(
        "--check-outputs",
        action="store_true",
        help="Check for expected output .npy/.npz files",
    )
    return parser.parse_args()


def check_path(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"OK: {label} -> {path}")


def check_import(module_name: str) -> None:
    __import__(module_name)
    print(f"OK: import {module_name}")


def main():
    args = parse_args()
    cfg, base_dir = load_config(args.config)

    ply_path = Path(resolve_path(cfg["forward"].get("ply_path", ""), base_dir))
    scene_path = Path(resolve_path(cfg["sionna"].get("scene_xml", ""), base_dir))

    if not ply_path.as_posix():
        raise ValueError("forward.ply_path is empty in config")

    check_path("PLY file", ply_path)
    check_path("Scene XML", scene_path)

    for module in [
        "numpy",
        "torch",
        "tensorflow",
        "sionna",
        "plyfile",
        "scipy",
        "sklearn",
        "matplotlib",
    ]:
        check_import(module)

    out_dir = Path(resolve_path(args.output_dir, base_dir))
    if args.check_outputs:
        expected = [
            "rfdt_beam_energy_grid.npy",
            "rfdt_beam_gradient_grid.npy",
            "sionna_beam_energy_grid_aligned.npy",
            "rfdt_uq_mu_lin.npy",
            "rfdt_uq_sigma_best_lin.npy",
            "uq_results.npz",
        ]
        for name in expected:
            check_path(f"Output {name}", out_dir / name)

    print("\nSmoke tests passed.")


if __name__ == "__main__":
    main()
