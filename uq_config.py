#!/usr/bin/env python3
"""Shared configuration helpers for RFDT beamspace UQ scripts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_CONFIG: Dict[str, Any] = {
    "forward": {
        "ply_path": "",
        "freq_hz": 3.5e9,
        "tx_pos_m": [3.5, 2.5, 2.7],
        "sh_order": 3,
        "gaussian_chunk": 32768,
        "rx_grid": {
            "x_min": 0.3,
            "x_max": 6.7,
            "x_count": 10,
            "y_min": 0.3,
            "y_max": 4.7,
            "y_count": 7,
            "z_height": 1.2,
        },
        "array": {
            "nx": 8,
            "ny": 8,
            "az": 33,
            "el": 11,
            "ang_sigma": 0.15,
        },
    },
    "sionna": {
        "scene_xml": "scenes/room_with_cube.xml",
        "max_depth": 3,
        "reflection": True,
        "scattering": True,
        "diffraction": False,
        "scat_keep_prob": 0.5,
        "num_samples": 500000,
        "global_scattering_coeff": 4.0,
        "materials": {
            "mat_concrete_scat": {
                "epsilon_r": 5.24,
                "sigma": 0.0462,
                "scattering_coeff": 0.1,
                "alpha_r": 5,
            },
            "mat_wood_scat": {
                "epsilon_r": 1.99,
                "sigma": 0.0047,
                "scattering_coeff": 0.2,
                "alpha_r": 3,
            },
            "mat_glass_scat": {
                "epsilon_r": 6.31,
                "sigma": 0.0036,
                "scattering_coeff": 0.025,
                "alpha_r": 10,
            },
            "mat_metal_scat": {
                "epsilon_r": 1.0,
                "sigma": 1e7,
                "scattering_coeff": 0.025,
                "alpha_r": 10,
            },
        },
        "object_material_map": [
            {"keywords": ["floor", "walls", "ceiling", "pillar"], "material": "mat_concrete_scat"},
            {"keywords": ["furniture", "door"], "material": "mat_wood_scat"},
            {"keywords": ["window"], "material": "mat_glass_scat"},
            {"keywords": ["tv", "led", "metal", "cube"], "material": "mat_metal_scat"},
        ],
    },
    "preprocess": {
        "visual": {
            "input_models_dir": "meshes_d",
            "output_dataset_dir": "dataset_visual_v2_with_object",
            "num_images": 300,
            "resolution": 800,
            "camera_lens_mm": 20.0,
            "room_min": [0.5, 0.5, 0.0],
            "room_max": [6.5, 4.5, 3.0],
            "cycles_samples": 96,
            "max_bounces": 3,
            "use_denoising": True,
            "denoiser": "OPENIMAGEDENOISE",
            "use_gpu": True,
            "gpu_backend": "CUDA",
            "seed": 1234,
            "test_fraction": 0.1,
            "strategy_split": {
                "perimeter": 0.34,
                "detail": 0.33,
                "topdown": 0.33,
            },
            "focus_orbit": {
                "enabled": True,
                "frames_per_object": 25,
                "keywords": ["chair", "table", "sofa", "tv", "desk"],
            },
        },
        "rf_dataset": {
            "output_dir": "dataset_ideal_mpc",
            "spectrum_type": "mpc",
            "mvdr_m": 4,
            "width": 600,
            "height": 600,
            "h_fov_deg": 120,
            "time_interval": 0.1,
            "stats_sample_count": 50,
        },
        "rf_multitx": {
            "rf_root": "dataset_custom_scene_ideal_mpc_with_object_txmulti",
            "images_per_tx": 200,
            "views_per_rx": 4,
            "train_tx_max": 2,
            "train_rx_max": 40,
        },
    },
    "uq": {
        "rx_std_xyz_m": [0.02, 0.02, 0.01],
        "mean_mode_std_m": 0.010,
        "sh_mode_std_frac": 0.08,
        "alpha_mode_std_frac": 0.06,
        "scale_mode_std_frac": 0.05,
        "dir_std_rad": 0.08,
        "smooth_std": 0.1,
        "drop_prob": 0.1,
        "drop_width_frac": 0.1,
        "enable_directional_uq": True,
        "enable_smoothing": True,
        "ut_alpha": 0.3,
        "ut_beta": 2.0,
        "ut_kappa": 0.0,
        "n_modes": 4,
        "mc_samples": 128,
        "mc_samples_list": [128, 256, 512],
        "seed": 1234,
        "high_error_quantile": 0.8,
        "report_top_n": 20,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> Tuple[Dict[str, Any], Path]:
    """Load config JSON and merge with defaults.

    If path is None, returns defaults and uses the current working directory
    as the base for relative paths. If UQ_CONFIG is set, it is used as a
    fallback path.
    """
    if not path:
        path = os.getenv("UQ_CONFIG", "")

    if path:
        cfg_path = Path(path).expanduser().resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        _deep_update(cfg, raw)
        return cfg, cfg_path.parent

    return copy.deepcopy(DEFAULT_CONFIG), Path.cwd()


def resolve_path(path_str: str, base_dir: Path) -> str:
    """Resolve a possibly-relative path against a base directory."""
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())
