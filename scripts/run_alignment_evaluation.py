#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path

C0 = 299792458.0


def build_rx_grid(rx_cfg):
    x_range = np.linspace(rx_cfg["x_min"], rx_cfg["x_max"], int(rx_cfg["x_count"]))
    y_range = np.linspace(rx_cfg["y_min"], rx_cfg["y_max"], int(rx_cfg["y_count"]))
    z_height = float(rx_cfg["z_height"])
    return [[x, y, z_height] for x in x_range for y in y_range]


def build_dirs(array_cfg):
    az_grid = np.linspace(-np.pi, np.pi, int(array_cfg["az"]), endpoint=False)
    el_grid = np.linspace(0, np.pi / 2, int(array_cfg["el"]))
    return np.array(
        [
            [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
            for th in el_grid
            for ph in az_grid
        ]
    )


def build_codebook(array_cfg, dirs_grid, freq_hz):
    lam = C0 / float(freq_hz)
    k0 = 2 * np.pi / lam
    d = lam / 2
    nx = int(array_cfg["nx"])
    ny = int(array_cfg["ny"])
    coords = np.array(
        [
            [(i - (nx - 1) / 2) * d, (j - (ny - 1) / 2) * d, 0.0]
            for i in range(nx)
            for j in range(ny)
        ]
    )
    steering = np.exp(1j * k0 * (coords @ dirs_grid.T))
    beams = steering / np.linalg.norm(steering, axis=0, keepdims=True)
    beam_gain = np.abs(beams.conj().T @ steering) ** 2
    return coords, beams, beam_gain, k0


def build_materials(material_cfg, global_scattering_coeff):
    mats = {}
    for name, cfg in material_cfg.items():
        mats[name] = RadioMaterial(
            name,
            cfg["epsilon_r"],
            cfg["sigma"],
            scattering_coefficient=cfg["scattering_coeff"] * global_scattering_coeff,
            scattering_pattern=sionna.rt.DirectivePattern(alpha_r=cfg["alpha_r"]),
        )
    return mats


def apply_materials(scene, materials, object_material_map):
    for mat in materials.values():
        if mat.name not in scene.radio_materials:
            scene.add(mat)

    if not object_material_map:
        return

    for obj_name, obj in scene.objects.items():
        name = obj_name.lower()
        for rule in object_material_map:
            if any(k in name for k in rule["keywords"]):
                obj.radio_material = rule["material"]
                break


def compute_sionna_beam_energy(scene, rx_pos, beam_gain, dirs_grid, sionna_cfg, ang_sigma):
    if "rx" in scene.receivers:
        scene.remove("rx")

    scene.add(Receiver(name="rx", position=rx_pos))

    paths = scene.compute_paths(
        max_depth=int(sionna_cfg["max_depth"]),
        reflection=bool(sionna_cfg["reflection"]),
        scattering=bool(sionna_cfg["scattering"]),
        diffraction=bool(sionna_cfg["diffraction"]),
        scat_keep_prob=float(sionna_cfg["scat_keep_prob"]),
        num_samples=int(sionna_cfg["num_samples"]),
    )

    a = paths.a.numpy().reshape(-1)
    theta = paths.theta_t.numpy().reshape(-1)
    phi = paths.phi_t.numpy().reshape(-1)

    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )

    valid = np.isfinite(a) & np.isfinite(dirs).all(axis=1)
    a = a[valid]
    dirs = dirs[valid]

    P = np.zeros(dirs_grid.shape[0])
    sigma = float(ang_sigma)

    for p in range(len(a)):
        dot = dirs_grid @ dirs[p]
        w = np.exp(-(1 - dot) ** 2 / (2 * sigma ** 2))
        P += np.abs(a[p]) ** 2 * w

    E = beam_gain @ P
    return 10 * np.log10(E + 1e-12)


def normalize_db(E):
    E = E - np.max(E)
    E_lin = 10 ** (E / 10)
    E_lin /= np.sum(E_lin) + 1e-12
    return 10 * np.log10(E_lin + 1e-12)


def dominant_dir(E, dirs_grid):
    return dirs_grid[np.argmax(E)]


def rodrigues(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align RFDT beam energies with Sionna and report agreement metrics."
    )
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument("--rfdt", default="rfdt_beam_energy_grid.npy", help="RFDT .npy input")
    parser.add_argument(
        "--output",
        default="sionna_beam_energy_grid_aligned.npy",
        help="Output aligned Sionna .npy file",
    )
    parser.add_argument("--scene-xml", default="", help="Override scene XML path")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--topk", type=int, default=10, help="Top-k beams for overlap metric")
    parser.add_argument("--thresh", type=float, default=0.99, help="Region overlap threshold")
    parser.add_argument("--num-samples", type=int, default=0, help="Override Sionna samples")
    parser.add_argument("--max-depth", type=int, default=0, help="Override max path depth")
    parser.add_argument("--scat-keep-prob", type=float, default=-1.0, help="Override scat_keep_prob")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    cfg, base_dir = load_config(args.config)
    if args.scene_xml:
        cfg["sionna"]["scene_xml"] = args.scene_xml
    if args.num_samples > 0:
        cfg["sionna"]["num_samples"] = int(args.num_samples)
    if args.max_depth > 0:
        cfg["sionna"]["max_depth"] = int(args.max_depth)
    if args.scat_keep_prob >= 0.0:
        cfg["sionna"]["scat_keep_prob"] = float(args.scat_keep_prob)

    forward_cfg = cfg["forward"]
    sionna_cfg = cfg["sionna"]

    rfdt_path = Path(resolve_path(args.rfdt, base_dir))
    if not rfdt_path.exists():
        raise FileNotFoundError(f"RFDT input not found: {rfdt_path}")
    rfdt = np.load(rfdt_path)
    if rfdt.ndim == 1:
        rfdt = rfdt[None, :]

    rx_locs = build_rx_grid(forward_cfg["rx_grid"])
    nrx = len(rx_locs)
    if rfdt.shape[0] != nrx:
        raise ValueError(f"RFDT RX count {rfdt.shape[0]} does not match grid size {nrx}")

    dirs_grid = build_dirs(forward_cfg["array"])
    coords, beams, beam_gain, k0 = build_codebook(
        forward_cfg["array"], dirs_grid, forward_cfg["freq_hz"]
    )

    scene_path = resolve_path(sionna_cfg["scene_xml"], base_dir)
    if not Path(scene_path).exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")

    scene = load_scene(scene_path)
    scene.frequency = float(forward_cfg["freq_hz"])
    scene.synthetic_array = True

    wavelength = C0 / scene.frequency
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
        vertical_spacing=0.5 * wavelength,
        horizontal_spacing=0.5 * wavelength,
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
        vertical_spacing=0.5 * wavelength,
        horizontal_spacing=0.5 * wavelength,
    )

    materials = build_materials(sionna_cfg["materials"], float(sionna_cfg["global_scattering_coeff"]))
    apply_materials(scene, materials, sionna_cfg.get("object_material_map", []))

    scene.add(Transmitter(name="tx", position=forward_cfg["tx_pos_m"]))

    E_sionna = []
    for i, rx in enumerate(rx_locs):
        print(f"Computing Sionna at RX {i + 1}/{nrx}")
        E_sionna.append(
            compute_sionna_beam_energy(
                scene,
                rx,
                beam_gain,
                dirs_grid,
                sionna_cfg,
                forward_cfg["array"]["ang_sigma"],
            )
        )
    E_sionna = np.array(E_sionna)

    E_aligned = []
    for r in range(nrx):
        rf_cent = dominant_dir(rfdt[r], dirs_grid)
        si_cent = dominant_dir(normalize_db(E_sionna[r]), dirs_grid)

        R = rodrigues(si_cent, rf_cent)
        dirs_rot = (R @ dirs_grid.T).T
        steering_rot = np.exp(1j * k0 * (coords @ dirs_rot.T))
        beam_gain_rot = np.abs(beams.conj().T @ steering_rot) ** 2

        E_aligned.append(10 * np.log10(beam_gain_rot @ (10 ** (E_sionna[r] / 10.0)) + 1e-12))
    E_aligned = np.array(E_aligned)

    print("\n===== PER-RX AGREEMENT METRICS =====")
    arr_cos, arr_sp, arr_pr, arr_mae, arr_topk, arr_angle, arr_region = [], [], [], [], [], [], []

    thresh = float(args.thresh)
    topk = int(args.topk)

    for rx_idx in range(nrx):
        rfdt_rx = normalize_db(rfdt[rx_idx])
        si_rx = normalize_db(E_aligned[rx_idx])

        rfdt_lin = 10 ** (rfdt_rx / 10)
        si_lin = 10 ** (si_rx / 10)

        cos = np.dot(rfdt_lin, si_lin) / (np.linalg.norm(rfdt_lin) * np.linalg.norm(si_lin) + 1e-12)
        sp = spearmanr(rfdt_rx, si_rx)[0]
        pr = pearsonr(rfdt_rx, si_rx)[0]
        mae = np.mean(np.abs(rfdt_rx - si_rx))

        topk_rfdt = set(np.argsort(rfdt_lin)[-topk:])
        topk_sionna = set(np.argsort(si_lin)[-topk:])
        topk_overlap = len(topk_rfdt & topk_sionna) / topk

        cent_rfdt = dominant_dir(rfdt_rx, dirs_grid)
        cent_sionna = dominant_dir(si_rx, dirs_grid)
        angle = np.degrees(np.arccos(np.clip(np.dot(cent_rfdt, cent_sionna), -1, 1)))

        rfdt_thresh = set(np.where(rfdt_lin >= thresh * np.max(rfdt_lin))[0])
        sionna_thresh = set(np.where(si_lin >= thresh * np.max(si_lin))[0])
        region = len(rfdt_thresh & sionna_thresh) / max(1, len(rfdt_thresh | sionna_thresh))

        arr_cos.append(cos)
        arr_sp.append(sp)
        arr_pr.append(pr)
        arr_mae.append(mae)
        arr_topk.append(topk_overlap)
        arr_angle.append(angle)
        arr_region.append(region)

        print(
            "RX {idx:02d} | Cos={cos:.4f} | Spearman={sp:.4f} | Pearson={pr:.4f} | "
            "MAE={mae:.4f} dB | Top-{topk} overlap={overlap:.3f} | "
            "Angle={angle:.2f} deg | Region@{thr:.2f}={region:.3f}".format(
                idx=rx_idx,
                cos=cos,
                sp=sp,
                pr=pr,
                mae=mae,
                topk=topk,
                overlap=topk_overlap,
                angle=angle,
                thr=thresh,
                region=region,
            )
        )

    print("\n===== OVERALL METRICS =====")
    print(f"Cosine: {np.mean(arr_cos):.4f}")
    print(f"Spearman: {np.mean(arr_sp):.4f}")
    print(f"Pearson: {np.mean(arr_pr):.4f}")
    print(f"MAE: {np.mean(arr_mae):.4f}")
    print(f"Top-{topk}: {np.mean(arr_topk):.4f}")
    print(f"Angle: {np.mean(arr_angle):.4f}")
    print(f"Region: {np.mean(arr_region):.4f}")

    out_path = Path(resolve_path(args.output, base_dir))
    np.save(out_path, E_aligned)
    print(f"Saved aligned Sionna beam grid: {out_path}")


if __name__ == "__main__":
    main()
