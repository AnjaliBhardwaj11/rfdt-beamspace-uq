#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C0 = 299792458.0

# Module-level configuration is populated via configure_from_config().
PLY_PATH = ""
FREQ = 0.0
TX_POS = None
RX_GRID = None
NX = 0
NY = 0
AZ = 0
EL = 0
ANG_SIGMA = 0.0
SH_ORDER = 0
SH_COEFFS = 0
GAUSSIAN_CHUNK = 0
lam = 0.0
k0 = 0.0
dirs = None
NUM_DIRS = 0
coords = None
beam_gain = None

GLOBAL_QUAT = None
CONFIG_READY = False


def build_rx_grid(rx_cfg):
    x_range = np.linspace(rx_cfg["x_min"], rx_cfg["x_max"], int(rx_cfg["x_count"]))
    y_range = np.linspace(rx_cfg["y_min"], rx_cfg["y_max"], int(rx_cfg["y_count"]))
    z_height = float(rx_cfg["z_height"])
    return torch.tensor(
        [[x, y, z_height] for x in x_range for y in y_range],
        dtype=torch.float32,
        device=DEVICE,
    )


def build_dirs(array_cfg):
    az = np.linspace(-np.pi, np.pi, int(array_cfg["az"]), endpoint=False)
    el = np.linspace(0, np.pi / 2, int(array_cfg["el"]))
    return torch.tensor(
        [
            [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
            for th in el
            for ph in az
        ],
        dtype=torch.float32,
        device=DEVICE,
    )


def build_codebook(array_cfg, dirs_local, k0_local, lam_local):
    nx = int(array_cfg["nx"])
    ny = int(array_cfg["ny"])
    d = lam_local / 2.0
    coords_local = torch.tensor(
        [
            [(i - (nx - 1) / 2) * d, (j - (ny - 1) / 2) * d, 0.0]
            for i in range(nx)
            for j in range(ny)
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    steering = torch.exp(1j * k0_local * (coords_local @ dirs_local.T))
    beams = steering / torch.norm(steering, dim=0, keepdim=True)
    beam_gain_local = torch.abs(beams.conj().T @ steering) ** 2
    return coords_local, beam_gain_local


def configure_from_config(forward_cfg, base_dir):
    global PLY_PATH, FREQ, TX_POS, RX_GRID
    global NX, NY, AZ, EL, ANG_SIGMA
    global SH_ORDER, SH_COEFFS, GAUSSIAN_CHUNK
    global lam, k0, dirs, NUM_DIRS, coords, beam_gain
    global CONFIG_READY

    PLY_PATH = resolve_path(forward_cfg.get("ply_path", ""), base_dir)
    FREQ = float(forward_cfg["freq_hz"])
    TX_POS = torch.tensor(forward_cfg["tx_pos_m"], device=DEVICE, dtype=torch.float32)

    RX_GRID = build_rx_grid(forward_cfg["rx_grid"])

    NX = int(forward_cfg["array"]["nx"])
    NY = int(forward_cfg["array"]["ny"])
    AZ = int(forward_cfg["array"]["az"])
    EL = int(forward_cfg["array"]["el"])
    ANG_SIGMA = float(forward_cfg["array"]["ang_sigma"])

    SH_ORDER = int(forward_cfg.get("sh_order", 3))
    SH_COEFFS = (SH_ORDER + 1) ** 2
    GAUSSIAN_CHUNK = int(forward_cfg.get("gaussian_chunk", 32768))

    lam = C0 / FREQ
    k0 = 2 * np.pi / lam
    dirs = build_dirs(forward_cfg["array"])
    NUM_DIRS = dirs.shape[0]
    coords, beam_gain = build_codebook(forward_cfg["array"], dirs, k0, lam)

    CONFIG_READY = True


def ensure_configured():
    if not CONFIG_READY:
        raise RuntimeError("Configuration not loaded. Call configure_from_config() first.")


# ============================================================
# LOAD GAUSSIANS
# ============================================================

def load_scene(path):
    ensure_configured()
    global GLOBAL_QUAT
    ply = PlyData.read(path)["vertex"]

    means = torch.tensor(
        np.stack([ply["x"], ply["y"], ply["z"]], axis=1),
        device=DEVICE).float()

    # RF-3DGS stores opacity as logits in the PLY; convert to physical alpha.
    alpha_logits = torch.tensor(ply["opacity"], device=DEVICE).float()
    alpha = torch.sigmoid(alpha_logits)

    # RF-3DGS stores SH as f_dc_* (3 coeffs) and f_rest_* (remaining coeffs).
    # Reconstruct channel-major [N, 3, SH_COEFFS] exactly as in training code.
    names = ply.data.dtype.names
    dc_names = sorted([k for k in names if k.startswith("f_dc_")], key=lambda n: int(n.split("_")[-1]))
    rest_names = sorted([k for k in names if k.startswith("f_rest_")], key=lambda n: int(n.split("_")[-1]))

    if len(dc_names) != 3:
        raise ValueError(f"Expected 3 DC SH fields, found {len(dc_names)}")

    expected_rest = 3 * (SH_COEFFS - 1)
    if len(rest_names) < expected_rest:
        raise ValueError(f"Expected at least {expected_rest} f_rest fields, found {len(rest_names)}")

    dc = np.stack([ply[k] for k in dc_names], axis=1).astype(np.float32)[:, :, None]
    rest = np.stack([ply[k] for k in rest_names[:expected_rest]], axis=1).astype(np.float32)
    rest = rest.reshape(-1, 3, SH_COEFFS - 1)
    sh_rgb_np = np.concatenate([dc, rest], axis=2)

    sh_rgb = torch.tensor(sh_rgb_np, device=DEVICE).float()
    sh = torch.linalg.norm(sh_rgb, dim=1) / np.sqrt(3)

    scales = torch.tensor(
        np.stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]], axis=1),
        device=DEVICE).float()

    quat = torch.tensor(
        np.stack([ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]], axis=1),
        device=DEVICE).float()

    GLOBAL_QUAT = quat

    return means, alpha, sh, scales, quat


# ============================================================
# SH RADIANCE
# ============================================================

def sh_radiance(d, sh):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]

    val = (
        0.282095 * sh[:, 0]
        - 0.488603 * y * sh[:, 1]
        + 0.488603 * z * sh[:, 2]
        - 0.488603 * x * sh[:, 3]
        + 1.092548 * x * y * sh[:, 4]
        - 1.092548 * y * z * sh[:, 5]
        + 0.315392 * (3 * z * z - 1) * sh[:, 6]
        - 1.092548 * x * z * sh[:, 7]
        + 0.546274 * (x * x - y * y) * sh[:, 8]
        - 0.590044 * y * (3 * x * x - y * y) * sh[:, 9]
        + 2.890611 * x * y * z * sh[:, 10]
        - 0.457046 * y * (5 * z * z - 1) * sh[:, 11]
        + 0.373176 * z * (5 * z * z - 3) * sh[:, 12]
        - 0.457046 * x * (5 * z * z - 1) * sh[:, 13]
        + 1.445306 * z * (x * x - y * y) * sh[:, 14]
        - 0.590044 * x * (x * x - 3 * y * y) * sh[:, 15]
    )
    return torch.relu(val)


def build_inverse_covariance(scales, quat):
    q = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)

    phys = torch.exp(scales)
    Sigma = R @ torch.diag_embed(phys ** 2) @ R.transpose(1, 2)
    eye = torch.eye(3, device=DEVICE, dtype=Sigma.dtype)
    return torch.linalg.inv(Sigma + 1e-6 * eye)


# ============================================================
# GAUSSIAN KERNEL + GRADIENT
# ============================================================

def gaussian_kernel_and_grad(means, scales, quat, rx):
    inv = build_inverse_covariance(scales, quat)

    d = rx.unsqueeze(1) - means.unsqueeze(0)

    inv_d = torch.einsum("nij,rnj->rni", inv, d)
    quad = torch.sum(d * inv_d, dim=2)
    K = torch.exp(-0.5 * quad)

    grad = -K.unsqueeze(2) * inv_d

    return K, grad


def compute_kernel(rx, means, scales, quat=None):
    """Recompute the attenuated kernel used by the forward model."""
    ensure_configured()
    if quat is None:
        global GLOBAL_QUAT
        quat = GLOBAL_QUAT

    inv = build_inverse_covariance(scales, quat)

    num_rx = rx.shape[0]
    num_pts = means.shape[0]

    # Keep only the final attenuated kernel to avoid large intermediate tensors.
    KA = torch.empty((num_rx, num_pts), device=DEVICE, dtype=means.dtype)

    dist_tx = torch.norm(means - TX_POS, dim=1).clamp_min(1e-6)
    A_tx = 1.0 / dist_tx

    for start in range(0, num_pts, GAUSSIAN_CHUNK):
        end = min(start + GAUSSIAN_CHUNK, num_pts)

        means_c = means[start:end]
        inv_c = inv[start:end]
        A_tx_c = A_tx[start:end]

        d_vec = rx.unsqueeze(1) - means_c.unsqueeze(0)
        inv_d = torch.einsum("nij,rnj->rni", inv_c, d_vec)
        quad = torch.sum(d_vec * inv_d, dim=2)
        K = torch.exp(-0.5 * quad)

        dist_rx = torch.norm(d_vec, dim=2).clamp_min(1e-6)
        A_rx = 1.0 / dist_rx
        KA[:, start:end] = K * A_rx * A_tx_c.unsqueeze(0)

    return KA


# ============================================================
# FORWARD + GRADIENT
# ============================================================

def compute_beam_energy_and_grad(means, alpha, sh, scales, quat):
    ensure_configured()
    inv = build_inverse_covariance(scales, quat)
    dist_tx = torch.norm(means - TX_POS, dim=1).clamp_min(1e-6)
    A_tx = 1.0 / dist_tx

    E_all = []
    dE_all = []
    num_pts = means.shape[0]

    for r in range(RX_GRID.shape[0]):
        rx = RX_GRID[r]
        m = torch.zeros(NUM_DIRS, device=DEVICE, dtype=means.dtype)
        dm = torch.zeros(NUM_DIRS, 3, device=DEVICE, dtype=means.dtype)

        # Process gaussians in chunks to bound peak GPU memory.
        for start in range(0, num_pts, GAUSSIAN_CHUNK):
            end = min(start + GAUSSIAN_CHUNK, num_pts)

            means_c = means[start:end]
            alpha_c = alpha[start:end]
            sh_c = sh[start:end]
            inv_c = inv[start:end]
            A_tx_c = A_tx[start:end]

            d_vec = rx.unsqueeze(0) - means_c
            dist_rx = torch.norm(d_vec, dim=1).clamp_min(1e-6)
            A_rx = 1.0 / dist_rx
            gradA_rx = -d_vec / (dist_rx.unsqueeze(1) ** 3)

            # Use actual arrival direction from each Gaussian to current RX.
            dir_rx = d_vec / dist_rx.unsqueeze(1)
            g_rx = sh_radiance(dir_rx, sh_c)

            coeff_base = alpha_c * A_tx_c * g_rx

            inv_d = torch.einsum("nij,nj->ni", inv_c, d_vec)
            quad = torch.sum(d_vec * inv_d, dim=1)
            K = torch.exp(-0.5 * quad)
            gradK = -K.unsqueeze(1) * inv_d

            coeff = coeff_base * K * A_rx

            dot = dir_rx @ dirs.T
            ang_w = torch.exp(-((1.0 - dot) ** 2) / (2.0 * (ANG_SIGMA ** 2)))
            ang_w = ang_w / (torch.sum(ang_w, dim=1, keepdim=True) + 1e-12)

            m = m + torch.sum(coeff.unsqueeze(1) * ang_w, dim=0)

            base_grad = coeff_base.unsqueeze(1) * (
                gradK * A_rx.unsqueeze(1) + K.unsqueeze(1) * gradA_rx
            )
            dm = dm + (ang_w.T @ base_grad)

        # m is already a non-negative linear-domain power accumulation.
        # Keep it linear to preserve RX-dependent contrast.
        P = m + 1e-12

        E = beam_gain @ P
        E_db = 10 * torch.log10(E + 1e-12)
        w = (10.0 / np.log(10.0)) * (beam_gain / (E.unsqueeze(0) + 1e-12))
        dE = torch.sum(w.unsqueeze(2) * dm.unsqueeze(0), dim=1)
        E_all.append(E_db)
        dE_all.append(dE)

    return torch.stack(E_all), torch.stack(dE_all)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RFDT beam energy and gradients over a multi-RX grid."
    )
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument("--ply-path", default="", help="Override PLY path")
    parser.add_argument("--output-dir", default=".", help="Directory for output .npy files")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Force device selection instead of auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global DEVICE
    if args.device != "auto":
        DEVICE = torch.device(args.device)

    cfg, base_dir = load_config(args.config)
    if args.ply_path:
        cfg["forward"]["ply_path"] = args.ply_path

    configure_from_config(cfg["forward"], base_dir)

    if not PLY_PATH:
        raise ValueError("PLY path is empty. Set forward.ply_path in config or pass --ply-path.")

    ply_path = Path(PLY_PATH)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    means, alpha, sh, scales, quat = load_scene(str(ply_path))
    E, dE = compute_beam_energy_and_grad(means, alpha, sh, scales, quat)

    out_dir = Path(resolve_path(args.output_dir, base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    energy_path = out_dir / "rfdt_beam_energy_grid.npy"
    grad_path = out_dir / "rfdt_beam_gradient_grid.npy"
    np.save(energy_path, E.detach().cpu().numpy())
    np.save(grad_path, dE.detach().cpu().numpy())

    E_np = E.detach().cpu().numpy()
    uniq_rows = np.unique(np.round(E_np, 4), axis=0).shape[0]
    max_row_diff = float(np.max(np.abs(E_np[0] - E_np[-1])))

    print("RFDT multi-Rx forward + gradient saved.")
    print(f"Energy: {energy_path}")
    print(f"Gradient: {grad_path}")
    print(f"RX row uniqueness (rounded 1e-4): {uniq_rows}/{E_np.shape[0]}")
    print(f"Max abs diff between first and last RX rows: {max_row_diff:.6f} dB")


if __name__ == "__main__":
    main()
