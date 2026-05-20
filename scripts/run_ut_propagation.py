#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import run_rfdt_rendering as rfdt_fwd
from uq_config import load_config, resolve_path

DEVICE = rfdt_fwd.DEVICE


@dataclass
class UQConfig:
    # Physical uncertainty priors
    rx_std_xyz_m: tuple = (0.02, 0.02, 0.01)
    mean_mode_std_m: float = 0.010
    sh_mode_std_frac: float = 0.08
    alpha_mode_std_frac: float = 0.06
    scale_mode_std_frac: float = 0.05

    # Directional / Model-form priors
    dir_std_rad: float = 0.08
    smooth_std: float = 0.1
    drop_prob: float = 0.1
    drop_width_frac: float = 0.1
    enable_directional_uq: bool = True
    enable_smoothing: bool = True

    # UT setup
    ut_alpha: float = 0.3
    ut_beta: float = 2.0
    ut_kappa: float = 0.0
    n_modes: int = 4

    # MC reference for UQ calibration diagnostics
    mc_samples: int = 128
    mc_samples_list: tuple = (128, 256, 512)
    seed: int = 1234

    # Error-detection metric threshold (top-k high-error beams)
    high_error_quantile: float = 0.8
    report_top_n: int = 20

    def apply_dict(self, data):
        """Update config fields from a dict, preserving defaults for missing keys."""
        for f in fields(self):
            if f.name in data:
                setattr(self, f.name, data[f.name])

        # Normalize list-like fields to tuples to keep type stability.
        if isinstance(self.rx_std_xyz_m, list):
            self.rx_std_xyz_m = tuple(self.rx_std_xyz_m)
        if isinstance(self.mc_samples_list, list):
            self.mc_samples_list = tuple(self.mc_samples_list)
        return self


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def _save_npy_replace(path, arr):
    _remove_if_exists(path)
    np.save(path, arr)


def _parse_int_list(text):
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _make_modes(base, seed, n_modes):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    means_mode = torch.randn((n_modes,) + tuple(base["means"].shape), generator=g, device=DEVICE)
    means_mode = means_mode / (torch.norm(means_mode, dim=2, keepdim=True) + 1e-9)

    sh_mode = torch.randn((n_modes,) + tuple(base["sh"].shape), generator=g, device=DEVICE)
    sh_mode = sh_mode / (torch.std(sh_mode) + 1e-9)

    alpha_mode = torch.randn((n_modes,) + tuple(base["alpha"].shape), generator=g, device=DEVICE)
    alpha_mode = alpha_mode / (torch.std(alpha_mode) + 1e-9)

    scale_mode = torch.randn((n_modes,) + tuple(base["scales"].shape), generator=g, device=DEVICE)
    scale_mode = scale_mode / (torch.std(scale_mode) + 1e-9)

    return {
        "means_mode": means_mode,
        "sh_mode": sh_mode,
        "alpha_mode": alpha_mode,
        "scale_mode": scale_mode,
    }


def _build_theta_and_cov(cfg: UQConfig):
    sx, sy, sz = cfg.rx_std_xyz_m
    nm = cfg.n_modes
    theta_dim = 3 + 4 * nm + 2
    theta0 = torch.zeros(theta_dim, dtype=torch.float32, device=DEVICE)

    s_mode = 1.0 / np.sqrt(max(1, nm))
    means_stds = [cfg.mean_mode_std_m * s_mode] * nm
    sh_stds = [cfg.sh_mode_std_frac * s_mode] * nm
    alpha_stds = [cfg.alpha_mode_std_frac * s_mode] * nm
    scale_stds = [cfg.scale_mode_std_frac * s_mode] * nm

    stds = torch.tensor(
        [
            sx,
            sy,
            sz,
            *means_stds,
            *sh_stds,
            *alpha_stds,
            *scale_stds,
            cfg.dir_std_rad,
            cfg.smooth_std,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    Sigma = torch.diag(stds**2)
    return theta0, Sigma


def _unpack_theta(theta, n_modes):
    idx = 0
    rx = theta[idx : idx + 3]
    idx += 3
    means_c = theta[idx : idx + n_modes]
    idx += n_modes
    sh_c = theta[idx : idx + n_modes]
    idx += n_modes
    alpha_c = theta[idx : idx + n_modes]
    idx += n_modes
    scale_c = theta[idx : idx + n_modes]
    idx += n_modes
    dir_c = theta[idx]
    idx += 1
    smooth_c = theta[idx]
    return rx, means_c, sh_c, alpha_c, scale_c, dir_c, smooth_c


def build_sigma_points(theta0, Sigma, alpha, beta, kappa):
    n = int(theta0.numel())
    lam = alpha**2 * (n + kappa) - n
    c = n + lam

    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    evals, evecs = torch.linalg.eigh(Sigma_sym)
    evals = torch.clamp(evals, min=0.0)
    sqrt_Sigma = evecs @ torch.diag(torch.sqrt(evals))

    pts = [theta0]
    s = np.sqrt(c)
    for i in range(n):
        d = s * sqrt_Sigma[:, i]
        pts.append(theta0 + d)
        pts.append(theta0 - d)
    sigma_points = torch.stack(pts, dim=0)

    Wm = torch.full((2 * n + 1,), 1.0 / (2.0 * c), dtype=theta0.dtype, device=theta0.device)
    Wc = Wm.clone()
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return sigma_points, Wm, Wc


def normalize(v):
    return v / (torch.norm(v, dim=1, keepdim=True) + 1e-9)


def angular_smoothing(P, dirs, sigma):
    dot = dirs @ dirs.T
    W = torch.exp(-(1 - dot) ** 2 / (2 * sigma**2))
    W = W / (torch.sum(W, dim=1, keepdim=True) + 1e-9)
    return W @ P


def _forward_from_theta(base, modes, theta, cfg: UQConfig):
    rx_c, means_c, sh_c, alpha_c, scale_c, dir_c, smooth_c = _unpack_theta(
        theta, modes["means_mode"].shape[0]
    )

    base_rx_grid = rfdt_fwd.RX_GRID.clone()
    rfdt_fwd.RX_GRID = base_rx_grid + rx_c

    means = base["means"] + torch.einsum("m,mnj->nj", means_c, modes["means_mode"])
    sh = torch.clamp(
        base["sh"] * (1.0 + torch.einsum("m,mnj->nj", sh_c, modes["sh_mode"])), min=0.0
    )
    alpha = torch.clamp(
        base["alpha"] * (1.0 + torch.einsum("m,mn->n", alpha_c, modes["alpha_mode"])), 0.0, 1.0
    )
    scales = base["scales"] + torch.einsum("m,mnj->nj", scale_c, modes["scale_mode"])

    codebook_dirs = rfdt_fwd.dirs.clone()
    if cfg.enable_directional_uq:
        # Deterministic directional perturbation for UT-consistent mapping.
        global_shift = dir_c * torch.ones_like(codebook_dirs[:1])
        rx_dirs = normalize(rfdt_fwd.RX_GRID - torch.mean(rfdt_fwd.RX_GRID, dim=0))
        align_full = rx_dirs @ codebook_dirs.T
        align = torch.mean(align_full, dim=0, keepdim=True).T
        align = align + 0.1 * torch.std(align_full, dim=0, keepdim=True).T
        local_noise = dir_c * align * codebook_dirs
        incoming_dirs = codebook_dirs + global_shift + local_noise
        incoming_dirs = normalize(incoming_dirs)
    else:
        incoming_dirs = codebook_dirs

    coords = rfdt_fwd.coords
    k0 = rfdt_fwd.k0

    steering_codebook = torch.exp(1j * k0 * (coords @ codebook_dirs.T))
    beams = steering_codebook / torch.norm(steering_codebook, dim=0, keepdim=True)
    steering_incoming = torch.exp(1j * k0 * (coords @ incoming_dirs.T))

    beam_gain = torch.abs(beams.conj().T @ steering_incoming) ** 2

    K = rfdt_fwd.compute_kernel(rfdt_fwd.RX_GRID, means, scales)
    drop_strength = torch.clamp(
        torch.tensor(cfg.drop_prob * cfg.drop_width_frac, device=DEVICE, dtype=means.dtype),
        0.0,
        0.3,
    )

    E_all = []

    for r in range(rfdt_fwd.RX_GRID.shape[0]):
        m = torch.zeros(len(incoming_dirs), device=DEVICE)

        for k in range(len(incoming_dirs)):
            d = incoming_dirs[k].unsqueeze(0).repeat(means.shape[0], 1)
            g = rfdt_fwd.sh_radiance(d, sh)
            m[k] = torch.sum(alpha * g * K[r])

        P = 10 ** (m / 10.0)

        # Deterministic expected blockage effect for UT consistency.
        P = P * (1.0 - drop_strength)

        if cfg.enable_smoothing:
            smooth_sigma = torch.clamp(torch.abs(smooth_c), 0.02, 0.15)
            P = angular_smoothing(P, incoming_dirs, smooth_sigma)

        E = beam_gain @ P
        E_all.append(E)

    rfdt_fwd.RX_GRID = base_rx_grid
    return torch.stack(E_all, dim=0)


def ut_propagation(base, modes, theta0, Sigma, cfg: UQConfig):
    sigma_points, Wm, Wc = build_sigma_points(
        theta0, Sigma, alpha=cfg.ut_alpha, beta=cfg.ut_beta, kappa=cfg.ut_kappa
    )

    y0 = _forward_from_theta(base, modes, sigma_points[0], cfg)

    ys = [y0]
    for i in range(1, sigma_points.shape[0]):
        ys.append(_forward_from_theta(base, modes, sigma_points[i], cfg))

    # Sigma-point beamspace tensors in linear power.
    # Shape: [N_SP, N_RX, N_BEAMS]
    Y = torch.stack(ys, dim=0)
    mu = torch.sum(Wm[:, None, None] * Y, dim=0)
    dY = Y - mu[None, :, :]

    var = torch.sum(Wc[:, None, None] * (dY**2), dim=0) + 1e-12
    return mu, var, Y, Wm, Wc


def mc_propagation(base, modes, cfg: UQConfig):
    rng = np.random.default_rng(cfg.seed)
    sx, sy, sz = cfg.rx_std_xyz_m
    nm = cfg.n_modes
    theta_dim = 3 + 4 * nm + 2

    z = np.zeros((cfg.mc_samples, theta_dim), dtype=np.float32)
    z[:, 0] = rng.normal(0.0, sx, size=cfg.mc_samples)
    z[:, 1] = rng.normal(0.0, sy, size=cfg.mc_samples)
    z[:, 2] = rng.normal(0.0, sz, size=cfg.mc_samples)

    s_mode = 1.0 / np.sqrt(max(1, nm))
    idx = 3
    z[:, idx : idx + nm] = rng.normal(0.0, cfg.mean_mode_std_m * s_mode, size=(cfg.mc_samples, nm))
    idx += nm
    z[:, idx : idx + nm] = rng.normal(0.0, cfg.sh_mode_std_frac * s_mode, size=(cfg.mc_samples, nm))
    idx += nm
    z[:, idx : idx + nm] = rng.normal(0.0, cfg.alpha_mode_std_frac * s_mode, size=(cfg.mc_samples, nm))
    idx += nm
    z[:, idx : idx + nm] = rng.normal(0.0, cfg.scale_mode_std_frac * s_mode, size=(cfg.mc_samples, nm))
    idx += nm
    z[:, idx] = rng.normal(0.0, cfg.dir_std_rad, size=cfg.mc_samples)
    idx += 1
    z[:, idx] = np.abs(rng.normal(0.0, cfg.smooth_std, size=cfg.mc_samples))

    y0 = _forward_from_theta(base, modes, torch.tensor(z[0], dtype=torch.float32, device=DEVICE), cfg)

    ys = [y0.detach().cpu().numpy()]
    for i in range(1, cfg.mc_samples):
        ti = torch.tensor(z[i], dtype=torch.float32, device=DEVICE)
        yi = _forward_from_theta(base, modes, ti, cfg)
        ys.append(yi.detach().cpu().numpy())
    return np.stack(ys, axis=0)


def _gaussian_coverage(y_true, mu, sigma):
    z = (y_true - mu) / np.maximum(sigma, 1e-12)
    return {
        "p50": float(np.mean(np.abs(z) <= 0.67448975)),
        "p80": float(np.mean(np.abs(z) <= 1.28155157)),
        "p90": float(np.mean(np.abs(z) <= 1.64485363)),
    }


def _coverage_ace(coverage_dict):
    nominal = {"p50": 0.5, "p80": 0.8, "p90": 0.9}
    errs = [abs(coverage_dict[k] - nominal[k]) for k in ["p50", "p80", "p90"]]
    return float(np.mean(errs))


def _roc_auc_binary(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = np.sum(ranks[pos])
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def _risk_coverage_curve(sigma, abs_err_db):
    order = np.argsort(sigma)
    n = len(order)
    coverages = np.linspace(0.1, 1.0, 10)
    risks = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        keep = order[:k]
        risks.append(float(np.mean(abs_err_db[keep])))
    aurc = float(np.trapz(risks, coverages) / (coverages[-1] - coverages[0]))
    return coverages, np.array(risks), aurc


def _recommend_rejection_percentile(sigma_db, err_db):
    n = len(sigma_db)
    order = np.argsort(sigma_db)
    full = float(np.mean(err_db))

    candidates = [0.05, 0.10, 0.15, 0.20, 0.25]
    best = {"rej": 0.0, "risk": full, "gain_pct": 0.0}
    for rej in candidates:
        keep = max(1, int(round((1.0 - rej) * n)))
        sel = order[:keep]
        risk = float(np.mean(err_db[sel]))
        gain_pct = float((full - risk) / max(full, 1e-12) * 100.0)
        if gain_pct > best["gain_pct"]:
            best = {"rej": rej, "risk": risk, "gain_pct": gain_pct}
    return best


def validate_ut_vs_mc(mu_ut, var_ut, mc_samples):
    ut_mu = mu_ut.detach().cpu().numpy().reshape(-1)
    ut_sigma = np.sqrt(np.maximum(var_ut.detach().cpu().numpy().reshape(-1), 1e-15))

    mc_samples_flat = mc_samples.reshape(mc_samples.shape[0], -1)
    mc_mu = np.mean(mc_samples_flat, axis=0)
    mc_sigma = np.sqrt(np.maximum(np.var(mc_samples_flat, axis=0), 1e-15))

    ut_mu_db = 10.0 * np.log10(np.maximum(ut_mu, 1e-12))
    mc_mu_db = 10.0 * np.log10(np.maximum(mc_mu, 1e-12))

    ratio = ut_sigma / np.maximum(mc_sigma, 1e-15)
    z_abs = np.abs((mc_samples_flat - ut_mu[None, :]) / np.maximum(ut_sigma[None, :], 1e-12))

    return {
        "mean_mae_db": float(np.mean(np.abs(ut_mu_db - mc_mu_db))),
        "mean_spearman": float(spearmanr(ut_mu_db, mc_mu_db)[0]),
        "mean_pearson": float(pearsonr(ut_mu_db, mc_mu_db)[0]),
        "sigma_spearman": float(spearmanr(ut_sigma, mc_sigma)[0]),
        "sigma_pearson": float(pearsonr(ut_sigma, mc_sigma)[0]),
        "sigma_ratio_mean": float(np.mean(ratio)),
        "sigma_ratio_median": float(np.median(ratio)),
        "sigma_ratio_p10": float(np.quantile(ratio, 0.10)),
        "sigma_ratio_p90": float(np.quantile(ratio, 0.90)),
        "ut_cov_on_mc": {
            "p50": float(np.mean(z_abs <= 0.67448975)),
            "p80": float(np.mean(z_abs <= 1.28155157)),
            "p90": float(np.mean(z_abs <= 1.64485363)),
        },
        "mc_mu": mc_mu,
        "mc_sigma": mc_sigma,
    }


def validate_uncertainty_against_gt(mu_lin, sigma_lin, gt_db, high_error_quantile=0.8):
    y = 10.0 ** (gt_db / 10.0)
    mu = np.maximum(mu_lin, 1e-12)
    sigma = np.maximum(sigma_lin, 1e-15)

    err_lin = y - mu
    abs_err_lin = np.abs(err_lin)
    mu_db = 10.0 * np.log10(mu)
    err_db_signed = gt_db - mu_db
    bias_db = float(np.median(err_db_signed))
    err_db_raw = np.abs(err_db_signed)
    err_db = np.abs(err_db_signed - bias_db)

    nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (err_lin**2) / (sigma**2)
    cov = _gaussian_coverage(y, mu, sigma)

    gamma = float(np.sqrt(np.mean((err_lin**2) / (sigma**2))))
    sigma_db = 4.342944819 * sigma / np.maximum(mu, 1e-12)
    sp_unc = float(spearmanr(sigma_db, err_db)[0])
    pr_unc = float(pearsonr(sigma_db, err_db)[0])

    thr = float(np.quantile(err_db, high_error_quantile))
    high_err = (err_db >= thr).astype(int)
    auc = _roc_auc_binary(sigma_db, high_err)

    n = len(sigma)
    k = max(1, int(round((1.0 - high_error_quantile) * n)))
    top_uncertain = set(np.argsort(sigma_db)[-k:])
    top_error = set(np.argsort(err_db)[-k:])
    precision_at_k = float(len(top_uncertain.intersection(top_error)) / k)
    overlap_k = int(len(top_uncertain.intersection(top_error)))
    expected_random_overlap = float(k * k / n)
    lift_k = float(overlap_k / max(expected_random_overlap, 1e-12))

    idx_desc = np.argsort(sigma_db)[::-1]
    top_idx = idx_desc[:k]
    bottom_idx = idx_desc[-k:]
    top_err_db = float(np.mean(err_db[top_idx]))
    bottom_err_db = float(np.mean(err_db[bottom_idx]))
    contrast_gap_db = top_err_db - bottom_err_db

    cov_grid, risk_curve, aurc = _risk_coverage_curve(sigma_db, err_db)
    risk_at_50 = float(risk_curve[np.argmin(np.abs(cov_grid - 0.5))])
    risk_full = float(risk_curve[-1])

    ratio = np.abs(err_lin) / np.maximum(sigma, 1e-12)
    s80 = float(np.quantile(ratio, 0.80) / 1.28155157)
    sigma_s80 = np.maximum(s80 * sigma, 1e-12)
    cov_s80 = _gaussian_coverage(y, mu, sigma_s80)
    width80_db = 2.0 * 1.28155157 * sigma_db

    return {
        "spearman_unc_vs_abs_err": sp_unc,
        "pearson_unc_vs_abs_err": pr_unc,
        "auc_high_error_detection": auc,
        "precision_at_k": precision_at_k,
        "overlap_k": overlap_k,
        "lift_k": lift_k,
        "k_count": int(k),
        "top_err_db": top_err_db,
        "bottom_err_db": bottom_err_db,
        "contrast_gap_db": contrast_gap_db,
        "risk_coverage_aurc_db": aurc,
        "risk_at_50cov_db": risk_at_50,
        "risk_fullcov_db": risk_full,
        "avg_nll": float(np.mean(nll)),
        "coverage": cov,
        "coverage_ace": _coverage_ace(cov),
        "coverage_s80": cov_s80,
        "s80_scale": s80,
        "gamma": gamma,
        "sharpness80_db_mean": float(np.mean(width80_db)),
        "bias_db": bias_db,
        "err_db_for_uq": err_db,
        "err_db_raw": err_db_raw,
        "mu_db": mu_db,
        "sigma_db": sigma_db,
    }


def save_dashboard(mu_db, sigma_db, gt_db, out_path, high_error_quantile=0.8, err_db=None):
    if err_db is None:
        err_db = np.abs(gt_db - mu_db)
    else:
        err_db = np.asarray(err_db)
    n = len(err_db)
    k = max(1, int(round((1.0 - high_error_quantile) * n)))
    idx = np.argsort(sigma_db)[::-1]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    axs[0, 0].scatter(sigma_db, err_db, s=10, alpha=0.55)
    axs[0, 0].set_title("Global Uncertainty vs Error")
    axs[0, 0].set_xlabel("Predicted Uncertainty (sigma dB)")
    axs[0, 0].set_ylabel("Debiased Abs Error wrt GT (dB)")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(err_db[idx], label="Abs Error")
    axs[0, 1].plot(sigma_db[idx], label="Uncertainty")
    axs[0, 1].set_title("Ranked by Uncertainty")
    axs[0, 1].set_xlabel("Flattened Beam Rank (high->low uncertainty)")
    axs[0, 1].set_ylabel("dB")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    bins = np.linspace(0.0, 1.0, 6)
    qvals = np.quantile(sigma_db, bins)
    mean_err = []
    labels = []
    for i in range(len(qvals) - 1):
        lo, hi = qvals[i], qvals[i + 1]
        if i < len(qvals) - 2:
            m = (sigma_db >= lo) & (sigma_db < hi)
        else:
            m = (sigma_db >= lo) & (sigma_db <= hi)
        mean_err.append(float(np.mean(err_db[m])) if np.any(m) else np.nan)
        labels.append(f"Q{i + 1}")

    axs[1, 0].bar(labels, mean_err)
    axs[1, 0].set_title("Mean Error by Uncertainty Bin")
    axs[1, 0].set_xlabel("Uncertainty Quantile Bin")
    axs[1, 0].set_ylabel("Mean Debiased Abs Error (dB)")
    axs[1, 0].grid(True, axis="y", alpha=0.3)

    top = float(np.mean(err_db[idx[:k]]))
    bottom = float(np.mean(err_db[idx[-k:]]))
    axs[1, 1].bar(["Top uncertain", "Bottom uncertain"], [top, bottom])
    axs[1, 1].set_title(f"Top/Bottom {k} Uncertain Error")
    axs[1, 1].set_ylabel("Mean Debiased Abs Error (dB)")
    axs[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Global UQ Summary Dashboard: {os.path.basename(out_path)}", fontsize=14)
    fig.tight_layout()
    _remove_if_exists(out_path)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_beam_summary_table(mu_lin, sigma_lin, gt_db, num_beams, out_csv):
    mu_lin = np.maximum(mu_lin, 1e-12)
    sigma_lin = np.maximum(sigma_lin, 1e-15)
    mu_db = 10.0 * np.log10(mu_lin)
    sigma_db = 4.342944819 * sigma_lin / mu_lin
    gt_lin = 10.0 ** (gt_db / 10.0)
    abs_err_db = np.abs(gt_db - mu_db)
    abs_err_lin = np.abs(gt_lin - mu_lin)

    unc_rank_desc = np.argsort(np.argsort(-sigma_db)) + 1
    err_rank_desc = np.argsort(np.argsort(-abs_err_db)) + 1

    _remove_if_exists(out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "flat_index",
                "rx_index",
                "local_beam_index",
                "mu_db",
                "gt_db",
                "sigma_db",
                "mu_lin",
                "gt_lin",
                "sigma_lin",
                "abs_err_db",
                "abs_err_lin",
                "uncertainty_rank_desc",
                "error_rank_desc",
            ]
        )
        for i in range(len(mu_db)):
            rx_idx = int(i // num_beams)
            local_beam = int(i % num_beams)
            writer.writerow(
                [
                    i,
                    rx_idx,
                    local_beam,
                    float(mu_db[i]),
                    float(gt_db[i]),
                    float(sigma_db[i]),
                    float(mu_lin[i]),
                    float(gt_lin[i]),
                    float(sigma_lin[i]),
                    float(abs_err_db[i]),
                    float(abs_err_lin[i]),
                    int(unc_rank_desc[i]),
                    int(err_rank_desc[i]),
                ]
            )


def save_overlap_evidence_quantile(mu_lin, sigma_lin, gt_db, high_error_quantile, out_txt, out_json):
    mu_lin = np.maximum(mu_lin, 1e-12)
    sigma_lin = np.maximum(sigma_lin, 1e-15)
    mu_db = 10.0 * np.log10(mu_lin)
    abs_err_db = np.abs(gt_db - mu_db)
    sigma_db = 4.342944819 * sigma_lin / mu_lin

    n = len(mu_db)
    k = max(1, int(round((1.0 - high_error_quantile) * n)))
    idx_uncertain = np.argsort(sigma_db)[::-1][:k]
    idx_error = np.argsort(abs_err_db)[::-1][:k]

    set_unc = set(idx_uncertain.tolist())
    set_err = set(idx_error.tolist())
    overlap = len(set_unc.intersection(set_err))
    union = len(set_unc.union(set_err))
    jaccard = float(overlap / max(1, union))
    expected_random_overlap = float(k * k / n)
    lift = float(overlap / max(expected_random_overlap, 1e-12))

    metrics = {
        "num_beams_global": int(n),
        "high_error_quantile": float(high_error_quantile),
        "k_count": int(k),
        "overlap_count": int(overlap),
        "jaccard": jaccard,
        "expected_random_overlap": expected_random_overlap,
        "lift_over_random": lift,
    }

    _remove_if_exists(out_json)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    _remove_if_exists(out_txt)
    with open(out_txt, "w") as f:
        f.write("UQ Overlap Evidence (Metric-Consistent K)\n")
        f.write("=======================================\n")
        f.write(f"Total beams (flattened grid): {n}\n")
        f.write(f"K from quantile: {k} (high_error_quantile={high_error_quantile})\n")
        f.write(f"Overlap count: {overlap}\n")
        f.write(f"Jaccard: {jaccard:.4f}\n")
        f.write(f"Expected random overlap: {expected_random_overlap:.4f}\n")
        f.write(f"Lift over random: {lift:.4f}\n")

    return metrics


def score_model(m):
    """Global ranking score using only global metrics."""
    return 0.6 * m["auc_high_error_detection"] + 0.4 * max(0.0, m["spearman_unc_vs_abs_err"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UT-based UQ propagation and validation for multi-RX RFDT."
    )
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument("--ply-path", default="", help="Override PLY path")
    parser.add_argument(
        "--gt",
        default="sionna_beam_energy_grid_aligned.npy",
        help="Aligned Sionna .npy for validation",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for all outputs")
    parser.add_argument("--mc-samples", type=int, default=0, help="Override MC samples")
    parser.add_argument(
        "--mc-samples-list",
        default="",
        help="Comma-separated MC sample counts for sweep (e.g., 128,256,512)",
    )
    parser.add_argument("--n-modes", type=int, default=0, help="Override number of modes")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--disable-directional-uq",
        action="store_true",
        help="Disable directional uncertainty perturbations",
    )
    parser.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable angular smoothing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_dict, base_dir = load_config(args.config)
    if args.ply_path:
        cfg_dict["forward"]["ply_path"] = args.ply_path

    cfg = UQConfig().apply_dict(cfg_dict.get("uq", {}))
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.n_modes > 0:
        cfg.n_modes = int(args.n_modes)
    if args.mc_samples > 0:
        cfg.mc_samples = int(args.mc_samples)
    if args.mc_samples_list:
        parsed_list = _parse_int_list(args.mc_samples_list)
        if parsed_list:
            cfg.mc_samples_list = tuple(parsed_list)
    if args.disable_directional_uq:
        cfg.enable_directional_uq = False
    if args.disable_smoothing:
        cfg.enable_smoothing = False

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    rfdt_fwd.configure_from_config(cfg_dict["forward"], base_dir)

    if not rfdt_fwd.PLY_PATH:
        raise ValueError("PLY path is empty. Set forward.ply_path or pass --ply-path.")
    ply_path = Path(rfdt_fwd.PLY_PATH)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    out_dir = Path(resolve_path(args.output_dir, base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_file(name):
        return out_dir / name

    gt_file = resolve_path(args.gt, base_dir) if args.gt else ""

    print("Mode: MC is used only for validation; deployable model is UT.")
    means, alpha, sh, scales, quat = rfdt_fwd.load_scene(str(ply_path))
    base = {
        "means": means,
        "alpha": alpha,
        "sh": sh,
        "scales": scales,
        "quat": quat,
    }
    modes = _make_modes(base, cfg.seed, cfg.n_modes)
    theta0, Sigma = _build_theta_and_cov(cfg)

    print("Running UT propagation over Rx grid...")
    t0_ut = time.time()
    mu_ut_t, var_ut_t, ut_sp_beamspace_t, Wm, Wc = ut_propagation(base, modes, theta0, Sigma, cfg)
    t1_ut = time.time()
    ut_time = t1_ut - t0_ut

    mc_samples_primary = int(cfg.mc_samples)
    mc_samples_list = list(cfg.mc_samples_list) if cfg.mc_samples_list else [mc_samples_primary]
    mc_samples_list = [int(x) for x in mc_samples_list]
    if mc_samples_primary not in mc_samples_list:
        mc_samples_list.append(mc_samples_primary)

    mc_runs = []
    primary_ut_mc = None
    primary_mc_samples = None
    primary_mc_time = None

    for mc_n in mc_samples_list:
        cfg.mc_samples = int(mc_n)
        print(f"Running MC propagation over Rx grid (validation reference, N={cfg.mc_samples})...")
        t0_mc = time.time()
        mc_samples_i = mc_propagation(base, modes, cfg)
        t1_mc = time.time()
        mc_time_i = t1_mc - t0_mc
        ut_mc_i = validate_ut_vs_mc(mu_ut_t, var_ut_t, mc_samples_i)
        mc_runs.append((mc_n, mc_time_i, ut_mc_i, mc_samples_i))
        if mc_n == mc_samples_primary:
            primary_ut_mc = ut_mc_i
            primary_mc_samples = mc_samples_i
            primary_mc_time = mc_time_i

    cfg.mc_samples = mc_samples_primary
    if primary_ut_mc is None:
        mc_n, primary_mc_time, primary_ut_mc, primary_mc_samples = mc_runs[-1]

    ut_mc = primary_ut_mc
    mc_samples = primary_mc_samples
    mc_time = primary_mc_time

    # Flatten the outputs for uniform evaluation
    NRX, NUM_BEAMS = mu_ut_t.shape
    mu_ut_flat = mu_ut_t.detach().cpu().numpy().reshape(-1)
    sigma_ut_flat = np.sqrt(np.maximum(var_ut_t.detach().cpu().numpy().reshape(-1), 1e-15))

    sigma_ut_base = sigma_ut_flat

    print("\n===== UQ PROPAGATION STATS =====")
    print(f"Modes per factor: {cfg.n_modes}")
    print(f"UT Propagation Time: {ut_time:.2f} seconds")
    print(
        f"UT params: alpha={cfg.ut_alpha:.3f}, "
        f"beta={cfg.ut_beta:.3f}, kappa={cfg.ut_kappa:.3f}"
    )
    print(f"MC Propagation Time: {mc_time:.2f} seconds")
    print(f"Speedup multiplier: {(mc_time / max(ut_time, 1e-6)):.2f}x")
    print("\n===== UT vs MC AGREEMENT (SWEEP) =====")
    for mc_n, mc_time_i, ut_mc_i, _ in mc_runs:
        print(f"\nMC samples: {mc_n}")
        print(f"UT-MC mean MAE (dB): {ut_mc_i['mean_mae_db']:.4f}")
        print(
            f"UT-MC mean Spearman/Pearson (dB): "
            f"{ut_mc_i['mean_spearman']:.4f} / {ut_mc_i['mean_pearson']:.4f}"
        )
        print(
            f"UT-MC sigma Spearman/Pearson: "
            f"{ut_mc_i['sigma_spearman']:.4f} / {ut_mc_i['sigma_pearson']:.4f}"
        )
        print(
            f"UT/MC sigma ratio (mean, median): "
            f"{ut_mc_i['sigma_ratio_mean']:.4f}, {ut_mc_i['sigma_ratio_median']:.4f}"
        )

    print(f"\nUsing MC samples N={mc_samples_primary} for downstream validation outputs.")

    if gt_file and Path(gt_file).exists():
        gt_db_full = np.load(gt_file)
        gt_db_flat = gt_db_full.reshape(-1)

        # ---------------------------------------------------------
        # 1. GENERATE ALL CANDIDATES & STABILIZE UT
        # ---------------------------------------------------------
        # Winsorize the calibrated UT standard deviation to prevent extreme blowouts
        p_low, p_high = 5, 95
        lo = np.percentile(sigma_ut_base, p_low)
        hi = np.percentile(sigma_ut_base, p_high)
        sigma_ut = np.clip(sigma_ut_base, lo, hi)

        print(f"Sigma clipped to [{p_low}, {p_high}] percentiles")
        print(f"Sigma stats (UT min/max): {np.min(sigma_ut):.6e} / {np.max(sigma_ut):.6e}")

        mc_mu_flat = ut_mc["mc_mu"]

        # Explicit (mu, sigma) pairs
        candidates = {
            "mc": (mc_mu_flat, ut_mc["mc_sigma"]),
            "ut": (mu_ut_flat, sigma_ut),
        }

        # ============================================================
        # SAVE ALL SIGMA ARRAYS FOR DOWNSTREAM EVALUATION
        # ============================================================

        print("\nSaving sigma arrays for all models...")

        np.save(out_file("rfdt_uq_sigma_mc_lin.npy"), ut_mc["mc_sigma"].reshape(NRX, NUM_BEAMS))
        np.save(out_file("rfdt_uq_sigma_ut_lin.npy"), sigma_ut.reshape(NRX, NUM_BEAMS))

        print("Saved:")
        print(f" - {out_file('rfdt_uq_sigma_mc_lin.npy')}")
        print(f" - {out_file('rfdt_uq_sigma_ut_lin.npy')}")

        # ---------------------------------------------------------
        # 2. EVALUATE & SAVE ALL CANDIDATES EXHAUSTIVELY
        # ---------------------------------------------------------
        print("\nEvaluating UT (deployable) with MC retained for validation diagnostics...")
        metrics_all = {}

        for name, (mu_val, sig_val) in candidates.items():
            m = validate_uncertainty_against_gt(
                mu_lin=mu_val,
                sigma_lin=sig_val,
                gt_db=gt_db_flat,
                high_error_quantile=cfg.high_error_quantile,
            )
            metrics_all[name] = m

            # Save independent diagnostics for each method
            save_dashboard(
                m["mu_db"],
                m["sigma_db"],
                gt_db_flat,
                out_file(f"uq_summary_dashboard_{name}.png"),
                cfg.high_error_quantile,
                m["err_db_for_uq"],
            )

            save_beam_summary_table(
                mu_val,
                sig_val,
                gt_db_flat,
                NUM_BEAMS,
                out_file(f"uq_beam_summary_{name}.csv"),
            )

            save_overlap_evidence_quantile(
                mu_val,
                sig_val,
                gt_db_flat,
                cfg.high_error_quantile,
                out_file(f"uq_overlap_{name}.txt"),
                out_file(f"uq_overlap_{name}.json"),
            )

        # ---------------------------------------------------------
        # 3. UT IS THE DEPLOYABLE MODEL (MC IS VALIDATION-ONLY)
        # ---------------------------------------------------------
        mu_best_lin_flat, sigma_best_lin_flat = candidates["ut"]
        m_best = metrics_all["ut"]
        err_db_best = m_best["err_db_for_uq"]

        # Reshape to [NRX, NUM_BEAMS] to keep arrays consistent with RFDT outputs
        _save_npy_replace(out_file("rfdt_uq_mu_lin.npy"), mu_best_lin_flat.reshape(NRX, NUM_BEAMS))
        _save_npy_replace(out_file("rfdt_uq_sigma_best_lin.npy"), sigma_best_lin_flat.reshape(NRX, NUM_BEAMS))
        _save_npy_replace(
            out_file("rfdt_uq_score_best_db.npy"),
            (4.342944819 * sigma_best_lin_flat / np.maximum(mu_best_lin_flat, 1e-12)).reshape(
                NRX, NUM_BEAMS
            ),
        )

        print("\n===== UT (DEPLOYABLE MODEL) =====")
        rej_rec = _recommend_rejection_percentile(m_best["sigma_db"], err_db_best)
        print(f"Uncertainty-Error Spearman: {m_best['spearman_unc_vs_abs_err']:.4f}")
        print(f"High-error detection AUC: {m_best['auc_high_error_detection']:.4f}")
        print(f"Contrast gap (top-bottom, dB): {m_best['contrast_gap_db']:.4f}")
        print(f"Risk-coverage AURC (dB, lower better): {m_best['risk_coverage_aurc_db']:.4f}")
        print(
            "Recommended uncertainty rejection (% beams): {rej}% (risk {risk:.4f} dB, gain {gain:.2f}%)".format(
                rej=int(rej_rec["rej"] * 100),
                risk=rej_rec["risk"],
                gain=rej_rec["gain_pct"],
            )
        )

        # ---------------------------------------------------------
        # 4. RISK-AWARE BEAM SELECTION
        # ---------------------------------------------------------
        lambda_risk = 1.0
        top_k = 5

        mu_db = 10.0 * np.log10(np.maximum(mu_best_lin_flat, 1e-12))
        sigma_db = 4.342944819 * sigma_best_lin_flat / np.maximum(mu_best_lin_flat, 1e-12)
        score = mu_db - lambda_risk * sigma_db
        score_2d = score.reshape(NRX, NUM_BEAMS)
        selected_beams = np.argsort(score_2d, axis=1)[:, -top_k:]

        threshold = np.percentile(sigma_best_lin_flat, 75)
        mask = sigma_best_lin_flat < threshold
        filtered_score = score * mask
        filtered_score_2d = filtered_score.reshape(NRX, NUM_BEAMS)
        selected_beams_filtered = np.argsort(filtered_score_2d, axis=1)[:, -top_k:]

        _save_npy_replace(out_file("rfdt_selected_beams_topk.npy"), selected_beams)
        _save_npy_replace(out_file("rfdt_selected_beams_topk_filtered.npy"), selected_beams_filtered)
        _save_npy_replace(out_file("rfdt_beam_score_riskaware.npy"), score_2d)

        print("\n===== BEAM SELECTION =====")
        print(f"Risk-aware score: mu_db - lambda*sigma_db, lambda={lambda_risk:.2f}")
        print(f"Top-K selected beams per RX: K={top_k}")
        print(f"Uncertainty filter threshold (75th percentile): {threshold:.6e}")
        print("Saved beam-selection arrays:")
        print(f" - {out_file('rfdt_selected_beams_topk.npy')}")
        print(f" - {out_file('rfdt_selected_beams_topk_filtered.npy')}")
        print(f" - {out_file('rfdt_beam_score_riskaware.npy')}")

        print("\nFinal array files saved (using best candidate):")
        print(f" - {out_file('rfdt_uq_mu_lin.npy')}")
        print(f" - {out_file('rfdt_uq_sigma_best_lin.npy')}")
        print(f" - {out_file('rfdt_uq_score_best_db.npy')}")
        print("\nNote: Individual dashboards and summaries were saved for ALL candidates.")

        # ============================================================
        # SAVE COMPLETE STOCHASTIC RF TENSORS
        # FOR ROBUST RFDT OPTIMIZATION
        # ============================================================

        print("\n================================================")
        print("Saving complete stochastic RF tensors...")
        print("================================================")

        # ------------------------------------------------------------
        # Reshape tensors back to [N_RX, N_BEAMS]
        # ------------------------------------------------------------

        mu_ut_2d = mu_ut_flat.reshape(NRX, NUM_BEAMS)

        sigma_ut_2d = sigma_best_lin_flat.reshape(NRX, NUM_BEAMS)

        # ------------------------------------------------------------
        # MC samples already available from propagation
        # Shape:
        # [N_MC, N_RX, N_BEAMS]
        # ------------------------------------------------------------

        mc_samples_3d = mc_samples

        # ------------------------------------------------------------
        # Deterministic RFDT nominal prediction
        # ------------------------------------------------------------

        rfdt_nominal_2d = mu_ut_2d.copy()

        # ------------------------------------------------------------
        # Sionna GT
        # Convert GT dB -> linear
        # ------------------------------------------------------------

        sionna_gt_linear = 10.0 ** (gt_db_full / 10.0)

        # ------------------------------------------------------------
        # Save everything together
        # ------------------------------------------------------------

        np.savez_compressed(
            out_file("uq_results.npz"),
            # UT statistics
            mu_ut=mu_ut_2d.astype(np.float32),
            sigma_ut=sigma_ut_2d.astype(np.float32),
            # UT sigma-point beamspace + weights (quadrature)
            # Shapes:
            #   B_ut_sp: [N_SP, N_RX, N_BEAMS]
            #   Wm, Wc : [N_SP]
            B_ut_sp=ut_sp_beamspace_t.detach().cpu().numpy().astype(np.float32),
            Wm=Wm.detach().cpu().numpy().astype(np.float32),
            Wc=Wc.detach().cpu().numpy().astype(np.float32),
            # Monte-Carlo rollout
            mc_samples=mc_samples_3d.astype(np.float32),
            # Deterministic RFDT
            rfdt_nominal=rfdt_nominal_2d.astype(np.float32),
            # Sionna GT
            sionna_gt=sionna_gt_linear.astype(np.float32),
            # Metadata
            rx_positions=rfdt_fwd.RX_GRID.detach().cpu().numpy().astype(np.float32),
            beam_dirs=rfdt_fwd.dirs.detach().cpu().numpy().astype(np.float32),
        )

        print("\nSaved:")
        print(f" - {out_file('uq_results.npz')}")

        print("\nTensor Shapes:")

        print("mu_ut:", mu_ut_2d.shape)
        print("sigma_ut:", sigma_ut_2d.shape)
        print("mc_samples:", mc_samples_3d.shape)
        print("rfdt_nominal:", rfdt_nominal_2d.shape)
        print("sionna_gt:", sionna_gt_linear.shape)

        print("\n================================================")
        print("UQ tensor export completed successfully.")
        print("Ready for robust RFDT optimization.")
        print("================================================")

    else:
        print(f"\nSionna GT not found: {gt_file}")
        print("Run scripts/run_alignment_evaluation.py to enable GT-side validation metrics.")


if __name__ == "__main__":
    main()
