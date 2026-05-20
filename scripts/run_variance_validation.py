#!/usr/bin/env python3
"""UT vs MC beamspace variance geometry analysis."""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate UT vs MC beamspace variance geometry (marginal variances)."
    )
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument("--mu-path", default="rfdt_uq_mu_lin.npy", help="UT mean (linear) .npy")
    parser.add_argument("--sigma-path", default="rfdt_uq_sigma_best_lin.npy", help="UT sigma (linear) .npy")
    parser.add_argument("--npz-path", default="uq_results.npz", help="NPZ file containing MC samples")
    parser.add_argument("--out-dir", default="ut_mc_geometry_results", help="Output directory for plots")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k beams for energy concentration")
    return parser.parse_args()


def configure_plots() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.5,
            "grid.alpha": 0.25,
            "font.family": "serif",
        }
    )


def load_inputs(mu_path: Path, sigma_path: Path, npz_path: Path):
    print("=" * 60)
    print("Loading tensors...")
    print("=" * 60)

    mu_ut = np.load(mu_path)
    sigma_ut = np.load(sigma_path)
    uq_data = np.load(npz_path)
    mc_samples = uq_data["mc_samples"]

    print(f"mu_ut shape      : {mu_ut.shape}")
    print(f"sigma_ut shape   : {sigma_ut.shape}")
    print(f"mc_samples shape : {mc_samples.shape}")

    return mu_ut, sigma_ut, mc_samples


def compute_mc_variances(mc_samples: np.ndarray) -> np.ndarray:
    print("\n" + "=" * 60)
    print("Computing MC marginal variances...")
    print("=" * 60)

    n_rx = mc_samples.shape[1]
    mc_variances = []

    for r in range(n_rx):
        samples_r = mc_samples[:, r, :]
        var_r = np.var(samples_r, axis=0)
        mc_variances.append(var_r)

    mc_variances = np.array(mc_variances)
    print(f"MC variance shape : {mc_variances.shape}")
    return mc_variances


def compute_metrics(mc_variances: np.ndarray, ut_variances: np.ndarray, top_k: int):
    variance_cosine = []
    variance_pearson = []
    topk_energy_mc = []
    topk_energy_ut = []
    topk_energy_gap = []

    print("\n" + "=" * 60)
    print("Analyzing variance geometry...")
    print("=" * 60)

    n_rx = mc_variances.shape[0]
    for r in range(n_rx):
        mc_var = mc_variances[r]
        ut_var = ut_variances[r]

        cos_sim = cosine_similarity(mc_var.reshape(1, -1), ut_var.reshape(1, -1))[0, 0]
        variance_cosine.append(cos_sim)

        pr = pearsonr(mc_var, ut_var)[0]
        variance_pearson.append(pr)

        mc_sorted = np.sort(mc_var)[::-1]
        ut_sorted = np.sort(ut_var)[::-1]

        mc_energy = np.sum(mc_sorted[:top_k]) / (np.sum(mc_sorted) + 1e-12)
        ut_energy = np.sum(ut_sorted[:top_k]) / (np.sum(ut_sorted) + 1e-12)
        gap = np.abs(mc_energy - ut_energy)

        topk_energy_mc.append(mc_energy)
        topk_energy_ut.append(ut_energy)
        topk_energy_gap.append(gap)

    return (
        np.array(variance_cosine),
        np.array(variance_pearson),
        np.array(topk_energy_mc),
        np.array(topk_energy_ut),
        np.array(topk_energy_gap),
    )


def summarize_metrics(
    variance_cosine: np.ndarray,
    variance_pearson: np.ndarray,
    topk_energy_mc: np.ndarray,
    topk_energy_ut: np.ndarray,
    topk_energy_gap: np.ndarray,
):
    return {
        "variance_cosine_mean": float(np.mean(variance_cosine)),
        "variance_cosine_std": float(np.std(variance_cosine)),
        "variance_pearson_mean": float(np.mean(variance_pearson)),
        "variance_pearson_std": float(np.std(variance_pearson)),
        "mc_topk_energy_mean": float(np.mean(topk_energy_mc)),
        "ut_topk_energy_mean": float(np.mean(topk_energy_ut)),
        "topk_energy_gap_mean": float(np.mean(topk_energy_gap)),
        "topk_energy_gap_std": float(np.std(topk_energy_gap)),
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("FINAL GEOMETRY VALIDATION")
    print("=" * 60)

    for k, v in summary.items():
        print(f"{k:30s}: {v:.6f}")


def plot_variance_cosine(values: np.ndarray, result_dir: Path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o", linewidth=2)
    plt.xlabel("RX Index")
    plt.ylabel("Variance Cosine Similarity")
    plt.title("UT vs MC Variance Structure Similarity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(result_dir / "variance_cosine_similarity.png", dpi=300)
    plt.close(fig)


def plot_variance_pearson(values: np.ndarray, result_dir: Path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o", linewidth=2)
    plt.xlabel("RX Index")
    plt.ylabel("Variance Pearson Correlation")
    plt.title("UT vs MC Marginal Variance Correlation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(result_dir / "variance_pearson.png", dpi=300)
    plt.close(fig)


def plot_topk_energy(
    mc_energy: np.ndarray, ut_energy: np.ndarray, result_dir: Path
) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(mc_energy, label="MC", linewidth=2)
    plt.plot(ut_energy, label="UT", linewidth=2)
    plt.xlabel("RX Index")
    plt.ylabel("Top-K Variance Energy")
    plt.title("Variance Energy Concentration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(result_dir / "topk_variance_energy.png", dpi=300)
    plt.close(fig)


def plot_topk_gap(values: np.ndarray, result_dir: Path) -> None:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o", linewidth=2)
    plt.xlabel("RX Index")
    plt.ylabel("Energy Gap")
    plt.title("Top-K Variance Energy Gap")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(result_dir / "topk_energy_gap.png", dpi=300)
    plt.close(fig)


def save_summary(summary: dict, result_dir: Path) -> None:
    np.save(result_dir / "summary.npy", summary)
    with (result_dir / "summary.txt").open("w") as f:
        f.write("UT vs MC VARIANCE GEOMETRY ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        for k, v in summary.items():
            f.write(f"{k:30s}: {v:.6f}\n")


def _resolve_input_path(path_str: str, base_dir: Path, out_dir: Path) -> Path:
    resolved = Path(resolve_path(path_str, base_dir))
    if resolved.exists():
        return resolved

    fallback = out_dir / path_str
    if fallback.exists():
        return fallback

    outputs_fallback = base_dir / "outputs" / path_str
    if outputs_fallback.exists():
        return outputs_fallback

    return resolved


def main() -> None:
    args = parse_args()
    _, base_dir = load_config(args.config)

    result_dir = Path(resolve_path(args.out_dir, base_dir))
    result_dir.mkdir(parents=True, exist_ok=True)

    configure_plots()

    mu_path = _resolve_input_path(args.mu_path, base_dir, result_dir.parent)
    sigma_path = _resolve_input_path(args.sigma_path, base_dir, result_dir.parent)
    npz_path = _resolve_input_path(args.npz_path, base_dir, result_dir.parent)

    mu_ut, sigma_ut, mc_samples = load_inputs(mu_path, sigma_path, npz_path)

    if mu_ut.shape != sigma_ut.shape:
        raise ValueError("mu and sigma shapes do not match")

    n_mc, n_rx, n_beams = mc_samples.shape
    if mu_ut.shape != (n_rx, n_beams):
        raise ValueError("mu/sigma shapes do not match MC samples")

    print("\nSanity checks passed.")
    print(f"N_MC    : {n_mc}")
    print(f"N_RX    : {n_rx}")
    print(f"N_BEAMS : {n_beams}")

    mc_variances = compute_mc_variances(mc_samples)
    ut_variances = sigma_ut ** 2

    (
        variance_cosine,
        variance_pearson,
        topk_energy_mc,
        topk_energy_ut,
        topk_energy_gap,
    ) = compute_metrics(mc_variances, ut_variances, args.top_k)

    summary = summarize_metrics(
        variance_cosine,
        variance_pearson,
        topk_energy_mc,
        topk_energy_ut,
        topk_energy_gap,
    )

    print_summary(summary)

    plot_variance_cosine(variance_cosine, result_dir)
    plot_variance_pearson(variance_pearson, result_dir)
    plot_topk_energy(topk_energy_mc, topk_energy_ut, result_dir)
    plot_topk_gap(topk_energy_gap, result_dir)

    save_summary(summary, result_dir)

    print("\nSaved outputs to:")
    print(str(result_dir))
    print("\nDone.")


if __name__ == "__main__":
    main()
