#!/usr/bin/env python3
"""Plot RFDT vs Sionna beam energies with Top-K highlights."""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from uq_config import load_config, resolve_path


def normalize_db(E: np.ndarray) -> np.ndarray:
    E = E - np.max(E)
    E_lin = 10 ** (E / 10.0)
    E_lin /= np.sum(E_lin) + 1e-12
    return 10 * np.log10(E_lin + 1e-12)


def parse_indices(text: str) -> Optional[List[int]]:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def coerce_grid(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def load_grid(path: Path) -> np.ndarray:
    return coerce_grid(np.load(path))


def resolve_path_local(path_str: str, base_dir: Path) -> Path:
    return Path(resolve_path(path_str, base_dir))


def select_rx_indices(nrx: int, rx_indices: Optional[List[int]], rx_count: int) -> List[int]:
    if rx_indices is not None and len(rx_indices) > 0:
        return [i for i in rx_indices if 0 <= i < nrx]
    if rx_count <= 0:
        return []
    if rx_count >= nrx:
        return list(range(nrx))
    idx = np.linspace(0, nrx - 1, rx_count, dtype=int)
    return sorted(set(idx.tolist()))


def plot_rx(
    rx_idx: int,
    rfdt: np.ndarray,
    sionna: np.ndarray,
    topk: int,
    normalize: bool,
    out_dir: Optional[Path],
) -> None:
    rfdt_plot = normalize_db(rfdt) if normalize else rfdt
    sionna_plot = normalize_db(sionna) if normalize else sionna

    rfdt_lin = 10 ** (rfdt_plot / 10.0)
    sionna_lin = 10 ** (sionna_plot / 10.0)

    topk_rfdt = np.argsort(rfdt_lin)[-topk:]
    topk_sionna = np.argsort(sionna_lin)[-topk:]

    x = np.arange(rfdt_plot.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))

    rfdt_label = "RFDT Beam Energy (Norm)" if normalize else "RFDT Beam Energy"
    sionna_label = "Sionna Beam Energy (Norm)" if normalize else "Sionna Beam Energy"

    ax.plot(x, rfdt_plot, "o-", ms=4, lw=1.0, label=rfdt_label)
    ax.plot(x, sionna_plot, "x-", ms=4, lw=1.0, label=sionna_label)
    ax.scatter(
        topk_rfdt,
        rfdt_plot[topk_rfdt],
        s=90,
        facecolors="none",
        edgecolors="b",
        label=f"RFDT Top-{topk}",
    )
    ax.scatter(
        topk_sionna,
        sionna_plot[topk_sionna],
        s=110,
        marker="*",
        color="orange",
        label=f"Sionna Top-{topk}",
    )

    ax.set_title(f"Beam Energies at RX index {rx_idx} (Top-{topk} Highlighted)")
    ax.set_xlabel("Beam Index")
    ax.set_ylabel("Beam Energy (dB)")
    ax.grid(True, alpha=0.4)
    ax.legend()

    fig.tight_layout()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"rx_{rx_idx:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RFDT vs Sionna beam energies with Top-K highlights."
    )
    parser.add_argument("--config", default="", help="Path to uq_config.json")
    parser.add_argument(
        "--rfdt",
        default="rfdt_beam_energy_grid.npy",
        help="Path to RFDT beam energy grid (.npy)",
    )
    parser.add_argument(
        "--sionna",
        default="sionna_beam_energy_grid_aligned.npy",
        help="Path to Sionna beam energy grid (.npy)",
    )
    parser.add_argument(
        "--rx-indices",
        default="",
        help="Comma-separated RX indices to plot, e.g., 0,5,10",
    )
    parser.add_argument(
        "--rx-count",
        type=int,
        default=5,
        help="Number of RX indices to plot if --rx-indices is empty",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-k beams to highlight")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-RX normalization",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="If set, save plots to this directory instead of showing",
    )
    args = parser.parse_args()

    _, base_dir = load_config(args.config)
    rfdt = load_grid(resolve_path_local(args.rfdt, base_dir))
    sionna = load_grid(resolve_path_local(args.sionna, base_dir))

    if rfdt.shape != sionna.shape:
        raise ValueError(f"Shape mismatch: rfdt {rfdt.shape} vs sionna {sionna.shape}")

    nrx = rfdt.shape[0]
    rx_indices = parse_indices(args.rx_indices)
    selected = select_rx_indices(nrx, rx_indices, args.rx_count)

    out_dir = resolve_path_local(args.save_dir, base_dir) if args.save_dir else None

    for rx_idx in selected:
        plot_rx(
            rx_idx,
            rfdt[rx_idx],
            sionna[rx_idx],
            topk=args.topk,
            normalize=not args.no_normalize,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
