# Low-Rank Nonlinear Beamspace Uncertainty Propagation for RF Digital Twins

This repository is a reproducible pipeline for beamspace uncertainty propagation in RF Digital Twins. It renders beamspace power from RF-3DGS point clouds, aligns with Sionna ray tracing, and propagates uncertainty using low-rank perturbations, Unscented Transform (UT), and Monte Carlo (MC) validation. The outputs include reliability metrics and risk-aware beam selection for downstream evaluation.

**Important prerequisite**: the `point_cloud.ply` file used by this pipeline is produced by the RF-3DGS project. Please read the RF-3DGS repository and paper before running these scripts.

- RF-3DGS code: https://github.com/SunLab-UGA/RF-3DGS
- RF-3DGS paper (arXiv): https://arxiv.org/abs/2411.19420
- IEEE TWC page (accepted version): https://ieeexplore.ieee.org/document/11355734

## What This Repo Provides

- Beamspace forward rendering from RF-3DGS PLY files.
- UT-based uncertainty propagation with MC diagnostics.
- Sionna alignment with per-RX agreement metrics.
- Reliability and risk-aware beam selection outputs.
- Figures and plots for reporting.

## Repository Structure

```
.
├── scripts/                       # Main pipeline entry points
├── scripts/preprocessing/         # RF-3DGS visual/RF dataset generation
├── scenes/                        # Sionna scene XMLs (expects scenes/meshes_d)
├── figures/                       # Output figures used in README/reports
├── requirements.txt               # Python dependencies
├── uq_config.example.json         # Example config (copy to uq_config.json)
├── installation_usage.md          # Full setup and usage guide
└── README.md
```

## Quick Start

1. Create a config file and set paths:

   ```bash
   cp uq_config.example.json uq_config.json
   # Edit uq_config.json to set forward.ply_path and sionna.scene_xml
   ```

2. Run the pipeline (from repo root):

   ```bash
   python scripts/run_rfdt_rendering.py --config uq_config.json --output-dir outputs
   python scripts/run_alignment_evaluation.py --config uq_config.json --rfdt outputs/rfdt_beam_energy_grid.npy --output outputs/sionna_beam_energy_grid_aligned.npy
   python scripts/run_ut_propagation.py --config uq_config.json --gt outputs/sionna_beam_energy_grid_aligned.npy --output-dir outputs
   python scripts/run_variance_validation.py --config uq_config.json --out-dir outputs/ut_mc_geometry_results
   python scripts/plot_rfdt_vs_sionna_comparison.py --config uq_config.json --rfdt outputs/rfdt_beam_energy_grid.npy --sionna outputs/sionna_beam_energy_grid_aligned.npy --save-dir outputs/plots
   ```

3. Optional smoke test:

   ```bash
   python scripts/run_smoke_tests.py --config uq_config.json --output-dir outputs --check-outputs
   ```

See `installation_usage.md` for the full environment setup and troubleshooting notes.

## Configuration Notes

- All hardcoded parameters have been moved to `uq_config.json` (or CLI overrides).
- Relative paths are resolved against the config file location.
- The Sionna alignment uses the *same* scene, frequency, array geometry, and angular grid as the RFDT forward pass, so consistency is enforced through the shared config.
- Preprocessing scripts read `preprocess.visual`, `preprocess.rf_dataset`, and `preprocess.rf_multitx`.

## RF-3DGS Preprocessing (Optional)

These scripts reproduce the visual and RF datasets used in RF-3DGS training. They are optional for the RFDT pipeline but useful for regenerating the inputs.

Prerequisites:
- Blender for visual dataset generation.
- RF-3DGS meshes copied to `scenes/meshes_d` (or update paths in `uq_config.json`).

Commands (from repo root):

```bash
blender -b -P scripts/preprocessing/generate_visual_dataset.py -- --config uq_config.json
python scripts/preprocessing/sionna_onetx.py --config uq_config.json --ideal --spectrum-type mpc --output-dir dataset_ideal_mpc
python scripts/preprocessing/prepare_rf_data_multx.py --config uq_config.json --rf-root dataset_custom_scene_ideal_mpc_with_object_txmulti
```

`scenes/room_with_cube.xml` is based on the RF-3DGS scene file. It expects mesh assets in `scenes/meshes_d`.

## Outputs (Key Files)

- `rfdt_beam_energy_grid.npy`: RFDT beam energy in dB for all RX positions.
- `rfdt_beam_gradient_grid.npy`: RFDT energy gradients (per-RX, per-beam).
- `sionna_beam_energy_grid_aligned.npy`: Sionna-aligned beam energy grid.
- `rfdt_uq_mu_lin.npy`, `rfdt_uq_sigma_best_lin.npy`: UT mean and uncertainty (linear).
- `rfdt_selected_beams_topk.npy`: risk-aware beam selection.
- `uq_results.npz`: full UT and MC tensors for downstream analysis.

## Figures

### Pipeline Overview

![Pipeline Overview](figures/pipeline.png)

### RF-3DGS Output (Reference PLY Rendering)

![RF-3DGS Output](figures/rf3dgs_output.png)

### RFDT vs Sionna Beam Energies

![RFDT vs Sionna RX 000](figures/rx_000.png)

### UQ Dashboard (UT)

![UT UQ Dashboard](figures/uq_summary_dashboard_ut.png)

### Variance Geometry Metric

![Variance Cosine Similarity](figures/variance_cosine_similarity.png)

Additional figures are available in the `figures/` directory.

## Tested Environment

The following environment was used for the latest runs in this workspace:

- Conda env: `rf-3dgs`
- Python: 3.10.19
- Sionna: 0.19.2 (RF-3DGS upstream notes 0.19.1 compatibility)
- TensorFlow: 2.15.1
- PyTorch: 2.10.0.dev20251212+cu130
- NumPy: 1.26.4

If you need maximum compatibility with RF-3DGS training, follow their README (Python 3.8 and CUDA 11.7/11.8).

## Citation

Please cite RF-3DGS if you use the PLY generation pipeline or any RF-3DGS outputs:

```bibtex
@article{rf3dgs2024,
  title={RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting},
  author={Lihao Zhang and Haijian Sun and Samuel Berweger and Camillo Gentile and Rose Qingyang Hu},
  journal={arXiv preprint arXiv:2411.19420},
  year={2024}
}
```

For the accepted IEEE TWC version, see: https://ieeexplore.ieee.org/document/11355734

## License

MIT License (see `LICENSE`).
