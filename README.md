# Diffract and Conquer — Hyperspectral Imaging from a Commodity RGB Camera

This repository is a **code skeleton** for the paper *Diffract and Conquer: Hyperspectral Imaging from Any RGB Camera via Optical Encoding and Learning* (IJCAI). The system combines **diffractive optical encoding** (a 4×4 array of harmonic diffractive lenses, Bayer CFA, up to 48 raw channels) with **neural reconstruction** of the hyperspectral cube (31 bands, 400–700 nm, 10 nm step).

## Repository layout

```
configs/           # experiment and optical-model hyperparameters
docs/              # architecture and data-flow notes
scripts/           # utilities (forward-model simulation, etc.)
src/diffract_conquer_hsi/
  optical/         # lens selection, PSF, forward model (Eqs. 3–6)
  data/            # NTIRE / ICVL / CAVE / CZ-HSDB, normalization
  models/          # GGPIR, cmKAN++, Gaussian primitives, spectral blocks
  processing/      # SAM / PSNR metrics, config helpers
  training/        # train / eval stubs until full pipeline is wired
tests/
```

## Components (from the paper)

### Processing and algorithms (non-neural)

- **Optical model:** scene integration with per-HDL PSF and Bayer sensitivity \(T_c(\lambda)\) (Sec. 3–4).
- **Noise:** Poisson plus Gaussian components (Eq. 6).
- **Microrelief selection:** candidate discretization and **greedy maximum coverage** over the 31-band grid (Sec. 3.1, Eq. 8).
- **Metrics:** SAM, PSNR, SSIM (Sec. 4.1); per-band normalization using training-set statistics.

### Neural networks

- **cmKAN++ baseline:** spectral projector and rearranger around a simplified backbone (Sec. 3.2; see cmKANlight).
- **GGPIR:** spectral projector with \(x + x \cdot \mathrm{conv2d}(x)^2\), FFN bottleneck, **generated Gaussian primitives** (Eq. 10), expansion back to 31 channels.
- **Full paper variant:** independent hypernetworks per Gaussian output channel, Illumination Estimator / Color Transformer blocks — see `docs/ARCHITECTURE.md`.

## Quick start

```bash
cd /path/to/repo
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

`training/train.py` and `scripts/simulate_forward.py` are still **stubs**; connect them to datasets and a full Fresnel / PSF implementation.

## Citation

```bibtex
@inproceedings{pronin2026diffract,
  title     = {Diffract and Conquer: Hyperspectral Imaging from Any {RGB} Camera via Optical Encoding and Learning},
  author    = {Pronin, Alexey and Vladimirov, Daniil and Korepanov, Andrei and others},
  booktitle = {Proceedings of IJCAI},
  year      = {2026},
  note      = {Replace with official proceedings BibTeX when available}
}
```

`IJCAI.pdf` in the repo root is listed in `.gitignore` so it is not pushed by default. For a public release, link to the camera-ready or arXiv version instead of committing large PDFs.

## License

MIT — see `LICENSE`. Confirm with authors and funders if needed (e.g. Ministry of Economic Development of the Russian Federation, as acknowledged in the paper).
