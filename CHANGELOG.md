# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-06
### Added
- `benchmarks/test_pipeline_performance.py`: the preprocessing around the model
  measured at three source sizes and two target sizes, so a regression shows up
  as a number.
- `docs/quality-iso25010.md`: all eight ISO/IEC 25010 characteristics assessed
  against measured evidence, including the known defects.
- A screenshot of a real classification run in the README.
- Downloadable release artefacts: a wheel and an sdist with checksums, built by
  `scripts/build_release_artifact.py` and attached to the GitHub Release.

### Fixed
- `scripts/collect_quality_metrics.py` read only `.github/workflows/ci.yml`, so
  this repository scored zero for portability while it was testing on a
  platform.
- The classifier tests depended on the machine's GPU: `TimmClassifier` rebinds
  `self.model` from `.half()` when CUDA is present, and a plain `MagicMock`
  returns a different mock from that call. Three tests passed on the CPU-only
  runner and failed on any machine with a graphics card.

### Changed
- CI watches `release`, `dev` and `test`; `master` and twelve fully-merged
  `feat/task*` branches are gone.

## [0.1.1] - 2026-04-17

### Added
- **Clean Architecture**: Refactored core logic into Domain, Application, and Infrastructure layers for better testability and maintainability.
- **Hardware Acceleration**: Added automated detection and support for CUDA (NVIDIA) and MPS (Apple Silicon) in `TimmClassifier`.
- **Mixed Precision**: Implementation of FP16 (half-precision) inference for significant performance gains on supported GPUs.
- **Dynamic Configuration**: Added automated input size detection from model metadata (e.g., `vit_base_patch16_384` automatically uses 384x384).
- **Optimization Suite**: Added profiling tools and result tracking (`profile_results_optimized.txt`).
- **Documentation Engine**: Scripted HTML documentation generation using `pdoc`.

### Changed
- Improved `main.py` with `rich` console integration for better UX and error reporting.
- Standardized project configuration using `pyproject.toml` and Ruff/Mypy.

## [0.1.0] - 2026-04-17

### Added
- Initial project structure for Computer Vision classification.
- Integration with `timm` for Vision Transformer models.
- Support for mushroom image classification (agaric, bolete, stinkhorn).
- Packaging metadata and build system configuration.

