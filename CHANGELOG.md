# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

