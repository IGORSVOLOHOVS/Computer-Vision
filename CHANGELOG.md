# Changelog

## v1.1.0 (2026-11-08)

### Commits since v1.0.0

- af367ca docs(docs): name revision 4.0.0 and its exceptions in the README
- 1ba8d2a ci(ci): add the five lab workflows and the issue and pull request templates
- ff501bf chore(core): add the AI environment, its hooks and the lab-ai-env skill
- d4bfc9d docs(docs): add the README, licence, rules and manual
- c26fe68 build(build): add recipe.ps1 and the test suites it runs
- 302867b refactor(core): point the quality metrics script at the lab's workflows
- 9191016 build(build): pin the toolchain, and decide line endings and ignores
- e9aaa14 chore: bring the pre-push check up to date
- 26ac91f chore: stop the secret scan running on a weekly schedule
- b667241 chore: add the pre-push check that mirrors CI
- 2b68868 chore: add the pre-push check that mirrors CI

### Issues

- #1 Pin the toolchain, and decide line endings and ignores in computer-vision (open)
- #2 Bring the sources of computer-vision under the pinned tools (open)
- #3 Add recipe.ps1 and its test suites to computer-vision (open)
- #4 Add the README, licence, rules and manual to computer-vision (open)
- #5 Add the AI environment to computer-vision (open)
- #6 Add the gates and templates to computer-vision (open)
- #7 State the revision 4.0.0 compliance line in computer-vision (open)

### Pull requests

- #14 build(build): pin the toolchain, and decide line endings and ignores (closed)
- #15 refactor(core): point the quality metrics script at the lab's workflows (closed)
- #16 build(build): add recipe.ps1 and the test suites it runs (closed)
- #17 docs(docs): add the README, licence, rules and manual (closed)
- #18 chore(core): add the AI environment, its hooks and the lab-ai-env skill (closed)
- #19 ci(ci): add the five lab workflows and the issue and pull request templates (closed)
- #20 docs(docs): name revision 4.0.0 and its exceptions in the README (closed)

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

