# computer-vision

A console tool that classifies one image with a pretrained `timm` vision model. It is laid out as ports and adapters: `src/domain/` holds the models and the two ports, `src/application/` the service that wires them, and `src/infrastructure/` the adapters for `timm` and for PIL with torchvision. The default model is `vit_base_patch16_384`, and the input size is read from the model itself.

![A classification run](docs/images/classification-run.png)

*A real run: ViT-Base at 384 px identifies an ImageNet validation photograph as class 992, `agaric`, with 0.88 confidence. Labels appear as indices because this model's `default_cfg` carries no `label_names`.*

The weights are not in this repository. The first real run downloads them from Hugging Face. The tests never do: they run offline and stand a mock in for `timm.create_model`.

## Use

```powershell
.\recipe.ps1 run path\to\image.jpg
```

`docs/install.md` says what to install. `docs/manual.md` describes the command and what it prints. `.\recipe.ps1 test` runs the tests, mypy and ruff.

## Measured: everything around the model

Inference is the model's cost. What this repository controls is the preprocessing around it, which `benchmarks/test_pipeline_performance.py` measures, median of many rounds, numbers from before the lab:

| What is measured | Median |
| --- | ---: |
| Missing file, the failure path | 18.8 µs |
| Preprocess a 224 × 224 JPEG | 964 µs |
| Preprocess a 512 × 512 JPEG | 4.30 ms |
| Service: load and orchestrate, model stubbed (512 px source) | 4.26 ms |
| Preprocess to a 384 px target (512 px source) | 5.75 ms |
| Preprocess a 2048 × 2048 JPEG | 56.9 ms |

The service measures 4.26 ms against 4.30 ms for the load it contains, within run-to-run noise, so the layering costs nothing measurable. Cost is roughly linear in source area: a 2048 × 2048 photograph spends about 57 ms on the CPU before the model sees it. The benchmarks stub the model, so they load no weights:

```powershell
uv run --frozen python -m pytest benchmarks --benchmark-only --no-cov
```

## What is here

- `main.py` is the command line.
- `src/` is the package, imported as `src.domain`, `src.application` and `src.infrastructure`.
- `tests/` holds the tests, `benchmarks/` the benchmarks above.
- `docs/` holds the architecture notes in `docs/architecture.md`, the ISO/IEC 25010 assessment in `docs/quality-iso25010.md`, the generated API pages and `docs/branching.md`.
- `scripts/` holds the helpers kept from before the lab. `collect_quality_metrics.py` produces the numbers of the assessment and `build_release_artifact.py` a PyInstaller build with checksums. `install_dependencies.py` is the pip route from before `uv.lock`. `profile_application.py` came from a template and imports a package this repository does not have, so it does not run.

## Licence

MIT, see `LICENSE`.

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#8), R-09 (#9), R-12 (#10), R-37 (#11), R-41 (#12), R-42 (#13).
