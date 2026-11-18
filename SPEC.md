# Software Requirements Specification: Computer-Vision

This specification was reverse-engineered from commit `515d10a` (branch `dev`,
2026-08-06) on 2026-11-18. Every requirement describes behaviour that is in the
code and cites it. Statements marked **inference** are conclusions drawn from
the cited evidence, not stated in code. Claims from documents that the code
does not implement are listed apart, in 3.8. The later commits on
`origin/release` (tag `v1.1.0`) and `origin/test` are not covered.

## 1. Purpose and scope

**Purpose.** The program classifies a single image file with a pretrained
`timm` model and reports the top-1 class label and its softmax confidence. It
is available as a command-line program (`main.py`) and as a layered Python
package (`src/`).

**In scope:**

- image loading and preprocessing (`src/infrastructure/image_adapter.py`);
- classification through timm (`src/infrastructure/timm_adapter.py`);
- the orchestration service and the domain types and ports (`src/application/`, `src/domain/`);
- the command-line program `main.py`;
- the unit tests and benchmarks (`tests/`, `benchmarks/`);
- the repository tooling that ships with the project: `scripts/`,
  `.github/workflows/`, `.pre-commit-config.yaml`.

**Out of scope.** None of the following is in the code at `515d10a`:

- model training, fine-tuning, or a dataset loader;
- a mushroom-specific model or label set. Classes come from the pretrained
  model (`timm_adapter.py:33-35`);
- batch classification, object detection, segmentation;
- a GUI, a network service or an HTTP API;
- pinning or verifying downloaded weights (`docs/quality-iso25010.md:141-142`
  states this is not implemented);
- the contents of `.agent/`. This is a third-party agent kit, excluded from
  ruff (`pyproject.toml:25-26`). Two of its files are used by the project:
  `.agent/scripts/check_commit_msg.py` and `.agent/scripts/docs_python.py`.

## 2. Context

**Actors.**

| Actor | Interaction | Source |
| --- | --- | --- |
| CLI user | runs `python main.py <image>` | `main.py:59-62` |
| Library caller | builds `TimmClassifier`, `PILImageLoader`, `ClassifierService` in Python | `src/` |
| Maintainer | runs tests, benchmarks, helper scripts, pre-commit hooks | `CONTRIBUTING.md`, `scripts/` |
| GitHub Actions | runs CI, release and secret-scan workflows | `.github/workflows/` |

**External systems.**

| System | Use | Source |
| --- | --- | --- |
| timm and its pretrained weights | `timm.create_model(name, pretrained=True)`. Downloads weights when the model is built. The repository does not set the source or the cache path | `timm_adapter.py:25`, `benchmarks/test_pipeline_performance.py:9-10` |
| PyTorch devices | CUDA, then MPS, then CPU | `timm_adapter.py:17-23` |
| Local file system | input images; outputs of the helper scripts | `image_adapter.py:25-29`, `scripts/` |
| GitHub Releases | `softprops/action-gh-release@v2` publishes the files | `.github/workflows/release.yml:60-66` |
| gitleaks | `gitleaks/gitleaks-action@v2` scans the git history | `.github/workflows/secret-scan.yml:24-26` |
| uv installer | CI downloads it with `curl ... astral.sh/uv/install.sh` | `.github/workflows/python-ci.yml:20-21` |

**Runtime environment.**

- Language runtime: Python `>=3.10` (`pyproject.toml:6`). CI uses 3.10
  (`.github/workflows/python-ci.yml:18`).
- OS: the code has no restriction. CI runs on `ubuntu-latest`
  (`python-ci.yml:11`). The scripts handle Windows paths
  (`scripts/build_release_artifact.py:72`, `scripts/install_dependencies.py:42-45`).
- Hardware: CPU. Optionally an NVIDIA GPU (CUDA) or Apple Silicon (MPS)
  (`timm_adapter.py:18-21`).

## 3. Functional requirements

### 3.1 Image loading (`PILImageLoader`)

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-1 | If the image path does not exist, the system shall raise `FileNotFoundError` with the message `Image not found at <path>`, before any decode is attempted. | `src/infrastructure/image_adapter.py:25-27` | `test_image_loader_not_found`; benchmark `test_missing_file_fails_fast` |
| FR-2 | The system shall open the file with Pillow and convert it to RGB. | `image_adapter.py:29` | no test (benchmarks use RGB JPEGs only) |
| FR-3 | The system shall resize the image so that its shorter side equals `size`, centre-crop it to `size` × `size`, convert it to a tensor and normalise each channel with mean 0.5 and std 0.5. | `image_adapter.py:15-22` | shape only: `test_preprocessing_scales_with_image_size`, `test_a_larger_target_size_costs_more`; the normalisation values are not tested |
| FR-4 | The system shall return a tensor of shape `[1, 3, size, size]` (batch dimension added). | `image_adapter.py:30-32` | `test_preprocessing_scales_with_image_size` (224), `test_a_larger_target_size_costs_more` (384) |
| FR-5 | When no size is given, the system shall use a `size` of 224. | `image_adapter.py:14` | `test_preprocessing_scales_with_image_size` |

### 3.2 Classification adapter (`TimmClassifier`)

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-6 | The system shall create the model with `timm.create_model(model_name, pretrained=pretrained)`, with `pretrained` defaulting to `True`, and put the model in eval mode. | `src/infrastructure/timm_adapter.py:13-14`, `:25-27` | `test_timm_adapter_init` |
| FR-7 | The system shall choose the device in the order CUDA, then MPS, then CPU, and move the model to that device. | `timm_adapter.py:17-26` | no test |
| FR-8 | When `use_fp16` is true (the default) and the device is not CPU, the system shall convert the model and each input tensor to half precision. | `timm_adapter.py:29-31`, `:39-40` | no test |
| FR-9 | The system shall read class names from `default_cfg["label_names"]`, or use an empty list when they are missing. | `timm_adapter.py:33-35` | `test_timm_classifier_classify[True]`, `[False]` |
| FR-10 | The system shall run the forward pass under `torch.no_grad()`, apply softmax to the first batch element's logits, and take the maximum as the top class and its confidence. | `timm_adapter.py:37-46` | `test_timm_classifier_classify` (confidence ≈ 1.0) |
| FR-11 | The system shall label the top class with `label_names[index]` when that name exists, and otherwise with `class_<index>`. | `timm_adapter.py:49-54` | `test_timm_classifier_classify[True]` (`cat`), `[False]` (`class_0`) |
| FR-12 | The system shall return a `ClassificationResult` whose `all_predictions` is a one-element list holding `top_prediction`, together with the model name. | `timm_adapter.py:56-65` | `test_timm_classifier_classify` (only `top_prediction` is checked) |
| FR-13 | The system shall report the input size as the last element of `default_cfg["input_size"]`, and as 224 when that key is missing. | `timm_adapter.py:70-75` | `test_timm_classifier_get_input_size` (299) |
| FR-14 | The system shall return the configured model name from `get_model_name()`. | `timm_adapter.py:67-68` | no test |

### 3.3 Orchestration (`ClassifierService`)

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-15 | For a path, the system shall call `loader.load_and_transform(path)` once, pass the resulting tensor to `classifier.classify`, and return that result unchanged. | `src/application/service.py:14-25` | `test_service_orchestration`; benchmark `test_service_overhead` |

### 3.4 Domain types and ports

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-16 | The system shall represent results as immutable `Prediction(label, confidence)` and `ClassificationResult(top_prediction, all_predictions, model_name)` dataclasses. | `src/domain/models.py:4-18` | no test of immutability |
| FR-17 | The system shall define the ports `ImageClassifier` (`classify`, `get_model_name`, `get_input_size`) and `ImageLoader` (`load_and_transform`) as `runtime_checkable` Protocols. | `src/domain/interfaces.py:9-32` | no test |

### 3.5 Command-line program (`main.py`)

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-18 | The system shall take the image path from the first command-line argument, and use `image.png` when there is none. | `main.py:59-62` | no test |
| FR-19 | The system shall classify with the model `vit_base_patch16_384`. The command line cannot change it. | `main.py:17`, `:24`, `:62` | no test |
| FR-20 | The system shall build the image loader with the input size reported by the classifier. | `main.py:27-30` | no test |
| FR-21 | The system shall print a table titled `Classification Results for '<path>'` with the rows `Model`, `Top Prediction` and `Confidence` (4 decimal places). | `main.py:40-48` | no test |
| FR-22 | On `FileNotFoundError`, the system shall print `Error: <message>` and exit with code 1. On any other exception, it shall print `Unexpected Error: <message>` and the traceback and exit with code 1. | `main.py:50-56` | no test |

### 3.6 Repository tooling

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-23 | `build_release_artifact.py` shall build a wheel and an sdist with `python -m build`, copy them into a freshly created `release/`, and write a `<file>.sha256` next to each one. | `scripts/build_release_artifact.py:39-59`, `:104-122` | no test |
| FR-24 | Unless `--skip-verify` is given, `build_release_artifact.py` shall install the wheel into a temporary venv with `--no-deps`, run `import src`, and exit with an error if the import fails. | `scripts/build_release_artifact.py:62-91`, `:115-117` | no test |
| FR-25 | `check_before_push.py` shall collect the `ruff`, `mypy` and `pytest` commands from `.github/workflows/*.y*ml`, run them with the local interpreter, skip tools that are not installed, and exit with 1 if any command fails. | `scripts/check_before_push.py:44`, `:87-97`, `:180-214` | no test |
| FR-26 | `check_before_push.py` shall remove `--cov*` flags when `pytest_cov` is missing. It shall report a pytest run as skipped, not failed, when the output shows a collection-time `ModuleNotFoundError`, or both `Required test coverage` and `Total coverage: 0.00%`. | `scripts/check_before_push.py:100-127`, `:148-151` | no test |
| FR-27 | `check_before_push.py --install-hook` shall write `.git/hooks/pre-push`, which runs the script with `--quiet`. | `scripts/check_before_push.py:35-40`, `:160-177` | no test |
| FR-28 | `collect_quality_metrics.py` shall report the test pass rate, coverage, radon complexity / maintainability index with a lint penalty, and CI OS / Python version counts as 0-100 scores, plus their mean. It shall write JSON with `--output` and exit with 1 when the mean is below `--fail-under`. | `scripts/collect_quality_metrics.py:61-177`, `:180-233` | no test |
| FR-29 | `enforce_branch_policy.py` shall report every local branch other than `release`, `dev` and `test` and, with `--remote`, every such branch on that remote too, plus missing required branches (judged on the remote when `--remote` is given, otherwise locally). Any extra branch, local or remote, makes it exit with 1. With `--delete-extra` it shall delete only local branches whose tip is already contained in an allowed branch. | `scripts/enforce_branch_policy.py:19`, `:54-64`, `:67-123` | no test |
| FR-30 | `install_dependencies.py` shall install the project editable (with `[dev]` when `--dev` is given), regenerate `requirements-dev.lock` from a clean venv with `--lock`, and install from that lock with `--from-lock`. | `scripts/install_dependencies.py:30-72`, `:75-111` | no test |
| FR-31 | The `commit-msg` hook shall reject a subject that does not match `type(scope): text.`, where `type` is one of `feat`, `feature`, `fix`, `bugfix`, `docs`, `refactor`, `test`, `chore`, the text is ASCII, and the subject ends with a period. | `.agent/scripts/check_commit_msg.py:21-49`, `.pre-commit-config.yaml:20-26` | no test |
| FR-32 | `.agent/scripts/docs_python.py` shall generate HTML API docs into `docs/` with `python -m pdoc src -o docs`, using `clean_venv`'s Python if that exists. | `.agent/scripts/docs_python.py:11-25` | no test |

### 3.7 CI and release automation

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| FR-33 | On every push and pull request to `release`, `dev` or `test`, CI shall run `ruff format --check .`, `ruff check .`, `mypy --strict .` and `pytest --cov=src --cov-report=xml` after `uv sync --all-extras`. | `.github/workflows/python-ci.yml:3-36` | the workflow itself |
| FR-34 | On a tag `v*.*.*`, the release workflow shall build the artefacts with `scripts/build_release_artifact.py` and publish a GitHub Release with the `.whl`, `.tar.gz` and `.sha256` files and `CHANGELOG.md` as the body. A manual run shall build but not publish. | `.github/workflows/release.yml:5-66` | the workflow itself |
| FR-35 | The secret scan shall run gitleaks over the full history on every push, every pull request, every Monday at 06:00 UTC (cron `0 6 * * 1`) and on manual dispatch. | `.github/workflows/secret-scan.yml:6-26` | the workflow itself |

### 3.8 Declared but not implemented

| Claim | Declared in | State at `515d10a` |
| --- | --- | --- |
| The result includes the full ranking | `docs/quality-iso25010.md:27-28` | `all_predictions = [top_prediction]` (`timm_adapter.py:58-59`) |
| `main.py` supports `--help` | `docs/quality-iso25010.md:103` | no argument parser; `--help` is taken as an image path (`main.py:61`) |
| Mushroom classification (agaric, bolete, stinkhorn) | `CHANGELOG.md:51`, `pyproject.toml:4` | no mushroom-specific code; generic pretrained timm classes |
| `task3.py` for prototyping | `README.md:80` | not in the repository (`.gitignore:63` ignores `task*.py`) |
| Profiling results `profile_results_optimized.txt` | `README.md:24`, `CHANGELOG.md:39` | not in the repository (`.gitignore:56`) |
| Profiling script for this project | `scripts/profile_application.py` | imports `quality_template.core`, which does not exist; fails with `ModuleNotFoundError` |
| A console entry point in the wheel | `scripts/build_release_artifact.py:13-14` | no `[project.scripts]` in `pyproject.toml` |
| A branch-policy workflow on every push and every Monday | `docs/branching.md:30-32` | no `branch-policy.yml` in `.github/workflows/` |
| The release workflow runs the tests and builds an executable and a zip | `docs/branching.md:69-71` | `release.yml` builds a wheel and an sdist and runs no tests |
| Coverage gate of 85 % (`fail_under`) | `CONTRIBUTING.md:31` | not configured in `pyproject.toml` |
| Complexity limit (ruff `C90`) | `CONTRIBUTING.md:32` | ruff selects `E F B I UP` (`pyproject.toml:31`) |
| gitleaks and `detect-private-key` in pre-commit | `CONTRIBUTING.md:33`, `SECURITY.md:14`, `:16` | not in `.pre-commit-config.yaml` |
| Bandit rules (`S`) in ruff | `SECURITY.md:17`, `docs/quality-iso25010.md:134` | not selected (`pyproject.toml:31`) |
| Pinned dependencies in `requirements-dev.lock` | `SECURITY.md:18`, `scripts/install_dependencies.py:21` | file not in the repository |
| Domain layer with no external dependencies | `README.md:15` | `src/domain/interfaces.py:4` imports `torch` |
| A manual release run takes a required `version` input | `.github/workflows/release.yml:8-12` | no step reads the input; the build gets `github.ref_name` (`release.yml:34`) |
| The wheel installs a top-level `src` package | `docs/quality-iso25010.md:91-97` | measured (see 9.4): the wheel holds top-level `domain`, `application`, `infrastructure` and a stray `__init__.py`, no `src` |
| The release script checks that the installed wheel imports | `scripts/build_release_artifact.py:63-67` | `import src` runs in the caller's working directory (`:80-85`, no `cwd`); from the checkout root it finds the checkout's `src/`, so it passes although the wheel has no `src` (measured, see 9.4) |
| `pytest benchmarks --benchmark-only` runs the benchmarks | `README.md:53`, `CONTRIBUTING.md:20`, `benchmarks/test_pipeline_performance.py:12-13` | started through the `pytest` console script, collection fails with `No module named 'src'` (measured, see 9.5); `python -m pytest benchmarks ...` or a bare `pytest` works |

## 4. Interfaces

### 4.1 Command line: `main.py`

| Element | Value | Source |
| --- | --- | --- |
| Invocation | `python main.py [IMAGE_PATH]` | `main.py:59-62` |
| `IMAGE_PATH` | path of the image file; default `image.png` | `main.py:61` |
| Standard output | status lines, then a `rich` table (`Model`, `Top Prediction`, `Confidence`) | `main.py:23-48` |
| Exit code 0 | classification printed | `main.py:16-48` (normal return) |
| Exit code 1 | missing file, or any other exception | `main.py:50-56` |

### 4.2 Helper scripts

The flags were confirmed with `--help` on 2026-11-18, except where marked.

| Script | Flags | Effect | Source |
| --- | --- | --- | --- |
| `scripts/build_release_artifact.py` | `--version VERSION` (log only, default `v0.0.0-dev`), `--skip-verify` | wheel + sdist + `.sha256` into `release/` | `:94-127` |
| `scripts/check_before_push.py` | `--install-hook`, `--quiet` | runs the CI checks locally / installs `pre-push` | `:180-214` |
| `scripts/collect_quality_metrics.py` | `--output PATH`, `--fail-under FLOAT` | ISO/IEC 25010 scores | `:180-233` |
| `scripts/enforce_branch_policy.py` | `--remote NAME`, `--delete-extra` | branch report / clean-up | `:67-123` |
| `scripts/install_dependencies.py` | `--dev`, `--lock`, `--from-lock` | installs dependencies / lock file | `:75-111` |
| `scripts/profile_application.py` | `--paragraphs`, `--repeats`, `--rows`, `--sort`, `--save` (from source; `--help` fails on import) | none, broken | `:38-46`, `:22` |
| `.agent/scripts/docs_python.py` | none | pdoc HTML into `docs/` | `:7-34` |
| `.agent/scripts/check_commit_msg.py` | `<commit-msg-file>` (positional) | exit 1 on a bad subject | `:6-52` |

### 4.3 Python API

| Element | Signature | Returns / raises | Source |
| --- | --- | --- | --- |
| `ImageClassifier.classify` | `(input_tensor: torch.Tensor)` | `ClassificationResult` | `src/domain/interfaces.py:13-15` |
| `ImageClassifier.get_model_name` | `()` | `str` | `interfaces.py:17-19` |
| `ImageClassifier.get_input_size` | `()` | `int` | `interfaces.py:21-23` |
| `ImageLoader.load_and_transform` | `(image_path: Path \| str)` | `torch.Tensor` | `interfaces.py:30-32` |
| `ClassifierService` | `(classifier: ImageClassifier, loader: ImageLoader)`; `.classify_image(image_path: Path \| str)` | `ClassificationResult`; passes through the adapters' exceptions | `src/application/service.py:10-25` |
| `TimmClassifier` | `(model_name: str, pretrained: bool = True, use_fp16: bool = True)` | implements `ImageClassifier` | `src/infrastructure/timm_adapter.py:10-75` |
| `PILImageLoader` | `(size: int = 224)` | implements `ImageLoader`; raises `FileNotFoundError` | `src/infrastructure/image_adapter.py:11-32` |

### 4.4 File formats

| File | Format | Source |
| --- | --- | --- |
| Input image | any file that `PIL.Image.open` can read, converted to RGB. Only JPEG is used in the benchmarks | `image_adapter.py:29`, `benchmarks/test_pipeline_performance.py:55-58` |
| `release/*.whl`, `release/*.tar.gz` | outputs of `python -m build` | `scripts/build_release_artifact.py:54-59` |
| `release/<file>.sha256` | one line: `<sha256 hex>  <file name>` | `scripts/build_release_artifact.py:119-122` |
| Quality report (`--output`) | JSON with keys `standard`, `measured_characteristics`, `assessed_by_hand`, `overall_measured_score`, `note` | `scripts/collect_quality_metrics.py:200-224` |
| `requirements-dev.lock` | `pip freeze` output after a three-line header (generated, not committed) | `scripts/install_dependencies.py:53-72` |
| `.git/hooks/pre-push` | `/bin/sh` script that runs `check_before_push.py --quiet` | `scripts/check_before_push.py:35-40` |

### 4.5 Workflows and hooks

| Name | Trigger | Steps | Source |
| --- | --- | --- | --- |
| Python CI | push / PR to `release`, `dev`, `test` | uv sync, ruff format check, ruff check, mypy strict, pytest with coverage | `.github/workflows/python-ci.yml` |
| Release | tag `v*.*.*`; manual (`version` input) | build job, then publish job (tags only) | `.github/workflows/release.yml` |
| Secret scan | push, PR, weekly cron, manual | gitleaks over the full history | `.github/workflows/secret-scan.yml` |
| pre-commit | `git commit` (after `pre-commit install`) | ruff `--fix`, ruff-format, whitespace / EOF / YAML / large-file checks | `.pre-commit-config.yaml:1-18` |
| commit-msg | `git commit` (commit-msg stage) | `check_commit_msg.py` | `.pre-commit-config.yaml:20-26` |

## 5. Data

**Entities** (`src/domain/models.py`, frozen dataclasses):

| Entity | Fields | Notes |
| --- | --- | --- |
| `Prediction` | `label: str`, `confidence: float` | `confidence` is a softmax probability (`timm_adapter.py:45-56`) |
| `ClassificationResult` | `top_prediction: Prediction`, `all_predictions: list[Prediction]`, `model_name: str` | at `515d10a`, `all_predictions` has exactly one element |

**Tensors.** The loader output is a float tensor of shape `[1, 3, size, size]`
(`image_adapter.py:30-31`). Inference: `ToTensor` scales the pixels to
[0, 1], and `Normalize(0.5, 0.5)` then maps them to [-1, 1]. On a GPU device
with `use_fp16`, the tensor is cast to half precision before the forward pass
(`timm_adapter.py:39-40`).

**Storage.** The project code has no database and no persistent store. Files
it writes:

- `release/` and `dist/` (`scripts/build_release_artifact.py:27-28`);
- `build/` and `src/Computer_Vision.egg-info/`, written by `python -m build`
  (seen in a build in a temporary copy, see 9.4);
- a temporary `build/verify-venv`, deleted after the check (`:68-86`);
- a temporary `.coverage.json`, deleted after it is read
  (`scripts/collect_quality_metrics.py:87-96`);
- the report path given to `--output`;
- `.git/hooks/pre-push`;
- `requirements-dev.lock`;
- `docs/*.html` (pdoc).

timm keeps model weights outside the repository; this repository does not set
the location.

**Gitignored data paths.** `data/`, `runs/`, `val2017/`, `celeba/` and
`results/` are ignored (`.gitignore:2-6`). No code at `515d10a` reads them.
**Inference:** they are left over from earlier task work in the history. The
first commits (2026-02-23 to 2026-02-28) are about traffic analysis, a DIoU box
loss and an annotation visualisation, and the tags `archive/task1`,
`archive/task2`, `archive/task15`, `archive/task17` and
`archive/xray-vit-classifier` exist (`git log --reverse`, `git tag`).

## 6. Non-functional requirements

| ID | Requirement | Source |
| --- | --- | --- |
| NFR-1 | The package shall support Python 3.10 and later. Ruff targets `py310`, mypy checks as 3.10. | `pyproject.toml:6`, `:28`, `:35` |
| NFR-2 | The code shall pass `mypy --strict` (missing third-party stubs are ignored). | `pyproject.toml:34-42`, `.github/workflows/python-ci.yml:33` |
| NFR-3 | The code shall pass `ruff format --check` and `ruff check` with rules `E F B I UP`, line length 88. | `pyproject.toml:27-32`, `python-ci.yml:27-30` |
| NFR-4 | Inference shall use a GPU when one is available (CUDA, then MPS) and half precision on non-CPU devices. | `timm_adapter.py:17-31` |
| NFR-5 | Inference shall run without gradient tracking. | `timm_adapter.py:42` |
| NFR-6 | A missing input file shall be rejected by a `Path.exists()` check before decoding. A benchmark measures this path (`README.md:34` reports a median of 18.8 µs; that is the author's measurement and was not reproduced). | `image_adapter.py:26-27`, `benchmarks/test_pipeline_performance.py:95-107` |
| NFR-7 | The preprocessing time is measured by benchmarks at source sizes 224, 512 and 2048 px and target sizes 224 and 384. **No time limit is enforced**; the benchmarks only check output shapes. | `benchmarks/test_pipeline_performance.py:63-92` |
| NFR-8 | The project code sets no timeout or retry for the weight download or for subprocesses (grep for `timeout`/`retry` found nothing). | `main.py`, `src/`, `scripts/` |
| NFR-9 | CI shall test on `ubuntu-latest` with Python 3.10. No other OS, Python version or GPU is tested. | `.github/workflows/python-ci.yml:11`, `:15-18` |
| NFR-10 | The git history shall be scanned for credentials (`fetch-depth: 0`) at least weekly, with read-only repository permission. | `.github/workflows/secret-scan.yml:9-10`, `:13-14`, `:21-24` |
| NFR-11 | Every release file shall have a SHA-256 checksum file next to it. | `scripts/build_release_artifact.py:31-36`, `:119-122`; `release.yml:57` |
| NFR-12 | The wheel shall be installed and imported in an isolated venv before release, without pulling dependencies (`--no-deps`). As implemented, the import check passes from the checkout root even for a wheel without `src` (see 3.8). | `scripts/build_release_artifact.py:62-91` |
| NFR-13 | The release workflow shall have `contents: write` and shall publish only from a tag ref. | `.github/workflows/release.yml:14-15`, `:47` |
| NFR-14 | Only the branches `release`, `dev` and `test` shall exist. This is checked by `scripts/enforce_branch_policy.py` when someone runs it; no workflow runs it at `515d10a`. | `scripts/enforce_branch_policy.py:19`, `docs/branching.md:3` |
| NFR-15 | Commit subjects shall follow `type(scope): text.`. This is enforced only where `pre-commit install` was run. | `.pre-commit-config.yaml:20-26`, `.agent/scripts/check_commit_msg.py:31` |

## 7. Constraints and dependencies

No lock file is committed in `dev`: there is no `uv.lock` and no
`requirements-dev.lock` (`git ls-files`). CI therefore resolves the latest
versions that match these declarations.

| Package | Constraint | Group | Source |
| --- | --- | --- | --- |
| `torch` | none | runtime | `pyproject.toml:8` |
| `torchvision` | none | runtime | `pyproject.toml:9` |
| `timm` | none | runtime | `pyproject.toml:10` |
| `pytest`, `pytest-cov`, `pytest-mock`, `ruff`, `mypy`, `pytest-benchmark` | none | `dev` extra | `pyproject.toml:14-21` |
| `pyinstaller` | `>=6.0` | `release` extra; no script uses it | `pyproject.toml:22`, grep |
| `setuptools`, `wheel` | `>=61.0`, none | build system | `pyproject.toml:44-46` |
| `rich` | **not declared** | imported by `main.py:3-5` | `main.py` |
| Pillow (`PIL`), `numpy` | **not declared** | imported by `image_adapter.py:5`, `benchmarks/test_pipeline_performance.py:21`. Inference: they come in through torch/torchvision (not verified) | — |
| `radon`, `build`, `pdoc`, `pre-commit` | **not declared** | used by `collect_quality_metrics.py:108-109`, `build_release_artifact.py:41`, `docs_python.py:24`, `CONTRIBUTING.md:9` | — |
| ruff-pre-commit / pre-commit-hooks | `v0.4.1` / `v4.6.0` | pre-commit | `.pre-commit-config.yaml:4`, `:13` |
| GitHub Actions | `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`, `actions/download-artifact@v4`, `softprops/action-gh-release@v2`, `gitleaks/gitleaks-action@v2` | CI | `.github/workflows/*.yml` |

Licence: MIT (`LICENSE:1`).

## 8. Acceptance

Automated checks. These are the commands CI runs (`.github/workflows/python-ci.yml:24-36`):

```bash
uv sync --all-extras
uv run ruff format --check .
uv run ruff check .
uv run mypy --strict .
uv run pytest --cov=src --cov-report=xml
```

Locally, `python scripts/check_before_push.py` runs the `ruff`, `mypy` and
`pytest` commands from this list, without `uv sync`, and skips the tools that
are not installed (`scripts/check_before_push.py:15`, `:44`, `:100-110`). The FRs marked with a test in section 3
(FR-1, FR-3 to FR-6, FR-9 to FR-13, FR-15) are accepted when those tests pass.

Manual checks for the FRs without a test:

| FR | Check |
| --- | --- |
| FR-18, FR-19, FR-21 | `python main.py <existing image>` prints the table with model `vit_base_patch16_384` and exits with 0. The first run needs network access for the weights. |
| FR-1, FR-22 | `python main.py does-not-exist.jpg` prints `Error: Image not found at does-not-exist.jpg` and exits with 1. |
| FR-2 | A grayscale or RGBA image is classified without an error. |
| FR-7, FR-8 | On a CUDA or MPS machine, `TimmClassifier(...).device.type` is `cuda` / `mps` and `use_fp16` is `True`. On CPU, `use_fp16` is `False`. |
| FR-23, FR-24 | `python scripts/build_release_artifact.py --version vX.Y.Z` leaves `.whl`, `.tar.gz` and two `.sha256` files in `release/` and prints `the wheel installs and imports as src`. Run from the checkout root, this message does not prove the wheel imports (see 3.8). |
| FR-29, NFR-14 | `python scripts/enforce_branch_policy.py --remote origin` exits with 0, run in a clone whose local branches are only `release`, `dev` and `test` (any other local branch also makes it exit with 1). |
| NFR-7 | `python -m pytest benchmarks --benchmark-only` runs 6 benchmarks (the plain `pytest benchmarks ...` form fails to import `src`, see 3.8). Compare them with `README.md:32-39` (there is no threshold). |

## 9. Open questions

1. **Users and purpose.** The code does not say who uses this, or whether
   "mushroom classification" means anything beyond the pretrained model's
   ImageNet classes (`README.md:7-9` shows ImageNet class 992). The
   `archive/task*` tags and `docs/quality-iso25010.md:108-109` ("teaching
   repository") point to coursework. That is an inference.
2. **Normalisation.** The loader uses a fixed mean/std of 0.5
   (`image_adapter.py:20`) and does not read `default_cfg`. Is this intended
   for every model that can be passed in?
3. **Batch input.** `classify` uses only `output[0]` (`timm_adapter.py:45`).
   Is input with more than one image meant to be supported?
4. **Wheel contents. Answered by measurement.** `pyproject.toml` configures no
   package discovery, so setuptools treats `src/` as a src-layout root. A wheel
   built from a temporary copy of `515d10a` (`python -m build --wheel
   --no-isolation`, setuptools 84.0.0, offline) contains top-level `domain/`,
   `application/`, `infrastructure/` and a stray `__init__.py`, and no `src`
   package. Installed with `--no-deps` into a fresh venv: `import src` fails
   outside a checkout (`ModuleNotFoundError`); `import application.service`
   fails with `attempted relative import beyond top-level package`; from the
   checkout root `import src` succeeds, because it finds the checkout's `src/`.
   An editable install adds `<checkout>/src` to `sys.path`, not the checkout
   root. CI builds with whichever setuptools build isolation resolves; that
   version was not tested. Still open: is the wheel meant to be importable as
   `src`?
5. **Tests outside CI. Answered by measurement.** `benchmarks/` has no
   `__init__.py`, and there is no root `conftest.py` and no pytest
   `pythonpath` setting. A probe on a reduced temporary copy (pytest 9.1.1;
   `src/domain`, `tests/__init__.py`, a conftest and a benchmark that import
   `src`) gave: `pytest benchmarks` through the console script fails with
   `No module named 'src'`, also with an editable install; `python -m pytest
   benchmarks` passes; a bare `pytest` passes, because it loads
   `tests/conftest.py`, which puts the checkout root on `sys.path`. CI runs a
   bare `pytest`, so it is not affected. The real test files were not run
   (torch and the plugins are not installed here).
6. **CI status.** It is not known whether CI passed at `515d10a`; GitHub was
   not queried.
7. **Weights.** Which timm version, weight tag and download source are
   expected; where the cache lives; whether offline use is supported.
8. **Hardware.** Which GPUs, if any, the CUDA, MPS and FP16 paths were run on.
   CI covers CPU only.
9. **`rich`.** It is not declared. Is it expected to arrive as a transitive
   dependency, or is the declaration missing?
10. **`pyinstaller` extra.** It is declared (`pyproject.toml:22`), but
    `build_release_artifact.py:9-14` explains that no frozen executable is
    built. Is the extra still needed?
11. **Target performance.** No latency or throughput target is defined; the
    benchmarks have no thresholds.
12. **Deployment.** Apart from the GitHub Release artefacts
    (`release.yml`), no deployment target is described.
13. **Branch divergence.** `origin/release` (tag `v1.1.0`, 2026-11-08) and
    `origin/test` contain changes that are not in `dev`. Which branch is
    authoritative for this specification?
