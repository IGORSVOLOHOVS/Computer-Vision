# Quality assessment — ISO/IEC 25010

ISO/IEC 25010 defines eight product quality characteristics. Four are
**measured** by `scripts/collect_quality_metrics.py`; the other four need
judgement and are argued below, citing those measurements rather than asserting
quality.

```bash
python scripts/collect_quality_metrics.py --output quality-report.json
```

Latest run, 2026-08-06 — overall measured score **85.3**:

| Characteristic | Score | Evidence |
| --- | --- | --- |
| Functional suitability | 100.0 | 12 tests, 0 failures |
| Reliability | 96.6 | 96.6 % branch coverage |
| Maintainability | 94.6 | mean cyclomatic complexity 1.84, maintainability index 94.6, 0 lint findings |
| Portability | 50.0 | one operating system and one interpreter version in CI |

---

## 1. Functional suitability — measured

*Does it do what it claims, correctly and completely?*

The claim is narrow: load an image, classify it with a timm model, return the
top prediction and the full ranking. `tests/test_classification.py` covers each
step of that — the adapter's call into timm, the classification path with and
without label names, input-size extraction from the model config, and the
service that orchestrates the two.

**Known gap:** no test loads real pretrained weights, so nothing verifies that a
recognisable photograph gets a sensible label. The tests prove the wiring, not
the model.

## 2. Performance efficiency — assessed, with measurements

*Time behaviour, resource use, capacity.*

Inference belongs to the model. What this repository controls is the
preprocessing around it, measured by `benchmarks/test_pipeline_performance.py`:

| Operation | Median |
| --- | --- |
| Missing file — the failure path | 18.8 µs |
| Preprocess a 224 × 224 JPEG | 964 µs |
| Preprocess a 512 × 512 JPEG | 4.30 ms |
| Service: load + orchestrate, model stubbed (512 px source) | 4.26 ms |
| Preprocess to a 384 px target (512 px source) | 5.75 ms |
| Preprocess a 2048 × 2048 JPEG | 56.9 ms |

Three things fall out of those numbers.

**The layering is free.** The service measures 4.26 ms against 4.30 ms for the
load it contains — the two are within run-to-run noise, so ports, Protocols and
the orchestrating class cost nothing measurable. The architecture is not what is
slow.

**Cost is roughly linear in source area.** 224 → 512 px is 5.2× the pixels for
4.5× the time; 512 → 2048 px is 16× the pixels for 13× the time. A 2048 × 2048
photograph therefore spends ~57 ms on the CPU before the model sees anything —
comparable to the inference it is feeding. Callers holding large images should
downscale before the loader, not after.

**The target size matters as much as the source.** Asking for 384 px instead of
224 px from the same 512 px file costs 5.75 ms against 4.30 ms, a third more,
because the resize dominates.

The failure path costs 18.8 µs — a `Path.exists()` check rather than a decode
attempt, so a mistyped path fails fifty times faster than a successful load.

## 3. Compatibility — assessed

*Co-existence and interoperability.*

The interface is the ecosystem's own currency: torch tensors in, dataclasses
out, timm model names as configuration. Any of the several hundred timm
architectures can be named without a code change, because `TimmClassifier` takes
the name as a constructor argument and reads the input size from the model's own
`default_cfg`.

The package holds no global state and installs as an ordinary wheel, so it can
sit beside other torch projects in one environment.

**Known limit:** `torch` is unpinned beyond a floor, and torch does change minor
behaviour between versions. That is deliberate — pinning it hard would fight
whatever CUDA build the machine already has — but it means the version actually
tested is the one CI resolves.

**Known defect:** the distribution's import name is `src`. The package directory
is `src/` itself rather than `src/<name>/`, so installing the wheel puts a
top-level `src` module into site-packages, where it collides with any other
project that made the same choice. It is invisible while the repository is used
from a checkout and only bites on install. Fixing it means renaming the
directory and touching every import, so it is recorded here rather than done
quietly.

## 4. Usability — assessed

*Can someone who did not write it get a result?*

`main.py` is one entry point with `--help`, and the README opens with a
screenshot of a real classification run, so the output format is visible before
installing anything. `docs/architecture.md` explains where a new classifier or a
new image source would be plugged in.

**Known gap:** no GUI, and no batch mode — one image per invocation. For a
teaching repository that is the right scope; for anything at volume the
per-process model load (~100 MB from cache) would dominate.

## 5. Reliability — measured

*Does it keep working?*

96.6 % branch coverage over 12 tests. The failure that matters most is handled
explicitly rather than by exception soup: a missing image raises
`FileNotFoundError` from `PILImageLoader` with the path in the message, checked
before any decode is attempted.

One reliability defect was found and fixed while this document was being
written, and it is worth recording because of how it hid. Three tests passed on
the CPU-only runner and failed on any machine with a GPU: `TimmClassifier.__init__`
rebinds `self.model` from `.half()` when CUDA is present, and a plain
`MagicMock` returns a *different* mock from that call — one with no
`default_cfg` and no logits. The suite was therefore green in CI and red on the
developer's own machine. Fixed in the fixture, which now returns itself from
`.half()`, `.to()` and `.eval()`.

## 6. Security — assessed

*Confidentiality, integrity, resistance to misuse.*

Ruff's `flake8-bandit` rules run over the tree on every push; current
outstanding findings: **0**. A scheduled `secret-scan` workflow searches the
full git history for credentials, not only the working tree.

Images are read from local paths and never transmitted. The one network action
is timm's weight download at model construction.

**Known limit:** downloaded weights are trusted implicitly. Pinning a revision
hash per model would close that and is not implemented.

## 7. Maintainability — measured

*Can it be changed safely?*

Mean cyclomatic complexity **1.84** and maintainability index **94.6** — the
highest of any repository in this account — with zero outstanding lint findings
and `mypy --strict` passing over the package.

The reason is structural rather than lucky: the domain layer is two frozen
dataclasses and two `Protocol` definitions, so the decisions live in small
adapters that can be replaced one at a time. The most complex function in the
project is an `__init__` at complexity 5.

## 8. Portability — measured

*Where does it run?*

Score 50.0: CI exercises **ubuntu-latest with Python 3.10 only**, while the
package declares support for more. Windows and macOS are untested in CI, though
the code itself is pure Python over torch and has no platform-specific paths.

Device selection is handled at runtime — CUDA, then MPS, then CPU — so all three
code paths exist, but CI has no GPU and only the CPU path is exercised
automatically. That gap is exactly what let the fixture defect above survive.

**To raise this honestly:** add a Python 3.12 cell on ubuntu. It is one line and
tests a claim the metadata already makes.
