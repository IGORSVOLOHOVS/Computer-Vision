# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13. The pins are locked for 3.13 only.
2. Install [Git LFS](https://git-lfs.com/): the PNG images are stored in it.
3. Clone this repository and run `uv sync --frozen` at its root.

On Windows the locked `torch` is the CPU build from PyPI, so the model runs on the CPU there. On a machine with CUDA or Apple's MPS the adapter uses that device, in half precision.

The first `.\recipe.ps1 run` downloads the weights of `vit_base_patch16_384`, several hundred megabytes, into the Hugging Face cache, so it needs a network connection. `.\recipe.ps1 test` needs none of that: it sets `HF_HUB_OFFLINE`, and a test that reached for the weights would fail instead of fetching them.

`.\recipe.ps1 build` packs `src` into a wheel in `dist\`. `main.py` is not in the wheel: start the program from a clone, with `.\recipe.ps1 run`.
