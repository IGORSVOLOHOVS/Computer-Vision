# Manual

```powershell
.\recipe.ps1 run [image]
```

Run it from the root of the repository. `image` is the path of one image file; without it the program reads `image.png` at the root.

The program loads `vit_base_patch16_384` through `timm`, reads the input size from the model, resizes and centre-crops the image to that size, and prints a table with the model's name, the top prediction and its confidence. The label is the model's own label name where its `default_cfg` has one, and `class_<index>` where it has none, as for this model.

| what happens | exit code |
| --- | --- |
| the image was classified | 0 |
| the image does not exist | 1, after `Error: Image not found at <path>` |
| anything else failed, the download of the weights included | 1, after `Unexpected Error:` and the traceback |

The model is loaded before the image is read, so a missing image is reported only once the weights are in place. The model name is not an option yet: it is the default of `main()` in `main.py`.
