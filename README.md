# 🍄 Mushroom Computer Vision Project

A high-performance image classification pipeline for mushroom detection, built with modern Python standards and Clean Architecture.

## 🏗️ Architecture

This project follows the **Clean (Hexagonal) Architecture** pattern to ensure the domain logic is isolated from external frameworks.

- **Domain**: Contains core models (`ClassificationResult`, `Prediction`) and interfaces (`ImageClassifier`, `ImageLoader`). Zero dependencies on external libraries.
- **Application**: Orchestrates workflows via `ClassifierService`.
- **Infrastructure**: Adapters for external libraries like `timm` and `PIL`.

## 🚀 Performance & Optimization

The codebase is optimized for latency and throughput:
- **Mixed Precision (FP16)**: Inference is performed in half-precision on supported hardware to reduce memory bandwidth and increase speed.
- **Dynamic Input Sizing**: Models like ViT automatically scale to their optimal input resolution (e.g., 224, 384).
- **Profiling**: Detailed profiling results are available in `profile_results_optimized.txt`.

## 💻 Hardware Requirements

- **GPU Support**: Automatic detection of **CUDA** (NVIDIA) and **MPS** (Apple Silicon).
- **Fallback**: Graceful fallback to CPU if no GPU is available.

## 🛠️ Usage

### Installation
```bash
# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"
```

### Classification
Use `main.py` for advanced classification using the service layer:
```bash
python main.py data/data/agaric/example.jpg
```

### Rapid Prototyping
Use `task3.py` for a simpler, script-based approach to ImageNet classification.

## 📚 Documentation

Automatically generated HTML documentation is available in the `docs/` directory. 
To regenerate:
```bash
python .agent/scripts/docs_python.py
```
