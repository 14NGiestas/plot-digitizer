# AGENTS.md - OpenCode Session Guidance for plot-digitizer

This document provides high-signal instructions for OpenCode agents working on the `plot-digitizer` project. It aims to prevent common pitfalls and accelerate understanding of the project's structure, build processes, and key commands.

## Project Overview
- **Type**: Python project for digitizing plots, including machine learning (ML) components for tasks like object detection and segmentation.
- **Primary Tool**: `digitizer` CLI.
- **Language**: Python (requires >=3.10,<3.13).

## Environment Setup & Dependencies

### Nix (Recommended for Development & CI)
- **Environment Setup**: Use `nix develop` to enter the default (CPU) development shell.
- **Named GPU shells** — pick the one that matches your hardware:

  | Shell | Command | Target |
  |---|---|---|
  | `default` / `cpu-only` | `nix develop` | CPU inference, CI |
  | `rocm` | `nix develop .#rocm` | AMD GPU (ROCm/HIP) |
  | `cuda` | `nix develop .#cuda` | NVIDIA GPU (CUDA) |

- **Build**: `nix build .#default`
- **ROCm / AMD APU note**: The `rocm` shell targets gfx1103 (Radeon 780M / Ryzen 8000-series Hawk Point APU).
  After entering the shell, install PyTorch for ROCm:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
  ```
- **CUDA / NVIDIA note**: After entering `.#cuda`, install the matching wheel:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

### uv (Alternative Python Package Management)
- If Nix is not used, `uv` is the preferred tool for Python dependency management.
- **Installation**: Follow instructions in `README.md`.
- **Install Dependencies**: `uv pip install -e '.[dev,ai]'` (or without `ai` for CPU-only).

## Key Commands

### Testing
- **Run Unit Tests**: `nix develop --command sh -c "PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v"`
  - Alternatively (if not using nix develop): Ensure `PYTHONPATH=src` is set and `python -m unittest discover -s tests -p 'test_*.py' -v` is run.

### CLI Usage (`digitizer`)
The main entry point is the `digitizer` command.
- **General Help**: `digitizer --help`
- **Digitize an image**:
  ```bash
  digitizer digitize <IMAGE_PATH> \
    --output <OUTPUT_JSON_PATH> \
    --model-dir <MODEL_WEIGHTS_DIR>
  ```
  - Example (from `README.md`): `digitizer digitize bandstructure_target.png --output digitized_data.json`
- **Generate Synthetic Data**:
  ```bash
  digitizer generate \
    --config-path <CONFIG_YAML> \
    --output-dir <OUTPUT_DIR> \
    --num-samples <NUMBER>
  ```
- **Train Models**:
  ```bash
  digitizer train \
    --config-path <CONFIG_YAML> \
    --output-dir <OUTPUT_DIR> \
    --device <DEVICE> # e.g., cpu, cuda, mps
  ```
- **Validate Models**:
  ```bash
  digitizer validate \
    --config-path <CONFIG_YAML> \
    --output-dir <OUTPUT_DIR> \
    --model-dir <MODEL_WEIGHTS_DIR>
  ```

## Project Structure & Conventions

- **Source Code**: Primarily located in the `src/` directory.
- **Tests**: Located in the `tests/` directory, following `test_*.py` naming convention and using the `unittest` framework.
- **Configuration**:
    - `pyproject.toml`: Defines project metadata and core Python dependencies.
    - `args.yaml`, `dataset.yaml`: Found in `runs/`, `training-runs/`, `synthetic-data/`, `data/` directories, indicating ML model arguments and dataset definitions.
- **ML Components**: The project uses `ultralytics`, `torch`, `torchvision` as optional `ai`/`ai-cpu` dependencies. Be aware of `args.yaml` and `dataset.yaml` files when working with ML-related tasks.

## Specific Task Guidance

- **Digitizing `bandstructure_target.png`**: The target image is `/home/pauli/plot-digitizer/bandstructure_target.png`.
  - Use the `digitizer digitize` command.
  - The model weights directory might need to be specified; check `README.md` or existing training runs for default locations if not explicitly provided in the request.
  - Expect JSON output containing digitized data.
