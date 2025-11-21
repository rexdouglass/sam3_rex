# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `sam3/`: `model/` for detectors/trackers, `agent/` for prompt handling, `sam/` for legacy wrappers, `train/` for Hydra configs and trainers, and `eval/` for metrics.
- Notebooks and usage demos sit in `examples/`; static assets (figures, gifs, toy data) live in `assets/`.
- Training configs are under `sam3/train/configs/` (Roboflow, ODinW, SA-Co variants); helper scripts are in `scripts/`.

## Build, Test, and Development Commands
- Install core: `pip install -e .`; add lint/pytest tooling: `pip install -e ".[dev]"`; training stack: `pip install -e ".[train]"` (or `.[train,dev]`).
- Unit tests (current coverage is perflib): `pytest sam3/perflib/tests`; target subsets with `pytest -k <pattern>`.
- Format and imports: `python -m black sam3` then `python -m usort format sam3`; type-check hot paths with `python -m mypy sam3`.
- Local training example: `python sam3/train/train.py -c sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indent, Black line length 88. Keep functions small and typed (`disallow_untyped_defs` is enabled).
- Prefer snake_case for modules/functions, PascalCase for classes, UPPER_SNAKE_CASE for constants. Align file names with primary class/module.
- Follow Ruff/ufmt defaults; sort imports with usort. Keep seeds/data loader options deterministic in training code.

## Testing Guidelines
- Use `pytest`; mirror package structure when adding tests (e.g., `sam3/<area>/tests/` or a new repo-level `tests/` directory).
- Add regression tests for new model utilities, data transforms, and trainer hooks. Only check in tiny fixtures from `assets/veval` when shareable.
- For substantial changes, run `pytest --cov=sam3` and note coverage risks in the PR when gaps remain.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (e.g., “Add video tracker seed control” matches existing history).
- Branch from `main`; keep PRs scoped. Include a brief “what/why” plus test commands in the description. Link related issues and note any config or checkpoint expectations.
- Update docs/notebooks/configs alongside code changes; add screenshots or metrics tables for UI or model-quality changes.
- Do not commit checkpoints, large datasets, or private tokens. Hugging Face auth is required for model weights—keep tokens local and documented only as instructions.

## Security & Configuration Tips
- Store dataset roots and experiment paths outside the repo; point configs via Hydra variables (`paths.*` in `sam3/train/configs`).
- Verify licenses for external assets before adding to `assets/`. Remove PII from logs before sharing. Keep `.gitignore` entries for generated artifacts intact.
