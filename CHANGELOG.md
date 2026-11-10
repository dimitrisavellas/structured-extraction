# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- GitHub Actions CI: ruff check, ruff format check, and pytest on Python 3.11 and 3.12 for every pull request and push to `main`.
- `uv.lock` and a `lint` dependency group pinning ruff; `uv sync --extra dev --group lint` sets up a dev environment.

### Changed
- Ruff config now lives in `pyproject.toml`. Existing files are exempt from the type-hint, docstring, line-length and complexity rules through an explicit per-file list; new files get every rule.
- Code reformatted with `ruff format`, and the remaining lint findings fixed. No change in behaviour.

## [0.2.1] - 2026-11-10

### Added
- `--local` flag for small local models (e.g. 9B on Ollama): one extraction call per section, thinking off, 30-minute request timeout.
- `EXTRACT_PER_SECTION` setting: one extraction call per matched section, which stops small models looping on rows repeated across tables.
- `REASONING_EFFORT` setting, sent as `reasoning_effort` on schema and extraction calls (`none` disables thinking).
- `LLM_TIMEOUT` setting for the schema and extraction request timeout (default 600 seconds, as before).

### Changed
- README restructured, with a new section on running with local models.
- Package metadata now points to the GitHub repository (https://github.com/dimitrisavellas/structured-extraction).

## [0.2.0] - 2026-08-04

### Added
- `--schema <file.json>` passes a custom extraction schema, so schema discovery is skipped.
- `fixed` on a sub-field constrains the model to a single value (`Literal`), for example to extract only one specific condition.

## [0.1.0] - 2026-07-20

### Added
- First public release.

[Unreleased]: https://github.com/dimitrisavellas/structured-extraction/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/dimitrisavellas/structured-extraction/releases/tag/v0.2.1
[0.2.0]: https://github.com/dimitrisavellas/structured-extraction/releases/tag/v0.2.0
[0.1.0]: https://github.com/dimitrisavellas/structured-extraction/releases/tag/v0.1.0
