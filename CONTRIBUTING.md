# Contributing

Thanks for your interest in improving this project. It is a research codebase, so
contributions are evaluated primarily on **scientific correctness and
reproducibility**, then on code quality.

## Ground rules (non-negotiable)

1. **Never fabricate or simulate results.** Every reported number must come from a
   measurable, reproducible computation. Do not add simulated surveys, synthetic
   "validation", or hard-coded metrics.
2. **No results in code or docs without a script that produces them.** Numbers in
   the thesis/paper are filled from `results/` by `scripts/render_report.py`.
3. **Keep experiments configurable.** New behaviour goes through `configs/`, not
   hard-coded constants.

## Development setup

```bash
make setup            # venv + dependencies
make test             # run the test suite (must stay green)
make smoke            # end-to-end pipeline on synthetic data
```

## Coding standards

- Python ≥ 3.10, **PEP 8**, type hints on public functions, NumPy-style or concise
  docstrings on modules/classes/functions.
- Format/lint with `ruff` and `black` (config in `pyproject.toml`).
- Keep all experimental logic in `src/` (importable, testable). Notebooks may only
  orchestrate calls into `src/` — no duplicated logic.

## Pull requests

1. Add or update tests in `tests/` for any new behaviour.
2. Ensure `make test` passes and `make smoke` completes end-to-end.
3. Describe what you changed and, for anything affecting results, how to reproduce.
4. Update `README.md` / `reports/` if you change the workflow or interfaces.

## Reporting issues

Please include: OS, Python version, `pip freeze` (or `conda list`), the exact
command, and the full traceback. For result discrepancies, attach the relevant
`results/metrics/*.json` and `*_metadata.json`.
