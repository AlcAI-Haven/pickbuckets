# Contributing

Thanks for helping make `pickbuckets` useful.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
```

## Principles

- Keep the base package dependency-free.
- Put integrations behind optional extras.
- Fit once, transform from saved rules.
- Add serialization round-trip tests for every new bucketer.
- Prefer clear exceptions over implicit coercion.

## Before opening a pull request

```bash
ruff check .
mypy src/pickbuckets
pytest
```

