# Project instructions

- Support Python 3.10+. Use uv and update `uv.lock` when dependencies change.
  Keep GPU runtime dependencies optional.
- Use Conventional Commits as described in `CONTRIBUTING.md`.
- Select runtime candidates by model capabilities and task, without hardcoded
  Hugging Face model IDs. Treat metadata-based suggestions as unverified.
- Validate application outputs against a reference before recommending a
  configuration. Include scores, thresholds and parsed fields when relevant.
- Quality and performance datasets are independent and may be private. Do not
  assume fixed IDs or schemas. Never log or commit tokens or private records.
- For benchmark results, record model and dataset revisions, environment,
  workload, measurement boundary and runtime settings. Separate warm-up from
  measurement.

For setup and verification commands, see `CONTRIBUTING.md` and `pyproject.toml`.
