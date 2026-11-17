# Contributing to CAELUM

Thanks for your interest in CAELUM! This project is in early development, and contributions
are welcome.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you
agree to uphold it.

## How to contribute

1. **Open an issue first** for anything beyond a small fix, so we can agree on the approach.
2. **Fork** the repository and create a branch from `main`:
   - `feat/<short-name>` for new features
   - `fix/<short-name>` for bug fixes
3. **Make your change**, with tests.
4. **Run the checks** locally (lint, type check, tests) before opening a pull request.
5. **Open a pull request** against `main` with a clear description of what and why.

## Commit messages

We use the format `type(scope): description`, where `type` is one of:
`feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

Examples:

```text
feat(cache): add exact cache with SQLite backend
fix(openai): preserve streaming behaviour
```

## Code style

- **Python**: formatted with `ruff format`, linted with `ruff check`, type-checked with
  `pyright`. Line length 100. Public functions have Google-style docstrings.
- **TypeScript**: formatted with `prettier`. Line length 100. Public APIs have JSDoc.
- Non-obvious decisions get a `# WHY:` comment explaining the reasoning.

## Tests

- Unit tests must not call real APIs (strict mocks).
- Integration tests live in `tests/integration/` and run separately.
- Target coverage: 80% on core modules.

## Core principles

Every change must respect these:

- **Drop-in replacement**: `from caelum import OpenAI` must behave exactly like
  `from openai import OpenAI`. No signature changes, no required options.
- **Privacy-first**: features that change responses (caching, routing) are opt-in.
  Data never leaves the user's machine without explicit consent.
- **Minimal overhead**: target p95 < 5 ms added latency on cache miss, hard ceiling 10 ms.
- **Lightweight**: no heavy mandatory dependencies.

## Contributor License Agreement

Before your first pull request can be merged, you sign our
[Contributor License Agreement](CLA.md) once, by replying to the bot on your pull request.
You keep the copyright in your work; the CLA lets Issouf KANAZOE, who maintains CAELUM,
distribute it, including under other licenses.

## License

CAELUM is released under the [Apache License 2.0](LICENSE). The CAELUM name and logo are
covered by our [trademark policy](TRADEMARKS.md).
