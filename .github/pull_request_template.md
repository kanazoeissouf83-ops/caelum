<!-- markdownlint-disable-file MD041 -->
## What

Short description of the change.

## Why

The problem this solves, or a link to the related issue (e.g. `Closes #123`).

## How it was tested

- [ ] Unit tests added or updated (no real API calls)
- [ ] `ruff format`, `ruff check`, `pyright` and `pytest` pass locally

## Checklist

- [ ] Drop-in compatibility preserved (`from caelum import OpenAI` behaves like `openai.OpenAI`)
- [ ] Features that change responses stay opt-in (privacy-first)
- [ ] No heavy mandatory dependency added
- [ ] Docs and `CHANGELOG.md` updated if user-facing
