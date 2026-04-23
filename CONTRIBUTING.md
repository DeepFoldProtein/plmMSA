# Contributing to plmMSA

## Branching

- Work on feature branches cut from `main`.
- Keep PRs small and focused; link the relevant bullet from `PLAN.md`.

## Commit style

Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) are
encouraged but not enforced.

## Dev setup

Requires [uv](https://github.com/astral-sh/uv) and Docker with Compose v2.

```bash
uv sync
cp .env.example .env
cp settings.example.toml settings.toml
./bin/up.sh
```

## Pre-merge checks

There is no CI workflow — run these locally before opening a PR:

- `uv run ruff check`
- `uv run ruff format --check`
- `uv run pyright`
- `uv run pytest`

## Updating the plan

If a change touches architecture, API shape, or the service layout, update
the relevant section of `PLAN.md` in the same PR.
