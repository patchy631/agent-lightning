# Contributing Guide

Agent Lightning welcomes contributions of all sizes—from bug fixes to new features and documentation.
This guide walks you through getting a local environment ready, following our branching strategy, and opening a high-quality pull request.

## 1. Prepare Your Environment

### Prerequisites

- **Python**: Version 3.10 or newer (the project is tested on 3.10–3.13).
- **uv**: We use [uv](https://docs.astral.sh/uv/) for dependency management and packaging speed. Install it via the instructions in the [official docs](https://docs.astral.sh/uv/getting-started/installation/).
- **Git**: Ensure you have Git installed and configured with your GitHub account.

### Clone the Repository

Fork the repository on GitHub, then clone your fork and add the upstream remote:

```bash
git clone git@github.com:<your-username>/agent-lightning.git
cd agent-lightning
git remote add upstream https://github.com/microsoft/agent-lightning.git
```

### Sync Dependencies

Install the development toolchain with uv:

```bash
uv sync --group dev
```

Need the full experience (algorithms, examples, GPU extras)? Enable the appropriate groups in one command:

```bash
uv sync --frozen \
    --extra apo \
    --extra verl \
    --group dev \
    --group torch-cpu \
    --group torch-stable \
    --group trl \
    --group agents \
    --no-default-groups
```

After `uv sync`, either prefix commands with `uv run …` or activate the virtual environment from `.venv/`.

## 2. Install and Run Pre-commit

Code style and linting are enforced via [pre-commit](https://pre-commit.com/). Install the hooks once and run them before you push:

```bash
uv run pre-commit install
uv run pre-commit run --all-files --show-diff-on-failure --color=always
```

Running the hooks locally saves you from CI failures and keeps diffs clean.

## 3. Branching Workflow

1. Always start from an up-to-date `main`:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```
2. Create a topic branch using one of the naming prefixes below:
   - `feature/<short-description>` — new features
   - `fix/<short-description>` — bug fixes
   - `docs/<short-description>` — documentation-only updates
   - `chore/<short-description>` — tooling or maintenance tweaks

Use lowercase words separated by hyphens (e.g. `feature/async-runner-hooks`).

## 4. Build and Validate Documentation

If your change touches documentation, verify it builds cleanly:

```bash
uv run mkdocs serve --strict  # live reload during editing
uv run mkdocs build --strict  # CI-equivalent check
```

The `--strict` flag matches our CI settings and turns warnings into errors so you can catch issues early.

## 5. Before You Push

- Run the pre-commit hooks (`uv run pre-commit run --all-files`).
- Execute the relevant tests, for example:
  ```bash
  uv run pytest -v tests
  ```
- For changes affecting examples, follow the README inside each `examples/<name>/` directory to validate manually as needed.

## 6. Open a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin <branch-name>
   ```
2. Open a pull request against `microsoft/agent-lightning:main`.
3. Fill out the PR description with:
   - A summary of the change.
   - A checklist of tests or commands you ran.
   - Links to related issues (use `Fixes #123` to auto-close).
4. Add screenshots or terminal output when they help reviewers understand the change.
5. Respond to review feedback promptly; keep commits focused and consider using fixup commits (`git commit --fixup`) for clarity.

Thank you for contributing—your improvements help the entire Agent Lightning community!
