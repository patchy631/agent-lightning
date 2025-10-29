# Maintainer Guide

This guide documents the day-to-day workflows for the Agent Lightning maintainers, including how to cut a release, interact with CI, and backport fixes.

## Release Workflow

Follow this checklist when preparing a new release.

1. **Plan the scope**
   - Collect the PRs and issues targeted for the release.
   - Verify that required documentation updates are complete.

2. **Create a release branch**
   ```bash
   git fetch upstream
   git checkout -b release/vX.Y.Z upstream/main
   ```

3. **Bump versions**
   - Update `pyproject.toml` (and `agentlightning/__init__.py` if present) using the helper script:
     ```bash
     ./scripts/bump_version.sh X.Y.Z
     ```
   - Commit the version bump separately so it is easy to cherry-pick later.

4. **Update release collateral**
   - Ensure the examples catalog at `examples/README.md` accurately reflects maintenance status and CI badges.
   - Review the API reference pages in `docs/reference/` for outdated signatures or docstrings.
   - Draft release notes in `docs/releases/` (create a new file for the version if necessary) and summarize key changes. Copy the same content into the GitHub release body later.

5. **Run local checks**
   - Install/update dependencies (`uv sync --group dev`).
   - Run formatting and linting: `uv run pre-commit run --all-files --show-diff-on-failure`.
   - Execute targeted tests, especially those touched by the release.
   - Build docs exactly like CI: `uv run mkdocs build --strict`.

6. **Open the release PR**
   - Push the branch to the upstream repository if you have permission, otherwise to your fork.
   - Title suggestion: `Release vX.Y.Z`.
   - Link all relevant issues and include a checklist of validation steps.
   - Apply the appropriate CI labels (see below) and comment `/ci` to exercise GPU/examples pipelines when needed.

7. **Finalize the release**
   - After the PR merges, create a signed tag that matches the version:
     ```bash
     git checkout main
     git pull upstream main
     git tag -s vX.Y.Z -m "Release vX.Y.Z"
     git push upstream vX.Y.Z
     ```
   - Pushing the tag triggers the PyPI publish and documentation deployment workflows.
   - Create the GitHub release using the prepared notes, attach built artifacts if desired, and verify that the docs site shows the new version.

## Working with CI Labels and `/ci`

Long-running jobs such as GPU training or example end-to-end runs are opt-in on pull requests. To trigger them:

1. Add one or more of the following labels to the PR before issuing the command:
   - `ci-all` — run every repository-dispatch aware workflow.
   - `ci-gpu` — run the GPU integration tests (`tests-full.yml`).
   - `ci-apo`, `ci-calc-x`, `ci-spider`, `ci-unsloth`, `ci-compat` — run the corresponding example pipelines.
2. Comment `/ci` on the pull request. The `issue-comment` workflow will acknowledge the command and track results inline.
3. Remove labels once the signal is collected to avoid accidental re-triggers.

Use `/ci` whenever a change affects shared infrastructure, dependencies, or training logic that requires extra validation beyond the default PR checks.

## Backporting Pull Requests

We rely on automated backports for supported stable branches.

1. Decide which stable branch should receive the fix (for example, `stable/v0.2`).
2. Before or immediately after merging the PR into `main`, add a label matching `stable/<series>` (e.g., `stable/v0.2`).
3. The `backport.yml` workflow creates a new PR named `backport/<original-number>/<target-branch>` authored by `agent-lightning-bot`.
4. Review the generated backport PR, ensure CI passes, and merge it into the target branch.
5. If conflicts arise, push manual fixes directly to the backport branch and re-run `/ci` as needed.

Keep the stable branches healthy by cherry-picking only critical fixes and ensuring their documentation and example metadata stay in sync with the release lines.
