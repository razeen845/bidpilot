# Continuous integration

`github-actions-ci.yml` runs `ruff` and `pytest` on every push. GitHub only runs workflow files from
`.github/workflows/`, and uploading files there needs the `workflow` permission on your GitHub login.
To switch CI on:

```bash
gh auth refresh -h github.com -s workflow     # one-time, approve in the browser
mkdir -p .github/workflows && git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "enable CI" && git push
```
