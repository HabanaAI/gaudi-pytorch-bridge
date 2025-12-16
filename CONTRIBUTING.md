# Contributing to pytorch-integration
In order to introduce any changes to the code present in this repository please fork it.
Any external pull requests won't be reviewed and merged.

## Python linters
In order to maintain standardized code formatting and early error detection below linters are used:
* [black](https://github.com/psf/black)
* [ruff](https://github.com/astral-sh/ruff)

### pre-commit
To save your time, you can run all above linters automatically on your workstation before committing a change.
To do this please run below commands which install [pre-commit](https://github.com/pre-commit/pre-commit) hook:

    pip install pre-commit
    pre-commit install

Above commands will install pre-commit hooks which will run all linters against all staged files.
If you want to run linters on all files in the repository please run:

    pre-commit run --files tests/*
