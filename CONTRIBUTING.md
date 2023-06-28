# Contributing to pytorch-integration

## Python linters
Each patchset uploaded to Gerrit is automatically run against below linters to keep high code quality and consistent look and feel:
* [black](https://github.com/psf/black)
* [flake8](https://github.com/PyCQA/flake8)
* [isort](https://github.com/PyCQA/isort)
* [mypy](https://github.com/python/mypy)

### pre-commit
To save your time, you can run all above linters automatically on your workstation before committing a change.
To do this please run below commands which install [pre-commit](https://github.com/pre-commit/pre-commit) hook:

    pip install pre-commit
    pre-commit install

Above commands will install pre-commit hooks which will run all linters against all staged files.
If you want to run linters on all files in the repository please run:

    pre-commit run --files tests/*
