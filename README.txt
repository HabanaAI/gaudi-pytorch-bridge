# Habana PyTorch Modules

## Detailed building instructions can be found on sharepoint:
https://habanalabs.sharepoint.com/SitePages/Habana-support-for-pytorch.aspx

## Linters and formatters

For C++, use clang-format and cmake-format for formatting C++ and CMake. Use
clang-tidy and cmake-lint for linting those languages.
For Python, use Black for formatting and flake8, mypy, and pyflakes for linting.
To install them, run:
```
pip install flake8 mypy flake8-bugbear flake8-comprehensions flake8-executable flake8-pyi mccabe black pyflakes
```

You can set up all of the above except mypy and pyflakes by installing
`pre-commit` and running `pre-commit install`. This way they will run each time
you commit changes.

## Building

For CMake to find PyTorch, `CMAKE_PREFIX_PATH` must point to the directory it
is installed in. You can use the following commands to build the project:

```bash
mkdir build
cd build
CMAKE_PREFIX_PATH=`python3 -c "import os, torch; print(os.path.dirname(torch.__file__))"` cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .
```
