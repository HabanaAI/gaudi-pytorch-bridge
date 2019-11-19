# Habana PyTorch Modules

## Prerequisites

This repo should be used in python3.6 venv with following pytorch version:
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

## Building

For CMake to find PyTorch, `CMAKE_PREFIX_PATH` must point to the directory it
is installed in. You can use the following commands to build the project:

```bash
mkdir build
cd build
CMAKE_PREFIX_PATH=`python3 -c "import os, torch; print(os.path.dirname(torch.__file__))"` cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .
```

