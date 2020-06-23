# Habana Mixed Precision
This tool is used to can be used to evaluate and create mixed precision models at operator level granularity for PyTorch framework. Dtypes supported: BF16, FP32
This is achieved by adding cast nodes dynamically to the original model

# Installation:
python setup.py bdist_wheel
pip install dist/*.whl

# Usage
from hmp import hmp
hmp.convert()