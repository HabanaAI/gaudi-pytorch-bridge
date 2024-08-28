# User CustomOp tests

To execute tests, first compile sources that create custom operators:

```
python setup.py install
```

Then run test in `test_hpu_custom_op.py` and `test_hpu_legacy_custom_op.py` (only lazy)

```
pytest test_hpu_custom_op.py
PT_HPU_LAZY_MODE=0 pytest test_hpu_custom_op.py
pytest test_hpu_legacy_custom_op.py
```
