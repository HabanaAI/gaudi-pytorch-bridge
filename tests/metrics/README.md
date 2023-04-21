# Metrics tests

To execute tests, first compile sources that create custom operator and kernel, with calls to metrics' `EventDispatcher`:

```
./build.sh
```

Then run test in `metrics_tests.py`

```
pytest metrics_tests.py
```
