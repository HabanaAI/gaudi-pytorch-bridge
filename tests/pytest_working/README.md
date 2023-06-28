Guidelines:
- all seeds (numpy, random, torch) will be automatically set before each test, no need to do it on your own
- test shall not affect env settings. Env is read once during torch initialization and the it's impossible to change it's state in cpp process.
- test shall specifacally not set `PT_HPU_LAZY_MODE` env var anywhere. If multiple tests with different settings are run within single pytest process then .so library will be loaded only in the first test. It may cause some hard to reproduce issues. It's ok to use assert/@pytest.skipf to check flag status

Directories content:
- `lazy/` tests to run only with `PT_HPU_LAZY_MODE=1`
- `eager/` tests to run only with `PT_HPU_LAZY_MODE=0`
- `compile/` TODO
- `fused_ops/` TODO