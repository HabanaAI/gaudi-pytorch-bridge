This folder contains the custom optimizers specific to habana.
Usually these are implemented as fused single kernels.


Unit tests:
1. adagrad_ut.py -- Unit test for fused adagrad optimizer for dense tensors
2. adamw_ut.py -- Unit test for fused adamw
3. lamb_ut.py -- Unit test for lamb optimizer
4. norm_ut.py -- Unit test for norm
5. norm_ut_nn.py -- Unit test for norm nn
6. sgd_ut.py -- Unit test for fused sgd optimizer
