# Generation of code for PyTorch hpu op through yaml

This [script](./gen_op.py) reads the input [yaml file](./hpu_op.yaml) and generates C++
 code for the ops defined.

To add an op, add the op's name defined in RegistrationDeclarations.h. For simple ops,
 define its guid (without the dtype suffix) and dtypes supported. Unsupported dtypes
 will use cpu fallback to perform the operation. For regular operations, define
 `out_ids` to allocate output tensor(s) with the properties of the tensor(s) with the
 indices in the list, and for inplace operations, define `inplace_ids` to denote the
 inplace tensor(s).
For example:
```
cos:
  dtypes: [BFloat16, Float]
  guid: cos_fwd
  out_ids: [0]
```

The generated code is broadly divided into
* Frontend lazy
* Output shape function
* TPC params function
* Lowering/Backend Kernel
* Registration to Lowering Kernel Registry
* Registration to PyTorch
* Op Template - Reduction

When additional code specific for an op is necessary, there is a provision for writing
 that part manually and getting those manually written code invoked in the generated code
 except for the two registrations above (lowering kernel registration still has a provision
 to be in `hpu` namespace with the use of `func_ns_hpu = true`). For example, for `cumsum_`
 there is a need to fill the TPC params structure manually and thus the function
 `FillCumsumParams` is specified to the key `custom_fill_params` for `cumsum_`.

When there is a need to chain multiple guids to realize a single PyTorch op, it can be
 achieved by writing `AddNode` of the lowering/backend kernel manually.
 One such example can be found in
```
logcumsumexp:
  dtypes: [BFloat16, Float]
  op_base_class: LogCumsumExp
  out_ids: [0]
```

And `LogCumsumExp::AddNode` method is written to chain guids - `log`, `cumsum` and `exp`.
The same can be found in [logcumsumexp_gen.cpp](../hpu_ops/logcumsumexp_gen.cpp).


Define Reduction template by adding `op_template` key to make use of common reduction op
 functions in Lowering/backend, frontend and OutputShape computation. Helper functions
 mainly handle dim, keepdim and dtype args which can be optional or required depending on
 the Op and makes it optimal to prune the category at once.
An example is as follows:
```
prod.int_out:
  dtypes: [BFloat16, Float]
  guid: reduce_prod_fwd
  op_template: reduction
```

Reduction helper functions like HandleReductionDimAndKeepdim and HandleReductionDtype can
 also be used explicitly if chain of guids have atleast one reduction guid and they consume
 dim and keepdim args for the former -- to cast dtype based on user input for the latter.
Reference can be found in the below specified files:
  HandleReductionDtype - [norm_gen.cpp](../hpu_ops/norm_gen.cpp)
  HandleReductionDimAndKeepdim - [all_dim_gen.cpp](../hpu_ops/all_dim_gen.cpp)
