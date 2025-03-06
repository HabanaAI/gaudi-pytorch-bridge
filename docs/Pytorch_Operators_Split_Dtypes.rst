
.. _pytorch-operators:

****************************
PyTorch Operators
****************************



Overview
========

This document provides an overview of PyTorch-supported operators for the
Intel® Gaudi® AI accelerator. Note that the operators listed below support
only selected variants and limited optional parameters for Gaudi.

For details on Fused Ops, see :ref:`custom_operators`.

PyTorch Operators Support Summary - Floating Point types
========================================================

.. rst-class:: datatable

====================================  ======== ======== ======== ======= ======================
**PyTorch Operator**                  **FP32** **BF16** **FP16** **FP8** **Operator Type**
====================================  ======== ======== ======== ======= ======================
adaptive_avg_pool1d                      Yes      Yes      Yes     No    torch.nn.functional
adaptive_avg_pool2d                      Yes      Yes      Yes     No    torch.nn.functional
adaptive_avg_pool3d                      Yes      Yes      Yes     No    torch.nn.functional
avg_pool1d                               Yes      Yes      Yes     No    torch.nn.functional
avg_pool2d                               Yes      Yes      Yes     No    torch.nn.functional
avg_pool3d                               Yes      Yes      Yes     No    torch.nn.functional
batch_norm                               Yes      Yes      Yes     No    torch.nn.functional
binary_cross_entropy                     Yes      Yes      Yes     No    torch.nn.functional
binary_cross_entropy_with_logits         Yes      Yes      Yes     No    torch.nn.functional
channel_shuffle                          Yes      Yes      Yes     No    torch.nn.functional
conv1d                                   Yes      Yes      Yes     Yes   torch.nn.functional
conv2d                                   Yes      Yes      Yes     Yes   torch.nn.functional
conv3d                                   Yes      Yes      Yes     Yes   torch.nn.functional
conv_transpose1d                         Yes      Yes      Yes     Yes   torch.nn.functional
conv_transpose2d                         Yes      Yes      Yes     Yes   torch.nn.functional
conv_transpose3d                         Yes      Yes      Yes     Yes   torch.nn.functional
dropout                                  Yes      Yes      Yes     No    torch.nn.functional
elu                                      Yes      Yes      Yes     No    torch.nn.functional
elu\_                                    Yes      Yes      Yes     No    torch.nn.functional
embedding                                Yes      Yes      Yes     Yes   torch.nn.functional
embedding_bag                            Yes      Yes      Yes     No    torch.nn.functional
gelu                                     Yes      Yes      Yes     No    torch.nn.functional
gelu\_                                   Yes      Yes      Yes     No    torch.nn.functional
glu                                      Yes      Yes      Yes     No    torch.nn.functional
grid_sample                              Yes      Yes      Yes     No    torch.nn.functional
hardshrink                               Yes      Yes      Yes     No    torch.nn.functional
hardsigmoid                              Yes      Yes      Yes     No    torch.nn.functional
hardsigmoid\_                            Yes      Yes      Yes     No    torch.nn.functional
hardtanh                                 Yes      Yes      Yes     Yes   torch.nn.functional
hardtanh\_                               Yes      Yes      Yes     Yes   torch.nn.functional
huber_loss                               Yes      Yes      Yes     No    torch.nn.functional
instance_norm                            Yes      Yes      No      No    torch.nn.functional
kl_div                                   Yes      Yes      No      No    torch.nn.functional
l1_loss                                  Yes      Yes      No      No    torch.nn.functional
layer_norm                               Yes      Yes      Yes     No    torch.nn.functional
leaky_relu                               Yes      Yes      Yes     No    torch.nn.functional
leaky_relu\_                             Yes      Yes      Yes     No    torch.nn.functional
linear                                   Yes      Yes      Yes     No    torch.nn.functional
log_softmax                              Yes      Yes      Yes     No    torch.nn.functional
logsigmoid                               Yes      Yes      Yes     No    torch.nn.functional
max_pool2d                               Yes      Yes      Yes     No    torch.nn.functional
max_pool2d_with_indices                  Yes      Yes      Yes     No    torch.nn.functional
max_pool3d                               Yes      Yes      Yes     No    torch.nn.functional
max_pool3d_with_indices                  Yes      Yes      Yes     No    torch.nn.functional
mish                                     Yes      Yes      Yes     No    torch.nn.functional
mish\_                                   Yes      Yes      Yes     No    torch.nn.functional
mse_loss                                 Yes      Yes      Yes     No    torch.nn.functional
multi_margin_loss                        Yes      Yes      Yes     No    torch.nn.functional
multilabel_margin_loss                   Yes      Yes      Yes     No    torch.nn.functional
nll_loss                                 Yes      Yes      Yes     No    torch.nn.functional
one_hot                                  No       No       No      No    torch.nn.functional
pad                                      Yes      Yes      No      Yes   torch.nn.functional
pixel_shuffle                            Yes      Yes      Yes     No    torch.nn.functional
prelu                                    Yes      Yes      No      No    torch.nn.functional
relu                                     Yes      Yes      Yes     No    torch.nn.functional
relu\_                                   Yes      Yes      Yes     No    torch.nn.functional
rrelu                                    Yes      Yes      Yes     No    torch.nn.functional
rrelu\_                                  Yes      Yes      Yes     No    torch.nn.functional
sigmoid                                  Yes      Yes      Yes     No    torch.nn.functional
sigmoid\_                                Yes      Yes      Yes     No    torch.nn.functional
silu                                     Yes      Yes      No      No    torch.nn.functional
silu\_                                   Yes      Yes      No      No    torch.nn.functional
smooth_l1_loss                           Yes      Yes      Yes     No    torch.nn.functional
softmax                                  Yes      Yes      Yes     No    torch.nn.functional
softplus                                 Yes      Yes      Yes     No    torch.nn.functional
softshrink                               Yes      Yes      Yes     No    torch.nn.functional
tanh                                     Yes      Yes      Yes     No    torch.nn.functional
threshold                                Yes      Yes      Yes     No    torch.nn.functional
threshold\_                              Yes      Yes      Yes     No    torch.nn.functional
upsample                                 Yes      Yes      Yes     No    torch.nn.functional
upsample_bilinear                        Yes      Yes      Yes     No    torch.nn.functional
upsample_nearest                         Yes      Yes      Yes     No    torch.nn.functional
cross                                    Yes      Yes      Yes     Yes   torch.linalg
vector_norm                              Yes      Yes      Yes     No    torch.linalg
_adaptive_avg_pool2d                     Yes      Yes      Yes     No    torch
_adaptive_avg_pool3d                     Yes      Yes      Yes     No    torch
_add_relu                                Yes      Yes      Yes     No    torch
_add_relu\_                              Yes      Yes      Yes     No    torch
_addmm_activation                        Yes      Yes      Yes     No    torch
_aminmax                                 Yes      Yes      Yes     Yes   torch
_ctc_loss                                Yes      No       No      No    torch
_efficientzerotensor                     Yes      Yes      Yes     Yes   torch
_foreach_abs                             Yes      Yes      Yes     No    torch
_foreach_abs\_                           Yes      Yes      Yes     No    torch
_foreach_acos                            Yes      Yes      Yes     No    torch
_foreach_acos\_                          Yes      Yes      Yes     No    torch
_foreach_add                             Yes      Yes      Yes     Yes   torch
_foreach_add\_                           Yes      Yes      Yes     Yes   torch
_foreach_addcdiv                         Yes      Yes      Yes     No    torch
_foreach_addcdiv\_                       Yes      Yes      Yes     No    torch
_foreach_addcmul                         Yes      Yes      Yes     No    torch
_foreach_addcmul\_                       Yes      Yes      Yes     No    torch
_foreach_asin                            Yes      Yes      Yes     No    torch
_foreach_asin\_                          Yes      Yes      Yes     No    torch
_foreach_atan                            Yes      Yes      Yes     No    torch
_foreach_atan\_                          Yes      Yes      Yes     No    torch
_foreach_ceil                            Yes      Yes      Yes     No    torch
_foreach_ceil\_                          Yes      Yes      Yes     No    torch
_foreach_clamp_max                       Yes      Yes      Yes     Yes   torch
_foreach_clamp_max\_                     Yes      Yes      Yes     Yes   torch
_foreach_clamp_min                       Yes      Yes      Yes     Yes   torch
_foreach_clamp_min\_                     Yes      Yes      Yes     Yes   torch
_foreach_cos                             Yes      Yes      Yes     No    torch
_foreach_cos\_                           Yes      Yes      Yes     No    torch
_foreach_cosh                            Yes      Yes      Yes     No    torch
_foreach_cosh\_                          Yes      Yes      Yes     No    torch
_foreach_div                             Yes      Yes      Yes     No    torch
_foreach_div\_                           Yes      Yes      Yes     No    torch
_foreach_erf                             Yes      Yes      Yes     No    torch
_foreach_erf\_                           Yes      Yes      Yes     No    torch
_foreach_erfc                            Yes      Yes      Yes     Yes   torch
_foreach_erfc\_                          Yes      Yes      Yes     Yes   torch
_foreach_exp                             Yes      Yes      Yes     No    torch
_foreach_exp\_                           Yes      Yes      Yes     No    torch
_foreach_expm1                           Yes      Yes      Yes     No    torch
_foreach_expm1\_                         Yes      Yes      Yes     No    torch
_foreach_floor                           Yes      Yes      Yes     No    torch
_foreach_floor\_                         Yes      Yes      Yes     No    torch
_foreach_frac                            Yes      Yes      Yes     No    torch
_foreach_frac\_                          Yes      Yes      Yes     No    torch
_foreach_lerp                            Yes      Yes      Yes     Yes   torch
_foreach_lerp\_                          Yes      Yes      Yes     Yes   torch
_foreach_lgamma                          Yes      Yes      Yes     No    torch
_foreach_lgamma\_                        Yes      Yes      Yes     No    torch
_foreach_log                             Yes      Yes      Yes     No    torch
_foreach_log10                           Yes      Yes      Yes     No    torch
_foreach_log10\_                         Yes      Yes      Yes     No    torch
_foreach_log1p                           Yes      Yes      Yes     No    torch
_foreach_log1p\_                         Yes      Yes      Yes     No    torch
_foreach_log2                            Yes      Yes      Yes     No    torch
_foreach_log2\_                          Yes      Yes      Yes     No    torch
_foreach_log\_                           Yes      Yes      Yes     No    torch
_foreach_maximum                         Yes      Yes      Yes     Yes   torch
_foreach_maximum\_                       Yes      Yes      Yes     Yes   torch
_foreach_minimum                         Yes      Yes      Yes     Yes   torch
_foreach_minimum\_                       Yes      Yes      Yes     Yes   torch
_foreach_mul                             Yes      Yes      Yes     Yes   torch
_foreach_mul\_                           Yes      Yes      Yes     Yes   torch
_foreach_neg                             Yes      Yes      Yes     No    torch
_foreach_neg\_                           Yes      Yes      Yes     No    torch
_foreach_pow                             Yes      Yes      Yes     No    torch
_foreach_pow\_                           Yes      Yes      Yes     No    torch
_foreach_reciprocal                      Yes      Yes      Yes     Yes   torch
_foreach_reciprocal\_                    Yes      Yes      Yes     Yes   torch
_foreach_round                           Yes      Yes      Yes     No    torch
_foreach_round\_                         Yes      Yes      Yes     No    torch
_foreach_sigmoid                         Yes      Yes      Yes     No    torch
_foreach_sigmoid\_                       Yes      Yes      Yes     No    torch
_foreach_sign                            Yes      Yes      Yes     No    torch
_foreach_sign\_                          Yes      Yes      Yes     No    torch
_foreach_sin                             Yes      Yes      Yes     No    torch
_foreach_sin\_                           Yes      Yes      Yes     No    torch
_foreach_sinh                            Yes      Yes      Yes     No    torch
_foreach_sinh\_                          Yes      Yes      Yes     No    torch
_foreach_sqrt                            Yes      Yes      Yes     No    torch
_foreach_sqrt\_                          Yes      Yes      Yes     No    torch
_foreach_sub                             Yes      Yes      Yes     Yes   torch
_foreach_sub\_                           Yes      Yes      Yes     Yes   torch
_foreach_tan                             Yes      Yes      Yes     No    torch
_foreach_tan\_                           Yes      Yes      Yes     No    torch
_foreach_tanh                            Yes      Yes      Yes     No    torch
_foreach_tanh\_                          Yes      Yes      Yes     No    torch
_foreach_trunc                           Yes      Yes      Yes     No    torch
_foreach_trunc\_                         Yes      Yes      Yes     No    torch
_foreach_zero\_                          Yes      Yes      Yes     Yes   torch
_fused_dropout                           Yes      Yes      Yes     No    torch
_logcumsumexp                            Yes      Yes      Yes     No    torch
_masked_scale                            Yes      Yes      Yes     Yes   torch
_native_batch_norm_legit                 Yes      Yes      Yes     No    torch
_native_batch_norm_legit_no_training     Yes      Yes      Yes     No    torch
_resize_output\_                         Yes      Yes      Yes     Yes   torch
_unique                                  Yes      No       No      No    torch
_unique2                                 Yes      No       No      No    torch
_weight_norm_interface                   Yes      Yes      Yes     No    torch
abs                                      Yes      Yes      Yes     No    torch
abs\_                                    Yes      Yes      Yes     No    torch
absolute                                 Yes      Yes      Yes     No    torch
acos                                     Yes      Yes      No      No    torch
acos\_                                   Yes      Yes      No      No    torch
acosh                                    Yes      Yes      No      No    torch
acosh\_                                  Yes      Yes      No      No    torch
add                                      Yes      Yes      Yes     Yes   torch
addbmm                                   Yes      Yes      Yes     Yes   torch
addcdiv                                  Yes      Yes      Yes     No    torch
addcmul                                  Yes      Yes      Yes     No    torch
addmm                                    Yes      Yes      Yes     Yes   torch
addmv                                    Yes      Yes      Yes     No    torch
addmv\_                                  Yes      Yes      Yes     No    torch
addr                                     Yes      Yes      Yes     No    torch
all                                      Yes      Yes      Yes     No    torch
amax                                     Yes      Yes      Yes     Yes   torch
amin                                     Yes      Yes      Yes     Yes   torch
aminmax                                  Yes      Yes      Yes     Yes   torch
any                                      Yes      Yes      Yes     Yes   torch
arange                                   Yes      Yes      Yes     No    torch
arccos                                   Yes      Yes      No      No    torch
arccos\_                                 Yes      Yes      No      No    torch
arccosh                                  Yes      Yes      No      No    torch
arccosh\_                                Yes      Yes      No      No    torch
arcsin                                   Yes      Yes      No      No    torch
arcsin\_                                 Yes      Yes      No      No    torch
arcsinh                                  Yes      Yes      No      No    torch
arcsinh\_                                Yes      Yes      No      No    torch
arctan                                   Yes      Yes      No      No    torch
arctan2                                  Yes      Yes      Yes     No    torch
arctan\_                                 Yes      Yes      No      No    torch
arctanh                                  Yes      Yes      No      No    torch
arctanh\_                                Yes      Yes      No      No    torch
argmax                                   Yes      Yes      Yes     No    torch
argmin                                   Yes      Yes      Yes     No    torch
argsort                                  Yes      Yes      Yes     No    torch
as_strided                               Yes      Yes      Yes     No    torch
as_strided\_                             Yes      Yes      Yes     No    torch
asin                                     Yes      Yes      No      No    torch
asin\_                                   Yes      Yes      No      No    torch
asinh                                    Yes      Yes      No      No    torch
asinh\_                                  Yes      Yes      No      No    torch
atan                                     Yes      Yes      No      No    torch
atan2                                    Yes      Yes      Yes     No    torch
atan\_                                   Yes      Yes      No      No    torch
atanh                                    Yes      Yes      No      No    torch
atanh\_                                  Yes      Yes      No      No    torch
baddbmm                                  Yes      Yes      Yes     Yes   torch
batch_norm                               Yes      Yes      Yes     No    torch
bernoulli                                Yes      Yes      Yes     No    torch
binary_cross_entropy_with_logits         Yes      Yes      Yes     No    torch
bincount                                 No       No       No      No    torch
bitwise_and                              No       No       No      No    torch
bitwise_left_shift                       No       No       No      No    torch
bitwise_not                              No       No       No      No    torch
bitwise_or                               No       No       No      No    torch
bitwise_right_shift                      No       No       No      No    torch
bitwise_xor                              No       No       No      No    torch
bmm                                      Yes      Yes      Yes     Yes   torch
broadcast_tensors                        Yes      Yes      No      No    torch
bucketize                                Yes      Yes      Yes     No    torch
cat                                      Yes      Yes      Yes     Yes   torch
ceil                                     Yes      Yes      Yes     No    torch
ceil\_                                   Yes      Yes      Yes     No    torch
channel_shuffle                          Yes      Yes      Yes     No    torch
cholesky                                 Yes      No       No      No    torch
chunk                                    Yes      Yes      No      No    torch
clamp                                    Yes      Yes      Yes     Yes   torch
clamp\_                                  Yes      Yes      Yes     Yes   torch
clamp_max                                Yes      Yes      Yes     Yes   torch
clamp_max\_                              Yes      Yes      Yes     Yes   torch
clamp_min                                Yes      Yes      Yes     Yes   torch
clamp_min\_                              Yes      Yes      Yes     Yes   torch
clip                                     Yes      Yes      No      No    torch
clip\_                                   Yes      Yes      No      No    torch
clone                                    Yes      Yes      Yes     Yes   torch
conj                                     Yes      Yes      No      No    torch
constant_pad_nd                          Yes      Yes      Yes     Yes   torch
conv1d                                   Yes      Yes      Yes     Yes   torch
conv2d                                   Yes      Yes      Yes     Yes   torch
conv3d                                   Yes      Yes      Yes     Yes   torch
conv_transpose1d                         Yes      Yes      Yes     Yes   torch
conv_transpose2d                         Yes      Yes      Yes     Yes   torch
conv_transpose3d                         Yes      Yes      Yes     Yes   torch
copysign                                 Yes      Yes      Yes     No    torch
cos                                      Yes      Yes      Yes     No    torch
cos\_                                    Yes      Yes      Yes     No    torch
cosh                                     Yes      Yes      Yes     No    torch
cosh\_                                   Yes      Yes      Yes     No    torch
count_nonzero                            Yes      Yes      Yes     No    torch
cross                                    Yes      Yes      Yes     Yes   torch
cumprod                                  Yes      Yes      Yes     No    torch
cumsum                                   Yes      Yes      Yes     No    torch
diag                                     Yes      Yes      Yes     No    torch
div                                      Yes      Yes      Yes     No    torch
divide                                   Yes      Yes      Yes     No    torch
dot                                      Yes      Yes      Yes     Yes   torch
dropout                                  Yes      Yes      Yes     No    torch
embedding                                Yes      Yes      Yes     Yes   torch
embedding_bag                            Yes      Yes      Yes     No    torch
embedding_renorm\_                       Yes      Yes      Yes     No    torch
empty                                    Yes      Yes      Yes     Yes   torch
empty_like                               Yes      Yes      Yes     Yes   torch
empty_strided                            Yes      Yes      Yes     Yes   torch
eq                                       Yes      Yes      Yes     No    torch
equal                                    Yes      Yes      Yes     No    torch
erf                                      Yes      Yes      Yes     No    torch
erf\_                                    Yes      Yes      Yes     No    torch
erfc                                     Yes      Yes      No      No    torch
erfc\_                                   Yes      Yes      No      No    torch
erfinv                                   Yes      No       No      No    torch
exp                                      Yes      Yes      Yes     No    torch
exp2                                     Yes      Yes      Yes     No    torch
exp2\_                                   Yes      Yes      Yes     No    torch
exp\_                                    Yes      Yes      Yes     No    torch
expm1                                    Yes      Yes      No      No    torch
expm1\_                                  Yes      Yes      No      No    torch
eye                                      Yes      Yes      Yes     No    torch
fill                                     Yes      Yes      Yes     Yes   torch
fill\_                                   Yes      Yes      Yes     Yes   torch
flatten                                  Yes      Yes      No      No    torch
flip                                     Yes      Yes      Yes     Yes   torch
floor                                    Yes      Yes      Yes     No    torch
floor\_                                  Yes      Yes      Yes     No    torch
floor_divide                             Yes      Yes      Yes     No    torch
fmax                                     Yes      Yes      Yes     Yes   torch
fmin                                     Yes      Yes      Yes     Yes   torch
fmod                                     Yes      Yes      Yes     No    torch
frac                                     Yes      Yes      Yes     No    torch
frac\_                                   Yes      Yes      Yes     No    torch
frexp                                    Yes      Yes      Yes     No    torch
full                                     Yes      Yes      Yes     Yes   torch
full_like                                Yes      Yes      Yes     Yes   torch
gather                                   Yes      Yes      Yes     No    torch
ge                                       Yes      Yes      Yes     Yes   torch
greater                                  Yes      Yes      Yes     Yes   torch
greater_equal                            Yes      Yes      Yes     Yes   torch
grid_sampler_2d                          Yes      Yes      Yes     No    torch
grid_sampler_3d                          Yes      Yes      Yes     No    torch
gt                                       Yes      Yes      Yes     Yes   torch
hardshrink                               Yes      Yes      Yes     No    torch
heaviside                                Yes      Yes      Yes     No    torch
hypot                                    Yes      Yes      Yes     No    torch
index_copy                               Yes      Yes      Yes     Yes   torch
index_fill                               Yes      Yes      Yes     Yes   torch
index_put                                Yes      Yes      No      No    torch
index_put\_                              Yes      Yes      No      No    torch
index_reduce                             Yes      Yes      No      No    torch
index_select                             Yes      Yes      Yes     Yes   torch
instance_norm                            Yes      Yes      No      No    torch
is_complex                               Yes      Yes      No      No    torch
is_floating_point                        Yes      Yes      No      No    torch
is_nonzero                               Yes      Yes      No      No    torch
isfinite                                 Yes      Yes      Yes     No    torch
isin                                     Yes      Yes      Yes     No    torch
isinf                                    Yes      Yes      Yes     No    torch
isnan                                    Yes      Yes      Yes     No    torch
isneginf                                 Yes      Yes      Yes     No    torch
isposinf                                 Yes      Yes      Yes     No    torch
kl_div                                   Yes      Yes      No      No    torch
kthvalue                                 Yes      Yes      Yes     No    torch
layer_norm                               Yes      Yes      Yes     No    torch
le                                       Yes      Yes      Yes     Yes   torch
lerp                                     Yes      Yes      Yes     Yes   torch
less                                     Yes      Yes      Yes     Yes   torch
less_equal                               Yes      Yes      Yes     Yes   torch
lgamma                                   Yes      Yes      Yes     No    torch
linspace                                 Yes      Yes      Yes     Yes   torch
log                                      Yes      Yes      Yes     No    torch
log10                                    Yes      Yes      Yes     No    torch
log10\_                                  Yes      Yes      Yes     No    torch
log1p                                    Yes      Yes      Yes     No    torch
log1p\_                                  Yes      Yes      Yes     No    torch
log2                                     Yes      Yes      Yes     No    torch
log2\_                                   Yes      Yes      Yes     No    torch
log\_                                    Yes      Yes      Yes     No    torch
log_softmax                              Yes      Yes      Yes     No    torch
logaddexp                                Yes      Yes      No      No    torch
logaddexp2                               Yes      Yes      Yes     No    torch
logcumsumexp                             Yes      Yes      Yes     No    torch
logical_and                              Yes      Yes      Yes     No    torch
logical_not                              Yes      Yes      Yes     No    torch
logical_or                               Yes      Yes      Yes     No    torch
logical_xor                              Yes      Yes      Yes     No    torch
logit                                    Yes      Yes      Yes     No    torch
logit\_                                  Yes      Yes      Yes     No    torch
logspace                                 Yes      Yes      Yes     Yes   torch
logsumexp                                Yes      Yes      No      No    torch
lt                                       Yes      Yes      Yes     Yes   torch
masked_fill                              Yes      Yes      Yes     Yes   torch
masked_scatter                           Yes      Yes      Yes     No    torch
masked_select                            Yes      Yes      No      No    torch
matmul                                   Yes      Yes      No      No    torch
max                                      Yes      Yes      Yes     Yes   torch
max_pool2d                               Yes      Yes      Yes     No    torch
max_pool3d                               Yes      Yes      Yes     No    torch
maximum                                  Yes      Yes      Yes     Yes   torch
mean                                     Yes      Yes      Yes     No    torch
median                                   Yes      Yes      Yes     No    torch
meshgrid                                 Yes      Yes      No      No    torch
min                                      Yes      Yes      Yes     Yes   torch
minimum                                  Yes      Yes      Yes     Yes   torch
mm                                       Yes      Yes      Yes     Yes   torch
mul                                      Yes      Yes      Yes     Yes   torch
multinomial                              Yes      Yes      Yes     No    torch
mv                                       Yes      Yes      Yes     Yes   torch
nan_to_num                               Yes      Yes      Yes     No    torch
nan_to_num\_                             Yes      Yes      Yes     No    torch
nansum                                   Yes      Yes      Yes     No    torch
narrow                                   Yes      No       No      No    torch
native_dropout                           Yes      Yes      Yes     No    torch
native_group_norm                        Yes      Yes      Yes     No    torch
native_layer_norm                        Yes      Yes      Yes     No    torch
ne                                       Yes      Yes      Yes     Yes   torch
neg                                      Yes      Yes      Yes     No    torch
neg\_                                    Yes      Yes      Yes     No    torch
nextafter                                Yes      Yes      Yes     No    torch
nonzero                                  Yes      Yes      No      No    torch
norm                                     Yes      Yes      Yes     No    torch
normal                                   Yes      Yes      Yes     No    torch
not_equal                                Yes      Yes      Yes     Yes   torch
ones                                     Yes      Yes      No      Yes   torch
ones_like                                Yes      Yes      No      Yes   torch
permute                                  Yes      Yes      Yes     Yes   torch
pixel_shuffle                            Yes      Yes      Yes     No    torch
poisson                                  Yes      Yes      Yes     No    torch
pow                                      Yes      Yes      Yes     No    torch
prelu                                    Yes      Yes      No      No    torch
prod                                     Yes      Yes      Yes     No    torch
put                                      Yes      Yes      Yes     No    torch
randperm                                 Yes      Yes      Yes     Yes   torch
reciprocal                               Yes      Yes      Yes     Yes   torch
reciprocal\_                             Yes      Yes      Yes     Yes   torch
relu                                     Yes      Yes      Yes     No    torch
relu\_                                   Yes      Yes      Yes     No    torch
remainder                                Yes      Yes      Yes     No    torch
reshape                                  Yes      Yes      Yes     Yes   torch
resolve_conj                             Yes      Yes      No      No    torch
resolve_neg                              Yes      Yes      No      No    torch
result_type                              Yes      Yes      No      No    torch
roll                                     Yes      Yes      No      Yes   torch
round                                    Yes      Yes      Yes     No    torch
round\_                                  Yes      Yes      Yes     No    torch
rsqrt                                    Yes      Yes      Yes     No    torch
rsqrt\_                                  Yes      Yes      Yes     No    torch
rsub                                     Yes      Yes      Yes     Yes   torch
scalar_tensor                            Yes      Yes      Yes     Yes   torch
scatter                                  Yes      Yes      Yes     Yes   torch
scatter_add                              Yes      Yes      Yes     No    torch
scatter_reduce                           Yes      Yes      Yes     No    torch
searchsorted                             Yes      Yes      Yes     No    torch
select                                   Yes      Yes      Yes     Yes   torch
sgn                                      Yes      Yes      Yes     No    torch
sigmoid                                  Yes      Yes      Yes     No    torch
sigmoid\_                                Yes      Yes      Yes     No    torch
sign                                     Yes      Yes      Yes     No    torch
signbit                                  Yes      Yes      Yes     No    torch
sin                                      Yes      Yes      Yes     No    torch
sin\_                                    Yes      Yes      Yes     No    torch
sinc                                     Yes      Yes      Yes     No    torch
sinc\_                                   Yes      Yes      Yes     No    torch
sinh                                     Yes      Yes      Yes     No    torch
sinh\_                                   Yes      Yes      Yes     No    torch
softmax                                  Yes      Yes      Yes     No    torch
sort                                     Yes      Yes      Yes     No    torch
split_with_sizes                         Yes      Yes      No      Yes   torch
sqrt                                     Yes      Yes      Yes     No    torch
sqrt\_                                   Yes      Yes      Yes     No    torch
square                                   Yes      Yes      No      No    torch
square\_                                 Yes      Yes      No      No    torch
squeeze                                  Yes      Yes      Yes     Yes   torch
stack                                    Yes      Yes      No      No    torch
std                                      Yes      Yes      Yes     No    torch
std_mean                                 Yes      Yes      Yes     No    torch
sub                                      Yes      Yes      Yes     Yes   torch
sum                                      Yes      Yes      Yes     Yes   torch
t                                        Yes      Yes      Yes     Yes   torch
take                                     Yes      Yes      Yes     No    torch
tan                                      Yes      Yes      No      No    torch
tan\_                                    Yes      Yes      No      No    torch
tanh                                     Yes      Yes      Yes     No    torch
tanh\_                                   Yes      Yes      Yes     No    torch
threshold                                Yes      Yes      Yes     No    torch
threshold\_                              Yes      Yes      Yes     No    torch
topk                                     Yes      Yes      Yes     No    torch
trace                                    Yes      Yes      Yes     No    torch
transpose                                Yes      Yes      Yes     Yes   torch
tril                                     Yes      Yes      Yes     No    torch
tril_indices                             No       No       No      No    torch
triu                                     Yes      Yes      Yes     No    torch
triu_indices                             No       No       No      No    torch
trunc                                    Yes      Yes      Yes     No    torch
trunc\_                                  Yes      Yes      Yes     No    torch
unbind                                   Yes      Yes      No      No    torch
unique                                   Yes      No       No      No    torch
unsqueeze                                Yes      Yes      Yes     Yes   torch
var                                      Yes      Yes      Yes     No    torch
var_mean                                 Yes      Yes      Yes     No    torch
vdot                                     Yes      Yes      Yes     Yes   torch
where                                    Yes      Yes      Yes     Yes   torch
xlogy                                    Yes      Yes      Yes     No    torch
xlogy\_                                  Yes      Yes      Yes     No    torch
zero\_                                   Yes      Yes      Yes     Yes   torch
zeros                                    Yes      Yes      Yes     Yes   torch
zeros_like                               Yes      Yes      Yes     Yes   torch
glu_jvp                                  Yes      Yes      Yes     No    torch.ops.aten
AdaptiveAvgPool1d                        Yes      Yes      Yes     No    torch.nn
AdaptiveAvgPool2d                        Yes      Yes      Yes     No    torch.nn
AdaptiveAvgPool3d                        Yes      Yes      Yes     No    torch.nn
AvgPool1d                                Yes      Yes      Yes     No    torch.nn
AvgPool2d                                Yes      Yes      Yes     No    torch.nn
AvgPool3d                                Yes      Yes      Yes     No    torch.nn
BCELoss                                  Yes      Yes      Yes     No    torch.nn
BCEWithLogitsLoss                        Yes      Yes      Yes     No    torch.nn
BatchNorm1d                              Yes      Yes      Yes     No    torch.nn
BatchNorm2d                              Yes      Yes      Yes     No    torch.nn
ChannelShuffle                           Yes      Yes      Yes     No    torch.nn
ConstantPad1d                            Yes      Yes      Yes     No    torch.nn
Conv1d                                   Yes      Yes      Yes     Yes   torch.nn
Conv2d                                   Yes      Yes      Yes     Yes   torch.nn
Conv3d                                   Yes      Yes      Yes     Yes   torch.nn
ConvTranspose1d                          Yes      Yes      Yes     Yes   torch.nn
ConvTranspose2d                          Yes      Yes      Yes     Yes   torch.nn
ConvTranspose3d                          Yes      Yes      Yes     Yes   torch.nn
CrossEntropyLoss                         Yes      Yes      No      No    torch.nn
Dropout                                  Yes      Yes      Yes     No    torch.nn
ELU                                      Yes      Yes      Yes     No    torch.nn
EmbeddingBag                             Yes      Yes      Yes     No    torch.nn
GELU                                     Yes      Yes      Yes     No    torch.nn
GLU                                      Yes      Yes      Yes     No    torch.nn
Hardshrink                               Yes      Yes      Yes     No    torch.nn
Hardsigmoid                              Yes      Yes      Yes     No    torch.nn
Hardswish                                Yes      Yes      No      No    torch.nn
Hardtanh                                 Yes      Yes      Yes     Yes   torch.nn
HuberLoss                                Yes      Yes      Yes     No    torch.nn
InstanceNorm2d                           Yes      Yes      No      No    torch.nn
KLDivLoss                                Yes      Yes      No      No    torch.nn
LayerNorm                                Yes      Yes      Yes     No    torch.nn
LeakyReLU                                Yes      Yes      Yes     No    torch.nn
Linear                                   Yes      Yes      Yes     No    torch.nn
LogSigmoid                               Yes      Yes      Yes     No    torch.nn
LogSoftmax                               Yes      Yes      Yes     No    torch.nn
MSELoss                                  Yes      Yes      Yes     No    torch.nn
MaxPool2d                                Yes      Yes      Yes     No    torch.nn
MaxPool3d                                Yes      Yes      Yes     No    torch.nn
Mish                                     Yes      Yes      Yes     No    torch.nn
MultiLabelMarginLoss                     Yes      Yes      Yes     No    torch.nn
MultiMarginLoss                          Yes      Yes      Yes     No    torch.nn
NLLLoss                                  Yes      Yes      Yes     No    torch.nn
PReLU                                    Yes      Yes      No      No    torch.nn
PixelShuffle                             Yes      Yes      Yes     No    torch.nn
RReLU                                    Yes      Yes      Yes     No    torch.nn
ReLU                                     Yes      Yes      Yes     No    torch.nn
ReflectionPad1d                          Yes      Yes      Yes     Yes   torch.nn
ReflectionPad2d                          Yes      Yes      Yes     Yes   torch.nn
ReflectionPad3d                          Yes      Yes      Yes     Yes   torch.nn
ReplicationPad1d                         Yes      Yes      Yes     Yes   torch.nn
ReplicationPad2d                         Yes      Yes      Yes     Yes   torch.nn
ReplicationPad3d                         Yes      Yes      Yes     Yes   torch.nn
SiLU                                     Yes      Yes      No      No    torch.nn
Sigmoid                                  Yes      Yes      Yes     No    torch.nn
SmoothL1Loss                             Yes      Yes      Yes     No    torch.nn
Softmax                                  Yes      Yes      Yes     No    torch.nn
Softplus                                 Yes      Yes      Yes     No    torch.nn
Softshrink                               Yes      Yes      Yes     No    torch.nn
Tanh                                     Yes      Yes      Yes     No    torch.nn
Threshold                                Yes      Yes      Yes     No    torch.nn
Upsample                                 Yes      Yes      Yes     No    torch.nn
UpsamplingNearest2d                      Yes      Yes      Yes     No    torch.nn
weight_norm                              Yes      Yes      No      No    torch.nn.utils
T                                        Yes      Yes      Yes     Yes   torch.Tensor
__and__                                  No       No       No      No    torch.Tensor
__iand__                                 No       No       No      No    torch.Tensor
__ilshift__                              No       No       No      No    torch.Tensor
__ior__                                  No       No       No      No    torch.Tensor
__irshift__                              No       No       No      No    torch.Tensor
__ixor__                                 No       No       No      No    torch.Tensor
__lshift__                               No       No       No      No    torch.Tensor
__or__                                   No       No       No      No    torch.Tensor
__rshift__                               No       No       No      No    torch.Tensor
__xor__                                  No       No       No      No    torch.Tensor
abs                                      Yes      Yes      Yes     No    torch.Tensor
abs\_                                    Yes      Yes      Yes     No    torch.Tensor
absolute                                 Yes      Yes      Yes     No    torch.Tensor
absolute\_                               Yes      Yes      Yes     No    torch.Tensor
acos                                     Yes      Yes      No      No    torch.Tensor
acos\_                                   Yes      Yes      No      No    torch.Tensor
acosh                                    Yes      Yes      No      No    torch.Tensor
acosh\_                                  Yes      Yes      No      No    torch.Tensor
add                                      Yes      Yes      Yes     Yes   torch.Tensor
add\_                                    Yes      Yes      Yes     Yes   torch.Tensor
addbmm                                   Yes      Yes      Yes     Yes   torch.Tensor
addbmm\_                                 Yes      Yes      Yes     Yes   torch.Tensor
addcdiv                                  Yes      Yes      Yes     No    torch.Tensor
addcdiv\_                                Yes      Yes      Yes     No    torch.Tensor
addcmul                                  Yes      Yes      Yes     No    torch.Tensor
addcmul\_                                Yes      Yes      Yes     No    torch.Tensor
addmm                                    Yes      Yes      Yes     Yes   torch.Tensor
addmm\_                                  Yes      Yes      Yes     Yes   torch.Tensor
addmv                                    Yes      Yes      Yes     No    torch.Tensor
addmv\_                                  Yes      Yes      Yes     No    torch.Tensor
addr                                     Yes      Yes      Yes     No    torch.Tensor
addr\_                                   Yes      Yes      Yes     No    torch.Tensor
all                                      Yes      Yes      Yes     No    torch.Tensor
amax                                     Yes      Yes      Yes     Yes   torch.Tensor
amin                                     Yes      Yes      Yes     Yes   torch.Tensor
aminmax                                  Yes      Yes      Yes     Yes   torch.Tensor
any                                      Yes      Yes      Yes     Yes   torch.Tensor
arccos                                   Yes      Yes      No      No    torch.Tensor
arccos\_                                 Yes      Yes      No      No    torch.Tensor
arccosh                                  Yes      Yes      No      No    torch.Tensor
arccosh\_                                Yes      Yes      No      No    torch.Tensor
arcsin                                   Yes      Yes      No      No    torch.Tensor
arcsin\_                                 Yes      Yes      No      No    torch.Tensor
arcsinh                                  Yes      Yes      No      No    torch.Tensor
arcsinh\_                                Yes      Yes      No      No    torch.Tensor
arctan                                   Yes      Yes      No      No    torch.Tensor
arctan2                                  Yes      Yes      Yes     No    torch.Tensor
arctan2\_                                Yes      Yes      Yes     No    torch.Tensor
arctan\_                                 Yes      Yes      No      No    torch.Tensor
arctanh                                  Yes      Yes      No      No    torch.Tensor
arctanh\_                                Yes      Yes      No      No    torch.Tensor
argmax                                   Yes      Yes      Yes     No    torch.Tensor
argmin                                   Yes      Yes      Yes     No    torch.Tensor
argsort                                  Yes      Yes      Yes     No    torch.Tensor
as_strided                               Yes      Yes      Yes     No    torch.Tensor
as_strided\_                             Yes      Yes      Yes     No    torch.Tensor
asin                                     Yes      Yes      No      No    torch.Tensor
asin\_                                   Yes      Yes      No      No    torch.Tensor
asinh                                    Yes      Yes      No      No    torch.Tensor
asinh\_                                  Yes      Yes      No      No    torch.Tensor
atan                                     Yes      Yes      No      No    torch.Tensor
atan2                                    Yes      Yes      Yes     No    torch.Tensor
atan2\_                                  Yes      Yes      Yes     No    torch.Tensor
atan\_                                   Yes      Yes      No      No    torch.Tensor
atanh                                    Yes      Yes      No      No    torch.Tensor
atanh\_                                  Yes      Yes      No      No    torch.Tensor
baddbmm                                  Yes      Yes      Yes     Yes   torch.Tensor
baddbmm\_                                Yes      Yes      Yes     Yes   torch.Tensor
bernoulli                                Yes      Yes      Yes     No    torch.Tensor
bernoulli\_                              Yes      Yes      Yes     No    torch.Tensor
bincount                                 No       No       No      No    torch.Tensor
bitwise_and                              No       No       No      No    torch.Tensor
bitwise_and\_                            No       No       No      No    torch.Tensor
bitwise_left_shift                       No       No       No      No    torch.Tensor
bitwise_left_shift\_                     No       No       No      No    torch.Tensor
bitwise_not                              No       No       No      No    torch.Tensor
bitwise_not\_                            No       No       No      No    torch.Tensor
bitwise_or                               No       No       No      No    torch.Tensor
bitwise_or\_                             No       No       No      No    torch.Tensor
bitwise_right_shift                      No       No       No      No    torch.Tensor
bitwise_right_shift\_                    No       No       No      No    torch.Tensor
bitwise_xor                              No       No       No      No    torch.Tensor
bitwise_xor\_                            No       No       No      No    torch.Tensor
bmm                                      Yes      Yes      Yes     Yes   torch.Tensor
ceil                                     Yes      Yes      Yes     No    torch.Tensor
ceil\_                                   Yes      Yes      Yes     No    torch.Tensor
cholesky                                 Yes      No       No      No    torch.Tensor
chunk                                    Yes      Yes      No      No    torch.Tensor
clamp                                    Yes      Yes      Yes     Yes   torch.Tensor
clamp\_                                  Yes      Yes      Yes     Yes   torch.Tensor
clamp_max                                Yes      Yes      Yes     Yes   torch.Tensor
clamp_max\_                              Yes      Yes      Yes     Yes   torch.Tensor
clamp_min                                Yes      Yes      Yes     Yes   torch.Tensor
clamp_min\_                              Yes      Yes      Yes     Yes   torch.Tensor
clip                                     Yes      Yes      No      No    torch.Tensor
clip\_                                   Yes      Yes      No      No    torch.Tensor
clone                                    Yes      Yes      Yes     Yes   torch.Tensor
conj                                     Yes      Yes      No      No    torch.Tensor
copy\_                                   Yes      Yes      No      Yes   torch.Tensor
copysign                                 Yes      Yes      Yes     No    torch.Tensor
copysign\_                               Yes      Yes      Yes     No    torch.Tensor
cos                                      Yes      Yes      Yes     No    torch.Tensor
cos\_                                    Yes      Yes      Yes     No    torch.Tensor
cosh                                     Yes      Yes      Yes     No    torch.Tensor
cosh\_                                   Yes      Yes      Yes     No    torch.Tensor
count_nonzero                            Yes      Yes      Yes     No    torch.Tensor
cross                                    Yes      Yes      Yes     Yes   torch.Tensor
cumprod                                  Yes      Yes      Yes     No    torch.Tensor
cumprod\_                                Yes      Yes      Yes     No    torch.Tensor
cumsum                                   Yes      Yes      Yes     No    torch.Tensor
cumsum\_                                 Yes      Yes      Yes     No    torch.Tensor
diag                                     Yes      Yes      Yes     No    torch.Tensor
div                                      Yes      Yes      Yes     No    torch.Tensor
div\_                                    Yes      Yes      Yes     No    torch.Tensor
divide                                   Yes      Yes      Yes     No    torch.Tensor
divide\_                                 Yes      Yes      Yes     No    torch.Tensor
dot                                      Yes      Yes      Yes     Yes   torch.Tensor
eq                                       Yes      Yes      Yes     No    torch.Tensor
eq\_                                     Yes      Yes      Yes     No    torch.Tensor
equal                                    Yes      Yes      Yes     No    torch.Tensor
erf                                      Yes      Yes      Yes     No    torch.Tensor
erf\_                                    Yes      Yes      Yes     No    torch.Tensor
erfc                                     Yes      Yes      No      No    torch.Tensor
erfc\_                                   Yes      Yes      No      No    torch.Tensor
erfinv                                   Yes      No       No      No    torch.Tensor
erfinv\_                                 Yes      No       No      No    torch.Tensor
exp                                      Yes      Yes      Yes     No    torch.Tensor
exp2                                     Yes      Yes      Yes     No    torch.Tensor
exp2\_                                   Yes      Yes      Yes     No    torch.Tensor
exp\_                                    Yes      Yes      Yes     No    torch.Tensor
expand                                   Yes      Yes      Yes     No    torch.Tensor
expand_as                                Yes      Yes      Yes     No    torch.Tensor
expm1                                    Yes      Yes      No      No    torch.Tensor
expm1\_                                  Yes      Yes      No      No    torch.Tensor
exponential\_                            Yes      Yes      Yes     No    torch.Tensor
fill\_                                   Yes      Yes      Yes     Yes   torch.Tensor
flatten                                  Yes      Yes      No      No    torch.Tensor
flip                                     Yes      Yes      Yes     Yes   torch.Tensor
floor                                    Yes      Yes      Yes     No    torch.Tensor
floor\_                                  Yes      Yes      Yes     No    torch.Tensor
floor_divide                             Yes      Yes      Yes     No    torch.Tensor
floor_divide\_                           Yes      Yes      Yes     No    torch.Tensor
fmax                                     Yes      Yes      Yes     Yes   torch.Tensor
fmin                                     Yes      Yes      Yes     Yes   torch.Tensor
fmod                                     Yes      Yes      Yes     No    torch.Tensor
fmod\_                                   Yes      Yes      Yes     No    torch.Tensor
frac                                     Yes      Yes      Yes     No    torch.Tensor
frac\_                                   Yes      Yes      Yes     No    torch.Tensor
frexp                                    Yes      Yes      Yes     No    torch.Tensor
gather                                   Yes      Yes      Yes     No    torch.Tensor
ge                                       Yes      Yes      Yes     Yes   torch.Tensor
ge\_                                     Yes      Yes      Yes     Yes   torch.Tensor
geometric\_                              Yes      Yes      Yes     No    torch.Tensor
greater                                  Yes      Yes      Yes     Yes   torch.Tensor
greater\_                                Yes      Yes      Yes     Yes   torch.Tensor
greater_equal                            Yes      Yes      Yes     Yes   torch.Tensor
greater_equal\_                          Yes      Yes      Yes     Yes   torch.Tensor
gt                                       Yes      Yes      Yes     Yes   torch.Tensor
gt\_                                     Yes      Yes      Yes     Yes   torch.Tensor
hardshrink                               Yes      Yes      Yes     No    torch.Tensor
heaviside                                Yes      Yes      Yes     No    torch.Tensor
heaviside\_                              Yes      Yes      Yes     No    torch.Tensor
hypot                                    Yes      Yes      Yes     No    torch.Tensor
hypot\_                                  Yes      Yes      Yes     No    torch.Tensor
index_add\_                              Yes      Yes      Yes     No    torch.Tensor
index_copy                               Yes      Yes      Yes     Yes   torch.Tensor
index_copy\_                             Yes      Yes      Yes     Yes   torch.Tensor
index_fill                               Yes      Yes      Yes     Yes   torch.Tensor
index_fill\_                             Yes      Yes      Yes     Yes   torch.Tensor
index_put                                Yes      Yes      No      No    torch.Tensor
index_put\_                              Yes      Yes      No      No    torch.Tensor
index_reduce                             Yes      Yes      No      No    torch.Tensor
index_reduce\_                           Yes      Yes      No      No    torch.Tensor
index_select                             Yes      Yes      Yes     Yes   torch.Tensor
is_complex                               Yes      Yes      No      No    torch.Tensor
is_floating_point                        Yes      Yes      No      No    torch.Tensor
is_nonzero                               Yes      Yes      No      No    torch.Tensor
isfinite                                 Yes      Yes      Yes     No    torch.Tensor
isinf                                    Yes      Yes      Yes     No    torch.Tensor
isnan                                    Yes      Yes      Yes     No    torch.Tensor
isneginf                                 Yes      Yes      Yes     No    torch.Tensor
isposinf                                 Yes      Yes      Yes     No    torch.Tensor
item                                     Yes      No       No      No    torch.Tensor
kthvalue                                 Yes      Yes      Yes     No    torch.Tensor
le                                       Yes      Yes      Yes     Yes   torch.Tensor
le\_                                     Yes      Yes      Yes     Yes   torch.Tensor
lerp                                     Yes      Yes      Yes     Yes   torch.Tensor
lerp\_                                   Yes      Yes      Yes     Yes   torch.Tensor
less                                     Yes      Yes      Yes     Yes   torch.Tensor
less\_                                   Yes      Yes      Yes     No    torch.Tensor
less_equal                               Yes      Yes      Yes     Yes   torch.Tensor
less_equal\_                             Yes      Yes      Yes     Yes   torch.Tensor
log                                      Yes      Yes      Yes     No    torch.Tensor
log10                                    Yes      Yes      Yes     No    torch.Tensor
log10\_                                  Yes      Yes      Yes     No    torch.Tensor
log1p                                    Yes      Yes      Yes     No    torch.Tensor
log1p\_                                  Yes      Yes      Yes     No    torch.Tensor
log2                                     Yes      Yes      Yes     No    torch.Tensor
log2\_                                   Yes      Yes      Yes     No    torch.Tensor
log\_                                    Yes      Yes      Yes     No    torch.Tensor
log_normal\_                             Yes      Yes      Yes     Yes   torch.Tensor
log_softmax                              Yes      Yes      Yes     No    torch.Tensor
logaddexp                                Yes      Yes      No      No    torch.Tensor
logaddexp2                               Yes      Yes      Yes     No    torch.Tensor
logcumsumexp                             Yes      Yes      Yes     No    torch.Tensor
logical_and                              Yes      Yes      Yes     No    torch.Tensor
logical_and\_                            Yes      Yes      Yes     No    torch.Tensor
logical_not                              Yes      Yes      Yes     No    torch.Tensor
logical_not\_                            No       No       Yes     No    torch.Tensor
logical_or                               Yes      Yes      Yes     No    torch.Tensor
logical_or\_                             Yes      Yes      Yes     No    torch.Tensor
logical_xor                              Yes      Yes      Yes     No    torch.Tensor
logical_xor\_                            Yes      Yes      Yes     No    torch.Tensor
logit                                    Yes      Yes      Yes     No    torch.Tensor
logit\_                                  Yes      Yes      Yes     No    torch.Tensor
logsumexp                                Yes      Yes      No      No    torch.Tensor
lt                                       Yes      Yes      Yes     Yes   torch.Tensor
lt\_                                     Yes      Yes      Yes     No    torch.Tensor
masked_fill                              Yes      Yes      Yes     Yes   torch.Tensor
masked_fill\_                            Yes      Yes      Yes     Yes   torch.Tensor
masked_scatter                           Yes      Yes      Yes     No    torch.Tensor
masked_scatter\_                         Yes      Yes      Yes     No    torch.Tensor
masked_select                            Yes      Yes      No      No    torch.Tensor
matmul                                   Yes      Yes      No      No    torch.Tensor
max                                      Yes      Yes      Yes     Yes   torch.Tensor
maximum                                  Yes      Yes      Yes     Yes   torch.Tensor
mean                                     Yes      Yes      Yes     No    torch.Tensor
median                                   Yes      Yes      Yes     No    torch.Tensor
min                                      Yes      Yes      Yes     Yes   torch.Tensor
minimum                                  Yes      Yes      Yes     Yes   torch.Tensor
mm                                       Yes      Yes      Yes     Yes   torch.Tensor
mul                                      Yes      Yes      Yes     Yes   torch.Tensor
mul\_                                    Yes      Yes      Yes     Yes   torch.Tensor
multinomial                              Yes      Yes      Yes     No    torch.Tensor
mv                                       Yes      Yes      Yes     Yes   torch.Tensor
nan_to_num                               Yes      Yes      Yes     No    torch.Tensor
nan_to_num\_                             Yes      Yes      Yes     No    torch.Tensor
nansum                                   Yes      Yes      Yes     No    torch.Tensor
narrow                                   Yes      No       No      No    torch.Tensor
ne                                       Yes      Yes      Yes     Yes   torch.Tensor
ne\_                                     Yes      Yes      Yes     Yes   torch.Tensor
neg                                      Yes      Yes      Yes     No    torch.Tensor
neg\_                                    Yes      Yes      Yes     No    torch.Tensor
new_empty                                Yes      Yes      No      No    torch.Tensor
new_empty_strided                        Yes      Yes      No      No    torch.Tensor
new_full                                 Yes      Yes      No      No    torch.Tensor
new_ones                                 Yes      Yes      No      No    torch.Tensor
new_zeros                                Yes      Yes      Yes     Yes   torch.Tensor
nextafter                                Yes      Yes      Yes     No    torch.Tensor
nextafter\_                              Yes      Yes      Yes     No    torch.Tensor
nonzero                                  Yes      Yes      No      No    torch.Tensor
norm                                     Yes      Yes      Yes     No    torch.Tensor
normal\_                                 Yes      Yes      Yes     Yes   torch.Tensor
not_equal                                Yes      Yes      Yes     Yes   torch.Tensor
not_equal\_                              Yes      Yes      Yes     Yes   torch.Tensor
permute                                  Yes      Yes      Yes     Yes   torch.Tensor
pin_memory                               Yes      Yes      No      No    torch.Tensor
pow                                      Yes      Yes      Yes     No    torch.Tensor
pow\_                                    Yes      Yes      Yes     No    torch.Tensor
prod                                     Yes      Yes      Yes     No    torch.Tensor
put                                      Yes      Yes      Yes     No    torch.Tensor
put\_                                    Yes      Yes      Yes     No    torch.Tensor
random\_                                 Yes      Yes      Yes     No    torch.Tensor
reciprocal                               Yes      Yes      Yes     Yes   torch.Tensor
reciprocal\_                             Yes      Yes      Yes     Yes   torch.Tensor
relu                                     Yes      Yes      Yes     No    torch.Tensor
relu\_                                   Yes      Yes      Yes     No    torch.Tensor
remainder                                Yes      Yes      Yes     No    torch.Tensor
remainder\_                              Yes      Yes      Yes     No    torch.Tensor
repeat                                   Yes      Yes      Yes     Yes   torch.Tensor
repeat_interleave                        Yes      Yes      Yes     No    torch.Tensor
reshape                                  Yes      Yes      Yes     Yes   torch.Tensor
resize\_                                 Yes      Yes      Yes     Yes   torch.Tensor
resolve_conj                             Yes      Yes      No      No    torch.Tensor
resolve_neg                              Yes      Yes      No      No    torch.Tensor
roll                                     Yes      Yes      No      Yes   torch.Tensor
round                                    Yes      Yes      Yes     No    torch.Tensor
round\_                                  Yes      Yes      Yes     No    torch.Tensor
rsqrt                                    Yes      Yes      Yes     No    torch.Tensor
rsqrt\_                                  Yes      Yes      Yes     No    torch.Tensor
scatter                                  Yes      Yes      Yes     Yes   torch.Tensor
scatter\_                                Yes      Yes      Yes     Yes   torch.Tensor
scatter_add                              Yes      Yes      Yes     No    torch.Tensor
scatter_add\_                            Yes      Yes      Yes     No    torch.Tensor
scatter_reduce                           Yes      Yes      Yes     No    torch.Tensor
scatter_reduce\_                         Yes      Yes      Yes     No    torch.Tensor
select                                   Yes      Yes      Yes     Yes   torch.Tensor
sgn                                      Yes      Yes      Yes     No    torch.Tensor
sgn\_                                    Yes      Yes      Yes     No    torch.Tensor
sigmoid                                  Yes      Yes      Yes     No    torch.Tensor
sigmoid\_                                Yes      Yes      Yes     No    torch.Tensor
sign                                     Yes      Yes      Yes     No    torch.Tensor
sign\_                                   Yes      Yes      Yes     No    torch.Tensor
signbit                                  Yes      Yes      Yes     No    torch.Tensor
sin                                      Yes      Yes      Yes     No    torch.Tensor
sin\_                                    Yes      Yes      Yes     No    torch.Tensor
sinc                                     Yes      Yes      Yes     No    torch.Tensor
sinc\_                                   Yes      Yes      Yes     No    torch.Tensor
sinh                                     Yes      Yes      Yes     No    torch.Tensor
sinh\_                                   Yes      Yes      Yes     No    torch.Tensor
sort                                     Yes      Yes      Yes     No    torch.Tensor
split_with_sizes                         Yes      Yes      No      Yes   torch.Tensor
sqrt                                     Yes      Yes      Yes     No    torch.Tensor
sqrt\_                                   Yes      Yes      Yes     No    torch.Tensor
square                                   Yes      Yes      No      No    torch.Tensor
square\_                                 Yes      Yes      No      No    torch.Tensor
squeeze                                  Yes      Yes      Yes     Yes   torch.Tensor
squeeze\_                                Yes      Yes      Yes     Yes   torch.Tensor
std                                      Yes      Yes      Yes     No    torch.Tensor
sub                                      Yes      Yes      Yes     Yes   torch.Tensor
sub\_                                    Yes      Yes      Yes     Yes   torch.Tensor
sum                                      Yes      Yes      Yes     Yes   torch.Tensor
t                                        Yes      Yes      Yes     Yes   torch.Tensor
take                                     Yes      Yes      Yes     No    torch.Tensor
tan                                      Yes      Yes      No      No    torch.Tensor
tan\_                                    Yes      Yes      No      No    torch.Tensor
tanh                                     Yes      Yes      Yes     No    torch.Tensor
tanh\_                                   Yes      Yes      Yes     No    torch.Tensor
to                                       Yes      Yes      Yes     Yes   torch.Tensor
topk                                     Yes      Yes      Yes     No    torch.Tensor
trace                                    Yes      Yes      Yes     No    torch.Tensor
transpose                                Yes      Yes      Yes     Yes   torch.Tensor
tril                                     Yes      Yes      Yes     No    torch.Tensor
tril\_                                   Yes      Yes      Yes     No    torch.Tensor
triu                                     Yes      Yes      Yes     No    torch.Tensor
triu\_                                   Yes      Yes      Yes     No    torch.Tensor
trunc                                    Yes      Yes      Yes     No    torch.Tensor
trunc\_                                  Yes      Yes      Yes     No    torch.Tensor
unbind                                   Yes      Yes      No      No    torch.Tensor
uniform\_                                Yes      Yes      Yes     No    torch.Tensor
unique                                   Yes      No       No      No    torch.Tensor
unsqueeze                                Yes      Yes      Yes     Yes   torch.Tensor
unsqueeze\_                              Yes      Yes      Yes     Yes   torch.Tensor
var                                      Yes      Yes      Yes     No    torch.Tensor
vdot                                     Yes      Yes      Yes     Yes   torch.Tensor
view                                     Yes      Yes      Yes     Yes   torch.Tensor
where                                    Yes      Yes      Yes     Yes   torch.Tensor
xlogy                                    Yes      Yes      Yes     No    torch.Tensor
xlogy\_                                  Yes      Yes      Yes     No    torch.Tensor
zero\_                                   Yes      Yes      Yes     Yes   torch.Tensor
entr                                     Yes      Yes      Yes     No    torch.special
erf                                      Yes      Yes      Yes     No    torch.special
erfc                                     Yes      Yes      No      No    torch.special
erfcx                                    Yes      Yes      No      No    torch.special
erfinv                                   Yes      No       No      No    torch.special
exp2                                     Yes      Yes      Yes     No    torch.special
expit                                    Yes      Yes      No      No    torch.special
expm1                                    Yes      Yes      No      No    torch.special
log1p                                    Yes      Yes      Yes     No    torch.special
logit                                    Yes      Yes      Yes     No    torch.special
logsumexp                                Yes      Yes      No      No    torch.special
round                                    Yes      Yes      Yes     No    torch.special
sinc                                     Yes      Yes      Yes     No    torch.special
softmax                                  Yes      Yes      Yes     No    torch.special
xlog1py                                  Yes      Yes      Yes     No    torch.special
xlogy                                    Yes      Yes      Yes     No    torch.special
batched_nms                              Yes      Yes      No      No    torchvision.ops
deform_conv2d                            Yes      No       No      No    torchvision.ops
nms                                      Yes      Yes      No      No    torchvision.ops
roi_align                                Yes      No       No      No    torchvision.ops
_safe_softmax                            Yes      Yes      Yes     No    torch.ops
====================================  ======== ======== ======== ======= ======================

PyTorch Operators Support Summary - Integer types
=================================================

.. rst-class:: datatable

====================================  ========= ========= ========= ======== ========  ======================
**PyTorch Operator**                  **INT64** **INT32** **INT16** **INT8** **BOOL**  **Operator Type**
====================================  ========= ========= ========= ======== ========  ======================
adaptive_avg_pool1d                       No       No        No        No       No     torch.nn.functional
adaptive_avg_pool2d                       No       No        No        No       No     torch.nn.functional
adaptive_avg_pool3d                       No       No        No        No       No     torch.nn.functional
avg_pool1d                                No       No        No        No       No     torch.nn.functional
avg_pool2d                                No       No        No        No       No     torch.nn.functional
avg_pool3d                                No       No        No        No       No     torch.nn.functional
batch_norm                                No       No        No        No       No     torch.nn.functional
binary_cross_entropy                      No       No        No        No       No     torch.nn.functional
binary_cross_entropy_with_logits          No       No        No        No       No     torch.nn.functional
channel_shuffle                           Yes      Yes       Yes       Yes      Yes    torch.nn.functional
conv1d                                    No       No        No        No       No     torch.nn.functional
conv2d                                    No       No        No        No       No     torch.nn.functional
conv3d                                    No       No        No        No       No     torch.nn.functional
conv_transpose1d                          No       No        No        No       No     torch.nn.functional
conv_transpose2d                          No       No        No        No       No     torch.nn.functional
conv_transpose3d                          No       No        No        No       No     torch.nn.functional
dropout                                   No       No        No        No       No     torch.nn.functional
elu                                       No       No        No        No       No     torch.nn.functional
elu\_                                     No       No        No        No       No     torch.nn.functional
embedding                                 Yes      Yes       Yes       Yes      Yes    torch.nn.functional
embedding_bag                             No       No        No        No       No     torch.nn.functional
gelu                                      No       No        No        No       No     torch.nn.functional
gelu\_                                    No       No        No        No       No     torch.nn.functional
glu                                       No       No        No        No       No     torch.nn.functional
grid_sample                               No       No        No        No       No     torch.nn.functional
hardshrink                                No       No        No        No       No     torch.nn.functional
hardsigmoid                               No       No        No        No       No     torch.nn.functional
hardsigmoid\_                             No       No        No        No       No     torch.nn.functional
hardtanh                                  Yes      Yes       Yes       Yes      Yes    torch.nn.functional
hardtanh\_                                Yes      Yes       Yes       Yes      Yes    torch.nn.functional
huber_loss                                No       No        No        No       No     torch.nn.functional
instance_norm                             No       No        No        No       No     torch.nn.functional
kl_div                                    No       No        No        No       No     torch.nn.functional
l1_loss                                   No       No        No        No       No     torch.nn.functional
layer_norm                                No       No        No        No       No     torch.nn.functional
leaky_relu                                No       No        No        No       No     torch.nn.functional
leaky_relu\_                              No       No        No        No       No     torch.nn.functional
linear                                    No       No        No        No       No     torch.nn.functional
log_softmax                               No       No        No        No       No     torch.nn.functional
logsigmoid                                No       No        No        No       No     torch.nn.functional
max_pool2d                                No       No        No        No       No     torch.nn.functional
max_pool2d_with_indices                   No       No        No        No       No     torch.nn.functional
max_pool3d                                No       No        No        No       No     torch.nn.functional
max_pool3d_with_indices                   No       No        No        No       No     torch.nn.functional
mish                                      No       No        No        No       No     torch.nn.functional
mish\_                                    No       No        No        No       No     torch.nn.functional
mse_loss                                  No       No        No        No       No     torch.nn.functional
multi_margin_loss                         No       No        No        No       No     torch.nn.functional
multilabel_margin_loss                    No       No        No        No       No     torch.nn.functional
nll_loss                                  No       No        No        No       No     torch.nn.functional
one_hot                                   Yes      No        No        No       No     torch.nn.functional
pad                                       No       No        No        No       No     torch.nn.functional
pixel_shuffle                             Yes      Yes       Yes       Yes      Yes    torch.nn.functional
prelu                                     No       No        No        No       No     torch.nn.functional
relu                                      No       No        No        No       No     torch.nn.functional
relu\_                                    No       No        No        No       No     torch.nn.functional
rrelu                                     No       No        No        No       No     torch.nn.functional
rrelu\_                                   No       No        No        No       No     torch.nn.functional
sigmoid                                   Yes      Yes       Yes       Yes      Yes    torch.nn.functional
sigmoid\_                                 Yes      Yes       Yes       Yes      Yes    torch.nn.functional
silu                                      No       No        No        No       No     torch.nn.functional
silu\_                                    No       No        No        No       No     torch.nn.functional
smooth_l1_loss                            No       No        No        No       No     torch.nn.functional
softmax                                   No       No        No        No       No     torch.nn.functional
softplus                                  No       No        No        No       No     torch.nn.functional
softshrink                                No       No        No        No       No     torch.nn.functional
tanh                                      Yes      Yes       Yes       Yes      Yes    torch.nn.functional
threshold                                 No       No        No        No       No     torch.nn.functional
threshold\_                               No       No        No        No       No     torch.nn.functional
upsample                                  Yes      Yes       Yes       Yes      Yes    torch.nn.functional
upsample_bilinear                         Yes      Yes       Yes       Yes      Yes    torch.nn.functional
upsample_nearest                          Yes      Yes       Yes       Yes      Yes    torch.nn.functional
cross                                     Yes      Yes       Yes       Yes      Yes    torch.linalg
vector_norm                               No       No        No        No       No     torch.linalg
_adaptive_avg_pool2d                      No       No        No        No       No     torch
_adaptive_avg_pool3d                      No       No        No        No       No     torch
_add_relu                                 No       No        No        No       No     torch
_add_relu\_                               No       No        No        No       No     torch
_addmm_activation                         No       No        No        No       No     torch
_aminmax                                  Yes      Yes       Yes       Yes      Yes    torch
_ctc_loss                                 No       No        No        No       No     torch
_efficientzerotensor                      Yes      Yes       Yes       Yes      Yes    torch
_foreach_abs                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_abs\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_acos                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_acos\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_add                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_add\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_addcdiv                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_addcdiv\_                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_addcmul                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_addcmul\_                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_asin                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_asin\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_atan                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_atan\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_ceil                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_ceil\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_clamp_max                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_clamp_max\_                      Yes      Yes       Yes       Yes      Yes    torch
_foreach_clamp_min                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_clamp_min\_                      Yes      Yes       Yes       Yes      Yes    torch
_foreach_cos                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_cos\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_cosh                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_cosh\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_div                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_div\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_erf                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_erf\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_erfc                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_erfc\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_exp                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_exp\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_expm1                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_expm1\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_floor                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_floor\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_frac                             No       No        No        No       No     torch
_foreach_frac\_                           No       No        No        No       No     torch
_foreach_lerp                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_lerp\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_lgamma                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_lgamma\_                         Yes      Yes       Yes       Yes      Yes    torch
_foreach_log                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_log10                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_log10\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_log1p                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_log1p\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_log2                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_log2\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_log\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_maximum                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_maximum\_                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_minimum                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_minimum\_                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_mul                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_mul\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_neg                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_neg\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_pow                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_pow\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_reciprocal                       Yes      Yes       Yes       Yes      Yes    torch
_foreach_reciprocal\_                     Yes      Yes       Yes       Yes      Yes    torch
_foreach_round                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_round\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_sigmoid                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_sigmoid\_                        Yes      Yes       Yes       Yes      Yes    torch
_foreach_sign                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_sign\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_sin                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_sin\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_sinh                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_sinh\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_sqrt                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_sqrt\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_sub                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_sub\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_tan                              Yes      Yes       Yes       Yes      Yes    torch
_foreach_tan\_                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_tanh                             Yes      Yes       Yes       Yes      Yes    torch
_foreach_tanh\_                           Yes      Yes       Yes       Yes      Yes    torch
_foreach_trunc                            Yes      Yes       Yes       Yes      Yes    torch
_foreach_trunc\_                          Yes      Yes       Yes       Yes      Yes    torch
_foreach_zero\_                           Yes      Yes       Yes       Yes      Yes    torch
_fused_dropout                            No       No        No        No       No     torch
_logcumsumexp                             No       No        No        No       No     torch
_masked_scale                             Yes      Yes       Yes       Yes      Yes    torch
_native_batch_norm_legit                  No       No        No        No       No     torch
_native_batch_norm_legit_no_training      No       No        No        No       No     torch
_resize_output\_                          Yes      Yes       Yes       Yes      Yes    torch
_unique                                   No       Yes       No        No       No     torch
_unique2                                  No       Yes       No        No       No     torch
_weight_norm_interface                    No       No        No        No       No     torch
abs                                       Yes      Yes       Yes       Yes      Yes    torch
abs\_                                     Yes      Yes       Yes       Yes      Yes    torch
absolute                                  Yes      Yes       Yes       Yes      Yes    torch
acos                                      Yes      Yes       Yes       Yes      Yes    torch
acos\_                                    Yes      Yes       Yes       Yes      Yes    torch
acosh                                     Yes      Yes       Yes       Yes      Yes    torch
acosh\_                                   Yes      Yes       Yes       Yes      Yes    torch
add                                       Yes      Yes       Yes       Yes      Yes    torch
addbmm                                    No       No        No        No       No     torch
addcdiv                                   Yes      Yes       Yes       Yes      Yes    torch
addcmul                                   Yes      Yes       Yes       Yes      Yes    torch
addmm                                     No       No        No        No       No     torch
addmv                                     No       No        No        No       No     torch
addmv\_                                   No       No        No        No       No     torch
addr                                      No       No        No        No       No     torch
all                                       Yes      Yes       Yes       Yes      Yes    torch
amax                                      Yes      Yes       Yes       Yes      Yes    torch
amin                                      Yes      Yes       Yes       Yes      Yes    torch
aminmax                                   Yes      Yes       Yes       Yes      Yes    torch
any                                       Yes      Yes       Yes       Yes      Yes    torch
arange                                    Yes      Yes       Yes       Yes      Yes    torch
arccos                                    Yes      Yes       Yes       Yes      Yes    torch
arccos\_                                  Yes      Yes       Yes       Yes      Yes    torch
arccosh                                   Yes      Yes       Yes       Yes      Yes    torch
arccosh\_                                 Yes      Yes       Yes       Yes      Yes    torch
arcsin                                    Yes      Yes       Yes       Yes      Yes    torch
arcsin\_                                  Yes      Yes       Yes       Yes      Yes    torch
arcsinh                                   Yes      Yes       Yes       Yes      Yes    torch
arcsinh\_                                 Yes      Yes       Yes       Yes      Yes    torch
arctan                                    Yes      Yes       Yes       Yes      Yes    torch
arctan2                                   Yes      Yes       Yes       Yes      Yes    torch
arctan\_                                  Yes      Yes       Yes       Yes      Yes    torch
arctanh                                   No       No        No        No       No     torch
arctanh\_                                 No       No        No        No       No     torch
argmax                                    Yes      Yes       No        Yes      Yes    torch
argmin                                    Yes      Yes       No        Yes      Yes    torch
argsort                                   Yes      Yes       Yes       No       No     torch
as_strided                                Yes      Yes       No        Yes      Yes    torch
as_strided\_                              Yes      Yes       No        Yes      Yes    torch
asin                                      Yes      Yes       Yes       Yes      Yes    torch
asin\_                                    Yes      Yes       Yes       Yes      Yes    torch
asinh                                     Yes      Yes       Yes       Yes      Yes    torch
asinh\_                                   Yes      Yes       Yes       Yes      Yes    torch
atan                                      Yes      Yes       Yes       Yes      Yes    torch
atan2                                     Yes      Yes       Yes       Yes      Yes    torch
atan\_                                    Yes      Yes       Yes       Yes      Yes    torch
atanh                                     No       No        No        No       No     torch
atanh\_                                   No       No        No        No       No     torch
baddbmm                                   No       No        No        No       No     torch
batch_norm                                No       No        No        No       No     torch
bernoulli                                 No       No        No        No       No     torch
binary_cross_entropy_with_logits          No       No        No        No       No     torch
bincount                                  Yes      Yes       Yes       Yes      No     torch
bitwise_and                               Yes      Yes       Yes       Yes      Yes    torch
bitwise_left_shift                        Yes      Yes       Yes       Yes      Yes    torch
bitwise_not                               Yes      Yes       Yes       Yes      Yes    torch
bitwise_or                                Yes      Yes       Yes       Yes      Yes    torch
bitwise_right_shift                       Yes      Yes       Yes       Yes      Yes    torch
bitwise_xor                               Yes      Yes       Yes       Yes      Yes    torch
bmm                                       No       No        No        No       No     torch
broadcast_tensors                         No       No        No        No       No     torch
bucketize                                 Yes      Yes       No        No       No     torch
cat                                       Yes      Yes       Yes       Yes      Yes    torch
ceil                                      Yes      Yes       Yes       Yes      Yes    torch
ceil\_                                    Yes      Yes       Yes       Yes      Yes    torch
channel_shuffle                           Yes      Yes       Yes       Yes      Yes    torch
cholesky                                  No       No        No        No       No     torch
chunk                                     No       Yes       No        No       No     torch
clamp                                     Yes      Yes       Yes       Yes      Yes    torch
clamp\_                                   Yes      Yes       Yes       Yes      Yes    torch
clamp_max                                 Yes      Yes       Yes       Yes      Yes    torch
clamp_max\_                               Yes      Yes       Yes       Yes      Yes    torch
clamp_min                                 Yes      Yes       Yes       Yes      Yes    torch
clamp_min\_                               Yes      Yes       Yes       Yes      Yes    torch
clip                                      No       Yes       No        No       No     torch
clip\_                                    No       Yes       No        No       No     torch
clone                                     Yes      Yes       Yes       Yes      Yes    torch
conj                                      No       Yes       No        No       No     torch
constant_pad_nd                           Yes      Yes       Yes       Yes      Yes    torch
conv1d                                    No       No        No        No       No     torch
conv2d                                    No       No        No        No       No     torch
conv3d                                    No       No        No        No       No     torch
conv_transpose1d                          No       No        No        No       No     torch
conv_transpose2d                          No       No        No        No       No     torch
conv_transpose3d                          No       No        No        No       No     torch
copysign                                  Yes      Yes       Yes       Yes      Yes    torch
cos                                       Yes      Yes       Yes       Yes      Yes    torch
cos\_                                     Yes      Yes       Yes       Yes      Yes    torch
cosh                                      Yes      Yes       Yes       Yes      Yes    torch
cosh\_                                    Yes      Yes       Yes       Yes      Yes    torch
count_nonzero                             Yes      Yes       Yes       Yes      Yes    torch
cross                                     Yes      Yes       Yes       Yes      Yes    torch
cumprod                                   Yes      Yes       No        Yes      Yes    torch
cumsum                                    Yes      Yes       No        Yes      Yes    torch
diag                                      No       No        No        No       No     torch
div                                       Yes      Yes       Yes       Yes      Yes    torch
divide                                    Yes      Yes       Yes       Yes      Yes    torch
dot                                       Yes      Yes       No        No       No     torch
dropout                                   No       No        No        No       No     torch
embedding                                 Yes      Yes       Yes       Yes      Yes    torch
embedding_bag                             No       No        No        No       No     torch
embedding_renorm\_                        No       No        No        No       No     torch
empty                                     Yes      Yes       Yes       Yes      Yes    torch
empty_like                                Yes      Yes       Yes       Yes      Yes    torch
empty_strided                             Yes      Yes       Yes       Yes      Yes    torch
eq                                        Yes      Yes       Yes       Yes      Yes    torch
equal                                     Yes      Yes       Yes       Yes      Yes    torch
erf                                       Yes      Yes       Yes       Yes      Yes    torch
erf\_                                     Yes      Yes       Yes       Yes      Yes    torch
erfc                                      Yes      Yes       Yes       Yes      Yes    torch
erfc\_                                    Yes      Yes       Yes       Yes      Yes    torch
erfinv                                    Yes      Yes       Yes       Yes      Yes    torch
exp                                       Yes      Yes       Yes       Yes      Yes    torch
exp2                                      Yes      Yes       Yes       Yes      Yes    torch
exp2\_                                    Yes      Yes       Yes       Yes      Yes    torch
exp\_                                     Yes      Yes       Yes       Yes      Yes    torch
expm1                                     Yes      Yes       Yes       Yes      Yes    torch
expm1\_                                   Yes      Yes       Yes       Yes      Yes    torch
eye                                       Yes      Yes       Yes       Yes      Yes    torch
fill                                      Yes      Yes       Yes       Yes      Yes    torch
fill\_                                    Yes      Yes       Yes       Yes      Yes    torch
flatten                                   No       Yes       No        No       No     torch
flip                                      Yes      Yes       Yes       Yes      Yes    torch
floor                                     Yes      Yes       Yes       Yes      Yes    torch
floor\_                                   Yes      Yes       Yes       Yes      Yes    torch
floor_divide                              Yes      Yes       Yes       Yes      Yes    torch
fmax                                      Yes      Yes       Yes       Yes      Yes    torch
fmin                                      Yes      Yes       Yes       Yes      Yes    torch
fmod                                      Yes      Yes       Yes       Yes      Yes    torch
frac                                      No       No        No        No       No     torch
frac\_                                    No       No        No        No       No     torch
frexp                                     No       No        No        No       No     torch
full                                      Yes      Yes       Yes       Yes      Yes    torch
full_like                                 Yes      Yes       Yes       Yes      Yes    torch
gather                                    Yes      Yes       Yes       Yes      Yes    torch
ge                                        Yes      Yes       No        Yes      Yes    torch
greater                                   Yes      Yes       No        Yes      Yes    torch
greater_equal                             Yes      Yes       No        Yes      Yes    torch
grid_sampler_2d                           No       No        No        No       No     torch
grid_sampler_3d                           No       No        No        No       No     torch
gt                                        Yes      Yes       No        Yes      Yes    torch
hardshrink                                No       No        No        No       No     torch
heaviside                                 Yes      Yes       No        No       No     torch
hypot                                     No       No        No        No       No     torch
index_copy                                Yes      Yes       Yes       Yes      Yes    torch
index_fill                                Yes      Yes       Yes       Yes      Yes    torch
index_put                                 No       Yes       No        No       No     torch
index_put\_                               No       Yes       No        No       No     torch
index_reduce                              Yes      Yes       No        No       No     torch
index_select                              Yes      Yes       Yes       Yes      Yes    torch
instance_norm                             No       No        No        No       No     torch
is_complex                                No       Yes       No        No       No     torch
is_floating_point                         No       Yes       No        No       No     torch
is_nonzero                                No       Yes       No        No       No     torch
isfinite                                  Yes      Yes       Yes       Yes      Yes    torch
isin                                      Yes      Yes       Yes       Yes      Yes    torch
isinf                                     Yes      Yes       Yes       Yes      Yes    torch
isnan                                     Yes      Yes       Yes       Yes      Yes    torch
isneginf                                  Yes      Yes       Yes       Yes      Yes    torch
isposinf                                  Yes      Yes       Yes       Yes      Yes    torch
kl_div                                    No       No        No        No       No     torch
kthvalue                                  Yes      Yes       No        No       No     torch
layer_norm                                No       No        No        No       No     torch
le                                        Yes      Yes       No        Yes      Yes    torch
lerp                                      Yes      Yes       Yes       Yes      Yes    torch
less                                      Yes      Yes       No        Yes      Yes    torch
less_equal                                Yes      Yes       No        Yes      Yes    torch
lgamma                                    Yes      Yes       Yes       Yes      Yes    torch
linspace                                  Yes      Yes       Yes       Yes      Yes    torch
log                                       Yes      Yes       Yes       Yes      Yes    torch
log10                                     Yes      Yes       Yes       Yes      Yes    torch
log10\_                                   Yes      Yes       Yes       Yes      Yes    torch
log1p                                     Yes      Yes       Yes       Yes      Yes    torch
log1p\_                                   Yes      Yes       Yes       Yes      Yes    torch
log2                                      Yes      Yes       Yes       Yes      Yes    torch
log2\_                                    Yes      Yes       Yes       Yes      Yes    torch
log\_                                     Yes      Yes       Yes       Yes      Yes    torch
log_softmax                               No       No        No        No       No     torch
logaddexp                                 No       No        No        No       No     torch
logaddexp2                                No       No        No        No       No     torch
logcumsumexp                              No       No        No        No       No     torch
logical_and                               Yes      Yes       No        Yes      Yes    torch
logical_not                               No       Yes       Yes       Yes      Yes    torch
logical_or                                No       No        No        Yes      Yes    torch
logical_xor                               No       No        No        Yes      Yes    torch
logit                                     No       No        No        No       No     torch
logit\_                                   No       No        No        No       No     torch
logspace                                  Yes      Yes       Yes       Yes      Yes    torch
logsumexp                                 No       No        No        No       No     torch
lt                                        Yes      Yes       No        Yes      Yes    torch
masked_fill                               Yes      Yes       No        Yes      Yes    torch
masked_scatter                            Yes      Yes       No        Yes      Yes    torch
masked_select                             No       Yes       No        No       No     torch
matmul                                    No       No        No        No       No     torch
max                                       Yes      Yes       No        No       No     torch
max_pool2d                                No       No        No        No       No     torch
max_pool3d                                No       No        No        No       No     torch
maximum                                   Yes      Yes       Yes       Yes      Yes    torch
mean                                      Yes      Yes       Yes       Yes      Yes    torch
median                                    Yes      Yes       No        No       No     torch
meshgrid                                  No       No        No        No       No     torch
min                                       Yes      Yes       No        No       No     torch
minimum                                   Yes      Yes       Yes       Yes      Yes    torch
mm                                        No       No        No        No       No     torch
mul                                       Yes      Yes       Yes       Yes      No     torch
multinomial                               No       No        No        No       No     torch
mv                                        No       No        No        No       No     torch
nan_to_num                                Yes      Yes       Yes       Yes      Yes    torch
nan_to_num\_                              Yes      Yes       Yes       Yes      Yes    torch
nansum                                    Yes      Yes       Yes       Yes      Yes    torch
narrow                                    No       No        No        No       No     torch
native_dropout                            No       No        No        No       No     torch
native_group_norm                         No       No        No        No       No     torch
native_layer_norm                         No       No        No        No       No     torch
ne                                        Yes      Yes       Yes       Yes      Yes    torch
neg                                       Yes      Yes       Yes       Yes      Yes    torch
neg\_                                     Yes      Yes       Yes       Yes      Yes    torch
nextafter                                 No       No        No        No       No     torch
nonzero                                   No       Yes       No        No       Yes    torch
norm                                      No       No        No        No       No     torch
normal                                    No       No        No        No       No     torch
not_equal                                 Yes      Yes       Yes       Yes      Yes    torch
ones                                      No       Yes       No        Yes      Yes    torch
ones_like                                 No       Yes       Yes       Yes      Yes    torch
permute                                   Yes      Yes       Yes       Yes      Yes    torch
pixel_shuffle                             Yes      Yes       Yes       Yes      Yes    torch
poisson                                   No       No        No        No       No     torch
pow                                       Yes      Yes       Yes       Yes      Yes    torch
prelu                                     No       No        No        No       No     torch
prod                                      Yes      Yes       Yes       Yes      Yes    torch
put                                       No       No        No        No       No     torch
randperm                                  Yes      Yes       Yes       Yes      Yes    torch
reciprocal                                Yes      Yes       Yes       Yes      Yes    torch
reciprocal\_                              Yes      Yes       Yes       Yes      Yes    torch
relu                                      No       No        No        No       No     torch
relu\_                                    No       No        No        No       No     torch
remainder                                 Yes      Yes       Yes       Yes      Yes    torch
reshape                                   Yes      Yes       Yes       Yes      Yes    torch
resolve_conj                              No       No        No        No       No     torch
resolve_neg                               No       No        No        No       No     torch
result_type                               No       Yes       No        No       No     torch
roll                                      No       Yes       Yes       Yes      Yes    torch
round                                     Yes      Yes       Yes       Yes      Yes    torch
round\_                                   Yes      Yes       Yes       Yes      Yes    torch
rsqrt                                     Yes      Yes       Yes       Yes      Yes    torch
rsqrt\_                                   Yes      Yes       Yes       Yes      Yes    torch
rsub                                      Yes      Yes       Yes       Yes      Yes    torch
scalar_tensor                             Yes      Yes       Yes       Yes      Yes    torch
scatter                                   Yes      Yes       Yes       Yes      Yes    torch
scatter_add                               No       No        No        No       No     torch
scatter_reduce                            No       No        No        No       No     torch
searchsorted                              Yes      Yes       No        No       No     torch
select                                    Yes      Yes       Yes       Yes      Yes    torch
sgn                                       Yes      Yes       Yes       Yes      Yes    torch
sigmoid                                   Yes      Yes       Yes       Yes      Yes    torch
sigmoid\_                                 Yes      Yes       Yes       Yes      Yes    torch
sign                                      Yes      Yes       Yes       Yes      Yes    torch
signbit                                   Yes      Yes       Yes       Yes      Yes    torch
sin                                       Yes      Yes       Yes       Yes      Yes    torch
sin\_                                     Yes      Yes       Yes       Yes      Yes    torch
sinc                                      Yes      Yes       Yes       Yes      Yes    torch
sinc\_                                    Yes      Yes       Yes       Yes      Yes    torch
sinh                                      Yes      Yes       Yes       Yes      Yes    torch
sinh\_                                    No       No        No        No       No     torch
softmax                                   No       No        No        No       No     torch
sort                                      Yes      Yes       Yes       No       No     torch
split_with_sizes                          No       Yes       Yes       No       No     torch
sqrt                                      Yes      Yes       Yes       Yes      Yes    torch
sqrt\_                                    Yes      Yes       Yes       Yes      Yes    torch
square                                    No       Yes       No        Yes      Yes    torch
square\_                                  No       Yes       No        Yes      Yes    torch
squeeze                                   Yes      Yes       Yes       Yes      Yes    torch
stack                                     No       Yes       Yes       No       No     torch
std                                       No       No        No        No       No     torch
std_mean                                  No       No        No        No       No     torch
sub                                       Yes      Yes       Yes       Yes      Yes    torch
sum                                       Yes      Yes       Yes       Yes      Yes    torch
t                                         Yes      Yes       Yes       Yes      Yes    torch
take                                      Yes      Yes       No        No       No     torch
tan                                       Yes      Yes       Yes       Yes      Yes    torch
tan\_                                     Yes      Yes       Yes       Yes      Yes    torch
tanh                                      Yes      Yes       Yes       Yes      Yes    torch
tanh\_                                    Yes      Yes       Yes       Yes      Yes    torch
threshold                                 No       No        No        No       No     torch
threshold\_                               No       No        No        No       No     torch
topk                                      Yes      Yes       Yes       No       No     torch
trace                                     Yes      Yes       No        No       No     torch
transpose                                 Yes      Yes       Yes       Yes      Yes    torch
tril                                      No       No        No        Yes      Yes    torch
tril_indices                              Yes      Yes       No        No       No     torch
triu                                      No       No        No        Yes      Yes    torch
triu_indices                              Yes      Yes       No        No       No     torch
trunc                                     Yes      Yes       Yes       Yes      Yes    torch
trunc\_                                   Yes      Yes       Yes       Yes      Yes    torch
unbind                                    No       Yes       No        No       No     torch
unique                                    No       Yes       No        No       No     torch
unsqueeze                                 Yes      Yes       Yes       No       No     torch
var                                       No       No        No        No       No     torch
var_mean                                  No       No        No        No       No     torch
vdot                                      Yes      Yes       No        No       No     torch
where                                     Yes      Yes       Yes       Yes      Yes    torch
xlogy                                     Yes      Yes       Yes       Yes      Yes    torch
xlogy\_                                   Yes      Yes       Yes       Yes      Yes    torch
zero\_                                    Yes      Yes       Yes       Yes      Yes    torch
zeros                                     No       Yes       Yes       Yes      Yes    torch
zeros_like                                No       Yes       Yes       Yes      Yes    torch
glu_jvp                                   No       No        No        No       No     torch.ops.aten
AdaptiveAvgPool1d                         No       No        No        No       No     torch.nn
AdaptiveAvgPool2d                         No       No        No        No       No     torch.nn
AdaptiveAvgPool3d                         No       No        No        No       No     torch.nn
AvgPool1d                                 No       No        No        No       No     torch.nn
AvgPool2d                                 No       No        No        No       No     torch.nn
AvgPool3d                                 No       No        No        No       No     torch.nn
BCELoss                                   No       No        No        No       No     torch.nn
BCEWithLogitsLoss                         No       No        No        No       No     torch.nn
BatchNorm1d                               No       No        No        No       No     torch.nn
BatchNorm2d                               No       No        No        No       No     torch.nn
ChannelShuffle                            Yes      Yes       Yes       Yes      Yes    torch.nn
ConstantPad1d                             No       No        No        No       No     torch.nn
Conv1d                                    No       No        No        No       No     torch.nn
Conv2d                                    No       No        No        No       No     torch.nn
Conv3d                                    No       No        No        No       No     torch.nn
ConvTranspose1d                           No       No        No        No       No     torch.nn
ConvTranspose2d                           No       No        No        No       No     torch.nn
ConvTranspose3d                           No       No        No        No       No     torch.nn
CrossEntropyLoss                          No       No        No        No       No     torch.nn
Dropout                                   No       No        No        No       No     torch.nn
ELU                                       No       No        No        No       No     torch.nn
EmbeddingBag                              No       No        No        No       No     torch.nn
GELU                                      No       No        No        No       No     torch.nn
GLU                                       No       No        No        No       No     torch.nn
Hardshrink                                No       No        No        No       No     torch.nn
Hardsigmoid                               No       No        No        No       No     torch.nn
Hardswish                                 No       No        No        No       No     torch.nn
Hardtanh                                  Yes      Yes       Yes       Yes      Yes    torch.nn
HuberLoss                                 No       No        No        No       No     torch.nn
InstanceNorm2d                            No       No        No        No       No     torch.nn
KLDivLoss                                 No       No        No        No       No     torch.nn
LayerNorm                                 No       No        No        No       No     torch.nn
LeakyReLU                                 No       No        No        No       No     torch.nn
Linear                                    No       No        No        No       No     torch.nn
LogSigmoid                                No       No        No        No       No     torch.nn
LogSoftmax                                No       No        No        No       No     torch.nn
MSELoss                                   No       No        No        No       No     torch.nn
MaxPool2d                                 No       No        No        No       No     torch.nn
MaxPool3d                                 No       No        No        No       No     torch.nn
Mish                                      No       No        No        No       No     torch.nn
MultiLabelMarginLoss                      No       No        No        No       No     torch.nn
MultiMarginLoss                           No       No        No        No       No     torch.nn
NLLLoss                                   No       No        No        No       No     torch.nn
PReLU                                     No       No        No        No       No     torch.nn
PixelShuffle                              Yes      Yes       Yes       Yes      Yes    torch.nn
RReLU                                     No       No        No        No       No     torch.nn
ReLU                                      No       No        No        No       No     torch.nn
ReflectionPad1d                           Yes      Yes       Yes       Yes      Yes    torch.nn
ReflectionPad2d                           Yes      Yes       Yes       Yes      Yes    torch.nn
ReflectionPad3d                           Yes      Yes       Yes       Yes      Yes    torch.nn
ReplicationPad1d                          Yes      Yes       Yes       Yes      Yes    torch.nn
ReplicationPad2d                          Yes      Yes       Yes       Yes      Yes    torch.nn
ReplicationPad3d                          Yes      Yes       Yes       Yes      Yes    torch.nn
SiLU                                      No       No        No        No       No     torch.nn
Sigmoid                                   Yes      Yes       Yes       Yes      Yes    torch.nn
SmoothL1Loss                              No       No        No        No       No     torch.nn
Softmax                                   No       No        No        No       No     torch.nn
Softplus                                  No       No        No        No       No     torch.nn
Softshrink                                No       No        No        No       No     torch.nn
Tanh                                      Yes      Yes       Yes       Yes      Yes    torch.nn
Threshold                                 No       No        No        No       No     torch.nn
Upsample                                  Yes      Yes       Yes       Yes      Yes    torch.nn
UpsamplingNearest2d                       Yes      Yes       Yes       Yes      Yes    torch.nn
weight_norm                               No       No        No        No       No     torch.nn.utils
T                                         Yes      Yes       Yes       Yes      Yes    torch.Tensor
__and__                                   No       Yes       No        Yes      Yes    torch.Tensor
__iand__                                  No       Yes       No        Yes      Yes    torch.Tensor
__ilshift__                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
__ior__                                   No       Yes       No        Yes      Yes    torch.Tensor
__irshift__                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
__ixor__                                  No       Yes       No        Yes      Yes    torch.Tensor
__lshift__                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
__or__                                    No       Yes       No        Yes      Yes    torch.Tensor
__rshift__                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
__xor__                                   No       Yes       No        Yes      Yes    torch.Tensor
abs                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
abs\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
absolute                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
absolute\_                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
acos                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
acos\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
acosh                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
acosh\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
add                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
add\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
addbmm                                    No       No        No        No       No     torch.Tensor
addbmm\_                                  No       No        No        No       No     torch.Tensor
addcdiv                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
addcdiv\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
addcmul                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
addcmul\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
addmm                                     No       No        No        No       No     torch.Tensor
addmm\_                                   No       No        No        No       No     torch.Tensor
addmv                                     No       No        No        No       No     torch.Tensor
addmv\_                                   No       No        No        No       No     torch.Tensor
addr                                      No       No        No        No       No     torch.Tensor
addr\_                                    No       No        No        No       No     torch.Tensor
all                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
amax                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
amin                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
aminmax                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
any                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
arccos                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
arccos\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
arccosh                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
arccosh\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
arcsin                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
arcsin\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
arcsinh                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
arcsinh\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
arctan                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
arctan2                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
arctan2\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
arctan\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
arctanh                                   No       No        No        No       No     torch.Tensor
arctanh\_                                 No       No        No        No       No     torch.Tensor
argmax                                    Yes      Yes       No        Yes      Yes    torch.Tensor
argmin                                    Yes      Yes       No        Yes      Yes    torch.Tensor
argsort                                   Yes      Yes       Yes       No       No     torch.Tensor
as_strided                                Yes      Yes       No        Yes      Yes    torch.Tensor
as_strided\_                              Yes      Yes       No        Yes      Yes    torch.Tensor
asin                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
asin\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
asinh                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
asinh\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
atan                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
atan2                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
atan2\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
atan\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
atanh                                     No       No        No        No       No     torch.Tensor
atanh\_                                   No       No        No        No       No     torch.Tensor
baddbmm                                   No       No        No        No       No     torch.Tensor
baddbmm\_                                 No       No        No        No       No     torch.Tensor
bernoulli                                 No       No        No        No       No     torch.Tensor
bernoulli\_                               Yes      Yes       Yes       Yes      No     torch.Tensor
bincount                                  Yes      Yes       Yes       Yes      No     torch.Tensor
bitwise_and                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_and\_                             Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_left_shift                        Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_left_shift\_                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_not                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_not\_                             Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_or                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_or\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_right_shift                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_right_shift\_                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_xor                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
bitwise_xor\_                             Yes      Yes       Yes       Yes      Yes    torch.Tensor
bmm                                       No       No        No        No       No     torch.Tensor
ceil                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
ceil\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
cholesky                                  No       No        No        No       No     torch.Tensor
chunk                                     No       Yes       No        No       No     torch.Tensor
clamp                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_max                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_max\_                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_min                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_min\_                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
clip                                      No       Yes       No        No       No     torch.Tensor
clip\_                                    No       Yes       No        No       No     torch.Tensor
clone                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
conj                                      No       Yes       No        No       No     torch.Tensor
copy\_                                    No       Yes       No        Yes      Yes    torch.Tensor
copysign                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
copysign\_                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
cos                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
cos\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
cosh                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
cosh\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
count_nonzero                             Yes      Yes       Yes       Yes      Yes    torch.Tensor
cross                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
cumprod                                   Yes      Yes       No        Yes      Yes    torch.Tensor
cumprod\_                                 Yes      Yes       No        No       No     torch.Tensor
cumsum                                    Yes      Yes       No        Yes      Yes    torch.Tensor
cumsum\_                                  Yes      Yes       No        Yes      Yes    torch.Tensor
diag                                      No       No        No        No       No     torch.Tensor
div                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
div\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
divide                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
divide\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
dot                                       Yes      Yes       No        No       No     torch.Tensor
eq                                        Yes      Yes       Yes       Yes      Yes    torch.Tensor
eq\_                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
equal                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
erf                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
erf\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
erfc                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
erfc\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
erfinv                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
erfinv\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
exp                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
exp2                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
exp2\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
exp\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
expand                                    Yes      Yes       No        No       No     torch.Tensor
expand_as                                 Yes      Yes       No        Yes      Yes    torch.Tensor
expm1                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
expm1\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
exponential\_                             No       No        No        No       No     torch.Tensor
fill\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
flatten                                   No       Yes       No        No       No     torch.Tensor
flip                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
floor                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
floor\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
floor_divide                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
floor_divide\_                            Yes      Yes       Yes       Yes      Yes    torch.Tensor
fmax                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
fmin                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
fmod                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
fmod\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
frac                                      No       No        No        No       No     torch.Tensor
frac\_                                    No       No        No        No       No     torch.Tensor
frexp                                     No       No        No        No       No     torch.Tensor
gather                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
ge                                        Yes      Yes       No        Yes      Yes    torch.Tensor
ge\_                                      Yes      Yes       No        Yes      Yes    torch.Tensor
geometric\_                               No       No        No        No       No     torch.Tensor
greater                                   Yes      Yes       No        Yes      Yes    torch.Tensor
greater\_                                 Yes      Yes       No        Yes      Yes    torch.Tensor
greater_equal                             Yes      Yes       No        Yes      Yes    torch.Tensor
greater_equal\_                           Yes      Yes       No        Yes      Yes    torch.Tensor
gt                                        Yes      Yes       No        Yes      Yes    torch.Tensor
gt\_                                      Yes      Yes       No        Yes      Yes    torch.Tensor
hardshrink                                No       No        No        No       No     torch.Tensor
heaviside                                 Yes      Yes       No        No       No     torch.Tensor
heaviside\_                               Yes      Yes       No        No       No     torch.Tensor
hypot                                     No       No        No        No       No     torch.Tensor
hypot\_                                   No       No        No        No       No     torch.Tensor
index_add\_                               No       Yes       No        No       No     torch.Tensor
index_copy                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
index_copy\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
index_fill                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
index_fill\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
index_put                                 No       Yes       No        No       No     torch.Tensor
index_put\_                               No       Yes       No        No       No     torch.Tensor
index_reduce                              Yes      Yes       No        No       No     torch.Tensor
index_reduce\_                            Yes      Yes       No        No       No     torch.Tensor
index_select                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
is_complex                                No       Yes       No        No       No     torch.Tensor
is_floating_point                         No       Yes       No        No       No     torch.Tensor
is_nonzero                                No       Yes       No        No       No     torch.Tensor
isfinite                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
isinf                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
isnan                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
isneginf                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
isposinf                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
item                                      No       No        No        No       No     torch.Tensor
kthvalue                                  Yes      Yes       No        No       No     torch.Tensor
le                                        Yes      Yes       No        Yes      Yes    torch.Tensor
le\_                                      Yes      Yes       No        Yes      Yes    torch.Tensor
lerp                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
lerp\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
less                                      Yes      Yes       No        Yes      Yes    torch.Tensor
less\_                                    Yes      Yes       No        Yes      Yes    torch.Tensor
less_equal                                Yes      Yes       No        Yes      Yes    torch.Tensor
less_equal\_                              Yes      Yes       No        Yes      Yes    torch.Tensor
log                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
log10                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
log10\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
log1p                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
log1p\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
log2                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
log2\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
log\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
log_normal\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
log_softmax                               No       No        No        No       No     torch.Tensor
logaddexp                                 No       No        No        No       No     torch.Tensor
logaddexp2                                No       No        No        No       No     torch.Tensor
logcumsumexp                              No       No        No        No       No     torch.Tensor
logical_and                               Yes      Yes       No        Yes      Yes    torch.Tensor
logical_and\_                             Yes      Yes       No        Yes      Yes    torch.Tensor
logical_not                               No       Yes       Yes       Yes      Yes    torch.Tensor
logical_not\_                             No       No        No        Yes      Yes    torch.Tensor
logical_or                                No       No        No        Yes      Yes    torch.Tensor
logical_or\_                              No       No        No        Yes      Yes    torch.Tensor
logical_xor                               No       No        No        Yes      Yes    torch.Tensor
logical_xor\_                             No       No        No        Yes      Yes    torch.Tensor
logit                                     No       No        No        No       No     torch.Tensor
logit\_                                   No       No        No        No       No     torch.Tensor
logsumexp                                 No       No        No        No       No     torch.Tensor
lt                                        Yes      Yes       No        Yes      Yes    torch.Tensor
lt\_                                      Yes      Yes       No        Yes      Yes    torch.Tensor
masked_fill                               Yes      Yes       No        Yes      Yes    torch.Tensor
masked_fill\_                             Yes      Yes       No        Yes      Yes    torch.Tensor
masked_scatter                            Yes      Yes       No        Yes      Yes    torch.Tensor
masked_scatter\_                          Yes      Yes       No        Yes      Yes    torch.Tensor
masked_select                             No       Yes       No        No       No     torch.Tensor
matmul                                    No       No        No        No       No     torch.Tensor
max                                       Yes      Yes       No        No       No     torch.Tensor
maximum                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
mean                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
median                                    Yes      Yes       No        No       No     torch.Tensor
min                                       Yes      Yes       No        No       No     torch.Tensor
minimum                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
mm                                        No       No        No        No       No     torch.Tensor
mul                                       Yes      Yes       Yes       Yes      No     torch.Tensor
mul\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
multinomial                               No       No        No        No       No     torch.Tensor
mv                                        No       No        No        No       No     torch.Tensor
nan_to_num                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
nan_to_num\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
nansum                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
narrow                                    No       No        No        No       No     torch.Tensor
ne                                        Yes      Yes       Yes       Yes      Yes    torch.Tensor
ne\_                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
neg                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
neg\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
new_empty                                 No       Yes       No        Yes      No     torch.Tensor
new_empty_strided                         No       Yes       No        Yes      No     torch.Tensor
new_full                                  No       Yes       No        Yes      No     torch.Tensor
new_ones                                  No       Yes       No        Yes      No     torch.Tensor
new_zeros                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
nextafter                                 No       No        No        No       No     torch.Tensor
nextafter\_                               No       No        No        No       No     torch.Tensor
nonzero                                   No       Yes       No        No       Yes    torch.Tensor
norm                                      No       No        No        No       No     torch.Tensor
normal\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
not_equal                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
not_equal\_                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
permute                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
pin_memory                                No       Yes       No        Yes      Yes    torch.Tensor
pow                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
pow\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
prod                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
put                                       No       No        No        No       No     torch.Tensor
put\_                                     No       No        No        No       No     torch.Tensor
random\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
reciprocal                                Yes      Yes       Yes       Yes      Yes    torch.Tensor
reciprocal\_                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
relu                                      No       No        No        No       No     torch.Tensor
relu\_                                    No       No        No        No       No     torch.Tensor
remainder                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
remainder\_                               Yes      Yes       Yes       Yes      Yes    torch.Tensor
repeat                                    Yes      Yes       No        Yes      Yes    torch.Tensor
repeat_interleave                         Yes      Yes       No        Yes      Yes    torch.Tensor
reshape                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
resize\_                                  Yes      Yes       Yes       Yes      Yes    torch.Tensor
resolve_conj                              No       No        No        No       No     torch.Tensor
resolve_neg                               No       No        No        No       No     torch.Tensor
roll                                      No       Yes       Yes       Yes      Yes    torch.Tensor
round                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
round\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
rsqrt                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
rsqrt\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
scatter                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
scatter\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
scatter_add                               No       No        No        No       No     torch.Tensor
scatter_add\_                             No       No        No        No       No     torch.Tensor
scatter_reduce                            No       No        No        No       No     torch.Tensor
scatter_reduce\_                          No       No        No        No       No     torch.Tensor
select                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
sgn                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
sgn\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
sigmoid                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
sigmoid\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
sign                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
sign\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
signbit                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
sin                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
sin\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
sinc                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
sinc\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
sinh                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
sinh\_                                    No       No        No        No       No     torch.Tensor
sort                                      Yes      Yes       Yes       No       No     torch.Tensor
split_with_sizes                          No       Yes       Yes       No       No     torch.Tensor
sqrt                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
sqrt\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
square                                    No       Yes       No        Yes      Yes    torch.Tensor
square\_                                  No       Yes       No        Yes      Yes    torch.Tensor
squeeze                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
squeeze\_                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
std                                       No       No        No        No       No     torch.Tensor
sub                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
sub\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
sum                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
t                                         Yes      Yes       Yes       Yes      Yes    torch.Tensor
take                                      Yes      Yes       No        No       No     torch.Tensor
tan                                       Yes      Yes       Yes       Yes      Yes    torch.Tensor
tan\_                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
tanh                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
tanh\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
to                                        Yes      Yes       Yes       Yes      Yes    torch.Tensor
topk                                      Yes      Yes       Yes       No       No     torch.Tensor
trace                                     Yes      Yes       No        No       No     torch.Tensor
transpose                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
tril                                      No       No        No        Yes      Yes    torch.Tensor
tril\_                                    No       No        No        Yes      Yes    torch.Tensor
triu                                      No       No        No        Yes      Yes    torch.Tensor
triu\_                                    No       No        No        Yes      Yes    torch.Tensor
trunc                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
trunc\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
unbind                                    No       Yes       No        No       No     torch.Tensor
uniform\_                                 Yes      Yes       No        No       No     torch.Tensor
unique                                    No       Yes       No        No       No     torch.Tensor
unsqueeze                                 Yes      Yes       Yes       No       No     torch.Tensor
unsqueeze\_                               Yes      Yes       Yes       No       No     torch.Tensor
var                                       No       No        No        No       No     torch.Tensor
vdot                                      Yes      Yes       No        No       No     torch.Tensor
view                                      Yes      Yes       Yes       Yes      Yes    torch.Tensor
where                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
xlogy                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
xlogy\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
zero\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
entr                                      Yes      Yes       Yes       Yes      Yes    torch.special
erf                                       Yes      Yes       Yes       Yes      Yes    torch.special
erfc                                      Yes      Yes       Yes       Yes      Yes    torch.special
erfcx                                     Yes      Yes       Yes       Yes      Yes    torch.special
erfinv                                    Yes      Yes       Yes       Yes      Yes    torch.special
exp2                                      Yes      Yes       Yes       Yes      Yes    torch.special
expit                                     No       No        No        No       No     torch.special
expm1                                     Yes      Yes       Yes       Yes      Yes    torch.special
log1p                                     Yes      Yes       Yes       Yes      Yes    torch.special
logit                                     No       No        No        No       No     torch.special
logsumexp                                 No       No        No        No       No     torch.special
round                                     Yes      Yes       Yes       Yes      Yes    torch.special
sinc                                      Yes      Yes       Yes       Yes      Yes    torch.special
softmax                                   No       No        No        No       No     torch.special
xlog1py                                   Yes      Yes       Yes       Yes      Yes    torch.special
xlogy                                     Yes      Yes       Yes       Yes      Yes    torch.special
batched_nms                               No       No        No        No       No     torchvision.ops
deform_conv2d                             No       No        No        No       No     torchvision.ops
nms                                       No       No        No        No       No     torchvision.ops
roi_align                                 No       No        No        No       No     torchvision.ops
_safe_softmax                             No       No        No        No       No     torch.ops
====================================  ========= ========= ========= ======== ========  ======================
