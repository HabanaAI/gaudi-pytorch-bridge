
.. _pytorch-operators:

****************************
PyTorch Operators
****************************



Overview
========

This document summarizes the SynapseAI® Software PyTorch supported operators for
Habana® Gaudi®. Note that the operators listed below support only selected
variants and limited optional parameters for Gaudi.

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
conv_transpose2d                         Yes      Yes      Yes     No    torch.nn.functional
conv1d                                   Yes      Yes      Yes     No    torch.nn.functional
conv2d                                   Yes      Yes      Yes     No    torch.nn.functional
conv3d                                   Yes      Yes      Yes     No    torch.nn.functional
dropout                                  Yes      Yes      Yes     No    torch.nn.functional
embedding                                Yes      Yes      Yes     Yes   torch.nn.functional
embedding_bag                            Yes      Yes      Yes     No    torch.nn.functional
elu                                      Yes      Yes      Yes     No    torch.nn.functional
elu\_                                    Yes      Yes      Yes     No    torch.nn.functional
gelu                                     Yes      Yes      Yes     No    torch.nn.functional
glu                                      Yes      No       No      No    torch.nn.functional
glu_jvp                                  Yes      Yes      Yes     No    torch.nn.functional
grid_sample                              Yes      No       No      No    torch.nn.functional
hardshrink                               Yes      Yes      No      No    torch.nn.functional
hardsigmoid                              Yes      Yes      Yes     No    torch.nn.functional
hardtanh                                 Yes      No       No      No    torch.nn.functional
hardtanh\_                               Yes      No       No      No    torch.nn.functional
huber_loss                               Yes      Yes      No      No    torch.nn.functional
instance_norm                            Yes      Yes      No      No    torch.nn.functional
kl_div                                   Yes      Yes      No      No    torch.nn.functional
l1_loss                                  Yes      Yes      No      No    torch.nn.functional
layer_norm                               Yes      Yes      Yes     No    torch.nn.functional
leaky_relu                               Yes      Yes      Yes     No    torch.nn.functional
linear                                   Yes      Yes      Yes     No    torch.nn.functional
log_softmax                              Yes      Yes      Yes     No    torch.nn.functional
logsigmoid                               Yes      No       No      No    torch.nn.functional
max_pool2d                               Yes      Yes      Yes     No    torch.nn.functional
max_pool2d_with_indices                  Yes      Yes      Yes     No    torch.nn.functional
max_pool3d                               Yes      Yes      No      No    torch.nn.functional
max_pool3d_with_indices                  Yes      Yes      No      No    torch.nn.functional
mish                                     Yes      Yes      No      No    torch.nn.functional
mse_loss                                 Yes      Yes      Yes     No    torch.nn.functional
nll_loss                                 Yes      Yes      Yes     No    torch.nn.functional
one_hot                                  Yes      Yes      Yes     No    torch.nn.functional
pad                                      Yes      Yes      No      Yes   torch.nn.functional
pixel_shuffle                            Yes      Yes      Yes     No    torch.nn.functional
prelu                                    Yes      Yes      No      No    torch.nn.functional
relu                                     Yes      Yes      Yes     No    torch.nn.functional
relu\_                                   Yes      Yes      Yes     No    torch.nn.functional
rrelu                                    Yes      Yes      No      No    torch.nn.functional
sigmoid                                  Yes      Yes      Yes     No    torch.nn.functional
silu                                     Yes      Yes      No      No    torch.nn.functional
smooth_l1_loss                           Yes      Yes      No      No    torch.nn.functional
softmax                                  Yes      Yes      Yes     No    torch.nn.functional
softplus                                 Yes      Yes      No      No    torch.nn.functional
softshrink                               Yes      Yes      No      No    torch.nn.functional
tanh                                     Yes      Yes      Yes     No    torch.nn.functional
threshold                                Yes      Yes      No      No    torch.nn.functional
upsample                                 Yes      Yes      Yes     No    torch.nn.functional
upsample_nearest                         Yes      Yes      Yes     No    torch.nn.functional
vector_norm                              Yes      Yes      No      No    torch.linalg
abs                                      Yes      Yes      Yes     No    torch
absolute                                 Yes      Yes      Yes     No    torch
acos                                     Yes      Yes      No      No    torch
acosh                                    Yes      Yes      No      No    torch
add                                      Yes      Yes      Yes     No    torch
addbmm                                   Yes      Yes      Yes     No    torch
addcdiv                                  Yes      Yes      No      No    torch
addcmul                                  Yes      Yes      No      No    torch
addmm                                    Yes      Yes      Yes     No    torch
addmv                                    Yes      Yes      Yes     No    torch
addmv\_                                  Yes      Yes      Yes     No    torch
addr                                     Yes      Yes      Yes     No    torch
all                                      Yes      Yes      No      No    torch
amax                                     Yes      Yes      Yes     Yes   torch
amin                                     Yes      Yes      Yes     Yes   torch
aminmax                                  Yes      Yes      No      Yes   torch
any                                      Yes      Yes      Yes     No    torch
arange                                   Yes      Yes      Yes     No    torch
arccos                                   Yes      Yes      No      No    torch
arccosh                                  Yes      Yes      No      No    torch
arcsin                                   Yes      Yes      No      No    torch
arcsinh                                  Yes      Yes      No      No    torch
arctan                                   Yes      Yes      No      No    torch
arctanh                                  Yes      Yes      No      No    torch
argmax                                   Yes      Yes      Yes     No    torch
argmin                                   Yes      Yes      Yes     No    torch
asin                                     Yes      Yes      No      No    torch
asinh                                    Yes      Yes      No      No    torch
as_strided                               Yes      Yes      Yes     No    torch
atan                                     Yes      Yes      No      No    torch
atan2                                    Yes      Yes      Yes     No    torch
arctan2                                  Yes      Yes      No      No    torch
atanh                                    Yes      Yes      No      No    torch
baddbmm                                  Yes      Yes      Yes     No    torch
bernoulli                                Yes      Yes      Yes     No    torch
bitwise_and                              No       No       No      No    torch
bitwise_left_shift                       No       No       No      No    torch
bitwise_not                              No       No       No      No    torch
bitwise_or                               No       No       No      No    torch
bitwise_right_shift                      No       No       No      No    torch
bitwise_xor                              No       No       No      No    torch
bmm                                      Yes      Yes      Yes     No    torch
broadcast_tensors                        Yes      Yes      No      No    torch
cat                                      Yes      Yes      Yes     Yes   torch
ceil                                     Yes      Yes      Yes     No    torch
chunk                                    Yes      Yes      No      No    torch
clamp                                    Yes      Yes      Yes     Yes   torch
clip                                     Yes      Yes      No      No    torch
clone                                    Yes      Yes      Yes     Yes   torch
conj                                     Yes      Yes      No      No    torch
copysign                                 Yes      Yes      No      No    torch
cos                                      Yes      Yes      Yes     No    torch
cosh                                     Yes      Yes      Yes     No    torch
count_nonzero                            Yes      Yes      Yes     No    torch
cross                                    Yes      Yes      No      No    torch
cumsum                                   Yes      Yes      Yes     No    torch
cumprod                                  Yes      Yes      No      No    torch
diag                                     Yes      Yes      Yes     No    torch
div                                      Yes      Yes      Yes     No    torch
divide                                   Yes      Yes      Yes     No    torch
dot                                      Yes      Yes      Yes     No    torch
embedding                                Yes      Yes      Yes     Yes   torch
embedding_renorm\_                       Yes      Yes      Yes     No    torch
empty                                    Yes      Yes      Yes     Yes   torch
empty_like                               Yes      Yes      Yes     Yes   torch
empty_strided                            Yes      Yes      Yes     Yes   torch
eq                                       Yes      Yes      Yes     Yes   torch
equal                                    Yes      Yes      No      No    torch
erf                                      Yes      Yes      No      No    torch
erfc                                     Yes      Yes      No      No    torch
erfinv                                   Yes      No       No      No    torch
exp                                      Yes      Yes      Yes     No    torch
exp2                                     Yes      Yes      No      No    torch
expm1                                    Yes      No       No      No    torch
eye                                      Yes      Yes      No      No    torch
fill                                     Yes      Yes      Yes     Yes   torch
fill\_                                   Yes      Yes      Yes     Yes   torch
flatten                                  Yes      Yes      No      No    torch
flip                                     Yes      Yes      Yes     No    torch
floor                                    Yes      Yes      Yes     No    torch
floor_divide                             Yes      Yes      No      No    torch
fmax                                     Yes      Yes      Yes     Yes   torch
fmin                                     Yes      Yes      Yes     Yes   torch
fmod                                     Yes      Yes      No      No    torch
frac                                     Yes      Yes      No      No    torch
frexp                                    Yes      Yes      Yes     No    torch
full                                     Yes      Yes      Yes     Yes   torch
full_like                                Yes      Yes      Yes     Yes   torch
gather                                   Yes      Yes      Yes     No    torch
ge                                       Yes      Yes      Yes     Yes   torch
greater                                  Yes      Yes      Yes     No    torch
greater_equal                            Yes      Yes      Yes     No    torch
grid_sampler_2d                          Yes      Yes      No      No    torch
gt                                       Yes      Yes      Yes     Yes   torch
heaviside                                Yes      Yes      No      No    torch
hypot                                    Yes      Yes      No      No    torch
index_fill                               Yes      Yes      Yes     Yes   torch
index_select                             Yes      Yes      Yes     Yes   torch
is_complex                               Yes      Yes      No      No    torch
is_floating_point                        Yes      Yes      No      No    torch
is_nonzero                               Yes      Yes      No      No    torch
isfinite                                 Yes      Yes      Yes     No    torch
isinf                                    Yes      Yes      Yes     No    torch
isnan                                    Yes      Yes      Yes     No    torch
isneginf                                 Yes      Yes      Yes     No    torch
isposinf                                 Yes      Yes      Yes     No    torch
kthvalue                                 Yes      Yes      Yes     No    torch
le                                       Yes      Yes      Yes     Yes   torch
lerp                                     Yes      Yes      No      No    torch
less                                     Yes      Yes      Yes     No    torch
less_equal                               Yes      Yes      Yes     No    torch
linspace                                 Yes      Yes      Yes     No    torch
log                                      Yes      Yes      Yes     No    torch
log10                                    Yes      Yes      No      No    torch
log1p                                    Yes      Yes      No      No    torch
log2                                     Yes      Yes      Yes     No    torch
logaddexp                                Yes      Yes      No      No    torch
logaddexp2                               Yes      Yes      No      No    torch
logcumsumexp                             Yes      Yes      No      No    torch
logical_and                              Yes      Yes      Yes     No    torch
logical_not                              Yes      Yes      Yes     No    torch
logical_or                               Yes      Yes      Yes     No    torch
logical_xor                              Yes      Yes      Yes     No    torch
logit                                    Yes      Yes      No      No    torch
logspace                                 Yes      Yes      No      No    torch
logsumexp                                Yes      Yes      No      No    torch
lt                                       Yes      Yes      Yes     Yes   torch
masked_fill                              Yes      Yes      Yes     Yes   torch
masked_scatter                           Yes      Yes      Yes     No    torch
masked_select                            Yes      Yes      No      No    torch
matmul                                   Yes      Yes      No      No    torch
max                                      Yes      Yes      No      Yes   torch
maximum                                  Yes      Yes      Yes     Yes   torch
mean                                     Yes      Yes      Yes     No    torch
median                                   Yes      Yes      Yes     No    torch
meshgrid                                 Yes      Yes      No      No    torch
min                                      Yes      Yes      Yes     Yes   torch
minimum                                  Yes      Yes      Yes     Yes   torch
mm                                       Yes      Yes      Yes     No    torch
mul                                      Yes      Yes      Yes     No    torch
mv                                       Yes      Yes      Yes     No    torch
nan_to_num                               Yes      Yes      No      No    torch
nansum                                   Yes      Yes      Yes     No    torch
narrow                                   Yes      No       No      No    torch
native_group_norm                        Yes      Yes      No      No    torch
native_layer_norm                        Yes      Yes      Yes     No    torch
ne                                       Yes      Yes      Yes     Yes   torch
neg                                      Yes      Yes      Yes     No    torch
nextafter                                Yes      Yes      Yes     No    torch
nonzero                                  Yes      Yes      No      No    torch
norm                                     Yes      Yes      Yes     No    torch
normal                                   Yes      Yes      Yes     No    torch
not_equal                                Yes      Yes      Yes     No    torch
ones                                     Yes      Yes      No      Yes   torch
ones_like                                Yes      Yes      No      Yes   torch
pixel_shuffle                            Yes      Yes      Yes     No    torch
poisson                                  Yes      Yes      No      No    torch
pow                                      Yes      Yes      Yes     No    torch
prod                                     Yes      Yes      Yes     No    torch
randperm                                 No       No       Yes     No    torch
reciprocal                               Yes      Yes      Yes     Yes   torch
remainder                                Yes      Yes      No      No    torch
reshape                                  Yes      Yes      Yes     Yes   torch
resolve_conj                             Yes      Yes      No      No    torch
resolve_neg                              Yes      Yes      No      No    torch
result_type                              Yes      Yes      No      No    torch
roll                                     Yes      Yes      No      Yes   torch
round                                    Yes      Yes      Yes     No    torch
rsqrt                                    Yes      Yes      Yes     No    torch
rsub                                     Yes      Yes      Yes     No    torch
scalar_tensor                            Yes      Yes      Yes     No    torch
scatter                                  Yes      Yes      Yes     No    torch
scatter_add                              Yes      Yes      Yes     No    torch
scatter_reduce                           Yes      Yes      Yes     No    torch
searchsorted                             Yes      Yes      Yes     No    torch
select                                   Yes      Yes      Yes     Yes   torch
sgn                                      Yes      Yes      Yes     No    torch
sigmoid                                  Yes      Yes      Yes     No    torch
sign                                     Yes      Yes      Yes     No    torch
signbit                                  Yes      Yes      Yes     No    torch
sin                                      Yes      Yes      Yes     No    torch
sinc                                     Yes      Yes      No      No    torch
sinh                                     Yes      Yes      No      No    torch
sort                                     Yes      Yes      Yes     No    torch
split_with_sizes                         Yes      Yes      No      Yes   torch
sqrt                                     Yes      Yes      Yes     No    torch
square                                   Yes      Yes      No      No    torch
squeeze                                  Yes      Yes      No      Yes   torch
stack                                    Yes      Yes      No      No    torch
std                                      Yes      Yes      No      No    torch
std_mean                                 Yes      Yes      No      No    torch
sub                                      Yes      Yes      Yes     No    torch
sum                                      Yes      Yes      Yes     Yes   torch
t                                        Yes      Yes      Yes     Yes   torch
take                                     Yes      Yes      No      No    torch
tan                                      Yes      Yes      No      No    torch
tanh                                     Yes      Yes      Yes     No    torch
topk                                     Yes      Yes      Yes     No    torch
trace                                    Yes      Yes      No      No    torch
transpose                                Yes      Yes      Yes     Yes   torch
tril                                     Yes      Yes      No      No    torch
triu                                     Yes      Yes      No      No    torch
trunc                                    Yes      Yes      Yes     No    torch
unbind                                   Yes      Yes      No      No    torch
unique                                   Yes      No       No      No    torch
_unique                                  Yes      No       No      No    torch
_unique2                                 Yes      No       No      No    torch
unsqueeze                                Yes      Yes      No      Yes   torch
var                                      Yes      Yes      No      No    torch
var_mean                                 Yes      Yes      No      No    torch
vdot                                     Yes      Yes      No      No    torch
where                                    Yes      Yes      Yes     No    torch
xlogy                                    Yes      Yes      Yes     No    torch
zero\_                                   Yes      Yes      Yes     Yes   torch
zeros                                    Yes      Yes      Yes     Yes   torch
zeros_like                               Yes      Yes      Yes     Yes   torch
_adaptive_avg_pool2d                     Yes      Yes      Yes     No    torch
_adaptive_avg_pool3d                     Yes      Yes      Yes     No    torch
_efficientzerotensor                     Yes      Yes      Yes     No    torch
_fused_dropout                           Yes      Yes      Yes     No    torch
_masked_scale                            Yes      Yes      Yes     No    torch
_native_batch_norm_legit                 Yes      Yes      Yes     No    torch
_native_batch_norm_legit_no_training     Yes      Yes      Yes     No    torch
_resize_output\_                         Yes      Yes      No      No    torch
_weight_norm_interface                   Yes      Yes      Yes     No    torch
AdaptiveAvgPool1d                        Yes      Yes      Yes     No    torch.nn
AdaptiveAvgPool2d                        Yes      Yes      Yes     No    torch.nn
AdaptiveAvgPool3d                        Yes      Yes      Yes     No    torch.nn
AvgPool1d                                Yes      Yes      Yes     No    torch.nn
AvgPool2d                                Yes      Yes      Yes     No    torch.nn
AvgPool3d                                Yes      Yes      Yes     No    torch.nn
BatchNorm1d                              Yes      Yes      Yes     No    torch.nn
BatchNorm2d                              Yes      Yes      Yes     No    torch.nn
BCELoss                                  Yes      Yes      Yes     No    torch.nn
BCEWithLogitsLoss                        Yes      Yes      Yes     No    torch.nn
ChannelShuffle                           Yes      Yes      Yes     No    torch.nn
ConstantPad1d                            Yes      Yes      Yes     No    torch.nn
Conv1d                                   Yes      Yes      Yes     No    torch.nn
Conv2d                                   Yes      Yes      Yes     No    torch.nn
Conv3d                                   Yes      Yes      Yes     No    torch.nn
ConvTranspose2d                          Yes      Yes      Yes     No    torch.nn
ConvTranspose3d                          Yes      Yes      Yes     No    torch.nn
CrossEntropyLoss                         Yes      Yes      No      No    torch.nn
Dropout                                  Yes      Yes      Yes     No    torch.nn
ELU                                      Yes      Yes      Yes     No    torch.nn
Embedding                                Yes      Yes      Yes     Yes   torch.nn
EmbeddingBag                             Yes      Yes      Yes     No    torch.nn
GELU                                     Yes      Yes      Yes     No    torch.nn
GLU                                      Yes      No       No      No    torch.nn
Hardshrink                               Yes      Yes      No      No    torch.nn
Hardsigmoid                              Yes      Yes      Yes     No    torch.nn
Hardtanh                                 Yes      No       No      No    torch.nn
HuberLoss                                Yes      Yes      No      No    torch.nn
InstanceNorm2d                           Yes      Yes      No      No    torch.nn
KLDivLoss                                Yes      Yes      No      No    torch.nn
LayerNorm                                Yes      Yes      Yes     No    torch.nn
LeakyReLU                                Yes      Yes      Yes     No    torch.nn
Linear                                   Yes      Yes      Yes     No    torch.nn
LogSigmoid                               Yes      No       No      No    torch.nn
LogSoftmax                               Yes      Yes      Yes     No    torch.nn
MaxPool2d                                Yes      Yes      Yes     No    torch.nn
MaxPool3d                                Yes      Yes      No      No    torch.nn
Mish                                     Yes      Yes      No      No    torch.nn
MSELoss                                  Yes      Yes      Yes     No    torch.nn
NLLLoss                                  Yes      Yes      Yes     No    torch.nn
PixelShuffle                             Yes      Yes      Yes     No    torch.nn
PReLU                                    Yes      Yes      No      No    torch.nn
ReLU                                     Yes      Yes      Yes     No    torch.nn
ReflectionPad1d                          Yes      No       No      No    torch.nn
ReflectionPad2d                          Yes      No       No      No    torch.nn
ReflectionPad3d                          Yes      No       No      No    torch.nn
ReplicationPad1d                         Yes      Yes      No      No    torch.nn
ReplicationPad2d                         Yes      Yes      No      No    torch.nn
ReplicationPad3d                         Yes      Yes      No      No    torch.nn
RReLU                                    Yes      Yes      No      No    torch.nn
SiLU                                     Yes      Yes      No      No    torch.nn
Softmax                                  Yes      Yes      Yes     No    torch.nn
Softplus                                 Yes      Yes      No      No    torch.nn
Softshrink                               Yes      Yes      No      No    torch.nn
Threshold                                Yes      Yes      No      No    torch.nn
SmoothL1Loss                             Yes      Yes      No      No    torch.nn
Upsample                                 Yes      Yes      Yes     No    torch.nn
UpsamplingNearest2d                      Yes      Yes      Yes     No    torch.nn
weight_norm                              Yes      Yes      No      No    torch.nn.utils
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
acos                                     Yes      Yes      No      No    torch.Tensor
acos\_                                   Yes      Yes      No      No    torch.Tensor
acosh                                    Yes      Yes      No      No    torch.Tensor
acosh\_                                  Yes      Yes      No      No    torch.Tensor
add                                      Yes      Yes      Yes     No    torch.Tensor
add\_                                    Yes      Yes      Yes     No    torch.Tensor
addbmm                                   Yes      Yes      Yes     No    torch.Tensor
addbmm\_                                 Yes      Yes      Yes     No    torch.Tensor
addcdiv                                  Yes      Yes      No      No    torch.Tensor
addcdiv\_                                Yes      Yes      Yes     No    torch.Tensor
addcmul                                  Yes      Yes      No      No    torch.Tensor
addcmul\_                                Yes      Yes      Yes     No    torch.Tensor
addmm                                    Yes      Yes      Yes     No    torch.Tensor
addmm\_                                  Yes      Yes      Yes     No    torch.Tensor
addmv                                    Yes      Yes      Yes     No    torch.Tensor
addmv\_                                  Yes      Yes      Yes     No    torch.Tensor
addr                                     Yes      Yes      Yes     No    torch.Tensor
addr\_                                   Yes      Yes      Yes     No    torch.Tensor
all                                      Yes      Yes      No      No    torch.Tensor
amax                                     Yes      Yes      Yes     Yes   torch.Tensor
amin                                     Yes      Yes      Yes     Yes   torch.Tensor
aminmax                                  Yes      Yes      No      Yes   torch.Tensor
any                                      Yes      Yes      Yes     No    torch.Tensor
arccos                                   Yes      Yes      No      No    torch.Tensor
arccos\_                                 Yes      Yes      No      No    torch.Tensor
arccosh                                  Yes      Yes      No      No    torch.Tensor
arccosh\_                                Yes      Yes      No      No    torch.Tensor
arcsin                                   Yes      Yes      No      No    torch.Tensor
arcsin\_                                 Yes      Yes      No      No    torch.Tensor
arcsinh                                  Yes      Yes      No      No    torch.Tensor
arcsinh\_                                Yes      Yes      No      No    torch.Tensor
arctan                                   Yes      Yes      No      No    torch.Tensor
arctan\_                                 Yes      Yes      No      No    torch.Tensor
arctanh                                  Yes      Yes      No      No    torch.Tensor
arctanh\_                                Yes      Yes      No      No    torch.Tensor
argmax                                   Yes      Yes      Yes     No    torch.Tensor
argmin                                   Yes      Yes      Yes     No    torch.Tensor
asin                                     Yes      Yes      No      No    torch.Tensor
asin\_                                   Yes      Yes      No      No    torch.Tensor
asinh                                    Yes      Yes      No      No    torch.Tensor
asinh\_                                  Yes      Yes      No      No    torch.Tensor
atan2                                    Yes      Yes      No      No    torch.Tensor
atan2\_                                  Yes      Yes      No      No    torch.Tensor
arctan2                                  Yes      Yes      No      No    torch.Tensor
arctan2\_                                Yes      Yes      No      No    torch.Tensor
atanh                                    Yes      Yes      No      No    torch.Tensor
atanh\_                                  Yes      Yes      No      No    torch.Tensor
baddbmm                                  Yes      Yes      Yes     No    torch.Tensor
baddbmm\_                                Yes      Yes      Yes     No    torch.Tensor
bernoulli                                Yes      Yes      Yes     No    torch.Tensor
bitwise_and                              No       No       No      No    torch.Tensor
bitwise_left_shift                       No       No       No      No    torch.Tensor
bitwise_not                              No       No       No      No    torch.Tensor
bitwise_or                               No       No       No      No    torch.Tensor
bitwise_right_shift                      No       No       No      No    torch.Tensor
bitwise_xor                              No       No       No      No    torch.Tensor
bmm                                      Yes      Yes      Yes     No    torch.Tensor
ceil                                     Yes      Yes      Yes     No    torch.Tensor
clamp                                    Yes      Yes      Yes     Yes   torch.Tensor
clamp\_                                  Yes      Yes      Yes     Yes   torch.Tensor
clamp_max                                Yes      Yes      Yes     Yes   torch.Tensor
clamp_min                                Yes      Yes      Yes     Yes   torch.Tensor
clip                                     Yes      Yes      No      No    torch.Tensor
clip\_                                   Yes      Yes      No      No    torch.Tensor
clone                                    Yes      Yes      Yes     Yes   torch.Tensor
conj                                     Yes      Yes      No      No    torch.Tensor
copy\_                                   Yes      Yes      No      Yes   torch.Tensor
copysign                                 Yes      Yes      No      No    torch.Tensor
copysign\_                               Yes      Yes      No      No    torch.Tensor
cos                                      Yes      Yes      Yes     No    torch.Tensor
cos\_                                    Yes      Yes      Yes     No    torch.Tensor
cosh\_                                   Yes      Yes      Yes     No    torch.Tensor
count_nonzero                            Yes      Yes      Yes     No    torch.Tensor
cross                                    Yes      Yes      No      No    torch.Tensor
cumprod                                  Yes      Yes      No      No    torch.Tensor
cumsum                                   Yes      Yes      Yes     No    torch.Tensor
diag                                     Yes      Yes      Yes     No    torch.Tensor
div                                      Yes      Yes      Yes     No    torch.Tensor
div\_                                    Yes      Yes      Yes     No    torch.Tensor
dot                                      Yes      Yes      Yes     No    torch.Tensor
fill\_                                   Yes      Yes      Yes     Yes   torch.Tensor
eq                                       Yes      Yes      Yes     Yes   torch.Tensor
eq\_                                     Yes      Yes      Yes     Yes   torch.Tensor
equal                                    Yes      Yes      No      No    torch.Tensor
erf                                      Yes      Yes      No      No    torch.Tensor
erf\_                                    Yes      Yes      No      No    torch.Tensor
erfinv                                   Yes      No       No      No    torch.Tensor
erfinv\_                                 Yes      No       No      No    torch.Tensor
erfc\_                                   Yes      Yes      No      No    torch.Tensor
erfc                                     Yes      Yes      No      No    torch.Tensor
exp                                      Yes      Yes      Yes     No    torch.Tensor
exp\_                                    Yes      Yes      Yes     No    torch.Tensor
expand                                   Yes      Yes      Yes     No    torch.Tensor
expand_as                                Yes      Yes      Yes     No    torch.Tensor
expm1                                    Yes      No       No      No    torch.Tensor
expm1\_                                  Yes      No       No      No    torch.Tensor
exponential\_                            Yes      Yes      No      No    torch.Tensor
flatten                                  Yes      Yes      No      No    torch.Tensor
flip                                     Yes      Yes      Yes     No    torch.Tensor
floor                                    Yes      Yes      Yes     No    torch.Tensor
floor\_                                  Yes      Yes      Yes     No    torch.Tensor
floor_divide                             Yes      Yes      No      No    torch.Tensor
floor_divide\_                           Yes      Yes      No      No    torch.Tensor
fmax                                     Yes      Yes      Yes     Yes   torch.Tensor
fmin                                     Yes      Yes      Yes     Yes   torch.Tensor
fmod                                     Yes      Yes      No      No    torch.Tensor
fmod\_                                   Yes      Yes      No      No    torch.Tensor
frac                                     Yes      Yes      No      No    torch.Tensor
frexp                                    Yes      Yes      Yes     No    torch.Tensor
gather                                   Yes      Yes      Yes     No    torch.Tensor
ge                                       Yes      Yes      Yes     Yes   torch.Tensor
ge\_                                     Yes      Yes      Yes     Yes   torch.Tensor
geometric\_                              Yes      Yes      No      No    torch.Tensor
greater                                  Yes      Yes      Yes     No    torch.Tensor
greater_equal                            Yes      Yes      Yes     No    torch.Tensor
gt                                       Yes      Yes      Yes     Yes   torch.Tensor
hardshrink                               Yes      Yes      No      No    torch.Tensor
heaviside                                Yes      Yes      No      No    torch.Tensor
hypot                                    Yes      Yes      No      No    torch.Tensor
index_add\_                              Yes      Yes      Yes     No    torch.Tensor
index_copy\_                             Yes      Yes      Yes     Yes   torch.Tensor
index_fill                               Yes      Yes      Yes     Yes   torch.Tensor
index_fill\_                             Yes      Yes      Yes     Yes   torch.Tensor
index_put                                Yes      Yes      No      No    torch.Tensor
index_put\_                              Yes      Yes      No      No    torch.Tensor
index_select                             Yes      Yes      Yes     Yes   torch.Tensor
is_complex                               Yes      Yes      No      No    torch.Tensor
is_floating_point                        Yes      Yes      No      No    torch.Tensor
isfinite                                 Yes      Yes      Yes     No    torch.Tensor
isinf                                    Yes      Yes      Yes     No    torch.Tensor
isnan                                    Yes      Yes      Yes     No    torch.Tensor
isneginf                                 Yes      Yes      Yes     No    torch.Tensor
isposinf                                 Yes      Yes      Yes     No    torch.Tensor
item                                     Yes      No       No      No    torch.Tensor
kthvalue                                 Yes      Yes      Yes     No    torch.Tensor
le                                       Yes      Yes      Yes     Yes   torch.Tensor
le\_                                     Yes      Yes      Yes     Yes   torch.Tensor
lerp                                     Yes      Yes      No      No    torch.Tensor
lerp\_                                   Yes      Yes      No      No    torch.Tensor
less                                     Yes      Yes      Yes     No    torch.Tensor
less\_                                   Yes      Yes      Yes     No    torch.Tensor
less_equal                               Yes      Yes      Yes     No    torch.Tensor
less_equal\_                             Yes      Yes      Yes     No    torch.Tensor
log                                      Yes      Yes      Yes     No    torch.Tensor
log\_                                    Yes      Yes      Yes     No    torch.Tensor
log_normal\_                             Yes      Yes      No      No    torch.Tensor
log_softmax                              Yes      Yes      Yes     No    torch.Tensor
log10                                    Yes      Yes      No      No    torch.Tensor
log1p                                    Yes      Yes      No      No    torch.Tensor
log1p\_                                  Yes      Yes      No      No    torch.Tensor
log2                                     Yes      Yes      Yes     No    torch.Tensor
log2\_                                   Yes      Yes      Yes     No    torch.Tensor
logaddexp                                Yes      Yes      No      No    torch.Tensor
logaddexp2                               Yes      Yes      No      No    torch.Tensor
logcumsumexp                             Yes      Yes      No      No    torch.Tensor
logical_and                              Yes      Yes      Yes     No    torch.Tensor
logical_and\_                            Yes      Yes      Yes     No    torch.Tensor
logical_not                              Yes      Yes      Yes     No    torch.Tensor
logical_not\_                            Yes      Yes      Yes     No    torch.Tensor
logical_or                               Yes      Yes      Yes     No    torch.Tensor
logical_or\_                             Yes      Yes      Yes     No    torch.Tensor
logical_xor\_                            Yes      Yes      Yes     No    torch.Tensor
logical_xor\_                            Yes      Yes      Yes     No    torch.Tensor
logit                                    Yes      Yes      No      No    torch.Tensor
logsumexp                                Yes      Yes      No      No    torch.Tensor
lt                                       Yes      Yes      Yes     Yes   torch.Tensor
masked_fill                              Yes      Yes      Yes     Yes   torch.Tensor
masked_fill\_                            Yes      Yes      Yes     Yes   torch.Tensor
masked_scatter                           Yes      Yes      Yes     No    torch.Tensor
masked_scatter\_                         Yes      Yes      Yes     No    torch.Tensor
masked_select                            Yes      Yes      No      No    torch.Tensor
matmul                                   Yes      Yes      No      No    torch.Tensor
max                                      Yes      Yes      No      Yes   torch.Tensor
maximum                                  Yes      Yes      Yes     Yes   torch.Tensor
mean                                     Yes      Yes      Yes     No    torch.Tensor
median                                   Yes      Yes      Yes     No    torch.Tensor
min                                      Yes      Yes      Yes     Yes   torch.Tensor
minimum                                  Yes      Yes      Yes     Yes   torch.Tensor
mm                                       Yes      Yes      Yes     No    torch.Tensor
mul                                      Yes      Yes      Yes     No    torch.Tensor
mul\_                                    Yes      Yes      Yes     No    torch.Tensor
mv                                       Yes      Yes      Yes     No    torch.Tensor
nan_to_num                               Yes      Yes      No      No    torch.Tensor
nansum                                   Yes      Yes      Yes     No    torch.Tensor
narrow                                   Yes      No       No      No    torch.Tensor
ne                                       Yes      Yes      Yes     Yes   torch.Tensor
ne\_                                     Yes      Yes      Yes     Yes   torch.Tensor
neg                                      Yes      Yes      Yes     No    torch.Tensor
new_empty                                Yes      Yes      No      No    torch.Tensor
new_empty_strided                        Yes      Yes      No      No    torch.Tensor
new_full                                 Yes      Yes      No      No    torch.Tensor
new_ones                                 Yes      Yes      No      No    torch.Tensor
new_zeros                                Yes      Yes      Yes     No    torch.Tensor
nextafter                                Yes      Yes      Yes     No    torch.Tensor
nonzero                                  Yes      Yes      No      No    torch.Tensor
norm                                     Yes      Yes      Yes     No    torch.Tensor
normal\_                                 Yes      Yes      Yes     No    torch.Tensor
permute                                  Yes      Yes      No      No    torch.Tensor
pin_memory                               Yes      Yes      No      No    torch.Tensor
pow                                      Yes      Yes      Yes     No    torch.Tensor
pow\_                                    Yes      Yes      Yes     No    torch.Tensor
prod                                     Yes      Yes      Yes     No    torch.Tensor
put\_                                    Yes      Yes      Yes     No    torch.Tensor
random\_                                 Yes      Yes      Yes     No    torch.Tensor
reciprocal                               Yes      Yes      Yes     Yes   torch.Tensor
reciprocal\_                             Yes      Yes      Yes     Yes   torch.Tensor
remainder                                Yes      Yes      No      No    torch.Tensor
remainder\_                              Yes      Yes      No      No    torch.Tensor
repeat                                   Yes      Yes      Yes     Yes   torch.Tensor
repeat_interleave                        Yes      Yes      Yes     No    torch.Tensor
reshape                                  Yes      Yes      Yes     Yes   torch.Tensor
roll                                     Yes      Yes      No      Yes   torch.Tensor
resize\_                                 Yes      Yes      Yes     No    torch.Tensor
round                                    Yes      Yes      Yes     No    torch.Tensor
round\_                                  Yes      Yes      Yes     No    torch.Tensor
rsqrt                                    Yes      Yes      Yes     No    torch.Tensor
rsqrt\_                                  Yes      Yes      Yes     No    torch.Tensor
scatter                                  Yes      Yes      Yes     No    torch.Tensor
scatter\_                                Yes      Yes      Yes     No    torch.Tensor
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
sinh                                     Yes      Yes      No      No    torch.Tensor
sinh\_                                   Yes      Yes      No      No    torch.Tensor
sinc                                     Yes      Yes      No      No    torch.Tensor
sort                                     Yes      Yes      Yes     No    torch.Tensor
split_with_sizes                         Yes      Yes      No      Yes   torch.Tensor
sqrt                                     Yes      Yes      Yes     No    torch.Tensor
square                                   Yes      Yes      No      No    torch.Tensor
square\_                                 Yes      Yes      No      No    torch.Tensor
squeeze                                  Yes      Yes      No      Yes   torch.Tensor
squeeze\_                                Yes      Yes      No      Yes   torch.Tensor
std                                      Yes      Yes      No      No    torch.Tensor
sub                                      Yes      Yes      Yes     No    torch.Tensor
sub\_                                    Yes      Yes      Yes     No    torch.Tensor
sum                                      Yes      Yes      Yes     Yes   torch.Tensor
T                                        Yes      Yes      Yes     No    torch.Tensor
t                                        Yes      Yes      Yes     Yes   torch.Tensor
tan                                      Yes      Yes      No      No    torch.Tensor
tan\_                                    Yes      Yes      No      No    torch.Tensor
tanh                                     Yes      Yes      Yes     No    torch.Tensor
tanh\_                                   Yes      Yes      Yes     No    torch.Tensor
to                                       Yes      Yes      Yes     Yes   torch.Tensor
topk                                     Yes      Yes      Yes     No    torch.Tensor
trace                                    Yes      Yes      No      No    torch.Tensor
take                                     Yes      Yes      No      No    torch.Tensor
transpose                                Yes      Yes      Yes     Yes   torch.Tensor
tril                                     Yes      Yes      No      No    torch.Tensor
tril\_                                   Yes      Yes      No      No    torch.Tensor
triu                                     Yes      Yes      No      No    torch.Tensor
triu\_                                   Yes      Yes      No      No    torch.Tensor
trunc                                    Yes      Yes      Yes     No    torch.Tensor
trunc\_                                  Yes      Yes      Yes     No    torch.Tensor
unbind                                   Yes      Yes      No      No    torch.Tensor
unsqueeze                                Yes      Yes      No      Yes   torch.Tensor
uniform\_                                Yes      Yes      Yes     No    torch.Tensor
unique                                   Yes      No       No      No    torch.Tensor
var                                      Yes      Yes      No      No    torch.Tensor
vdot                                     Yes      Yes      No      No    torch.Tensor
view                                     Yes      No       No      No    torch.Tensor
where                                    Yes      Yes      Yes     No    torch.Tensor
xlogy                                    Yes      Yes      Yes     No    torch.Tensor
xlogy\_                                  Yes      Yes      Yes     No    torch.Tensor
zero\_                                   Yes      Yes      Yes     Yes   torch.Tensor
entr                                     Yes      No       No      No    torch.special
erf                                      Yes      Yes      No      No    torch.special
erfc                                     Yes      Yes      No      No    torch.special
exp2                                     Yes      Yes      No      No    torch.special
expit                                    Yes      Yes      No      No    torch.special
expm1                                    Yes      No       No      No    torch.special
erfinv                                   Yes      No       No      No    torch.special
log1p                                    Yes      Yes      No      No    torch.special
logit                                    Yes      Yes      No      No    torch.special
logsumexp                                Yes      Yes      No      No    torch.special
round                                    Yes      Yes      Yes     No    torch.special
sinc                                     Yes      Yes      No      No    torch.special
softmax                                  Yes      Yes      Yes     No    torch.special
xlog1py                                  Yes      Yes      No      No    torch.special
xlogy                                    Yes      Yes      Yes     No    torch.special
batched_nms                              Yes      Yes      No      No    torchvision.ops
nms                                      Yes      Yes      No      No    torchvision.ops
roi_align                                Yes      No       No      No    torchvision.ops
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
conv_transpose2d                          No       No        No        No       No     torch.nn.functional
conv1d                                    No       No        No        No       No     torch.nn.functional
conv2d                                    No       No        No        No       No     torch.nn.functional
conv3d                                    No       No        No        No       No     torch.nn.functional
dropout                                   No       No        No        No       No     torch.nn.functional
embedding                                 No       Yes       Yes       Yes      Yes    torch.nn.functional
embedding_bag                             No       No        No        No       No     torch.nn.functional
elu                                       No       No        No        No       No     torch.nn.functional
elu\_                                     No       No        No        No       No     torch.nn.functional
gelu                                      No       No        No        No       No     torch.nn.functional
glu                                       No       No        No        No       No     torch.nn.functional
glu_jvp                                   No       No        No        No       No     torch.nn.functional
grid_sample                               No       No        No        No       No     torch.nn.functional
hardshrink                                No       No        No        No       No     torch.nn.functional
hardsigmoid                               No       No        No        No       No     torch.nn.functional
hardtanh                                  No       No        No        No       No     torch.nn.functional
hardtanh\_                                No       No        No        No       No     torch.nn.functional
huber_loss                                No       No        No        No       No     torch.nn.functional
instance_norm                             No       No        No        No       No     torch.nn.functional
kl_div                                    No       No        No        No       No     torch.nn.functional
l1_loss                                   No       No        No        No       No     torch.nn.functional
layer_norm                                No       No        No        No       No     torch.nn.functional
leaky_relu                                No       No        No        No       No     torch.nn.functional
linear                                    No       No        No        No       No     torch.nn.functional
log_softmax                               No       No        No        No       No     torch.nn.functional
logsigmoid                                No       No        No        No       No     torch.nn.functional
max_pool2d                                No       No        No        No       No     torch.nn.functional
max_pool2d_with_indices                   No       No        No        No       No     torch.nn.functional
max_pool3d                                No       No        No        No       No     torch.nn.functional
max_pool3d_with_indices                   No       No        No        No       No     torch.nn.functional
mish                                      No       No        No        No       No     torch.nn.functional
mse_loss                                  No       No        No        No       No     torch.nn.functional
nll_loss                                  No       No        No        No       No     torch.nn.functional
one_hot                                   Yes      Yes       No        No       No     torch.nn.functional
pad                                       No       No        No        No       No     torch.nn.functional
pixel_shuffle                             No       Yes       Yes       Yes      Yes    torch.nn.functional
prelu                                     No       No        No        No       No     torch.nn.functional
relu                                      No       No        No        No       No     torch.nn.functional
relu\_                                    No       No        No        No       No     torch.nn.functional
rrelu                                     No       No        No        No       No     torch.nn.functional
sigmoid                                   No       No        No        No       No     torch.nn.functional
silu                                      No       No        No        No       No     torch.nn.functional
smooth_l1_loss                            No       No        No        No       No     torch.nn.functional
softmax                                   No       No        No        No       No     torch.nn.functional
softplus                                  No       No        No        No       No     torch.nn.functional
softshrink                                No       No        No        No       No     torch.nn.functional
tanh                                      No       No        No        No       No     torch.nn.functional
threshold                                 No       No        No        No       No     torch.nn.functional
upsample                                  No       No        No        No       No     torch.nn.functional
upsample_nearest                          No       No        No        No       No     torch.nn.functional
vector_norm                               No       No        No        No       No     torch.linalg
abs                                       No       No        Yes       No       No     torch
absolute                                  No       No        No        No       No     torch
acos                                      No       No        No        No       No     torch
acosh                                     No       No        No        No       No     torch
add                                       Yes      Yes       No        No       No     torch
addbmm                                    No       No        No        No       No     torch
addcdiv                                   No       No        Yes       No       No     torch
addcmul                                   No       No        Yes       No       No     torch
addmm                                     No       No        No        No       No     torch
addmv                                     No       No        No        No       No     torch
addmv\_                                   No       No        No        No       No     torch
addr                                      No       No        No        No       No     torch
all                                       No       Yes       Yes       No       Yes    torch
amax                                      No       Yes       Yes       No       No     torch
amin                                      No       Yes       Yes       No       No     torch
aminmax                                   Yes      Yes       Yes       Yes      No     torch
any                                       No       Yes       Yes       Yes      Yes    torch
arange                                    Yes      Yes       Yes       Yes      Yes    torch
arccos                                    No       No        No        No       No     torch
arccosh                                   No       No        No        No       No     torch
arcsin                                    No       No        No        No       No     torch
arcsinh                                   No       No        No        No       No     torch
arctan                                    No       No        No        No       No     torch
arctanh                                   No       No        No        No       No     torch
argmax                                    No       Yes       Yes       Yes      Yes    torch
argmin                                    No       Yes       Yes       Yes      Yes    torch
asin                                      No       No        No        No       No     torch
asinh                                     No       No        No        No       No     torch
as_strided                                Yes      Yes       No        Yes      Yes    torch
atan                                      No       No        No        No       No     torch
atan2                                     No       No        No        No       No     torch
arctan2                                   No       No        No        No       No     torch
atanh                                     No       No        No        No       No     torch
baddbmm                                   No       No        No        No       No     torch
bernoulli                                 No       No        No        No       No     torch
bitwise_and                               No       Yes       No        Yes      Yes    torch
bitwise_left_shift                        No       Yes       No        Yes      Yes    torch
bitwise_not                               No       No        Yes       No       Yes    torch
bitwise_or                                No       Yes       No        Yes      Yes    torch
bitwise_right_shift                       No       Yes       No        Yes      Yes    torch
bitwise_xor                               No       Yes       No        Yes      Yes    torch
bmm                                       No       No        No        No       No     torch
broadcast_tensors                         No       No        No        No       No     torch
cat                                       Yes      Yes       Yes       Yes      Yes    torch
ceil                                      No       Yes       Yes       Yes      No     torch
chunk                                     No       Yes       No        No       No     torch
clamp                                     Yes      Yes       Yes       Yes      Yes    torch
clip                                      No       Yes       No        No       No     torch
clone                                     No       Yes       Yes       Yes      Yes    torch
conj                                      No       Yes       No        No       No     torch
copysign                                  No       No        No        No       No     torch
cos                                       No       No        No        No       No     torch
cosh                                      No       No        No        No       No     torch
count_nonzero                             Yes      Yes       Yes       Yes      Yes    torch
cross                                     No       Yes       No        No       No     torch
cumsum                                    Yes      Yes       No        No       No     torch
cumprod                                   No       Yes       No        No       No     torch
diag                                      No       No        No        No       No     torch
div                                       No       Yes       No        Yes      No     torch
divide                                    No       Yes       No        Yes      No     torch
dot                                       No       Yes       Yes       No       No     torch
embedding                                 No       Yes       Yes       Yes      Yes    torch
embedding_renorm\_                        No       No        No        No       No     torch
empty                                     Yes      Yes       Yes       Yes      Yes    torch
empty_like                                Yes      Yes       Yes       Yes      Yes    torch
empty_strided                             Yes      Yes       Yes       Yes      Yes    torch
eq                                        Yes      Yes       No        Yes      Yes    torch
equal                                     Yes      Yes       Yes       Yes      Yes    torch
erf                                       No       No        No        No       No     torch
erfc                                      No       No        No        No       No     torch
erfinv                                    No       No        No        No       No     torch
exp                                       No       No        Yes       No       No     torch
exp2                                      No       No        No        No       No     torch
expm1                                     No       No        No        No       No     torch
eye                                       No       Yes       No        No       No     torch
fill                                      Yes      Yes       Yes       Yes      Yes    torch
fill\_                                    Yes      Yes       Yes       Yes      Yes    torch
flatten                                   No       Yes       No        No       No     torch
flip                                      No       Yes       Yes       Yes      Yes    torch
floor                                     No       Yes       Yes       Yes      No     torch
floor_divide                              Yes      Yes       No        Yes      Yes    torch
fmax                                      No       Yes       Yes       Yes      Yes    torch
fmin                                      No       Yes       Yes       Yes      Yes    torch
fmod                                      No       Yes       No        No       No     torch
frac                                      No       No        No        No       No     torch
frexp                                     No       No        No        No       No     torch
full                                      Yes      Yes       No        Yes      Yes    torch
full_like                                 Yes      Yes       Yes       Yes      No     torch
gather                                    No       Yes       Yes       Yes      Yes    torch
ge                                        Yes      Yes       No        No       No     torch
greater                                   Yes      Yes       No        Yes      No     torch
greater_equal                             Yes      Yes       No        Yes      No     torch
grid_sampler_2d                           No       No        No        No       No     torch
gt                                        Yes      No        No        No       No     torch
heaviside                                 No       Yes       No        No       No     torch
hypot                                     No       No        No        No       No     torch
index_fill                                No       Yes       No        Yes      Yes    torch
index_select                              Yes      Yes       Yes       Yes      Yes    torch
is_complex                                No       Yes       No        No       No     torch
is_floating_point                         No       Yes       No        No       No     torch
is_nonzero                                No       Yes       No        No       No     torch
isfinite                                  No       No        No        No       No     torch
isinf                                     No       No        No        No       No     torch
isnan                                     No       Yes       Yes       Yes      Yes    torch
isneginf                                  No       No        No        No       No     torch
isposinf                                  No       No        No        No       No     torch
kthvalue                                  No       Yes       No        No       No     torch
le                                        No       Yes       No        Yes      No     torch
lerp                                      No       Yes       No        No       No     torch
less                                      Yes      Yes       No        Yes      No     torch
less_equal                                No       Yes       No        Yes      No     torch
linspace                                  No       Yes       No        No       No     torch
log                                       No       No        No        No       No     torch
log10                                     No       No        No        No       No     torch
log1p                                     No       No        No        No       No     torch
log2                                      No       No        No        No       No     torch
logaddexp                                 No       No        No        No       No     torch
logaddexp2                                No       No        No        No       No     torch
logcumsumexp                              No       No        No        No       No     torch
logical_and                               Yes      Yes       No        Yes      Yes    torch
logical_not                               No       No        Yes       Yes      Yes    torch
logical_or                                No       No        No        Yes      Yes    torch
logical_xor                               No       No        No        Yes      Yes    torch
logit                                     No       No        No        No       No     torch
logspace                                  No       Yes       No        No       No     torch
logsumexp                                 No       No        No        No       No     torch
lt                                        Yes      Yes       No        Yes      No     torch
masked_fill                               No       Yes       No        Yes      Yes    torch
masked_scatter                            No       Yes       No        Yes      Yes    torch
masked_select                             No       Yes       No        No       No     torch
matmul                                    No       No        No        No       No     torch
max                                       No       Yes       Yes       No       No     torch
maximum                                   No       Yes       Yes       No       No     torch
mean                                      No       No        No        No       No     torch
median                                    No       Yes       No        No       No     torch
meshgrid                                  No       No        No        No       No     torch
min                                       No       Yes       Yes       No       No     torch
minimum                                   No       Yes       Yes       No       No     torch
mm                                        No       No        No        No       No     torch
mul                                       No       No        No        No       No     torch
mv                                        No       Yes       Yes       No       No     torch
nan_to_num                                No       Yes       No        No       No     torch
nansum                                    No       No        Yes       No       No     torch
narrow                                    No       No        No        No       No     torch
native_group_norm                         No       No        No        No       No     torch
native_layer_norm                         No       No        No        No       No     torch
ne                                        No       Yes       Yes       No       No     torch
neg                                       No       No        No        No       No     torch
nextafter                                 No       No        No        No       No     torch
nonzero                                   No       Yes       No        No       Yes    torch
norm                                      No       No        Yes       No       No     torch
normal                                    No       No        No        No       No     torch
not_equal                                 No       Yes       No        Yes      Yes    torch
ones                                      No       Yes       No        Yes      Yes    torch
ones_like                                 No       Yes       Yes       Yes      Yes    torch
pixel_shuffle                             No       Yes       Yes       Yes      Yes    torch
poisson                                   No       No        No        No       No     torch
pow                                       No       Yes       No        Yes      Yes    torch
prod                                      Yes      No        Yes       No       No     torch
randperm                                  Yes      Yes       No        No       No     torch
reciprocal                                No       No        No        No       No     torch
remainder                                 No       Yes       Yes       Yes      Yes    torch
reshape                                   Yes      Yes       Yes       Yes      Yes    torch
resolve_conj                              No       No        No        No       No     torch
resolve_neg                               No       No        No        No       No     torch
result_type                               No       Yes       No        No       No     torch
roll                                      No       Yes       Yes       Yes      Yes    torch
round                                     No       No        No        No       No     torch
rsqrt                                     No       No        No        No       No     torch
rsub                                      Yes      No        Yes       No       No     torch
scalar_tensor                             Yes      Yes       Yes       Yes      Yes    torch
scatter                                   No       Yes       No        Yes      Yes    torch
scatter_add                               No       No        No        No       No     torch
scatter_reduce                            No       No        No        No       No     torch
searchsorted                              Yes      Yes       No        No       No     torch
select                                    No       Yes       No        Yes      Yes    torch
sgn                                       No       No        No        No       No     torch
sigmoid                                   No       No        No        No       No     torch
sign                                      No       No        No        No       No     torch
signbit                                   No       Yes       No        Yes      Yes    torch
sin                                       No       No        No        No       No     torch
sinc                                      No       No        No        No       No     torch
sinh                                      No       No        No        No       No     torch
sort                                      No       No        Yes       No       No     torch
split_with_sizes                          No       Yes       Yes       No       No     torch
sqrt                                      No       No        No        No       No     torch
square                                    No       Yes       No        Yes      Yes    torch
squeeze                                   No       Yes       No        Yes      Yes    torch
stack                                     No       Yes       Yes       No       No     torch
std                                       No       No        No        No       No     torch
std_mean                                  No       No        No        No       No     torch
sub                                       No       No        Yes       No       No     torch
sum                                       No       Yes       No        No       Yes    torch
t                                         Yes      Yes       Yes       Yes      Yes    torch
take                                      No       Yes       No        No       No     torch
tan                                       No       No        No        No       No     torch
tanh                                      No       No        No        No       No     torch
topk                                      No       No        Yes       No       No     torch
trace                                     No       Yes       No        No       No     torch
transpose                                 Yes      Yes       Yes       Yes      Yes    torch
tril                                      No       No        No        Yes      Yes    torch
triu                                      No       No        No        Yes      Yes    torch
trunc                                     No       Yes       Yes       Yes      No     torch
unbind                                    No       Yes       No        No       No     torch
unique                                    No       Yes       No        No       No     torch
_unique                                   No       Yes       No        No       No     torch
_unique2                                  No       Yes       No        No       No     torch
unsqueeze                                 No       Yes       Yes       No       No     torch
var                                       No       No        No        No       No     torch
var_mean                                  No       No        No        No       No     torch
vdot                                      No       No        No        No       No     torch
where                                     Yes      Yes       No        No       No     torch
xlogy                                     No       No        No        No       No     torch
zero\_                                    No       Yes       Yes       Yes      Yes    torch
zeros                                     No       Yes       No        Yes      Yes    torch
zeros_like                                No       Yes       Yes       Yes      Yes    torch
_adaptive_avg_pool2d                      No       No        No        No       No     torch
_adaptive_avg_pool3d                      No       No        No        No       No     torch
_efficientzerotensor                      No       Yes       Yes       Yes      Yes    torch
_fused_dropout                            No       No        No        No       No     torch
_masked_scale                             No       Yes       Yes       Yes      Yes    torch
_native_batch_norm_legit                  No       No        No        No       No     torch
_native_batch_norm_legit_no_training      No       No        No        No       No     torch
_resize_output\_                          Yes      Yes       No        Yes      Yes    torch
_weight_norm_interface                    No       No        No        No       No     torch
AdaptiveAvgPool1d                         No       No        No        No       No     torch.nn
AdaptiveAvgPool2d                         No       No        No        No       No     torch.nn
AdaptiveAvgPool3d                         No       No        No        No       No     torch.nn
AvgPool1d                                 No       No        No        No       No     torch.nn
AvgPool2d                                 No       No        No        No       No     torch.nn
AvgPool3d                                 No       No        No        No       No     torch.nn
BatchNorm1d                               No       No        No        No       No     torch.nn
BatchNorm2d                               No       No        No        No       No     torch.nn
BCELoss                                   No       No        No        No       No     torch.nn
BCEWithLogitsLoss                         No       No        No        No       No     torch.nn
ChannelShuffle                            No       Yes       No        Yes      Yes    torch.nn
ConstantPad1d                             No       No        No        No       No     torch.nn
Conv1d                                    No       No        No        No       No     torch.nn
Conv2d                                    No       No        No        No       No     torch.nn
Conv3d                                    No       No        No        No       No     torch.nn
ConvTranspose2d                           No       No        No        No       No     torch.nn
ConvTranspose3d                           No       No        No        No       No     torch.nn
CrossEntropyLoss                          No       No        No        No       No     torch.nn
Dropout                                   No       No        No        No       No     torch.nn
ELU                                       No       No        No        No       No     torch.nn
Embedding                                 No       Yes       Yes       Yes      Yes    torch.nn
EmbeddingBag                              No       No        No        No       No     torch.nn
GELU                                      No       No        No        No       No     torch.nn
GLU                                       No       No        No        No       No     torch.nn
Hardshrink                                No       No        No        No       No     torch.nn
Hardsigmoid                               No       No        No        No       No     torch.nn
Hardtanh                                  No       No        No        No       No     torch.nn
HuberLoss                                 No       No        No        No       No     torch.nn
InstanceNorm2d                            No       No        No        No       No     torch.nn
KLDivLoss                                 No       No        No        No       No     torch.nn
LayerNorm                                 No       No        No        No       No     torch.nn
LeakyReLU                                 No       No        No        No       No     torch.nn
Linear                                    No       No        No        No       No     torch.nn
LogSigmoid                                No       No        No        No       No     torch.nn
LogSoftmax                                No       No        No        No       No     torch.nn
MaxPool2d                                 No       No        No        No       No     torch.nn
MaxPool3d                                 No       No        No        No       No     torch.nn
Mish                                      No       No        No        No       No     torch.nn
MSELoss                                   No       No        No        No       No     torch.nn
NLLLoss                                   No       No        No        No       No     torch.nn
PixelShuffle                              No       Yes       No        Yes      Yes    torch.nn
PReLU                                     No       No        No        No       No     torch.nn
ReLU                                      No       No        No        No       No     torch.nn
ReflectionPad1d                           No       No        No        No       No     torch.nn
ReflectionPad2d                           No       No        No        No       No     torch.nn
ReflectionPad3d                           No       No        No        No       No     torch.nn
ReplicationPad1d                          No       Yes       No        No       No     torch.nn
ReplicationPad2d                          No       Yes       No        No       No     torch.nn
ReplicationPad3d                          No       Yes       No        No       No     torch.nn
RReLU                                     No       No        No        No       No     torch.nn
SiLU                                      No       No        No        No       No     torch.nn
Softmax                                   No       No        No        No       No     torch.nn
Softplus                                  No       No        No        No       No     torch.nn
Softshrink                                No       No        No        No       No     torch.nn
Threshold                                 No       No        No        No       No     torch.nn
SmoothL1Loss                              No       No        No        No       No     torch.nn
Upsample                                  No       No        No        No       No     torch.nn
UpsamplingNearest2d                       No       No        No        No       No     torch.nn
weight_norm                               No       No        No        No       No     torch.nn.utils
__and__                                   No       Yes       No        Yes      Yes    torch.Tensor
__iand__                                  No       Yes       No        Yes      Yes    torch.Tensor
__ilshift__                               No       Yes       No        Yes      Yes    torch.Tensor
__ior__                                   No       Yes       No        Yes      Yes    torch.Tensor
__irshift__                               No       Yes       No        Yes      Yes    torch.Tensor
__ixor__                                  No       Yes       No        Yes      Yes    torch.Tensor
__lshift__                                No       Yes       No        Yes      Yes    torch.Tensor
__or__                                    No       Yes       No        Yes      Yes    torch.Tensor
__rshift__                                No       Yes       No        Yes      Yes    torch.Tensor
__xor__                                   No       Yes       No        Yes      Yes    torch.Tensor
abs                                       No       No        Yes       No       No     torch.Tensor
acos                                      No       No        No        No       No     torch.Tensor
acos\_                                    No       No        No        No       No     torch.Tensor
acosh                                     No       No        No        No       No     torch.Tensor
acosh\_                                   No       No        No        No       No     torch.Tensor
add                                       Yes      Yes       No        No       No     torch.Tensor
add\_                                     Yes      Yes       No        No       No     torch.Tensor
addbmm                                    No       No        No        No       No     torch.Tensor
addbmm\_                                  No       No        No        No       No     torch.Tensor
addcdiv                                   No       No        Yes       No       No     torch.Tensor
addcdiv\_                                 No       No        Yes       No       No     torch.Tensor
addcmul                                   No       No        Yes       No       No     torch.Tensor
addcmul\_                                 No       No        Yes       No       No     torch.Tensor
addmm                                     No       No        No        No       No     torch.Tensor
addmm\_                                   No       No        No        No       No     torch.Tensor
addmv                                     No       No        No        No       No     torch.Tensor
addmv\_                                   No       No        No        No       No     torch.Tensor
addr                                      No       No        No        No       No     torch.Tensor
addr\_                                    No       No        No        No       No     torch.Tensor
all                                       No       Yes       Yes       No       Yes    torch.Tensor
amax                                      No       No        Yes       No       No     torch.Tensor
amin                                      No       No        Yes       No       No     torch.Tensor
aminmax                                   Yes      Yes       Yes       Yes      No     torch.Tensor
any                                       No       Yes       Yes       Yes      Yes    torch.Tensor
arccos                                    No       No        No        No       No     torch.Tensor
arccos\_                                  No       No        No        No       No     torch.Tensor
arccosh                                   No       No        No        No       No     torch.Tensor
arccosh\_                                 No       No        No        No       No     torch.Tensor
arcsin                                    No       No        No        No       No     torch.Tensor
arcsin\_                                  No       No        No        No       No     torch.Tensor
arcsinh                                   No       No        No        No       No     torch.Tensor
arcsinh\_                                 No       No        No        No       No     torch.Tensor
arctan                                    No       No        No        No       No     torch.Tensor
arctan\_                                  No       No        No        No       No     torch.Tensor
arctanh                                   No       No        No        No       No     torch.Tensor
arctanh\_                                 No       No        No        No       No     torch.Tensor
argmax                                    No       Yes       Yes       Yes      Yes    torch.Tensor
argmin                                    No       Yes       Yes       Yes      Yes    torch.Tensor
asin                                      No       No        No        No       No     torch.Tensor
asin\_                                    No       No        No        No       No     torch.Tensor
asinh                                     No       No        No        No       No     torch.Tensor
asinh\_                                   No       No        No        No       No     torch.Tensor
atan2                                     No       No        No        No       No     torch.Tensor
atan2\_                                   No       No        No        No       No     torch.Tensor
arctan2                                   No       No        No        No       No     torch.Tensor
arctan2\_                                 No       No        No        No       No     torch.Tensor
atanh                                     No       No        No        No       No     torch.Tensor
atanh\_                                   No       No        No        No       No     torch.Tensor
baddbmm                                   No       No        No        No       No     torch.Tensor
baddbmm\_                                 No       No        No        No       No     torch.Tensor
bernoulli                                 No       No        No        No       No     torch.Tensor
bitwise_and                               No       No        No        No       Yes    torch.Tensor
bitwise_left_shift                        No       Yes       No        Yes      No     torch.Tensor
bitwise_not                               No       Yes       Yes       Yes      Yes    torch.Tensor
bitwise_or                                No       No        No        No       Yes    torch.Tensor
bitwise_right_shift                       No       Yes       No        Yes      No     torch.Tensor
bitwise_xor                               No       No        No        No       Yes    torch.Tensor
bmm                                       No       No        No        No       No     torch.Tensor
ceil                                      No       Yes       Yes       Yes      No     torch.Tensor
clamp                                     Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp\_                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_max                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
clamp_min                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
clip                                      No       Yes       No        No       No     torch.Tensor
clip\_                                    No       Yes       No        No       No     torch.Tensor
clone                                     No       Yes       Yes       Yes      Yes    torch.Tensor
conj                                      No       Yes       No        No       No     torch.Tensor
copy\_                                    No       Yes       No        Yes      Yes    torch.Tensor
copysign                                  No       No        No        No       No     torch.Tensor
copysign\_                                No       No        No        No       No     torch.Tensor
cos                                       No       No        No        No       No     torch.Tensor
cos\_                                     No       No        No        No       No     torch.Tensor
cosh\_                                    No       No        No        No       No     torch.Tensor
count_nonzero                             Yes      Yes       Yes       Yes      Yes    torch.Tensor
cross                                     No       Yes       No        No       No     torch.Tensor
cumprod                                   No       Yes       No        No       No     torch.Tensor
cumsum                                    Yes      Yes       No        No       No     torch.Tensor
diag                                      No       No        No        No       No     torch.Tensor
div                                       No       Yes       No        Yes      No     torch.Tensor
div\_                                     No       Yes       No        Yes      No     torch.Tensor
dot                                       No       Yes       Yes       No       No     torch.Tensor
fill\_                                    Yes      Yes       Yes       Yes      Yes    torch.Tensor
eq                                        Yes      Yes       No        Yes      Yes    torch.Tensor
eq\_                                      Yes      Yes       No        Yes      Yes    torch.Tensor
equal                                     Yes      Yes       Yes       Yes      No     torch.Tensor
erf                                       No       No        No        No       No     torch.Tensor
erf\_                                     No       No        No        No       No     torch.Tensor
erfinv                                    No       No        No        No       No     torch.Tensor
erfinv\_                                  No       No        No        No       No     torch.Tensor
erfc\_                                    No       No        No        No       No     torch.Tensor
erfc                                      No       No        No        No       No     torch.Tensor
exp                                       No       No        Yes       No       No     torch.Tensor
exp\_                                     No       No        Yes       No       No     torch.Tensor
expand                                    Yes      Yes       No        No       No     torch.Tensor
expand_as                                 Yes      Yes       No        Yes      Yes    torch.Tensor
expm1                                     No       No        No        No       No     torch.Tensor
expm1\_                                   No       No        No        No       No     torch.Tensor
exponential\_                             No       No        No        No       No     torch.Tensor
flatten                                   No       Yes       No        No       No     torch.Tensor
flip                                      No       Yes       Yes       Yes      Yes    torch.Tensor
floor                                     No       Yes       Yes       Yes      No     torch.Tensor
floor\_                                   No       Yes       Yes       Yes      No     torch.Tensor
floor_divide                              Yes      Yes       No        Yes      Yes    torch.Tensor
floor_divide\_                            Yes      Yes       No        Yes      Yes    torch.Tensor
fmax                                      No       Yes       Yes       Yes      Yes    torch.Tensor
fmin                                      No       Yes       Yes       Yes      Yes    torch.Tensor
fmod                                      No       Yes       No        No       No     torch.Tensor
fmod\_                                    No       Yes       No        No       No     torch.Tensor
frac                                      No       No        No        No       No     torch.Tensor
frexp                                     No       No        No        No       No     torch.Tensor
gather                                    No       Yes       Yes       Yes      Yes    torch.Tensor
ge                                        Yes      Yes       No        No       No     torch.Tensor
ge\_                                      Yes      Yes       No        No       No     torch.Tensor
geometric\_                               No       No        No        No       No     torch.Tensor
greater                                   Yes      Yes       No        Yes      No     torch.Tensor
greater_equal                             Yes      Yes       No        Yes      No     torch.Tensor
gt                                        Yes      Yes       No        Yes      No     torch.Tensor
hardshrink                                No       No        No        No       No     torch.Tensor
heaviside                                 No       Yes       No        No       No     torch.Tensor
hypot                                     No       No        No        No       No     torch.Tensor
index_add\_                               No       Yes       No        No       No     torch.Tensor
index_copy\_                              No       Yes       Yes       Yes      Yes    torch.Tensor
index_fill                                No       Yes       No        Yes      Yes    torch.Tensor
index_fill\_                              No       Yes       Yes       Yes      Yes    torch.Tensor
index_put                                 No       Yes       No        No       No     torch.Tensor
index_put\_                               No       Yes       No        No       No     torch.Tensor
index_select                              Yes      Yes       Yes       Yes      Yes    torch.Tensor
is_complex                                No       Yes       No        No       No     torch.Tensor
is_floating_point                         No       Yes       No        No       No     torch.Tensor
isfinite                                  No       No        No        No       No     torch.Tensor
isinf                                     No       No        No        No       No     torch.Tensor
isnan                                     No       Yes       Yes       No       No     torch.Tensor
isneginf                                  No       No        No        No       No     torch.Tensor
isposinf                                  No       No        No        No       No     torch.Tensor
item                                      No       No        No        No       No     torch.Tensor
kthvalue                                  No       Yes       No        No       No     torch.Tensor
le                                        No       Yes       No        Yes      No     torch.Tensor
le\_                                      No       Yes       No        Yes      No     torch.Tensor
lerp                                      No       Yes       No        No       No     torch.Tensor
lerp\_                                    No       Yes       No        No       No     torch.Tensor
less                                      Yes      Yes       No        Yes      No     torch.Tensor
less\_                                    Yes      Yes       No        Yes      No     torch.Tensor
less_equal                                No       Yes       No        Yes      No     torch.Tensor
less_equal\_                              No       Yes       No        Yes      No     torch.Tensor
log                                       No       No        No        No       No     torch.Tensor
log\_                                     No       No        No        No       No     torch.Tensor
log_normal\_                              No       No        No        No       No     torch.Tensor
log_softmax                               No       No        No        No       No     torch.Tensor
log10                                     No       No        No        No       No     torch.Tensor
log1p                                     No       No        No        No       No     torch.Tensor
log1p\_                                   No       No        No        No       No     torch.Tensor
log2                                      No       No        No        No       No     torch.Tensor
log2\_                                    No       No        No        No       No     torch.Tensor
logaddexp                                 No       No        No        No       No     torch.Tensor
logaddexp2                                No       No        No        No       No     torch.Tensor
logcumsumexp                              No       No        No        No       No     torch.Tensor
logical_and                               Yes      Yes       No        Yes      Yes    torch.Tensor
logical_and\_                             Yes      Yes       No        Yes      Yes    torch.Tensor
logical_not                               No       No        Yes       Yes      Yes    torch.Tensor
logical_not\_                             No       No        Yes       Yes      Yes    torch.Tensor
logical_or                                No       No        No        Yes      Yes    torch.Tensor
logical_or\_                              No       No        No        Yes      Yes    torch.Tensor
logical_xor\_                             No       No        No        Yes      Yes    torch.Tensor
logical_xor\_                             No       No        No        Yes      Yes    torch.Tensor
logit                                     No       No        No        No       No     torch.Tensor
logsumexp                                 No       No        No        No       No     torch.Tensor
lt                                        Yes      Yes       No        Yes      No     torch.Tensor
masked_fill                               No       Yes       No        Yes      Yes    torch.Tensor
masked_fill\_                             No       Yes       No        Yes      Yes    torch.Tensor
masked_scatter                            No       Yes       No        Yes      Yes    torch.Tensor
masked_scatter\_                          No       Yes       No        Yes      Yes    torch.Tensor
masked_select                             No       Yes       No        No       No     torch.Tensor
matmul                                    No       No        No        No       No     torch.Tensor
max                                       No       Yes       Yes       No       No     torch.Tensor
maximum                                   No       Yes       Yes       No       No     torch.Tensor
mean                                      No       No        No        No       No     torch.Tensor
median                                    No       Yes       No        No       No     torch.Tensor
min                                       No       Yes       Yes       No       No     torch.Tensor
minimum                                   No       Yes       Yes       No       No     torch.Tensor
mm                                        No       No        No        No       No     torch.Tensor
mul                                       No       No        No        No       No     torch.Tensor
mul\_                                     No       No        No        No       No     torch.Tensor
mv                                        No       Yes       Yes       No       No     torch.Tensor
nan_to_num                                No       Yes       No        No       No     torch.Tensor
nansum                                    No       No        Yes       No       No     torch.Tensor
narrow                                    No       No        No        No       No     torch.Tensor
ne                                        No       Yes       Yes       No       No     torch.Tensor
ne\_                                      No       Yes       Yes       No       No     torch.Tensor
neg                                       No       No        No        No       No     torch.Tensor
new_empty                                 No       Yes       No        Yes      No     torch.Tensor
new_empty_strided                         No       Yes       No        Yes      No     torch.Tensor
new_full                                  No       Yes       No        Yes      No     torch.Tensor
new_ones                                  No       Yes       No        Yes      No     torch.Tensor
new_zeros                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
nextafter                                 No       No        No        No       No     torch.Tensor
nonzero                                   No       Yes       No        No       Yes    torch.Tensor
norm                                      No       No        Yes       No       No     torch.Tensor
normal\_                                  No       No        No        No       No     torch.Tensor
permute                                   No       Yes       No        No       No     torch.Tensor
pin_memory                                No       Yes       No        Yes      Yes    torch.Tensor
pow                                       No       Yes       No        Yes      Yes    torch.Tensor
pow\_                                     No       Yes       No        Yes      Yes    torch.Tensor
prod                                      Yes      No        Yes       No       No     torch.Tensor
put\_                                     No       No        No        No       No     torch.Tensor
random\_                                  No       Yes       Yes       Yes      Yes    torch.Tensor
reciprocal                                No       No        No        No       No     torch.Tensor
reciprocal\_                              No       No        No        No       No     torch.Tensor
remainder                                 No       Yes       Yes       Yes      Yes    torch.Tensor
remainder\_                               No       Yes       Yes       Yes      Yes    torch.Tensor
repeat                                    No       Yes       No        Yes      Yes    torch.Tensor
repeat_interleave                         Yes      Yes       No        Yes      Yes    torch.Tensor
reshape                                   Yes      Yes       Yes       Yes      Yes    torch.Tensor
roll                                      No       Yes       Yes       Yes      Yes    torch.Tensor
resize\_                                  Yes      Yes       No        Yes      Yes    torch.Tensor
round                                     No       No        No        No       No     torch.Tensor
round\_                                   No       No        No        No       No     torch.Tensor
rsqrt                                     No       No        No        No       No     torch.Tensor
rsqrt\_                                   No       No        No        No       No     torch.Tensor
scatter                                   No       Yes       No        Yes      Yes    torch.Tensor
scatter\_                                 No       Yes       No        Yes      Yes    torch.Tensor
scatter_add                               No       No        No        No       No     torch.Tensor
scatter_add\_                             No       No        No        No       No     torch.Tensor
scatter_reduce                            No       No        No        No       No     torch.Tensor
scatter_reduce\_                          No       No        No        No       No     torch.Tensor
select                                    No       Yes       No        Yes      Yes    torch.Tensor
sgn                                       No       No        No        No       No     torch.Tensor
sgn\_                                     No       No        No        No       No     torch.Tensor
sigmoid                                   No       No        No        No       No     torch.Tensor
sigmoid\_                                 No       No        No        No       No     torch.Tensor
sign                                      No       No        No        No       No     torch.Tensor
sign\_                                    No       No        No        No       No     torch.Tensor
signbit                                   No       Yes       No        Yes      Yes    torch.Tensor
sin                                       No       No        No        No       No     torch.Tensor
sin\_                                     No       No        No        No       No     torch.Tensor
sinh                                      No       No        No        No       No     torch.Tensor
sinh\_                                    No       No        No        No       No     torch.Tensor
sinc                                      No       No        No        No       No     torch.Tensor
sort                                      No       No        Yes       No       No     torch.Tensor
split_with_sizes                          No       Yes       Yes       No       No     torch.Tensor
sqrt                                      No       No        No        No       No     torch.Tensor
square                                    No       No        No        No       No     torch.Tensor
square\_                                  No       No        No        No       No     torch.Tensor
squeeze                                   No       Yes       No        Yes      Yes    torch.Tensor
squeeze\_                                 No       Yes       No        Yes      Yes    torch.Tensor
std                                       No       No        No        No       No     torch.Tensor
sub                                       No       No        Yes       No       No     torch.Tensor
sub\_                                     No       No        Yes       No       No     torch.Tensor
sum                                       No       No        No        No       No     torch.Tensor
T                                         Yes      Yes       No        Yes      Yes    torch.Tensor
t                                         Yes      Yes       Yes       Yes      Yes    torch.Tensor
tan                                       No       No        No        No       No     torch.Tensor
tan\_                                     No       No        No        No       No     torch.Tensor
tanh                                      No       No        No        No       No     torch.Tensor
tanh\_                                    No       No        No        No       No     torch.Tensor
to                                        Yes      Yes       No        Yes      Yes    torch.Tensor
topk                                      No       No        Yes       No       No     torch.Tensor
trace                                     No       Yes       No        No       No     torch.Tensor
take                                      No       Yes       No        No       No     torch.Tensor
transpose                                 Yes      Yes       Yes       Yes      Yes    torch.Tensor
tril                                      No       No        No        Yes      Yes    torch.Tensor
tril\_                                    No       No        No        Yes      Yes    torch.Tensor
triu                                      No       No        No        Yes      Yes    torch.Tensor
triu\_                                    No       No        No        Yes      Yes    torch.Tensor
trunc                                     No       Yes       Yes       Yes      No     torch.Tensor
trunc\_                                   No       Yes       Yes       Yes      No     torch.Tensor
unbind                                    No       Yes       No        No       No     torch.Tensor
unsqueeze                                 No       Yes       Yes       No       No     torch.Tensor
uniform\_                                 No       No        No        No       No     torch.Tensor
unique                                    No       Yes       No        No       No     torch.Tensor
var                                       No       No        No        No       No     torch.Tensor
vdot                                      No       No        No        No       No     torch.Tensor
view                                      No       No        No        No       No     torch.Tensor
where                                     Yes      Yes       No        No       No     torch.Tensor
xlogy                                     No       No        No        No       No     torch.Tensor
xlogy\_                                   No       No        No        No       No     torch.Tensor
zero\_                                    No       Yes       Yes       Yes      Yes    torch.Tensor
entr                                      No       No        No        No       No     torch.special
erf                                       No       No        No        No       No     torch.special
erfc                                      No       No        No        No       No     torch.special
exp2                                      No       No        No        No       No     torch.special
expit                                     No       No        No        No       No     torch.special
expm1                                     No       No        No        No       No     torch.special
erfinv                                    No       No        No        No       No     torch.special
log1p                                     No       No        No        No       No     torch.special
logit                                     No       No        No        No       No     torch.special
logsumexp                                 No       No        No        No       No     torch.special
round                                     No       No        No        No       No     torch.special
sinc                                      No       No        No        No       No     torch.special
softmax                                   No       No        No        No       No     torch.special
xlog1py                                   No       No        No        No       No     torch.special
xlogy                                     No       No        No        No       No     torch.special
batched_nms                               No       No        No        No       No     torchvision.ops
nms                                       No       No        No        No       No     torchvision.ops
roi_align                                 No       No        No        No       No     torchvision.ops
====================================  ========= ========= ========= ======== ========  ======================
