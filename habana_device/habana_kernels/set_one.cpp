#include <torch/script.h>

torch::Tensor set_one(torch::Tensor image) {
  torch::Tensor output = image;
  for(size_t i = 0; i < image.numel(); ++i){
    output[i] = 1;
  }

  return output;
}

static auto registry =
  torch::RegisterOperators("habana_kernels::set_one", &set_one);