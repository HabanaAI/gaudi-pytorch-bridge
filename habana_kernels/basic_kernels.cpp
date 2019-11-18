#include <torch/script.h>

using namespace torch;

Tensor set_one(Tensor image) {
  Tensor output = image;
  for(size_t i = 0; i < image.numel(); ++i){
    output[i] = 1;
  }

  return output;
}

static auto registry = torch::RegisterOperators()
  .op("habana_kernels::set_one", &set_one);