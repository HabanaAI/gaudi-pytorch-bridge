namespace habana_lazy {
namespace ir {
class EmbeddingBagSum : public ir::Node {
 public:
  enum class EmbeddingBagSumParams { KERNEL_MODE_INDEX = 4 };
  EmbeddingBagSum() = delete;
  EmbeddingBagSum(
      const Tensor& input,
      const Tensor& indices,
      const Tensor& offsets,
      const Tensor& valid_count,
      int64_t kernel_mode)
      : Node(c10::Symbol::fromQualString("hpu::embedding_bag_sum")) {
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHABANA);
    auto hl_offsets = GetOrCreateHbLazyTensor(offsets, c10::kHABANA);
    auto hl_valid_count = GetOrCreateHbLazyTensor(valid_count, c10::kHABANA);

    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());
    AddInput(hl_offsets.GetIrValue());
    AddInput(hl_valid_count.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{input, indices, offsets, valid_count};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_mode,
        static_cast<size_t>(EmbeddingBagSumParams::KERNEL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_mode="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBagSumParams::KERNEL_MODE_INDEX));

    return ss.str();
  } // std::string ToString()
}; // class EmbeddingBagSum : public ir::Node

class EmbeddingBagSumBwd : public ir::Node {
 public:
  enum class EmbeddingBagSumBwdParams { KERNEL_MODE_INDEX = 5 };
  EmbeddingBagSumBwd() = delete;
  EmbeddingBagSumBwd(
      Tensor& out,
      const Tensor& input,
      const Tensor& indices,
      const Tensor& offsets,
      const Tensor& valid_count,
      int64_t kernel_mode)
      : Node(c10::Symbol::fromQualString("hpu::embedding_bag_sum_bwd_out")) {
    auto hl_out = GetOrCreateHbLazyTensor(out, c10::kHABANA);
    auto hl_input = GetOrCreateHbLazyTensor(input, c10::kHABANA);
    auto hl_indices = GetOrCreateHbLazyTensor(indices, c10::kHABANA);
    auto hl_offsets = GetOrCreateHbLazyTensor(offsets, c10::kHABANA);
    auto hl_valid_count = GetOrCreateHbLazyTensor(valid_count, c10::kHABANA);

    AddInput(hl_out.GetIrValue());
    AddInput(hl_input.GetIrValue());
    AddInput(hl_indices.GetIrValue());
    AddInput(hl_offsets.GetIrValue());
    AddInput(hl_valid_count.GetIrValue());

    std::vector<at::Tensor> input_pt_vec{
        out, input, indices, offsets, valid_count};
    AddInputPtTensors(input_pt_vec);

    m_meta_data.set(
        kernel_mode,
        static_cast<size_t>(EmbeddingBagSumBwdParams::KERNEL_MODE_INDEX));
  }

  std::string ToString() const override {
    std::stringstream ss;
    ss << Node::ToString() << ", kernel_mode="
       << m_meta_data.get(
              static_cast<size_t>(EmbeddingBagSumBwdParams::KERNEL_MODE_INDEX));

    return ss.str();
  } // std::string ToString()
}; // class EmbeddingBagSumBwd : public ir::Node
} // namespace ir
}; // namespace habana_lazy
