#include "habana_helpers/tensor_info.h"

void PtTensorInfo::populate_tinfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  ir_name_ = irn;
  syn_name_ = sn;

  buffer_ = pt_tensor.data_ptr();
  buffer_start_ = pt_tensor.storage().data_ptr().get();

  numel_ = pt_tensor.numel();
  size_ = pt_tensor.nbytes();
  shape_ = pt_tensor.sizes().vec();
  strides_ = pt_tensor.strides().vec();

  mf_ = pt_tensor.suggest_memory_format();
  topts_ = pt_tensor.options();

  offset_ = (get_buffer_syn() - get_buffer_start_syn());
  is_view_tensor_ = (offset_ != 0);

  dma_cb_ = dma_cb;

  tensor_type_ = stt;

  watch_ = wflag;

  update_shape_syn();
}

PtTensorInfo::PtTensorInfo(
    const synapse_helpers::tensor& st,
    const std::string& irn) {
  ir_name_ = irn;
  syn_name_ = st.name();
  tensor_type_ = SHAPE_TENSOR;

  // Populate the synapse shapes directly from the input syn tensor.
  numel_ = st.num_elements();
  size_ = st.size_bytes();
  shape_ = st.pt_shape();
  strides_ = st.pt_strides();

  update_shape_syn();
}

PtTensorInfo::PtTensorInfo(const IValPtrShared& ivpsh)
    : is_tensor_(false), iv_(*ivpsh) {}

PtTensorInfo::PtTensorInfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  populate_tinfo(pt_tensor, sn, irn, wflag, stt, dma_cb);
}

PtTensorInfo::PtTensorInfo(
    const IValPtrShared& ivpsh,
    const std::string& sn,
    const ValPtr& vp,
    const bool wflag,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  std::string irn = "%" + vp->debugName();
  auto pt_tensor = ivpsh->toTensor();
  populate_tinfo(pt_tensor, sn, irn, wflag, stt, dma_cb);
}

void PtTensorInfo::update_shape_syn() {
  switch (tensor_type_) {
    case DATA_TENSOR:
    case SHAPE_TENSOR:
    case DATA_TENSOR_DYNAMIC: {
      HABANA_ASSERT(SYN_GAUDI_MAX_TENSOR_DIM >= shape_.size());
      for (size_t i = 0; i < shape_.size(); ++i) {
        // Reverse PyTorch shapes for synapse tensor shape patching
        if (i < shape_.size()) {
          syn_shape_[i] = shape_[shape_.size() - 1 - i];
        }
      }
    } break;
    // TODO: Fix handling for shape tensors
    // case INPUT_DESCRIBING_SHAPE_TENSOR: {
    //  shape_ndim_ = pt_tensor.numel();
    //  at::Tensor pt_tensor_cpu =
    //      (pt_tensor.device().type() == at::kHPU ? pt_tensor.to(at::kCPU)
    //                                                : pt_tensor);
    //  for (uint64_t i = 0; i < shape_ndim_; i++) {
    //    // Reverse PyTorch shapes for synapse tensor shape patching
    //    auto val = pt_tensor_cpu[shape_ndim_ - 1 - i].item<int>();
    //    syn_shape_[i] = val;
    //  }
    //} break;
    case DEVICE_SHAPE_TENSOR:
      syn_shape_ = {SYN_MAX_TENSOR_DIM, 0, 0, 0, 0};
      break;
    case TENSOR_TYPE_MAX:
    default:
      TORCH_CHECK(false, "Unreachable condition.");
  }
}

std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t) {
  O << '<' << t.get_ir_name();
  if (t.is_tensor()) {
    O << ":[" << t.get_shape() << "]:[" << t.get_strides()
      << "]:" << t.get_numel() << ':' << '(' << t.get_size() << " b)"
      << " :: " << t.get_syn_name() << ':' << t.get_buffer() << '>'
      << " tensor type:" << t.tensor_type_;

    if (t.get_dma_cb() != nullptr) {
      O << " dma_cb : " << (void*)t.get_dma_cb();
      O << " dma_tensor_idx : " << t.get_dma_tensor_idx();
    }
    if (t.is_duplicate()) {
      O << " duplicate of " << t.get_parent_index();
    }

    if (ULONG_MAX != t.get_output_index()) {
      O << ", output index " << t.get_output_index();
    } else {
      O << ", non output";
    }

    O << ", is_restrided " << std::boolalpha << t.is_restrided();

    O << ", <" << t.get_buffer_start() << ", +" << t.offset_ << ">";
    if (t.offset_ != 0) {
      O << " nz offset view tensor ";
    }
  } else {
    O << "> non-tensor : ivalue :: " << t.iv_;
  }
  return O;
}

void PrintATenTensor(const at::Tensor& a) {
  std::ostream& O = std::cout;
  O << " Tensor -> ";
  if (a.has_storage()) {
    O << " @ " << (void*)a.storage().data_ptr().get() << " : " << a.data_ptr()
      << " : "
      << " dim " << a.dim() << " : " << a.sizes();
  } else {
    O << " does not have storage";
  }
  O << ',' << " use_count " << a.use_count() << '\n';
}

void PrintATenTensor(const IVal& a) {
  if (a.isTensor()) {
    PrintATenTensor(a.toTensor());
  } else {
    std::cout << "Non-tensor : ivalue :: " << a << '\n';
  }
}

void PrintATenTensor(const IValPtrShared& a) {
  if (a->isTensor()) {
    PrintATenTensor(a->toTensor());
  }
}
