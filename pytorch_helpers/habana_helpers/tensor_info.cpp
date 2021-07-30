#include "habana_helpers/tensor_info.h"
#include "synapse_helpers/habana_tensor.h"

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
  numel_ = pt_tensor.numel();
  size_ = pt_tensor.nbytes();
  storage_data_ptr_ = reinterpret_cast<synapse_helpers::device_ptr>(
      pt_tensor.storage().data_ptr().get());
  dma_cb_ = dma_cb;

  shape_ = pt_tensor.sizes().vec();
  tensor_type_ = stt;

  update_shape_values();
  strides_ = pt_tensor.strides().vec();
  topts_ = pt_tensor.options();
  mf_ = pt_tensor.suggest_memory_format();

  watch_ = wflag;

  synapse_helpers::device_ptr buffer_ptr =
      reinterpret_cast<synapse_helpers::device_ptr>(pt_tensor.data_ptr());
  is_view_tensor_ = (storage_data_ptr_ != buffer_ptr);
  offset_ = (buffer_ptr - storage_data_ptr_);
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

void PtTensorInfo::update_shape_values() {
  switch (tensor_type_) {
    case DATA_TENSOR:
      break;
    case SHAPE_TENSOR:
    case DATA_TENSOR_DYNAMIC: {
      HABANA_ASSERT(SYN_MAX_TENSOR_DIM >= shape_.size());
      for (size_t i = 0; i < shape_.size(); ++i) {
        // Reverse PyTorch shapes for synapse tensor shape patching
        if (i < shape_.size()) {
          shape_values_[i] = shape_[shape_.size() - 1 - i];
        }
      }
    } break;
    // TODO: Fix handling for shape tensors
    // case SHAPE_TENSOR:
    // case INPUT_DESCRIBING_SHAPE_TENSOR: {
    //  shape_ndim_ = pt_tensor.numel();
    //  at::Tensor pt_tensor_cpu =
    //      (pt_tensor.device().type() == at::kHPU ? pt_tensor.to(at::kCPU)
    //                                                : pt_tensor);
    //  for (uint64_t i = 0; i < shape_ndim_; i++) {
    //    // Reverse PyTorch shapes for synapse tensor shape patching
    //    auto val = pt_tensor_cpu[shape_ndim_ - 1 - i].item<int>();
    //    shape_values_[i] = val;
    //  }
    //} break;
    // case DEVICE_SHAPE_TENSOR:
    //  shape_values_ = {SYN_MAX_TENSOR_DIM, 0, 0, 0, 0};
    //  break;
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
