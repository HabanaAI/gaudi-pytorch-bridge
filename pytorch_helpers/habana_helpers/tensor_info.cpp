#include "habana_helpers/tensor_info.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"

void PtTensorInfo::populate_tinfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag,
    const uint64_t tensor_id,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  ir_name_ = irn;
  syn_name_ = sn;
  tensor_id_ = tensor_id;

  is_ZST_ = habana::is_ZST(pt_tensor);

  buffer_ = pt_tensor.data_ptr();
  buffer_start_ = pt_tensor.storage().data_ptr().get();

  numel_ = pt_tensor.numel();
  size_ = pt_tensor.nbytes();
  shape_ = pt_tensor.sizes().vec();
  strides_ = pt_tensor.strides().vec();

  mf_ = pt_tensor.suggest_memory_format();
  topts_ = pt_tensor.options();

  auto hb_internal_tensor = habana_lazy::GetHbInternalTensorImpl(pt_tensor);
  if (hb_internal_tensor != nullptr) {
    hb_internal_lf_ = hb_internal_tensor->GetTensorLayout();
  }

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
  tensor_type_ = st.tensor_type();
  // Populate the synapse shapes directly from the input syn tensor.
  numel_ = st.num_elements();
  size_ = st.size_bytes();
  shape_ = st.pt_shape();
  strides_ = st.pt_strides();
  tensor_id_ = st.id();

  update_shape_syn();
}

PtTensorInfo::PtTensorInfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag,
    const uint64_t tensor_id,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  populate_tinfo(pt_tensor, sn, irn, wflag, tensor_id, stt, dma_cb);
}

PtTensorInfo::PtTensorInfo(
    const IValPtrShared& ivpsh,
    const std::string& sn,
    const ValPtr& vp,
    const bool wflag,
    const uint64_t tensor_id,
    const synTensorType stt,
    const getDMAInputTensorCBType dma_cb) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  std::string irn = "%" + vp->debugName();
  auto pt_tensor = ivpsh->toTensor();
  populate_tinfo(pt_tensor, sn, irn, wflag, tensor_id, stt, dma_cb);
}

void PtTensorInfo::update_shape_syn() {
  switch (tensor_type_) {
    case DATA_TENSOR:
    case SHAPE_TENSOR:
    case DATA_TENSOR_DYNAMIC:
    case INPUT_DESCRIBING_SHAPE_TENSOR:
      HABANA_ASSERT(SYN_GAUDI_MAX_TENSOR_DIM >= shape_.size());
      for (size_t i = 0; i < shape_.size(); ++i) {
        // Reverse PyTorch shapes for synapse tensor shape patching
        if (i < shape_.size()) {
          syn_shape_[i] = shape_[shape_.size() - 1 - i];
        }
      }
      break;
    case DEVICE_SHAPE_TENSOR:
      syn_shape_ = {SYN_MAX_TENSOR_DIM, 0, 0, 0, 0};
      break;
    case TENSOR_TYPE_MAX:
    default:
      TORCH_CHECK(false, "Unreachable condition.");
  }
}

PtTensorInfo::PtTensorInfo(std::istream& is) {
  using namespace serialization;
  deserialize(is, is_ZST_);
  deserialize(is, is_view_tensor_);
  deserialize(is, is_restrided_);
  deserialize(is, offset_);
  deserialize(is, ir_name_);
  deserialize(is, syn_name_);
  deserialize(is, numel_);
  deserialize(is, size_);
  deserialize(is, is_duplicate_);
  deserialize(is, parent_index_);
  deserialize(is, output_index_);
  deserialize(is, watch_);
  deserialize(is, shape_);
  deserialize(is, strides_);
  deserialize(is, mf_);
  deserialize(is, topts_);
  deserialize(is, tensor_type_);
  deserialize(is, dma_tensor_idx_);
  deserialize(is, tensor_id_);

  update_shape_syn(); // constructs syn_shape_ according to shape_ and
                      // tensor_type_
}

void PtTensorInfo::Serialize(std::ostream& os) const {
  using namespace serialization;
  serialize(os, is_ZST_);
  serialize(os, is_view_tensor_);
  serialize(os, is_restrided_);
  serialize(os, offset_);
  serialize(os, ir_name_);
  serialize(os, syn_name_);
  serialize(os, numel_);
  serialize(os, size_);
  serialize(os, is_duplicate_);
  serialize(os, parent_index_);
  serialize(os, output_index_);
  serialize(os, watch_);
  serialize(os, shape_);
  serialize(os, strides_);
  serialize(os, mf_);
  serialize(os, topts_);
  serialize(os, tensor_type_);
  serialize(os, dma_tensor_idx_);
  serialize(os, tensor_id_);
}

std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t) {
  O << '<' << t.get_ir_name();
  O << ":[" << t.get_shape() << "]:[" << t.get_strides() << "]:#"
    << t.get_numel() << ':' << '(' << t.get_size() << " b):["
    << t.getHbInternalLayoutFormat() << "]"
    << " :: " << t.get_syn_name() << ':' << t.get_buffer() << '>'
    << " tensor type:" << t.tensor_type_ << " tensor id: " << t.tensor_id_;

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
