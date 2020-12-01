#include "habana_helpers/tensor_info.h"

void PtTensorInfo::populate_tinfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag) {
  ir_name_ = irn;
  syn_name_ = sn;

  std::ostringstream oss;
  oss << pt_tensor.sizes();
  shape_str_ = oss.str();

  buffer_ = pt_tensor.data_ptr();
  numel_ = pt_tensor.numel();
  size_ = pt_tensor.nbytes();

  watch_ = wflag;
}

PtTensorInfo::PtTensorInfo(
    const at::Tensor& pt_tensor,
    const std::string& sn,
    const std::string& irn,
    const bool wflag) {
  populate_tinfo(pt_tensor, sn, irn, wflag);
}

PtTensorInfo::PtTensorInfo(
    const IValPtrShared& ivpsh,
    const std::string& sn,
    const ValPtr& vp,
    const bool wflag) {
  TORCH_CHECK(ivpsh->isTensor(), "aten tensor is expected");
  std::string irn = "%" + vp->debugName();
  auto pt_tensor = ivpsh->toTensor();
  populate_tinfo(pt_tensor, sn, irn, wflag);
}

std::ostream& operator<<(std::ostream& O, const PtTensorInfo& t) {
  O << '<' << t.get_ir_name() << ':' << t.get_shape_str() << ':'
    << t.get_numel() << ':' << '(' << t.get_size() << " b)"
    << " :: " << t.get_syn_name() << ':' << t.get_buffer() << '>';

  if (t.is_duplicate()) {
    O << " duplicate";
  }

  return O;
}

void PrintATenTensor(const at::Tensor& a) {
  std::ostream& O = std::cout;
  O << " Tensor -> ";
  if (a.has_storage()) {
    O << " @ " << a.data_ptr() << " : "
      << " dim " << a.dim() << " : " << a.sizes();
  } else {
    O << " does not have storage";
  }
  O << ',' << " use_count " << a.use_count() << '\n';
}

void PrintATenTensor(const IValPtrShared& a) {
  if (a->isTensor()) {
    PrintATenTensor(a->toTensor());
  }
}
