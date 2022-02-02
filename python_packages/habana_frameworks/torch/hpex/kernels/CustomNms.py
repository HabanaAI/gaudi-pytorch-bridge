import torch
from typing import List
from habana_frameworks.torch import _hpex_C


# This class is deprecated in favor of TorchVision
class CustomNms:
    def __init__(self):

        self.nms = _hpex_C.custom_nms
        super(CustomNms, self).__init__()

    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.5
    ):

        assert boxes.shape[-1] == 4
        keep = self.nms(boxes, scores, iou_threshold)
        return keep

    def batched_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float = 0.5
    ):
        # Exactly same as what torchvision 0.8.0 is doing
        if boxes.numel() > 4_000:
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            for class_id in torch.jit.annotate(
                List[int], torch.unique(idxs).cpu().tolist()
            ):
                curr_indices = torch.nonzero(idxs == class_id, as_tuple=True)[0]
                curr_keep_indices = self.nms(
                    boxes[curr_indices],
                    scores[curr_indices],
                    iou_threshold
                )
                keep_mask[curr_indices[curr_keep_indices]] = True
            keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
            return keep_indices[scores[keep_indices].sort(descending=True)[1]]
        else:
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.int64, device=boxes.device)
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1.0).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]
            keep = self.nms(boxes_for_nms, scores, iou_threshold)
            return keep
