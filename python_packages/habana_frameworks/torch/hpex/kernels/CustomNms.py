from typing import List

import torch
from habana_frameworks.torch import _hpex_C


# This class is deprecated in favor of TorchVision
class CustomNms:
    def __init__(self):

        self.nms = _hpex_C.custom_nms
        self.bnms = _hpex_C.batched_nms
        super(CustomNms, self).__init__()

    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):

        assert boxes.shape[-1] == 4
        keep = self.nms(boxes, scores, iou_threshold)
        return keep

    def batched_nms(self, boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float = 0.5):
        return self.bnms(boxes, scores, idxs, iou_threshold)
