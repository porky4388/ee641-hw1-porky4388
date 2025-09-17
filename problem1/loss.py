import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model import yolo_decode
from utils import match_anchors_to_targets


class DetectionLoss(nn.Module):
    """
    Multi-task detection loss for EE641 HW1 Problem 1
    損失組成：
      - Localization: SmoothL1, weight=2.0
      - Objectness:   BCEWithLogits, weight=1.0
      - Classification: CrossEntropy, weight=1.0
    """
    def __init__(self, num_classes: int = 3, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.w_obj = 1.0
        self.w_cls = 1.0
        self.w_loc = 2.0

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def _build_targets_per_image(self, anchors_per_level, target_boxes, target_labels):
        """
        把 GT box 分配到 anchors，回傳 (matched_labels, matched_boxes, pos_mask, neg_mask) per level
        """
        outs = []
        for anchors in anchors_per_level:
            ml, mb, pm, nm = match_anchors_to_targets(
                anchors, target_boxes, target_labels,
                pos_threshold=0.5, neg_threshold=0.3
            )
            outs.append((ml, mb, pm, nm))
        return outs

    def hard_negative_mining(self, loss_neg, pos_mask, neg_mask, ratio=3):
        """
        針對 neg sample 做 HNM，選最多 pos*ratio 個最難負樣本
        """
        num_pos = pos_mask.sum().item()
        if num_pos == 0:
            k = min(100, int(neg_mask.sum().item()))
        else:
            k = min(int(num_pos * ratio), int(neg_mask.sum().item()))
        if k == 0:
            return neg_mask.new_zeros(neg_mask.shape, dtype=torch.bool)

        # 僅對 neg 取 loss 排序
        loss_copy = loss_neg.clone()
        loss_copy[~neg_mask] = -1e9
        _, idx = torch.topk(loss_copy, k)
        selected = neg_mask.new_zeros(neg_mask.shape, dtype=torch.bool)
        selected[idx] = True
        return selected

    def forward(self, preds: List[torch.Tensor], anchors_per_level: List[torch.Tensor],
                targets: List[dict]):
        """
        Args:
          preds: list of 3 tensors (p1, p2, p3)，shape [B, A*(5+C), H, W]
          anchors_per_level: list of 3 tensors，每層 anchors [H*W*A, 4]
          targets: list of dict，每個 batch 元素包含 {"boxes":[N,4], "labels":[N]}

        Returns:
          loss_total, dict(loss_loc, loss_obj, loss_cls)
        """
        device = preds[0].device
        B = preds[0].shape[0]

        # ✅ 改成 tensor 初始化，避免梯度斷掉
        total_loc = torch.tensor(0., device=device)
        total_obj = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)
        total_pos = 0
        total_neg = 0

        for b in range(B):
            target_boxes = targets[b]["boxes"].to(device)
            target_labels = targets[b]["labels"].to(device)

            # per image matching
            match_outs = self._build_targets_per_image(anchors_per_level, target_boxes, target_labels)

            for pred, anchors, (matched_labels, matched_boxes, pos_mask, neg_mask) in zip(
                preds, anchors_per_level, match_outs
            ):
                # decode 預測輸出
                box_deltas, obj_logits, cls_logits = yolo_decode(
                    pred[b:b+1], self.num_anchors, self.num_classes
                )
                # shape [1, N, ...] → [N,...]
                box_deltas = box_deltas.squeeze(0)
                obj_logits = obj_logits.squeeze(0)
                cls_logits = cls_logits.squeeze(0)

                anchors = anchors.to(device)
                matched_boxes = matched_boxes.to(device)
                matched_labels = matched_labels.to(device)

                # Localization loss (only positives)
                if pos_mask.any():
                    loc_loss = self.smooth_l1(box_deltas[pos_mask], matched_boxes[pos_mask])
                    total_loc = total_loc + loc_loss.mean(dim=1).sum()

                # Objectness loss
                obj_targets = pos_mask.float()
                obj_loss_all = self.bce(obj_logits.squeeze(-1), obj_targets)

                # HNM on negatives
                neg_selected = self.hard_negative_mining(obj_loss_all.detach(), pos_mask, neg_mask)
                obj_mask = pos_mask | neg_selected
                total_obj = total_obj + obj_loss_all[obj_mask].sum()

                # Classification loss (only positives)
                if pos_mask.any():
                    cls_loss = F.cross_entropy(
                        cls_logits[pos_mask], matched_labels[pos_mask] - 1, reduction="sum"
                    )
                    total_cls = total_cls + cls_loss

                total_pos += pos_mask.sum().item()
                total_neg += neg_selected.sum().item()

        # normalize
        norm = max(total_pos + total_neg, 1)
        loss_loc = self.w_loc * total_loc / norm
        loss_obj = self.w_obj * total_obj / norm
        loss_cls = self.w_cls * total_cls / max(total_pos, 1)

        loss_total = loss_loc + loss_obj + loss_cls

        # 保持 loss_total 是 tensor (可 backward)，log 部分轉 float
        return loss_total, {
            "loss_loc": float(loss_loc.detach()),
            "loss_obj": float(loss_obj.detach()),
            "loss_cls": float(loss_cls.detach()),
        }
