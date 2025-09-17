import torch
from typing import List, Tuple


# --------------------------------------------------
# Anchor generator
# --------------------------------------------------
def generate_anchors(
    feature_map_sizes: List[Tuple[int, int]],
    anchor_scales: List[List[int]],
    image_size: int = 224
) -> List[torch.Tensor]:
    """
    為每個尺度產生 anchors (xyxy 格式，絕對座標)。

    Args:
      feature_map_sizes: [(56,56), (28,28), (14,14)]
      anchor_scales: [[16,24,32], [48,64,96], [96,128,192]]
      image_size: 輸入影像大小 (224)

    Returns:
      anchors_per_level: List of tensors
        - 每層 shape = [H*W*num_scales, 4] (x1,y1,x2,y2)
    """
    anchors_per_level = []

    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / H
        stride_x = image_size / W

        # 建立網格中心點
        cy = (torch.arange(H, dtype=torch.float32) + 0.5) * stride_y
        cx = (torch.arange(W, dtype=torch.float32) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")  # [H,W]

        # 每個尺度對應多個 anchor 大小
        anchor_list = []
        for s in scales:
            w = torch.full_like(grid_x, float(s))
            h = torch.full_like(grid_y, float(s))
            x1 = grid_x - w * 0.5
            y1 = grid_y - h * 0.5
            x2 = grid_x + w * 0.5
            y2 = grid_y + h * 0.5
            anchor_list.append(torch.stack([x1, y1, x2, y2], dim=-1))  # [H,W,4]

        anchors_level = torch.stack(anchor_list, dim=2).reshape(-1, 4)  # [H*W*num_scales, 4]
        anchors_per_level.append(anchors_level)

    return anchors_per_level


# --------------------------------------------------
# IoU computation
# --------------------------------------------------
def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Args:
      boxes1: [N,4] xyxy
      boxes2: [M,4] xyxy

    Returns:
      iou: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    # Intersection
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou


# --------------------------------------------------
# Anchor-to-target matching
# --------------------------------------------------
def match_anchors_to_targets(
    anchors: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    pos_threshold: float = 0.5,
    neg_threshold: float = 0.3
):
    """
    Assign target boxes to anchors for training.

    Args:
      anchors: [A,4] xyxy
      target_boxes: [N,4] xyxy
      target_labels: [N]
      pos_threshold: IoU >= threshold → positive
      neg_threshold: IoU < threshold → negative

    Returns:
      matched_labels: [A]  (0=background, 1..C=classes)
      matched_boxes:  [A,4]
      pos_mask:       [A] bool
      neg_mask:       [A] bool
    """
    A = anchors.shape[0]
    device = anchors.device

    if target_boxes.numel() == 0:
        matched_labels = torch.zeros((A,), dtype=torch.long, device=device)
        matched_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((A,), dtype=torch.bool, device=device)
        neg_mask = torch.ones((A,), dtype=torch.bool, device=device)
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes)  # [A,N]
    max_iou, max_idx = iou.max(dim=1)         # [A]

    matched_boxes = target_boxes[max_idx]
    # label +1: 0 reserved for background
    matched_labels = target_labels[max_idx] + 1

    pos_mask = max_iou >= pos_threshold
    neg_mask = max_iou < neg_threshold
    ignore_mask = (~pos_mask) & (~neg_mask)

    matched_labels = torch.where(ignore_mask, torch.zeros_like(matched_labels), matched_labels)
    return matched_labels, matched_boxes, pos_mask, neg_mask
