# problem1/model.py
import torch
import torch.nn as nn
from typing import List, Tuple

# ------------------------------
# Small helpers
# ------------------------------
def conv_bn_relu(c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    """Conv -> BN -> ReLU block"""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class DetectionHead(nn.Module):
    """
    對應每個尺度的偵測頭：
      3×3 Conv(保持通道數) -> 1×1 Conv 到 A*(5+C)
    輸出 shape: [B, A*(5+C), H, W]
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.stem = conv_bn_relu(in_channels, in_channels, k=3, s=1, p=1)
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pred(x)
        return x


# ------------------------------
# Backbone + Heads (YOLO-like)
# ------------------------------
class MultiScaleDetector(nn.Module):
    """
    YOLO-style 單階段多尺度偵測器

    Architecture Requirements（輸入固定 224x224）:
      Backbone: 4 conv blocks
        Block 1 (Stem): Conv(3→32, s=1) → BN → ReLU → Conv(32→64, s=2) → BN → ReLU  [224→112]
        Block 2:        Conv(64→128, s=2) → BN → ReLU                                 [112→56]  -> Scale 1 feature
        Block 3:        Conv(128→256, s=2) → BN → ReLU                                [56→28]   -> Scale 2 feature
        Block 4:        Conv(256→512, s=2) → BN → ReLU                                [28→14]   -> Scale 3 feature

      Detection Heads (each scale):
        3×3 Conv(keep channels) → 1×1 Conv → num_anchors * (5 + num_classes)

      Output (per-location, per-anchor):
        4: bbox offsets (tx, ty, tw, th)
        1: objectness
        C: class scores
    """
    def __init__(self, num_classes: int = 3, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # --- Backbone ---
        # Block 1 (Stem): 224x224 -> 112x112，輸出 64 通道
        self.block1 = nn.Sequential(
            conv_bn_relu(3, 32, k=3, s=1, p=1),
            conv_bn_relu(32, 64, k=3, s=2, p=1),
        )
        # Block 2: 112x112 -> 56x56，輸出 128 通道（Scale 1）
        self.block2 = conv_bn_relu(64, 128, k=3, s=2, p=1)
        # Block 3: 56x56 -> 28x28，輸出 256 通道（Scale 2）
        self.block3 = conv_bn_relu(128, 256, k=3, s=2, p=1)
        # Block 4: 28x28 -> 14x14，輸出 512 通道（Scale 3）
        self.block4 = conv_bn_relu(256, 512, k=3, s=2, p=1)

        # --- Detection Heads ---
        self.head_s1 = DetectionHead(128, num_anchors, num_classes)  # for 56x56
        self.head_s2 = DetectionHead(256, num_anchors, num_classes)  # for 28x28
        self.head_s3 = DetectionHead(512, num_anchors, num_classes)  # for 14x14

    @torch.no_grad()
    def _check_shapes(self, x: torch.Tensor,
                      s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> None:
        """開發期尺寸檢查（inference/訓練不做任何改動）。"""
        _, _, H, W = x.shape
        assert (H, W) == (224, 224), f"Expected input 224x224, got {H}x{W}"
        assert s1.shape[-2:] == (56, 56), f"Scale1 feature must be 56x56, got {tuple(s1.shape[-2:])}"
        assert s2.shape[-2:] == (28, 28), f"Scale2 feature must be 28x28, got {tuple(s2.shape[-2:])}"
        assert s3.shape[-2:] == (14, 14), f"Scale3 feature must be 14x14, got {tuple(s3.shape[-2:])}"

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
          x: [B, 3, 224, 224]

        Returns:
          [p1, p2, p3]  # 依序對應 56x56、28x28、14x14
          其中每個 pi 形狀為 [B, A*(5+C), Hi, Wi]
        """
        # Backbone
        x1 = self.block1(x)   # [B, 64, 112, 112]
        s1 = self.block2(x1)  # [B, 128, 56, 56]
        s2 = self.block3(s1)  # [B, 256, 28, 28]
        s3 = self.block4(s2)  # [B, 512, 14, 14]

        # (可選) 尺寸健檢
        # self._check_shapes(x, s1, s2, s3)

        # Detection heads
        p1 = self.head_s1(s1)  # [B, A*(5+C), 56, 56]
        p2 = self.head_s2(s2)  # [B, A*(5+C), 28, 28]
        p3 = self.head_s3(s3)  # [B, A*(5+C), 14, 14]
        return [p1, p2, p3]


# ------------------------------
# Decoding helper
# ------------------------------
def yolo_decode(pred: torch.Tensor, num_anchors: int, num_classes: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    將單一尺度輸出 pred: [B, A*(5+C), H, W]
    解析為：
      box_deltas: [B, H*W*A, 4]  -> (tx, ty, tw, th)
      obj_logits: [B, H*W*A, 1]
      cls_logits: [B, H*W*A, C]

    注意：這裡僅做張量重排，不做 sigmoid/softmax，讓 loss 自行決定。
    """
    B, ch, H, W = pred.shape
    A, C = num_anchors, num_classes
    D = 5 + C
    assert ch == A * D, f"Channel must be A*(5+C). Got {ch}, expected {A}*({D})"

    # [B, A, D, H, W] -> [B, H, W, A, D] -> [B, H*W*A, D]
    t = pred.view(B, A, D, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, H * W * A, D)

    box_deltas = t[..., 0:4]           # (tx, ty, tw, th)
    obj_logits = t[..., 4:5]           # objectness
    cls_logits = t[..., 5: 5 + C]      # class scores
    return box_deltas, obj_logits, cls_logits
