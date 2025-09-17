import os
import json
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import nms
from collections import defaultdict

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector, yolo_decode
from utils import generate_anchors


CLASS_NAMES = ["circle", "square", "triangle"]


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)


def main():
    # -------------------------------
    # Config
    # -------------------------------
    val_image_dir = "datasets/detection/val"
    val_ann_file = "datasets/detection/val_annotations.json"

    results_dir = "problem1/results/visualizations"
    os.makedirs(results_dir, exist_ok=True)

    num_classes = 3
    num_anchors = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Dataset & Loader
    # -------------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ShapeDetectionDataset(val_image_dir, val_ann_file, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # -------------------------------
    # 讀取 categories → id2name
    # -------------------------------
    with open(val_ann_file, "r") as f:
        coco = json.load(f)
    id2name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # -------------------------------
    # Model
    # -------------------------------
    model = MultiScaleDetector(num_classes=num_classes, num_anchors=num_anchors).to(device)
    model.load_state_dict(torch.load(os.path.join("problem1", "results", "best_model.pth"), map_location=device))
    model.eval()

    # -------------------------------
    # Anchors
    # -------------------------------
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors_per_level = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors_per_level = [a.to(device) for a in anchors_per_level]

    # -------------------------------
    # 1. Visualize detections on 10 images
    # -------------------------------
    print("Generating detection visualizations...")
    indices = random.sample(range(len(dataset)), 10)
    for i, idx in enumerate(indices, start=1):
        img, target = dataset[idx]
        img_t = img.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_t)

        pred_boxes, pred_scores, pred_labels = [], [], []
        for pred, anchors in zip(preds, anchors_per_level):
            # decode + 去掉 batch 維度
            box_deltas, obj_logits, cls_logits = yolo_decode(pred, num_anchors, num_classes)
            box_deltas = box_deltas.squeeze(0)   # [N,4]
            obj_logits = obj_logits.squeeze(0)   # [N,1]
            cls_logits = cls_logits.squeeze(0)   # [N,C]

            obj_scores = torch.sigmoid(obj_logits).squeeze(-1)  # [N]
            cls_scores = torch.softmax(cls_logits, dim=-1)      # [N,C]
            conf, cls_idx = torch.max(cls_scores, dim=-1)       # [N]
            score = obj_scores * conf                           # [N]

            mask = score > 0.5
            if mask.any():
                pred_boxes.append(anchors[mask])
                pred_scores.append(score[mask])
                pred_labels.extend(cls_idx[mask].tolist())

        if pred_boxes:
            pred_boxes = torch.cat(pred_boxes, dim=0)
            pred_scores = torch.cat(pred_scores, dim=0)

            # 🔑 NMS 過濾多餘框
            keep = nms(pred_boxes, pred_scores, iou_threshold=0.4)
            pred_boxes = pred_boxes[keep].cpu()
            pred_scores = pred_scores[keep].cpu()
            pred_labels = [pred_labels[j] for j in keep.tolist()]
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros((0,))
            pred_labels = []

        # 🔑 每個類別保留前二名分數最高的紅框
        topk_per_class = defaultdict(list)
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            topk_per_class[label].append((box, score))

        for label in topk_per_class:
            topk_per_class[label] = sorted(
                topk_per_class[label], key=lambda x: x[1], reverse=True
            )[:2]  # 取前二名

        # 畫圖 (GT 綠框 + Pred 紅框)
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())

        # 畫 GT
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)
            class_name = id2name[int(label.item())]
            ax.text(x1, y1 - 2, class_name,
                    color="green", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, pad=0))

        # 畫 Pred (每類顯示前二名)
        for label, box_scores in topk_per_class.items():
            for box, score in box_scores:
                x1, y1, x2, y2 = box.tolist()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1 - 2, f"{CLASS_NAMES[label]} ({score:.2f})",
                        color="red", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.5, pad=0))

        plt.axis("off")
        save_path = os.path.join(results_dir, f"det{i}.png")  # ✅ det1.png ~ det10.png
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    # -------------------------------
    # 2. Anchor coverage visualization
    # -------------------------------
    print("Generating anchor coverage visualizations...")
    for scale_idx, anchors in enumerate(anchors_per_level):
        img, target = dataset[random.randint(0, len(dataset) - 1)]
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())

        # 畫 GT
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)

        # 隨機挑一些 anchors
        sample_idx = torch.randperm(len(anchors))[:50]
        for box in anchors[sample_idx]:
            x1, y1, x2, y2 = box.cpu().tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1, edgecolor="r", facecolor="none", alpha=0.3)
            ax.add_patch(rect)

        plt.axis("off")
        plt.savefig(os.path.join(results_dir, f"anchors_scale{scale_idx+1}.png"), bbox_inches="tight")
        plt.close()

    # -------------------------------
    # 3. Scale analysis
    # -------------------------------
    print("Generating scale analysis...")
    size_bins = {"small": 0, "medium": 0, "large": 0}
    scale_assignments = {1: 0, 2: 0, 3: 0}

    for i in range(min(200, len(dataset))):
        _, target = dataset[i]
        for box in target["boxes"]:
            w = (box[2] - box[0]).item()
            h = (box[3] - box[1]).item()
            size = max(w, h)
            if size < 32:
                size_bins["small"] += 1
                scale_assignments[1] += 1
            elif size < 96:
                size_bins["medium"] += 1
                scale_assignments[2] += 1
            else:
                size_bins["large"] += 1
                scale_assignments[3] += 1

    with open(os.path.join(results_dir, "scale_analysis.json"), "w") as f:
        json.dump({
            "object_size_bins": size_bins,
            "scale_assignments": scale_assignments
        }, f, indent=2)


if __name__ == "__main__":
    main()
