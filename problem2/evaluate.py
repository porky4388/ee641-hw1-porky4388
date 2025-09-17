import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_keypoints_from_heatmaps(heatmaps):
    """
    Args:
        heatmaps: [B, K, H, W]
    Returns:
        coords: [B, K, 2] in (x, y) order in heatmap pixel space
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)                  # [B, K]
    ys = (idx // W).float()
    xs = (idx % W).float()
    coords = torch.stack([xs, ys], dim=-1)     # [B, K, 2]
    return coords


def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    predictions, ground_truths: [N, K, 2] in pixel space
    thresholds: list of fractions w.r.t normalization factor
    """
    N, K, _ = predictions.shape
    dists = torch.norm(predictions - ground_truths, dim=2)  # [N, K]

    if normalize_by == 'bbox':
        x_min, _ = ground_truths[:, :, 0].min(dim=1)
        x_max, _ = ground_truths[:, :, 0].max(dim=1)
        y_min, _ = ground_truths[:, :, 1].min(dim=1)
        y_max, _ = ground_truths[:, :, 1].max(dim=1)
        diag = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)  # [N]
        norm = diag.unsqueeze(1).clamp(min=1e-6).repeat(1, K)
    elif normalize_by == 'torso':
        torso = torch.norm(ground_truths[:, 0] - ground_truths[:, 3], dim=1)  # example
        norm = torso.unsqueeze(1).clamp(min=1e-6).repeat(1, K)
    else:
        raise ValueError("normalize_by must be 'bbox' or 'torso'")

    pck = {}
    for t in thresholds:
        correct = (dists <= t * norm).float().mean().item()
        pck[t] = correct
    return pck


def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thresholds = sorted(pck_heatmap.keys())

    plt.figure()
    plt.plot(thresholds, [pck_heatmap[t] for t in thresholds], marker="o", label="Heatmap")
    plt.plot(thresholds, [pck_regression[t] for t in thresholds], marker="s", label="Regression")
    plt.xlabel("Threshold (fraction of normalization)")
    plt.ylabel("PCK")
    plt.title("PCK Curve: Heatmap vs Regression")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = image.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c="g", label="GT", marker="o")
    plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c="r", label="Pred", marker="x")
    plt.legend()
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()
