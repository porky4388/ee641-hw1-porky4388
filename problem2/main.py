import torch
from torch.utils.data import DataLoader
import json
import os

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from train import train_heatmap_model, train_regression_model
from evaluate import (
    extract_keypoints_from_heatmaps,
    compute_pck,
    plot_pck_curves,
    visualize_predictions
)
from baseline import ablation_study, analyze_failure_cases

# ---------------- 路徑設定（固定到 problem2/） ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))                      # .../problem2
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "keypoints"))  # .../datasets/keypoints


def run_training():
    train_dataset_heat = KeypointDataset(
        os.path.join(DATA_ROOT, "train"),
        os.path.join(DATA_ROOT, "train_annotations.json"),
        output_type="heatmap"
    )
    val_dataset_heat = KeypointDataset(
        os.path.join(DATA_ROOT, "val"),
        os.path.join(DATA_ROOT, "val_annotations.json"),
        output_type="heatmap"
    )

    train_loader_heat = DataLoader(train_dataset_heat, batch_size=32, shuffle=True)
    val_loader_heat = DataLoader(val_dataset_heat, batch_size=32, shuffle=False)

    heatmap_model = HeatmapNet(num_keypoints=5)
    heatmap_log = train_heatmap_model(heatmap_model, train_loader_heat, val_loader_heat, num_epochs=30)

    train_dataset_reg = KeypointDataset(
        os.path.join(DATA_ROOT, "train"),
        os.path.join(DATA_ROOT, "train_annotations.json"),
        output_type="regression"
    )
    val_dataset_reg = KeypointDataset(
        os.path.join(DATA_ROOT, "val"),
        os.path.join(DATA_ROOT, "val_annotations.json"),
        output_type="regression"
    )

    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32, shuffle=False)

    regression_model = RegressionNet(num_keypoints=5)
    regression_log = train_regression_model(regression_model, train_loader_reg, val_loader_reg, num_epochs=30)

    # 寫 training log 到 problem2/results/
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump({"heatmap": heatmap_log, "regression": regression_log}, f, indent=4)

    return heatmap_model, regression_model


def run_evaluation(heatmap_model, regression_model):
    val_dataset = KeypointDataset(
        os.path.join(DATA_ROOT, "val"),
        os.path.join(DATA_ROOT, "val_annotations.json"),
        output_type="regression"
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    preds_heat, preds_reg, gts = [], [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heatmap_model.to(device).eval()
    regression_model.to(device).eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            gt = targets.view(-1, 5, 2) * 128
            gts.append(gt.cpu())

            out_hm = heatmap_model(images)
            pred_hm = extract_keypoints_from_heatmaps(out_hm)
            scale_x = 128 / out_hm.shape[-1]
            scale_y = 128 / out_hm.shape[-2]
            pred_hm[..., 0] *= scale_x
            pred_hm[..., 1] *= scale_y
            preds_heat.append(pred_hm.cpu())

            out_reg = regression_model(images)
            pred_reg = out_reg.view(-1, 5, 2) * 128
            preds_reg.append(pred_reg.cpu())

            # 存一組示例可視化到 problem2/results/visualizations/
            if len(preds_heat) == 1:
                visualize_predictions(
                    images[0].cpu(), pred_hm[0].cpu(), gt[0].cpu(),
                    save_path=os.path.join(VIS_DIR, "prediction_heatmap.png")
                )
                visualize_predictions(
                    images[0].cpu(), pred_reg[0].cpu(), gt[0].cpu(),
                    save_path=os.path.join(VIS_DIR, "prediction_regression.png")
                )

    preds_heat = torch.cat(preds_heat, dim=0)
    preds_reg = torch.cat(preds_reg, dim=0)
    gts = torch.cat(gts, dim=0)

    # 報告要求的門檻
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_heat = compute_pck(preds_heat, gts, thresholds, normalize_by='bbox')
    pck_reg = compute_pck(preds_reg, gts, thresholds, normalize_by='bbox')

    plot_pck_curves(pck_heat, pck_reg, save_path=os.path.join(VIS_DIR, "pck_curve.png"))
    return pck_heat, pck_reg


def run_ablation():
    # 傳入 DATA_ROOT，避免 baseline.py 內部猜路徑
    return ablation_study(KeypointDataset, HeatmapNet, DATA_ROOT)


def run_failure_analysis(heatmap_model, regression_model):
    test_dataset = KeypointDataset(
        os.path.join(DATA_ROOT, "val"),
        os.path.join(DATA_ROOT, "val_annotations.json"),
        output_type="regression"
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return analyze_failure_cases(heatmap_model, regression_model, test_loader)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    heatmap_model, regression_model = run_training()

    pck_heat, pck_reg = run_evaluation(heatmap_model, regression_model)
    print("PCK Heatmap:", pck_heat)
    print("PCK Regression:", pck_reg)

    ablation_results = run_ablation()
    print("Ablation study results:", ablation_results)

    failure_cases = run_failure_analysis(heatmap_model, regression_model)
    print("Failure cases analyzed.")


if __name__ == "__main__":
    main()
