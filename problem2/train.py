import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# --- project imports ---
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import (
    extract_keypoints_from_heatmaps,
    compute_pck,
    plot_pck_curves,
    visualize_predictions,
)
from baseline import ablation_study, analyze_failure_cases

# ---------- fixed paths (always under problem2/) ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))                # .../problem2
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR     = os.path.join(RESULTS_DIR, "visualizations")
DATA_ROOT   = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "keypoints"))  # .../datasets/keypoints


# ============================
# Part C: training functions
# ============================
def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
    """Train heatmap model with MSE on heatmaps; save best to results/."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        print(f"[Heatmap] Epoch {epoch+1}/{num_epochs} - "
              f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "heatmap_model.pth"))

        # snapshots every 10 epochs
        if (epoch + 1) % 10 == 0:
            try:
                sample_img, _ = next(iter(val_loader))
                sample_img = sample_img.to(device)
                with torch.no_grad():
                    out = model(sample_img[:1])
                hm = out[0].detach().cpu().numpy()
                for k in range(min(5, hm.shape[0])):
                    plt.imshow(hm[k], cmap="hot")
                    plt.title(f"Epoch {epoch+1}, Keypoint {k}")
                    plt.savefig(os.path.join(VIS_DIR, f"heatmap_epoch{epoch+1}_kp{k}.png"))
                    plt.close()
            except StopIteration:
                pass

    return {"train_loss": train_losses, "val_loss": val_losses}


def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    """Train regression model with MSE on normalized (x,y); save best to results/."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        print(f"[Regression] Epoch {epoch+1}/{num_epochs} - "
              f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "regression_model.pth"))

    return {"train_loss": train_losses, "val_loss": val_losses}


# ============================
# Orchestration (moved from main.py)
# ============================
def _build_loaders(output_type="heatmap", batch_size=32, split="train"):
    img_dir = os.path.join(DATA_ROOT, split)
    ann_file = os.path.join(DATA_ROOT, f"{split}_annotations.json")
    ds = KeypointDataset(img_dir, ann_file, output_type=output_type)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))


def run_training():
    # heatmap
    train_loader_h = _build_loaders(output_type="heatmap", batch_size=32, split="train")
    val_loader_h   = _build_loaders(output_type="heatmap", batch_size=32, split="val")
    heatmap_model = HeatmapNet(num_keypoints=5)
    heatmap_log = train_heatmap_model(heatmap_model, train_loader_h, val_loader_h, num_epochs=30)

    # regression
    train_loader_r = _build_loaders(output_type="regression", batch_size=32, split="train")
    val_loader_r   = _build_loaders(output_type="regression", batch_size=32, split="val")
    regression_model = RegressionNet(num_keypoints=5)
    regression_log = train_regression_model(regression_model, train_loader_r, val_loader_r, num_epochs=30)

    # save logs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump({"heatmap": heatmap_log, "regression": regression_log}, f, indent=4)

    return heatmap_model, regression_model


def run_evaluation(heatmap_model, regression_model):
    val_loader = _build_loaders(output_type="regression", batch_size=32, split="val")
    preds_heat, preds_reg, gts = [], [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heatmap_model.to(device).eval()
    regression_model.to(device).eval()

    os.makedirs(VIS_DIR, exist_ok=True)

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            gt = targets.view(-1, 5, 2) * 128
            gts.append(gt.cpu())

            # heatmap
            out_hm  = heatmap_model(images)
            pred_hm = extract_keypoints_from_heatmaps(out_hm)
            scale_x = 128 / out_hm.shape[-1]
            scale_y = 128 / out_hm.shape[-2]
            pred_hm[..., 0] *= scale_x
            pred_hm[..., 1] *= scale_y
            preds_heat.append(pred_hm.cpu())

            # regression
            out_reg  = regression_model(images)
            pred_reg = out_reg.view(-1, 5, 2) * 128
            preds_reg.append(pred_reg.cpu())

            # one sample viz
            if len(preds_heat) == 1:
                visualize_predictions(
                    images[0].cpu(), pred_hm[0].cpu(), gt[0].cpu(),
                    save_path=os.path.join(VIS_DIR, "prediction_heatmap.png"),
                )
                visualize_predictions(
                    images[0].cpu(), pred_reg[0].cpu(), gt[0].cpu(),
                    save_path=os.path.join(VIS_DIR, "prediction_regression.png"),
                )

    preds_heat = torch.cat(preds_heat, dim=0)
    preds_reg  = torch.cat(preds_reg,  dim=0)
    gts        = torch.cat(gts,        dim=0)

    thresholds = [0.05, 0.1, 0.15, 0.2]  # per assignment
    pck_heat = compute_pck(preds_heat, gts, thresholds, normalize_by="bbox")
    pck_reg  = compute_pck(preds_reg,  gts, thresholds, normalize_by="bbox")

    plot_pck_curves(pck_heat, pck_reg, save_path=os.path.join(VIS_DIR, "pck_curve.png"))
    return pck_heat, pck_reg


def run_ablation():
    return ablation_study(KeypointDataset, HeatmapNet, DATA_ROOT)


def run_failure_analysis(heatmap_model, regression_model):
    test_loader = _build_loaders(output_type="regression", batch_size=16, split="val")
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
