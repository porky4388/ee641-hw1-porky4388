import torch
from torch.utils.data import DataLoader
import os

from evaluate import extract_keypoints_from_heatmaps, visualize_predictions

# 固定寫到 problem2/results/...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../problem2
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FAIL_DIR = os.path.join(RESULTS_DIR, "visualizations", "failure_cases")
# 資料根目錄（交由 main.py 傳入更穩，但也保留預設）
DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "keypoints"))


def ablation_study(dataset_class, model_class, data_root=DATA_ROOT, device="cuda"):
    """
    1) Heatmap resolution: 32/64/128
    2) Gaussian sigma: 1.0/2.0/3.0/4.0
    3) Skip connections: with vs without
    （此處為輕量檢查版：forward / 統計；如需完整短訓練做 PCK，可再擴展）
    """
    results = {"heatmap_resolution": {}, "sigma": {}, "skip_connections": {}}

    # ---------------- Experiment 1: Heatmap resolution ----------------
    for res in [32, 64, 128]:
        dataset = dataset_class(os.path.join(data_root, "train"),
                                os.path.join(data_root, "train_annotations.json"),
                                output_type="heatmap", heatmap_size=res)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = model_class(num_keypoints=5)
        images, _ = next(iter(loader))
        outputs = model(images)  # forward 一次檢查
        results["heatmap_resolution"][res] = list(outputs.shape)

    # ---------------- Experiment 2: Gaussian sigma ----------------
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        dataset = dataset_class(os.path.join(data_root, "train"),
                                os.path.join(data_root, "train_annotations.json"),
                                output_type="heatmap", heatmap_size=64, sigma=sigma)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        _, targets = next(iter(loader))
        avg_value = targets.mean().item()
        results["sigma"][sigma] = avg_value

    # ---------------- Experiment 3: Skip connections ----------------
    for use_skip in [True, False]:
        base_model = model_class(num_keypoints=5)

        if not use_skip:
            # 不使用 skip 的 decoder（避免 channel mismatch）
            class NoSkipModel(torch.nn.Module):
                def __init__(self, encoder, num_keypoints=5):
                    super().__init__()
                    self.conv1 = encoder.conv1
                    self.conv2 = encoder.conv2
                    self.conv3 = encoder.conv3
                    self.conv4 = encoder.conv4
                    self.deconv4 = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(inplace=True)
                    )
                    self.deconv3 = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU(inplace=True)
                    )
                    self.deconv2 = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.ReLU(inplace=True)
                    )
                    self.final = torch.nn.Conv2d(32, num_keypoints, kernel_size=1)

                def forward(self, x):
                    x1 = self.conv1(x)
                    x2 = self.conv2(x1)
                    x3 = self.conv3(x2)
                    x4 = self.conv4(x3)
                    d4 = self.deconv4(x4)
                    d3 = self.deconv3(d4)
                    d2 = self.deconv2(d3)
                    return self.final(d2)

            model = NoSkipModel(base_model, num_keypoints=5)
        else:
            model = base_model

        dummy_input = torch.randn(2, 1, 128, 128)
        out = model(dummy_input)
        results["skip_connections"][use_skip] = list(out.shape)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ablation_results.txt"), "w") as f:
        for key, val in results.items():
            f.write(f"{key}: {val}\n")

    print("Ablation study results saved to", os.path.join(RESULTS_DIR, "ablation_results.txt"))
    return results


def analyze_failure_cases(model_heatmap, model_regression, test_loader, device="cuda"):
    """
    存圖到 problem2/results/visualizations/failure_cases/
    """
    failure_cases = {"heatmap_success": [], "regression_success": [], "both_fail": []}

    model_heatmap.to(device).eval()
    model_regression.to(device).eval()

    strict_t = 0.05  # 影像對角線的比例

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            B, _, H, W = images.shape

            # heatmap predictions
            out_hm = model_heatmap(images)
            pred_hm = extract_keypoints_from_heatmaps(out_hm)
            scale_x = W / out_hm.shape[-1]
            scale_y = H / out_hm.shape[-2]
            pred_hm[..., 0] *= scale_x
            pred_hm[..., 1] *= scale_y

            # regression predictions
            out_reg = model_regression(images)
            pred_reg = out_reg.view(B, -1, 2)
            pred_reg[..., 0] *= W
            pred_reg[..., 1] *= H

            # ground truth
            gt = targets.view(B, -1, 2)
            gt[..., 0] *= W
            gt[..., 1] *= H

            diag = (W**2 + H**2) ** 0.5
            thr_pixels = strict_t * diag

            for i in range(B):
                d_h = torch.norm(pred_hm[i] - gt[i], dim=1).mean().item()
                d_r = torch.norm(pred_reg[i] - gt[i], dim=1).mean().item()
                correct_h = d_h < thr_pixels
                correct_r = d_r < thr_pixels

                if correct_h and not correct_r:
                    failure_cases["heatmap_success"].append((images[i].cpu(), pred_hm[i].cpu(), gt[i].cpu()))
                elif correct_r and not correct_h:
                    failure_cases["regression_success"].append((images[i].cpu(), pred_reg[i].cpu(), gt[i].cpu()))
                elif not correct_h and not correct_r:
                    failure_cases["both_fail"].append((images[i].cpu(), pred_hm[i].cpu(), gt[i].cpu()))

    os.makedirs(FAIL_DIR, exist_ok=True)
    for cat, cases in failure_cases.items():
        for idx, (img, pred, gt) in enumerate(cases[:5]):  # 每類存前 5 張
            save_path = os.path.join(FAIL_DIR, f"{cat}_{idx}.png")
            visualize_predictions(img, pred, gt, save_path)

    print("Failure cases visualized in", FAIL_DIR)
    return failure_cases
