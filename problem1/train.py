import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)


def main():
    # -------------------------------
    # Config
    # -------------------------------
    train_image_dir = "datasets/detection/train"
    train_ann_file = "datasets/detection/train_annotations.json"
    val_image_dir = "datasets/detection/val"
    val_ann_file = "datasets/detection/val_annotations.json"

    results_dir = "problem1/results"
    os.makedirs(results_dir, exist_ok=True)

    num_classes = 3
    num_anchors = 3
    batch_size = 16
    num_epochs = 50
    lr = 0.01
    weight_decay = 5e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Dataset & DataLoader
    # -------------------------------
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ShapeDetectionDataset(train_image_dir, train_ann_file, transform=transform)
    val_dataset = ShapeDetectionDataset(val_image_dir, val_ann_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # -------------------------------
    # Model, Loss, Optimizer
    # -------------------------------
    model = MultiScaleDetector(num_classes=num_classes, num_anchors=num_anchors).to(device)
    criterion = DetectionLoss(num_classes=num_classes, num_anchors=num_anchors)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Anchors
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors_per_level = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors_per_level = [a.to(device) for a in anchors_per_level]

    # -------------------------------
    # Training Loop
    # -------------------------------
    best_val_loss = float("inf")
    log_history = []

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(imgs)
            loss, _ = criterion(preds, anchors_per_level, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                preds = model(imgs)
                loss, _ = criterion(preds, anchors_per_level, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        # Logging
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }
        log_history.append(log_entry)
        print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        with open(os.path.join(results_dir, "training_log.json"), "w") as f:
            json.dump(log_history, f, indent=2)


if __name__ == "__main__":
    main()
