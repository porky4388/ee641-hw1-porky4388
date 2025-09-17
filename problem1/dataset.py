import json
import os
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

# 類別 ID 與名稱
CLASS_NAME_TO_ID = {
    "circle": 0,
    "square": 1,
    "triangle": 2,
}

class ShapeDetectionDataset(Dataset):
    """
    Dataset for EE641 HW1 Problem 1
    JSON 格式 (COCO-like, 已簡化):
      images: [{id, file_name, width, height}]
      annotations: [{id, image_id, category_id, bbox[x1,y1,x2,y2], area, visibility}]
      categories: [{id, name}]
    """

    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # 建立 image_id -> image_meta
        self.images = {img["id"]: img for img in coco["images"]}

        # 建立 image_id -> annotations
        self.img_id_to_anns: Dict[int, list] = {img_id: [] for img_id in self.images.keys()}
        for ann in coco["annotations"]:
            self.img_id_to_anns[ann["image_id"]].append(ann)

        self.ids = list(self.images.keys())

        # 類別映射 (確保 0=circle, 1=square, 2=triangle)
        self.cat_id_remap = {}
        if "categories" in coco:
            for c in coco["categories"]:
                name = c["name"].lower()
                if name in CLASS_NAME_TO_ID:
                    self.cat_id_remap[c["id"]] = CLASS_NAME_TO_ID[name]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_meta = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_meta["file_name"])
        img = Image.open(img_path).convert("RGB")  # 224x224

        boxes = []
        labels = []

        for ann in self.img_id_to_anns[img_id]:
            # 過濾掉完全不可見的物件
            if float(ann.get("visibility", 1.0)) == 0.0:
                continue

            # 注意：這份 JSON bbox 已經是 [x1,y1,x2,y2]
            bbox_xyxy = ann["bbox"]

            boxes.append(bbox_xyxy)
            lab = ann["category_id"]
            if self.cat_id_remap:
                lab = self.cat_id_remap.get(lab, lab)
            labels.append(lab)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return img, {"boxes": boxes, "labels": labels}
