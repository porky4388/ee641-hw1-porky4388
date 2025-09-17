import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T


class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.

        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.samples = data["images"]

        # Fixed image size for training
        self.input_size = (128, 128)
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(self.input_size),
            T.ToTensor()  # [0,1] normalized
        ])

    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.

        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap

        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)

        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0:  # skip invalid points
                continue

            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            # Gaussian centered at (x, y)
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
            heatmaps[i] = heatmap

        return torch.tensor(heatmaps, dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [K, 64, 64] (K heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [2K] (x,y for K keypoints, normalized to [0,1])
        """
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # [1, 128, 128]

        keypoints = np.array(sample["keypoints"], dtype=np.float32)  # [K,2]
        K = keypoints.shape[0]

        if self.output_type == 'heatmap':
            # scale keypoints to heatmap size
            scale_x = self.heatmap_size / self.input_size[0]
            scale_y = self.heatmap_size / self.input_size[1]
            keypoints_hm = np.stack([
                keypoints[:, 0] * scale_x,
                keypoints[:, 1] * scale_y
            ], axis=1)

            targets = self.generate_heatmap(keypoints_hm, self.heatmap_size, self.heatmap_size)

        elif self.output_type == 'regression':
            # normalize keypoints to [0,1]
            keypoints[:, 0] /= self.input_size[0]
            keypoints[:, 1] /= self.input_size[1]
            targets = torch.tensor(keypoints.flatten(), dtype=torch.float32)

        else:
            raise ValueError("output_type must be 'heatmap' or 'regression'")

        return image, targets

    def __len__(self):
        return len(self.samples)
