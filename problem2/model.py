import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Heatmap-based keypoint detection network.
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # ---------------- Encoder ----------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64 -> 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16 -> 8
        )

        # ---------------- Decoder ----------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)  # [B, K, 64, 64]

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # [B, 32, 64, 64]
        x2 = self.conv2(x1) # [B, 64, 32, 32]
        x3 = self.conv3(x2) # [B, 128, 16, 16]
        x4 = self.conv4(x3) # [B, 256, 8, 8]

        # Decoder with skip connections
        d4 = self.deconv4(x4)                # [B,128,16,16]
        d4 = torch.cat([d4, x3], dim=1)      # [B,256,16,16]

        d3 = self.deconv3(d4)                # [B,64,32,32]
        d3 = torch.cat([d3, x2], dim=1)      # [B,128,32,32]

        d2 = self.deconv2(d3)                # [B,32,64,64]
        out = self.final(d2)                 # [B,K,64,64]

        return out


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Direct regression network.
        """
        super().__init__()
        self.num_keypoints = num_keypoints

        # ---------------- Encoder ----------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 64 -> 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 16 -> 8
        )

        # ---------------- Regression Head ----------------
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_keypoints * 2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)   # [B,256,8,8]

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B,256,1,1]
        x = torch.flatten(x, 1)               # [B,256]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = torch.sigmoid(self.fc3(x))  # [B, num_keypoints*2], normalized [0,1]

        return x
