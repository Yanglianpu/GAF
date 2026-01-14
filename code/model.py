# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiModalFlameNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT 

        self.img_backbone = resnet18(weights=weights)
        self.img_backbone.fc = nn.Identity()   


        self.gaf_backbone = resnet18(weights=weights)
        self.gaf_backbone.fc = nn.Identity()

        self.gate_net = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()                 
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, gaf):

        if gaf.shape[1] == 1:
            gaf = gaf.repeat(1, 3, 1, 1)

        # 各自 backbone 提取特征
        feat_img = self.img_backbone(img)   # (B, 512)
        feat_gaf = self.gaf_backbone(gaf)   # (B, 512)

        # 拼接后输入 gate 网络
        feat_cat = torch.cat([feat_img, feat_gaf], dim=1)   # (B, 1024)
        gate = self.gate_net(feat_cat)                      # (B, 1), 每个样本一个 gate

        # 融合：gate 越大越偏向图像，越小越偏向 GAF
        fused_feat = gate * feat_img + (1.0 - gate) * feat_gaf   # (B, 512)

        # 分类
        logits = self.classifier(fused_feat)  # (B, num_classes)

        return logits

