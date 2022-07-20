import torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    def __init__(self, name="res50") -> None:
        super().__init__()

        if name == "res18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "res34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "res50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "effb1":
            self.model = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 5
            )
        elif name == "effb3":
            self.model = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 5
            )
        elif name == "effb4":
            self.model = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 5
            )
        elif name == "effb5":
            self.model = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 5
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
