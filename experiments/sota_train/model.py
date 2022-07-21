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
        elif name == "regnet_y_3_2gf":
            self.model = models.regnet_y_3_2gf(
                weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "regnet_y_8gf":
            self.model = models.regnet_y_8gf(
                weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V2
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "regnet_y_16gf":
            self.model = models.regnet_y_16gf(
                weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, 5)
        elif name == "effb0":
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, 5
            )
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
        elif name == "swin_t":
            self.model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.model.head = nn.Linear(self.model.head.in_features, 5)

        elif name == "convnext_s":
            self.model = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            )
            self.model.classifier[2] = nn.Linear(
                self.model.classifier[2].in_features, 5
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
