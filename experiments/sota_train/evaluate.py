import random
import os

import cv2
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import presets
import utils
from datasets import ImageDataset, prepare_data
from config import Config
from model import CNN

SEED = 42
sigmoid = torch.nn.Sigmoid()


def _compute_metric(data_true_batch, data_pred_batch, w=10496, h=10496):
    result_metric = 0
    for data_true, data_pred in zip(data_true_batch, data_pred_batch):
        x_center_true = (data_true[0] + data_true[2]) / 2
        y_center_true = (data_true[1] + data_true[3]) / 2
        x_center_pred = (data_pred[0] + data_pred[2]) / 2
        y_center_pred = (data_pred[1] + data_pred[3]) / 2

        x_metr = abs(x_center_true - x_center_pred)
        y_metr = abs(y_center_true - y_center_pred)
        angle_metr = abs(data_true[4] - data_pred[4])

        metr = 1 - (
            0.7 * 0.5 * (x_metr + y_metr) + 0.3 * min(angle_metr, abs(angle_metr - 360))
        )
        result_metric += metr
    return result_metric / (len(data_true_batch) + 1)


def set_seed(seed: int = 1234, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.set_printoptions(precision=precision)


set_seed(SEED)

cfg = Config()

train_df, valid_df = prepare_data(cfg)
valid_dataset = ImageDataset(
    cfg,
    valid_df,
    feature_extractor=None,
    transform=presets.PresetEval(
        crop_size=cfg.val_crop_size,
        resize_size=cfg.val_resize_size,
    ),
    cloud_transform=None,
    stage="valid",
)
data_loader_valid = DataLoader(
    valid_dataset,
    batch_size=cfg.batch_size,
    num_workers=cfg.workers,
    pin_memory=True,
)


# model = torchvision.models.__dict__["resnet50"](
#     weights=None
# )
# model.fc = torch.nn.Linear(model.fc.in_features, 5)
model = CNN("regnet_y_8gf")
weights = torch.load(
    "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf_clouds/best.pth"
)["model"]
model.load_state_dict(weights)
model.to(cfg.device)
model.eval()

all_preds = []
all_targets = []
with torch.inference_mode():
    for images, target in data_loader_valid:
        images = images.to(cfg.device)
        pred = model(images)

        all_preds.extend(pred.cpu().tolist())
        all_targets.extend(target.cpu().tolist())

print("Score:", _compute_metric(all_targets, all_preds))
