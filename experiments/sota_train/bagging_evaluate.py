import random
import os
import math
from copy import deepcopy

import pandas as pd
import cv2
import torch
import torchvision
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode

import transforms
import presets
import utils
from datasets import prepare_data, ImageDataset
from config import Config
from model import CNN
from losses import RegressionLoss


SEED = 42
IMG_SIZE = 10496
sigmoid = torch.torch.nn.Sigmoid()


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


def tta(img, angle):
    interpolation = F.InterpolationMode.BILINEAR
    tta_img = F.rotate(img, angle, interpolation)
    return tta_img


def inv_tta(pred, angle):

    pred[4] *= 360
    pred[:4] *= IMG_SIZE

    pred_angle = pred[4]
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = pred[:4]

    center_coords = (top_left_x + bottom_right_x) // 2, (
        top_left_y + bottom_right_y
    ) // 2

    matrix = cv2.getRotationMatrix2D(np.float32(center_coords), -angle, 1.0)

    top_left_x, top_left_y = cv2.transform(
        np.array([[[top_left_x, top_left_y]]]), matrix
    ).squeeze()
    bottom_right_x, bottom_right_y = cv2.transform(
        np.array([[[bottom_right_x, bottom_right_y]]]), matrix
    ).squeeze()

    return [
        top_left_x / IMG_SIZE,
        top_left_y / IMG_SIZE,
        bottom_right_x / IMG_SIZE,
        bottom_right_y / IMG_SIZE,
        ((360 + (pred_angle - angle)) % 360) / 360,
    ]


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
    batch_size=1,
    num_workers=cfg.workers,
    pin_memory=True,
)

cv_models = []

model = CNN("regnet_y_8gf")
weights = torch.load(
    "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf/best.pth"
)["model"]
model.load_state_dict(weights)
cv_models.append(model)

model = CNN("regnet_y_8gf")
weights = torch.load(
    "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf_clouds_autoaug/best.pth"
)["model"]
model.load_state_dict(weights)
cv_models.append(model)


# model = CNN("res50")
# weights = torch.load(
#     "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_res50/best.pth"
# )["model"]
# model.load_state_dict(weights)
# cv_models.append(model)


for model in cv_models:
    model.to(cfg.device)
    model.eval()


agg_preds = []
all_targets = []

for model_num, model in enumerate(cv_models[:]):
    all_preds = []
    with torch.inference_mode():
        for images, target in data_loader_valid:
            images = images.to(cfg.device)

            preds = []
            for angle in [0, 180]:
                tta_img = tta(images[0], angle).unsqueeze(0)
                pred = model(tta_img).squeeze(0).cpu().numpy()
                pred = inv_tta(pred, angle)
                preds.append(pred)
            preds = torch.tensor(preds).mean(0)

            all_preds.append(preds.cpu().tolist())

            if model_num == 0:
                all_targets.extend(target.cpu().tolist())
    agg_preds.append(all_preds)

if len(agg_preds) > 1:
    agg_preds = np.average(agg_preds, axis=0, weights=[9, 7])
    print("Score:", _compute_metric(all_targets, agg_preds))
else:
    print("Score:", _compute_metric(all_targets, agg_preds[0]))
