import random
from copy import deepcopy

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import cv2
import os

import torch
from torch.utils.data import Dataset

from PIL import Image

from torchvision.transforms.functional import InterpolationMode

import presets
from losses import RegressionLoss
from config import Config
from datasets import prepare_data
from model import CNN, Regressor


cfg = Config()
IMG_SIZE = cfg.original_img_size
SEED = cfg.seed
TRAIN_FOLDER = cfg.train_folder
interpolation = InterpolationMode(cfg.interpolation)
train_df, valid_df = prepare_data(cfg)


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


class ImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.data_df.iloc[idx]["id"]

        labels = [
            self.data_df.iloc[idx]["left_top_x"] / IMG_SIZE,
            self.data_df.iloc[idx]["left_top_y"] / IMG_SIZE,
            self.data_df.iloc[idx]["right_bottom_x"] / IMG_SIZE,
            self.data_df.iloc[idx]["right_bottom_y"] / IMG_SIZE,
            self.data_df.iloc[idx]["angle"] / 360,
        ]

        image = cv2.imread(f"../../data/{TRAIN_FOLDER}/img/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return int(image_name.split(".")[0]), image, torch.tensor(labels).float()

    def __len__(self):
        return len(self.data_df)


def load_models(cfg):
    cv_models = []
    img_sizes = [224, 224, 224, 272]

    model = CNN("regnet_y_8gf")
    weights = torch.load(
        "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf/best.pth"
    )["model"]
    model.load_state_dict(weights)
    cv_models.append(model)

    model = CNN("regnet_y_8gf")
    weights = torch.load(
        "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf_clouds_v2/regnet_y_8gf_best_weights.pth"
    )
    model.load_state_dict(weights)
    cv_models.append(model)

    model = CNN("regnet_y_3_2gf")
    weights = torch.load(
        "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_3_2gf/best.pth"
    )["model"]
    model.load_state_dict(weights)
    cv_models.append(model)

    model = CNN("regnet_y_3_2gf")
    weights = torch.load(
        "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_3_2gf_272_crop/best.pth"
    )["model"]
    model.load_state_dict(weights)
    cv_models.append(model)

    for model in cv_models:
        model.to(cfg.device)
        model.eval()

    return cv_models, img_sizes


def generate_preds_df(
    model_num,
    pred_df,
    model,
    dataset,
    cfg,
):
    rows = []

    with torch.inference_mode():
        for img_id, img, labels in tqdm(dataset):
            labels = labels.tolist()
            img = img.unsqueeze(0).to(cfg.device, non_blocking=True)

            pred = model(img).squeeze(0).cpu().flatten()
            row = {
                "id": img_id,
                **{str(i): p.item() for i, p in enumerate(pred)},
                "gt_left_top_x": labels[0],
                "gt_left_top_y": labels[1],
                "gt_right_bottom_x": labels[2],
                "gt_right_bottom_y": labels[3],
                "gt_angle": labels[4],
            }
            rows.append(row)
    data_df = pd.DataFrame(rows)
    if model_num == 0:
        pred_df = data_df
    else:
        data_df = data_df.drop(
            [
                "gt_left_top_x",
                "gt_left_top_y",
                "gt_right_bottom_x",
                "gt_right_bottom_y",
                "gt_angle",
            ],
            axis=1,
        )
        pred_df = pred_df.merge(data_df, on="id", suffixes=("", f"_{model_num}"))

    return pred_df


def gen_models_predictions(cv_models, img_sizes, cfg):

    pred_train_df = None
    pred_valid_df = None

    for model_num, (model, img_size) in enumerate(zip(cv_models[:], img_sizes)):

        if img_size == 272:
            val_crop_size = 272
            val_resize_size = 280
        else:
            val_crop_size = cfg.val_crop_size
            val_resize_size = cfg.val_resize_size

        test_transform = presets.PresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
        )

        train_dataset = ImageDataset(train_df, transform=test_transform)
        valid_dataset = ImageDataset(valid_df, transform=test_transform)

        pred_train_df = generate_preds_df(
            model_num, pred_train_df, model, train_dataset, cfg
        )
        pred_valid_df = generate_preds_df(
            model_num, pred_valid_df, model, valid_dataset, cfg
        )

        print(f"Model {model_num} predictions generated!")

    pred_train_data = pred_train_df.drop(
        [
            "id",
            "gt_left_top_x",
            "gt_left_top_y",
            "gt_right_bottom_x",
            "gt_right_bottom_y",
            "gt_angle",
        ],
        axis=1,
    )
    pred_valid_data = pred_valid_df.drop(
        [
            "id",
            "gt_left_top_x",
            "gt_left_top_y",
            "gt_right_bottom_x",
            "gt_right_bottom_y",
            "gt_angle",
        ],
        axis=1,
    )

    gt_train = pred_train_df[
        [
            "gt_left_top_x",
            "gt_left_top_y",
            "gt_right_bottom_x",
            "gt_right_bottom_y",
            "gt_angle",
        ]
    ]
    gt_valid = pred_valid_df[
        [
            "gt_left_top_x",
            "gt_left_top_y",
            "gt_right_bottom_x",
            "gt_right_bottom_y",
            "gt_angle",
        ]
    ]

    return pred_train_data, gt_train, pred_valid_data, gt_valid


def train_regressor(cfg, pred_train_data, gt_train, pred_valid_data, gt_valid):
    regressor = Regressor(pred_train_data.shape[-1])
    regressor.to(cfg.device)
    criterion = RegressionLoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5500, eta_min=1e-6
    )

    train_metrics, val_metrics = [], []
    best_score = 0
    for epoch in tqdm(range(5500)):
        regressor.train()
        train_pred = []
        data = torch.from_numpy(pred_train_data.values).float().to(cfg.device)
        labels = torch.from_numpy(gt_train.values).float().to(cfg.device)
        pred = regressor(data)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred.extend(pred.cpu().tolist())

        regressor.eval()
        val_pred = []
        with torch.no_grad():
            data = torch.from_numpy(pred_valid_data.values).float().to(cfg.device)
            pred = regressor(data)
            val_pred.extend(pred.cpu().tolist())

        train_metrics.append(_compute_metric(gt_train.values, train_pred))
        val_metrics.append(_compute_metric(gt_valid.values, val_pred))

        if val_metrics[-1] > best_score:
            best_regressor = deepcopy(regressor)
            best_score = val_metrics[-1]

        scheduler.step(-val_metrics[-1])

        if epoch % 1000 == 0:
            print(
                "Last Train score:",
                train_metrics[-1],
                " Best Val score:",
                max(val_metrics),
            )
    torch.save(best_regressor.state_dict(), "regressor.pth")
    return best_regressor


def main():

    cv_models, img_sizes = load_models(cfg)
    pred_train_data, gt_train, pred_valid_data, gt_valid = gen_models_predictions(
        cv_models, img_sizes, cfg
    )
    regressor = train_regressor(
        cfg, pred_train_data, gt_train, pred_valid_data, gt_valid
    )

    regressor.eval()
    with torch.inference_mode():
        val_prediction = (
            regressor(torch.from_numpy(pred_valid_data.values).float().to(cfg.device))
            .cpu()
            .tolist()
        )
    print("Valid Score:", _compute_metric(gt_valid.values, val_prediction))


main()
