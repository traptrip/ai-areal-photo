import random
import json
from copy import deepcopy

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import cv2
import os
from pathlib import Path

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


class TestData(Dataset):
    def __init__(self, images_paths, transform=None):
        self.images_paths = images_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return int(str(self.images_paths[idx].name).split(".")[0]), image

    def __len__(self):
        return len(self.images_paths)


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


def generate_preds_df(model_num, pred_df, model, dataset, cfg):
    rows = []
    with torch.inference_mode():
        for img_id, img in tqdm(dataset):
            img = img.to(cfg.device, non_blocking=True)
            pred = model(img.unsqueeze(0)).squeeze(0).cpu()
            row = {
                "id": img_id,
                **{str(i): p.item() for i, p in enumerate(pred)},
            }
            rows.append(row)
    data_df = pd.DataFrame(rows)
    if model_num == 0:
        pred_df = data_df
    else:
        pred_df = pred_df.merge(data_df, on="id", suffixes=("", f"_{model_num}"))

    return pred_df


def gen_models_predictions(cv_models, img_sizes, cfg):

    images_paths = sorted(
        list(Path("../../data/test").iterdir()), key=lambda p: int(p.name[:-4])
    )
    pred_test_df = None

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

        test_dataset = TestData(images_paths, test_transform)

        pred_test_df = generate_preds_df(
            model_num, pred_test_df, model, test_dataset, cfg
        )

        print(f"Model {model_num} predictions generated!")

    ids = pred_test_df.id

    return ids, pred_test_df.drop("id", axis=1)


def main(submit_folder):
    cv_models, img_sizes = load_models(cfg)
    img_ids, pred_test_data = gen_models_predictions(cv_models, img_sizes, cfg)
    regressor = Regressor(pred_test_data.shape[-1])
    weights = torch.load("regressor.pth")
    regressor.load_state_dict(weights)
    regressor.to(cfg.device)
    regressor.eval()

    with torch.inference_mode():
        prediction = (
            regressor(torch.from_numpy(pred_test_data.values).float().to(cfg.device))
            .cpu()
            .tolist()
        )

    # submit_folder = Path("../../data/agg_submit_regnets_v3/")
    for img_id, pred in tqdm(zip(img_ids, prediction), total=len(img_ids)):
        res = {
            "left_top": [pred[0] * IMG_SIZE, pred[1] * IMG_SIZE],
            "right_top": [pred[2] * IMG_SIZE, pred[1] * IMG_SIZE],
            "left_bottom": [pred[0] * IMG_SIZE, pred[3] * IMG_SIZE],
            "right_bottom": [pred[2] * IMG_SIZE, pred[3] * IMG_SIZE],
            "angle": pred[4] * 360,
        }

        with open(submit_folder / f"{img_id}.json", "w") as f:
            json.dump(res, f)


main(Path("../../data/agg_submit_regnets_v3/"))
