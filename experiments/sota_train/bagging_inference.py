import torch
import math
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from torchvision.transforms import functional as F

from model import CNN
import presets
from config import Config

SEED = 42
IMG_SIZE = 10496
TRAIN_IMG_SIZE = 224
SAVE_FOLDER = Path("/home/and/projects/hacks/ai-areal-photo/data/submit_bagging_v2")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 1234, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.set_printoptions(precision=precision)


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


cfg = Config()
cv_models = []


model = CNN("regnet_y_8gf")
weights = torch.load(
    "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf/best.pth"
)["model"]
model.load_state_dict(weights)
cv_models.append(model)

# model = CNN("regnet_y_8gf")
# weights = torch.load(
#     "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf_clouds_autoaug/best.pth"
# )["model"]
# model.load_state_dict(weights)
# cv_models.append(model)

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


for model in cv_models:
    model.to(cfg.device)
    model.eval()


test_transform = presets.PresetEval(
    crop_size=cfg.val_crop_size,
    resize_size=cfg.val_resize_size,
)

images_paths = sorted(
    list(Path("/home/and/projects/hacks/ai-areal-photo/data/test").iterdir()),
    key=lambda p: int(p.name[:-4]),
)
dataset = TestData(images_paths, test_transform)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.workers,
    pin_memory=True,
)


ids = []
agg_preds = []

for model_num, model in enumerate(cv_models):
    all_preds = []
    with torch.inference_mode():
        for img_id, images in data_loader:
            images = images.to(cfg.device)

            preds = []
            angles = [0, 180] if model_num in [0, 1] else [0]
            for angle in angles:
                tta_img = tta(images[0], angle).unsqueeze(0)
                pred = model(tta_img).squeeze(0).cpu().numpy()
                pred = inv_tta(pred, angle)
                preds.append(pred)
            preds = torch.tensor(preds).mean(0)

            all_preds.append(preds.cpu().tolist())

            if model_num == 0:
                ids.append(img_id.item())

    agg_preds.append(all_preds)

# weights = [10, 7, 9]
agg_preds = np.average(agg_preds, axis=0)  # , weights=weights)

for img_id, pred in zip(ids, agg_preds):
    res = {
        "left_top": [pred[0] * IMG_SIZE, pred[1] * IMG_SIZE],
        "right_top": [pred[2] * IMG_SIZE, pred[1] * IMG_SIZE],
        "left_bottom": [pred[0] * IMG_SIZE, pred[3] * IMG_SIZE],
        "right_bottom": [pred[2] * IMG_SIZE, pred[3] * IMG_SIZE],
        "angle": pred[4] * 360,
    }

    with open(SAVE_FOLDER / f"{img_id}.json", "w") as f:
        json.dump(res, f)
