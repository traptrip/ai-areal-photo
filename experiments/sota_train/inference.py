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
SAVE_FOLDER = Path("/home/and/projects/hacks/ai-areal-photo/data/submit/regnet_y_8gf")
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
models_weights = "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf/best.pth"
model = CNN("regnet_y_8gf")
model.load_state_dict(torch.load(models_weights)["model"])
model.to(cfg.device)
model.eval()

test_transform = presets.PresetEval(
    crop_size=cfg.val_crop_size, resize_size=cfg.val_resize_size
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


def tta(img, angle):
    interpolation = F.InterpolationMode.BILINEAR
    tta_img = F.rotate(torch.from_numpy(img), angle, interpolation)
    return tta_img.numpy()


def inv_tta(prediction, angle):

    pred_angle = prediction[4]
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = prediction[:4]

    center_coords = (top_left_x + bottom_right_x) // 2, (
        top_left_y + bottom_right_y
    ) // 2

    matrix = cv2.getRotationMatrix2D(center_coords, -angle, 1.0)

    top_left_x, top_left_y = cv2.transform(
        np.array([[[top_left_x, top_left_y]]]), matrix
    ).squeeze()
    bottom_right_x, bottom_right_y = cv2.transform(
        np.array([[[bottom_right_x, bottom_right_y]]]), matrix
    ).squeeze()

    return (
        top_left_x,
        top_left_y,
        bottom_right_x,
        bottom_right_y,
        math.degrees(math.radians(pred_angle) + math.radians(angle)),
    )


with torch.inference_mode():
    for img_id, batch in tqdm(data_loader):
        batch = batch.to(cfg.device)
        img_id = img_id.item()

        pred = model(batch).squeeze(0).cpu().tolist()
        res = {
            "left_top": [pred[0] * IMG_SIZE, pred[1] * IMG_SIZE],
            "right_top": [pred[2] * IMG_SIZE, pred[1] * IMG_SIZE],
            "left_bottom": [pred[0] * IMG_SIZE, pred[3] * IMG_SIZE],
            "right_bottom": [pred[2] * IMG_SIZE, pred[3] * IMG_SIZE],
            "angle": pred[4] * 360,
        }

        with open(SAVE_FOLDER / f"{img_id}.json", "w") as f:
            json.dump(res, f)
