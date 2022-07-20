import json
import os
import random

import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

from transforms import RandomRotate, RandomCutmix, RandomMixup


def prepare_data(cfg):
    json_dir = f"/home/and/projects/hacks/ai-areal-photo/data/{cfg.train_folder}/json/"

    rows = []
    for _, _, files in os.walk(json_dir):
        for x in files:
            if x.endswith(".json"):
                data = json.load(open(json_dir + x))
                new_row = {
                    "id": x.split(".")[0] + ".png",
                    "left_top_x": data["left_top"][0],
                    "left_top_y": data["left_top"][1],
                    "right_bottom_x": data["right_bottom"][0],
                    "right_bottom_y": data["right_bottom"][1],
                    "x_center": (data["left_top"][0] + data["right_bottom"][0]) / 2,
                    "y_center": (data["left_top"][1] + data["right_bottom"][1]) / 2,
                    "angle": data["angle"],
                }
                rows.append(new_row)

    data_df = pd.DataFrame(rows)
    train_df, valid_df = train_test_split(
        data_df, test_size=cfg.test_size, random_state=cfg.seed
    )

    return train_df, valid_df


class ImageDataset(Dataset):
    def __init__(
        self,
        cfg,
        data_df,
        feature_extractor=None,
        transform=None,
        cloud_transform=None,
        stage="valid",
    ):
        self.cfg = cfg
        self.data_df = data_df
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.cloud_transform = cloud_transform
        self.stage = stage

    def __getitem__(self, idx):
        # достаем имя изображения и ее лейбл
        image_name = self.data_df.iloc[idx]["id"]
        keypoints = np.array(
            [
                [
                    self.data_df.iloc[idx]["left_top_x"],
                    self.data_df.iloc[idx]["left_top_y"],
                    self.data_df.iloc[idx]["angle"],
                ],
                [
                    self.data_df.iloc[idx]["right_bottom_x"],
                    self.data_df.iloc[idx]["left_top_y"],
                    self.data_df.iloc[idx]["angle"],
                ],
                [
                    self.data_df.iloc[idx]["right_bottom_x"],
                    self.data_df.iloc[idx]["right_bottom_y"],
                    self.data_df.iloc[idx]["angle"],
                ],
                [
                    self.data_df.iloc[idx]["left_top_x"],
                    self.data_df.iloc[idx]["right_bottom_y"],
                    self.data_df.iloc[idx]["angle"],
                ],
            ]
        )

        # читаем картинку. read the image
        image = cv2.imread(
            f"/home/and/projects/hacks/ai-areal-photo/data/{self.cfg.train_folder}/img/{image_name}"
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # преобразуем, если нужно. transform it, if necessary
        if self.cloud_transform is not None and random.random() > 0.5:
            image = self.cloud_transform(image=image)

        if self.feature_extractor is not None:
            image = self.feature_extractor(
                images=Image.fromarray(image), return_tensors="pt"
            )
        elif self.transform:
            if self.stage == "train":
                center_coords = (keypoints[:, 0].min() + keypoints[:, 0].max()) // 2, (
                    keypoints[:, 1].min() + keypoints[:, 1].max()
                ) // 2

                rotate_transform = A.Compose(
                    [
                        RandomRotate(
                            center_coords=np.float32(center_coords), limit=360, p=0.9
                        ),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xya",
                        angle_in_degrees=True,
                        remove_invisible=False,
                        check_each_transform=False,
                    ),
                )
                transformed = rotate_transform(image=image, keypoints=keypoints)
                image, keypoints = transformed["image"], transformed["keypoints"]

            image = self.transform(Image.fromarray(image))

        labels = [
            keypoints[0][0] / self.cfg.original_img_size,  # top left x
            keypoints[0][1] / self.cfg.original_img_size,  # top left y
            keypoints[2][0] / self.cfg.original_img_size,  # right bottom x
            keypoints[2][1] / self.cfg.original_img_size,  # right bottom y
            keypoints[0][2] / 360,  # angle
        ]
        return image, torch.tensor(labels)

    def __len__(self):
        return len(self.data_df)
