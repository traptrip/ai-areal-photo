from dataclasses import dataclass

import torch.distributed as dist
from torchvision import models


@dataclass
class Config:
    seed = 42
    device = "cuda"
    use_deterministic_algorithms = True
    distributed = False
    train_folder = "train"
    test_size = 0.2
    original_img_size = 10496
    output_dir = "run"

    # Model
    # model = "resnet50"
    # weights = models.ResNet50_Weights.IMAGENET1K_V2
    # model = "efficientnet_b0"
    # weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model_name = "regnet_y_3_2gf"

    n_classes = 5

    # train options
    batch_size = 256
    workers = 4
    epochs = 800
    amp = True
    gpus = [0]
    resume = None  # "/home/and/projects/hacks/ai-areal-photo/experiments/sota_train/run_regnet_y_8gf/best.pth"  # resume model path
    test_only = False
    start_epoch = 0

    # Logger
    print_freq = 100

    # dataset options
    val_resize_size = 232
    val_crop_size = 224
    train_crop_size = 176
    interpolation = "bilinear"
    cache_dataset = True
    auto_augment_policy = None  # "ta_wide"
    random_erase_prob = 0.1

    ra_sampler = False
    ra_reps = 4

    # mixup
    mixup_alpha = 0.2
    cutmix_alpha = 1.0

    # optimizer
    optimizer = "adamw"
    lr = 1e-3
    bias_weight_decay = None
    transformer_embedding_decay = None
    weight_decay = 2e-05
    norm_weight_decay = 0.0

    # scheduler
    lr_scheduler = "CosineAnnealingLR"
    lr_min = 1e-6
    # lr_step_size = 0.1
    # lr_gamma = 0.1
    lr_warmup_epochs = 5
    lr_warmup_method = "linear"
    lr_warmup_decay = 0.01

    # EMA
    model_ema = True
    world_size = 1  # dist.get_world_size()
    model_ema_steps = 32
    model_ema_decay = 0.99998

    # Gradients
    clip_grad_norm = None
