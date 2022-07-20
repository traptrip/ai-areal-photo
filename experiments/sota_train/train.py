import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import transforms
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter

import presets
import utils
from sampler import RASampler
from datasets import ImageDataset, prepare_data
from transforms import cloud_transform
from losses import RegressionLoss
from config import Config
from model import CNN


sm = SummaryWriter(log_dir="tb_logs/run")


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    cfg,
    model_ema=None,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    train_epoch_score = 0
    train_epoch_size = 0

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, cfg.print_freq, header)
    ):

        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        train_score = -loss.item()

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

        if model_ema and i % cfg.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < cfg.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["score"].update(train_score, n=batch_size)
        # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        train_epoch_score += train_score
        train_epoch_size += 1

    return train_epoch_score / train_epoch_size


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            valid_score = -loss.item()

            # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["score"].update(valid_score, n=batch_size)
            # metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Score: {metric_logger.score.global_avg:.6f}"  # Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.score.global_avg


def load_data(cfg):
    # Data loading code
    print("Loading data")
    interpolation = InterpolationMode(cfg.interpolation)
    train_df, valid_df = prepare_data(cfg)

    print("Loading training data")
    st = time.time()
    dataset_train = ImageDataset(
        cfg,
        train_df,
        feature_extractor=None,
        transform=presets.PresetTrain(
            crop_size=cfg.train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=cfg.auto_augment_policy,
            random_erase_prob=cfg.random_erase_prob,
        ),
        cloud_transform=cloud_transform,
        stage="train",
    )
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_valid = ImageDataset(
        cfg,
        valid_df,
        feature_extractor=None,
        transform=presets.PresetEval(
            crop_size=cfg.val_crop_size,
            resize_size=cfg.val_resize_size,
            interpolation=interpolation,
        ),
        cloud_transform=None,
        stage="valid",
    )

    print("Creating data loaders")
    if cfg.distributed:
        if hasattr(cfg, "ra_sampler") and cfg.ra_sampler:
            train_sampler = RASampler(
                dataset_train, shuffle=True, repetitions=cfg.ra_reps
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_train
            )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_valid, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        valid_sampler = torch.utils.data.SequentialSampler(dataset_valid)

    return dataset_train, dataset_valid, train_sampler, valid_sampler


def main(cfg):

    utils.set_seed(cfg.seed)

    if cfg.output_dir:
        utils.mkdir(cfg.output_dir)

    utils.init_distributed_mode(cfg)
    print(cfg)

    device = torch.device(cfg.device)

    # if cfg.use_deterministic_algorithms:
    #     torch.backends.cudnn.benchmark = False
    #     torch.use_deterministic_algorithms(True)
    # else:
    #     torch.backends.cudnn.benchmark = True

    dataset_train, dataset_valid, train_sampler, valid_sampler = load_data(cfg)

    collate_fn = None
    num_classes = cfg.n_classes
    mixup_transforms = []
    if cfg.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=cfg.mixup_alpha)
        )
    if cfg.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes, p=1.0, alpha=cfg.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        sampler=valid_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    print("Creating model")
    # model = torchvision.models.__dict__[cfg.model](weights=cfg.weights)
    # model.fc = nn.Linear(model.fc.in_features, 5)
    model = CNN(cfg.model_name)
    model.to(device)

    if cfg.distributed and cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = RegressionLoss()

    custom_keys_weight_decay = []
    if cfg.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", cfg.bias_weight_decay))
    if cfg.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, cfg.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        cfg.weight_decay,
        norm_weight_decay=cfg.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    opt_name = cfg.optimizer.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {cfg.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    scaler = torch.cuda.amp.GradScaler() if cfg.amp else None

    cfg.lr_scheduler = cfg.lr_scheduler.lower()
    if cfg.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
    elif cfg.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.lr_warmup_epochs, eta_min=cfg.lr_min
        )
    elif cfg.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.lr_gamma
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{cfg.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if cfg.lr_warmup_epochs > 0:
        if cfg.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=cfg.lr_warmup_decay,
                total_iters=cfg.lr_warmup_epochs,
            )
        elif cfg.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=cfg.lr_warmup_decay,
                total_iters=cfg.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{cfg.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[cfg.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=cfg.gpus)
        model_without_ddp = model.module

    model_ema = None
    if cfg.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = cfg.world_size * cfg.batch_size * cfg.model_ema_steps / cfg.epochs
        alpha = 1.0 - cfg.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not cfg.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        cfg.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if cfg.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_valid, device=device, log_suffix="EMA"
            )
        else:
            evaluate(model, criterion, data_loader_valid, device=device)
        return

    print("Start training")
    start_time = time.time()
    best_score = 0
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        train_score = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader_train,
            device,
            epoch,
            cfg,
            model_ema,
            scaler,
        )
        lr_scheduler.step()
        valid_score = evaluate(model, criterion, data_loader_valid, device=device)
        if model_ema:
            valid_score_ema = evaluate(
                model_ema, criterion, data_loader_valid, device=device, log_suffix="EMA"
            )
        sm.add_scalar("Score/train", train_score, epoch)
        sm.add_scalar("Score/valid", valid_score, epoch)
        sm.add_scalar("Score/valid_ema", valid_score_ema, epoch)
        if cfg.output_dir and valid_score > best_score:
            best_score = valid_score
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, f"best.pth"))

    if cfg.output_dir:
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        }
        if model_ema:
            checkpoint["model_ema"] = model_ema.state_dict()
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(cfg.output_dir, f"last.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
