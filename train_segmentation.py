"""Train or evaluate LFDM for remote-sensing semantic segmentation."""

import argparse
import copy
import logging
import os

import numpy as np
from PIL import Image
import torch

import core.logger as Logger
import data as Data
import model as Model
from misc.torchutils import get_scheduler
from model.head_modules.mlp_seghead import FeatureFusionSegHead


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="config/segmentation_gid5.json",
        help="Experiment configuration file.",
    )
    parser.add_argument("-p", "--phase", choices=["train", "test"], default="train")
    parser.add_argument("-gpu", "--gpu_ids", default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_eval", action="store_true")
    return parser.parse_args()


def build_backbone(opt):
    if not opt["path"]["resume_state"]:
        raise ValueError("Set path.resume_state to a pretrained diffusion checkpoint prefix.")
    backbone_opt = copy.deepcopy(opt)
    # The frozen diffusion encoder only needs generator weights, not its optimizer state.
    backbone_opt["phase"] = "test"
    diffusion = Model.create_model(backbone_opt)
    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"][opt["phase"]], schedule_phase=opt["phase"]
    )
    diffusion.netG.eval()
    for parameter in diffusion.netG.parameters():
        parameter.requires_grad = False
    return diffusion


def build_head(opt, device):
    seg_opt = opt["model_seg"]
    diffusion_opt = opt["model"]["unet"]
    head = FeatureFusionSegHead(
        feat_scales=seg_opt["feat_scales"],
        num_classes=seg_opt["out_channels"],
        inner_channel=diffusion_opt["inner_channel"],
        channel_multiplier=diffusion_opt["channel_multiplier"],
        img_size=seg_opt["output_size"],
        time_steps=seg_opt["t"],
        embedding_dim=seg_opt.get("embedding_dim", 256),
    ).to(device)
    resume_state = seg_opt.get("resume_state")
    if resume_state:
        state = torch.load(resume_state, map_location=device)
        head.load_state_dict(state.get("model", state))
    return head


def extract_features(diffusion, batch, opt):
    diffusion.feed_data(batch)
    features = []
    for timestep in opt["model_seg"]["t"]:
        encoder_features, decoder_features = diffusion.get_feats(t=timestep)
        features.append(
            decoder_features if opt["model_seg"]["feat_type"] == "dec" else encoder_features
        )
    return features


def accumulate_confusion(confusion, logits, label, num_classes, ignore_index=255):
    prediction = logits.argmax(dim=1).detach().cpu()
    label = label.detach().cpu()
    valid = (label != ignore_index) & (label >= 0) & (label < num_classes)
    bins = num_classes * label[valid] + prediction[valid]
    confusion += torch.bincount(bins, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def summarize_confusion(confusion):
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid = union > 0
    iou = torch.zeros_like(intersection, dtype=torch.float64)
    iou[valid] = intersection[valid] / union[valid]
    total = confusion.sum()
    pixel_accuracy = intersection.sum().double() / total if total else torch.tensor(0.0)
    return {
        "mIoU": iou[valid].mean().item() if valid.any() else 0.0,
        "pixel_accuracy": pixel_accuracy.item(),
        "IoU": iou.tolist(),
    }


def save_predictions(logits, names, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    prediction = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
    for mask, name in zip(prediction, names):
        output_name = os.path.splitext(name)[0] + ".png"
        Image.fromarray(mask).save(os.path.join(result_dir, output_name))


def run_epoch(loader, diffusion, head, opt, criterion, optimizer=None, prediction_dir=None):
    is_training = optimizer is not None
    head.train(is_training)
    num_classes = opt["model_seg"]["out_channels"]
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0

    for batch in loader:
        features = extract_features(diffusion, batch, opt)
        logits = head(features)
        label = batch["label"].to(diffusion.device)
        loss = criterion(logits, label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        accumulate_confusion(confusion, logits, label, num_classes)
        if prediction_dir:
            save_predictions(logits, batch["name"], prediction_dir)

    metrics = summarize_confusion(confusion)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def save_checkpoint(head, optimizer, scheduler, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    from tensorboardX import SummaryWriter

    opt = Logger.dict_to_nonedict(Logger.parse(args))
    Logger.setup_logger(None, opt["path"]["log"], opt["phase"], screen=True)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))
    writer = SummaryWriter(log_dir=opt["path"]["tb_logger"])

    diffusion = build_backbone(opt)
    head = build_head(opt, diffusion.device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    if opt["phase"] == "test":
        test_set = Data.create_seg_dataset(opt["datasets"]["test"], "test", require_labels=True)
        test_loader = Data.create_dataloader(test_set, opt["datasets"]["test"], "test")
        metrics = run_epoch(
            test_loader,
            diffusion,
            head,
            opt,
            criterion,
            prediction_dir=os.path.join(opt["path"]["results"], "predictions"),
        )
        logger.info("Test metrics: %s", metrics)
        return

    train_set = Data.create_seg_dataset(opt["datasets"]["train"], "train", require_labels=True)
    val_set = Data.create_seg_dataset(opt["datasets"]["val"], "val", require_labels=True)
    train_loader = Data.create_dataloader(train_set, opt["datasets"]["train"], "train")
    val_loader = Data.create_dataloader(val_set, opt["datasets"]["val"], "val")
    optimizer = torch.optim.Adam(
        head.parameters(), lr=opt["train"]["optimizer"]["lr"]
    )
    scheduler = get_scheduler(optimizer, opt["train"])
    best_miou = -1.0

    for epoch in range(opt["train"]["n_epoch"]):
        train_metrics = run_epoch(train_loader, diffusion, head, opt, criterion, optimizer)
        writer.add_scalar("training/loss", train_metrics["loss"], epoch)
        writer.add_scalar("training/mIoU", train_metrics["mIoU"], epoch)
        logger.info("Epoch %d training metrics: %s", epoch, train_metrics)

        if (epoch + 1) % opt["train"]["val_freq"] == 0:
            with torch.no_grad():
                val_metrics = run_epoch(val_loader, diffusion, head, opt, criterion)
            writer.add_scalar("validation/loss", val_metrics["loss"], epoch)
            writer.add_scalar("validation/mIoU", val_metrics["mIoU"], epoch)
            logger.info("Epoch %d validation metrics: %s", epoch, val_metrics)
            checkpoint = os.path.join(opt["path"]["checkpoint"], "segmentation_latest.pth")
            save_checkpoint(head, optimizer, scheduler, epoch, checkpoint)
            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                save_checkpoint(
                    head,
                    optimizer,
                    scheduler,
                    epoch,
                    os.path.join(opt["path"]["checkpoint"], "segmentation_best.pth"),
                )
        scheduler.step()
    writer.close()


if __name__ == "__main__":
    main()
