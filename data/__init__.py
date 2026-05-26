"""Dataset and dataloader factories for diffusion pre-training and segmentation."""

import logging

import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    """Create a dataloader using the options associated with a split."""
    if phase not in {"train", "val", "test"}:
        raise NotImplementedError("Dataloader [{}] is not supported.".format(phase))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_opt.get("batch_size", 1),
        shuffle=dataset_opt.get("use_shuffle", phase == "train"),
        num_workers=dataset_opt.get("num_workers", 0),
        pin_memory=True,
    )


def create_image_dataset(dataset_opt, phase):
    """Create the unlabeled image dataset used to train the diffusion backbone."""
    from data.ImageDataset import ImageDataset

    dataset = ImageDataset(
        dataroot=dataset_opt["dataroot"],
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt["data_len"],
    )
    logging.getLogger("base").info(
        "Dataset [%s - %s] is created.", dataset.__class__.__name__, dataset_opt["name"]
    )
    return dataset


def create_seg_dataset(dataset_opt, phase, require_labels=True):
    """Create a semantic segmentation dataset from paired image/mask folders."""
    from data.SegmentationDataset import SegmentationDataset

    dataset = SegmentationDataset(
        image_root=dataset_opt["dataroot"],
        label_root=dataset_opt.get("labelroot"),
        resolution=dataset_opt["resolution"],
        split=phase,
        data_len=dataset_opt["data_len"],
        label_offset=dataset_opt.get("label_offset", 0),
        require_labels=require_labels,
    )
    logging.getLogger("base").info(
        "Dataset [%s - %s - %s] is created.",
        dataset.__class__.__name__,
        dataset_opt["name"],
        phase,
    )
    return dataset
