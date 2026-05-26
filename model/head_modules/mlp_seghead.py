"""Lightweight decoder for diffusion feature based semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_in_channels(scale, inner_channel, channel_multiplier):
    """Return the decoder channel count associated with a UNet feature level."""
    if scale < 3:
        multiplier = channel_multiplier[0]
    elif scale < 6:
        multiplier = channel_multiplier[1]
    elif scale < 9:
        multiplier = channel_multiplier[2]
    elif scale < 12:
        multiplier = channel_multiplier[3]
    elif scale < 15:
        multiplier = channel_multiplier[4]
    else:
        raise ValueError("Feature scale must be between 0 and 14.")
    return inner_channel * multiplier


class ProjectionBlock(nn.Sequential):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(
            nn.Conv2d(input_dim, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )


class FeatureFusionSegHead(nn.Module):
    """Fuse multi-scale features sampled from one or more diffusion time steps."""

    def __init__(
        self,
        feat_scales,
        num_classes,
        inner_channel,
        channel_multiplier,
        img_size=256,
        time_steps=None,
        embedding_dim=256,
    ):
        super().__init__()
        time_steps = time_steps or [0]
        self.feat_scales = sorted(feat_scales, reverse=True)
        self.output_size = (img_size, img_size)
        if not self.feat_scales:
            raise ValueError("At least one feature scale is required.")

        self.projections = nn.ModuleList()
        for scale in self.feat_scales:
            input_dim = get_in_channels(scale, inner_channel, channel_multiplier)
            input_dim *= len(time_steps)
            self.projections.append(ProjectionBlock(input_dim, embedding_dim))

        self.fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * len(self.feat_scales), embedding_dim, 1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embedding_dim, num_classes, 1),
        )

    def forward(self, features_by_timestep):
        target_size = features_by_timestep[0][self.feat_scales[-1]].shape[-2:]
        projected = []
        for scale, projection in zip(self.feat_scales, self.projections):
            timestep_features = [features[scale] for features in features_by_timestep]
            feature = projection(torch.cat(timestep_features, dim=1))
            projected.append(
                F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            )
        logits = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)


class pixel_classifierMLP(FeatureFusionSegHead):
    """Backward compatible name for checkpoints and external experiment scripts."""

    def __init__(
        self,
        feat_scales,
        numpy_class,
        inner_channel,
        channel_multiplier,
        img_size=256,
        time_steps=None,
        dim=256,
    ):
        super().__init__(
            feat_scales=feat_scales,
            num_classes=numpy_class,
            inner_channel=inner_channel,
            channel_multiplier=channel_multiplier,
            img_size=img_size,
            time_steps=time_steps,
            embedding_dim=dim,
        )
