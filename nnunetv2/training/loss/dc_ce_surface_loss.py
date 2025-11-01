import torch
import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.surface_loss import SurfaceLoss


class DC_CE_Surface_Combined(nn.Module):
    """
    total = w_dcce * (Dice+CE) + w_surf * SurfaceLoss
    We assume standard nnU-Net target format:
      net_output: logits (B, C, Z, Y, X)
      target:     int labels (B, Z, Y, X)
    """

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        aggregate_function=torch.mean,
        w_dcce: float = 1.0,
        w_surf: float = 0.5,
        ignore_background_for_surface: bool = False
    ):
        super().__init__()
        self.w_dcce = w_dcce
        self.w_surf = w_surf
        self.dcce = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate_function=aggregate_function
        )
        self.surf = SurfaceLoss(ignore_background=ignore_background_for_surface)

    def forward(self, net_output, target):
        loss_dcce = self.dcce(net_output, target)
        loss_surf = self.surf(net_output, target)
        return self.w_dcce * loss_dcce + self.w_surf * loss_surf
