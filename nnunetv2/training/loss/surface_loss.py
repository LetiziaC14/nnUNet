import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.ndimage as ndi


def compute_edts_for_penalized_loss(gt_np):
    """
    gt_np: shape (C, Z, Y, X), binary mask per class.
    Return: signed distance maps (C, Z, Y, X), float32.
      - outside surface: positive distances
      - inside surface: negative distances
      - 0 near boundary
    """
    C, Z, Y, X = gt_np.shape
    res = np.zeros_like(gt_np, dtype=np.float32)

    for c in range(C):
        gtc = gt_np[c] > 0.5
        if not np.any(gtc):
            # class not present -> leave zeros
            continue
        # distance outside
        dist_out = ndi.distance_transform_edt(~gtc)
        # distance inside
        dist_in = ndi.distance_transform_edt(gtc)
        signed = dist_out.astype(np.float32)
        signed[gtc] = -dist_in[gtc].astype(np.float32)
        res[c] = signed

    return res


class SurfaceLoss(nn.Module):
    """
    Surface/boundary-aware penalty.

    net_output: logits, shape (B, C, Z, Y, X)
    target: int labels, shape (B, Z, Y, X)

    We:
    - softmax logits → probs
    - one-hot target → per-class masks
    - build signed distance map per class
    - penalize probability far from the GT surface

    If ignore_background=True we skip channel 0 in the sum.
    """

    def __init__(self, ignore_background=False):
        super().__init__()
        self.ignore_background = ignore_background

    def forward(self, net_output, target):
        # convert logits to probabilities
        if net_output.shape[1] == 1:
            probs = torch.sigmoid(net_output)
        else:
            probs = F.softmax(net_output, dim=1)

        B, C, *spatial = probs.shape

        # one-hot GT
        with torch.no_grad():
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, target[:, None].long(), 1.0)

        losses_b = []
        for b in range(B):
            # distance transform on CPU/NumPy
            oh_np = one_hot[b].detach().cpu().numpy()  # (C, Z, Y, X)
            distmap = compute_edts_for_penalized_loss(oh_np)  # (C, Z, Y, X)
            dist_t = torch.from_numpy(distmap).to(probs.device)

            if self.ignore_background and C > 1:
                cls_slice = slice(1, C)
            else:
                cls_slice = slice(0, C)

            # |dist| * prob
            term = probs[b, cls_slice] * dist_t[cls_slice].abs()
            losses_b.append(term.mean())

        return torch.stack(losses_b).mean()
