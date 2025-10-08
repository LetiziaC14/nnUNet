import json, os
import nibabel as nib
import numpy as np
from pathlib import Path

def bbox_from_mask(mask):
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max())+1 for c in coords]
    return tuple(slice(mins[d], maxs[d]) for d in range(mask.ndim))

def dilate_bbox(sl, shape, pad_vox=(12,12,12)):
    out = []
    for s, n, p in zip(sl, shape, pad_vox):
        start = max(0, s.start - p)
        stop  = min(n, s.stop + p)
        out.append(slice(start, stop))
    return tuple(out)

def crop_case(full_img_p, coarse_mask_p, out_img_p, meta_json_p, pad_vox=(12,12,12)):
    img = nib.load(full_img_p)            # full FOV CT
    msk = nib.load(coarse_mask_p)         # coarse 0/1 mask
    I = img.get_fdata()
    M = (msk.get_fdata() > 0.5).astype(np.uint8)

    sl = bbox_from_mask(M)
    if sl is None:
        # fallback: keep whole image
        sl = (slice(0,I.shape[0]), slice(0,I.shape[1]), slice(0,I.shape[2]))
    else:
        sl = dilate_bbox(sl, I.shape, pad_vox)

    I_roi = I[sl]
    # IMPORTANT: we keep the original affine/header so nnU-Net sees a normal NIfTI.
    # We'll paste predictions back using voxel indices from meta.json.
    roi_img = nib.Nifti1Image(I_roi.astype(I.dtype), img.affine, img.header)
    Path(os.path.dirname(out_img_p)).mkdir(parents=True, exist_ok=True)
    nib.save(roi_img, out_img_p)

    meta = {
        "z": [sl[0].start, sl[0].stop],
        "y": [sl[1].start, sl[1].stop],
        "x": [sl[2].start, sl[2].stop],
        "orig_shape": list(I.shape)
    }
    Path(os.path.dirname(meta_json_p)).mkdir(parents=True, exist_ok=True)
    with open(meta_json_p, "w") as f:
        json.dump(meta, f)

# --- example batch driver ---
# Inputs:
#   full images: /nnUNet_raw/Dataset221_KidneyCoarse/imagesTs/CASE_0000.nii.gz
#   coarse segs: /outputs/221_coarse_pred/CASE.nii.gz
# Outputs (ROIs):
#   /work/roi_222/imagesTs/CASE_0000.nii.gz
#   /work/roi_223/imagesTs/CASE_0000.nii.gz
#   /work/roi_meta/CASE.json
