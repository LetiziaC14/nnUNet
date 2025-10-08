import json, os
import nibabel as nib
import numpy as np
from pathlib import Path

def paste_back(roi_pred_p, meta_json_p, out_full_p, label_dtype=np.uint8):
    # roi_pred_p: prediction in ROI space (NIfTI)
    # meta_json_p: contains z,y,x start/stop and orig_shape
    roi = nib.load(roi_pred_p)
    P = roi.get_fdata()

    with open(meta_json_p, "r") as f:
        meta = json.load(f)
    z0,z1 = meta["z"]; y0,y1 = meta["y"]; x0,x1 = meta["x"]
    Z,Y,X = meta["orig_shape"]

    full = np.zeros((Z,Y,X), dtype=label_dtype)
    full[z0:z1, y0:y1, x0:x1] = P.astype(label_dtype)

    # Reuse the original affine? We don't have it here.
    # Typically you take the *full image* header as reference:
    #   ref_img = nib.load("/nnUNet_raw/Dataset221_KidneyCoarse/imagesTs/CASE_0000.nii.gz")
    # For this stand-alone, we keep roi affine (acceptable for voxel-consistent pipelines),
    # but best practice is to save & reuse the full image affine/header.
    out = nib.Nifti1Image(full, roi.affine, roi.header)
    Path(os.path.dirname(out_full_p)).mkdir(parents=True, exist_ok=True)
    nib.save(out, out_full_p)
