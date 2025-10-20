# crop_from_prediction.py
import os, json, argparse
from pathlib import Path
import nibabel as nib
import numpy as np

def bbox_from_mask(mask):
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    return (slice(mins[0], maxs[0]), slice(mins[1], maxs[1]), slice(mins[2], maxs[2]))

def dilate_bbox(sl, shape, pad_vox=(12,12,12)):
    out = []
    for s, n, p in zip(sl, shape, pad_vox):
        start = max(0, s.start - p)
        stop  = min(n, s.stop + p)
        out.append(slice(start, stop))
    return tuple(out)

def crop_case(full_img_p, coarse_mask_p, out_img_222, out_img_223, meta_json_p, pad_vox=(12,12,12)):
    img = nib.load(full_img_p)            # full FOV CT (CASEID_0000.nii.gz)
    msk = nib.load(coarse_mask_p)         # coarse 0/1 mask (CASEID.nii.gz)
    I = np.asanyarray(img.dataobj)
    M = (np.asanyarray(msk.dataobj) > 0.5).astype(np.uint8)

    if I.shape != M.shape:
        raise ValueError(f"Shape mismatch: image {I.shape} vs mask {M.shape} for {full_img_p}")

    sl = bbox_from_mask(M)
    if sl is None:
        # fallback: keep whole image
        sl = (slice(0,I.shape[0]), slice(0,I.shape[1]), slice(0,I.shape[2]))
    else:
        sl = dilate_bbox(sl, I.shape, pad_vox)

    I_roi = I[sl]
    roi_img = nib.Nifti1Image(I_roi.astype(I.dtype), img.affine, img.header)

    # write for both 222 and 223
    Path(out_img_222).parent.mkdir(parents=True, exist_ok=True)
    Path(out_img_223).parent.mkdir(parents=True, exist_ok=True)
    nib.save(roi_img, out_img_222)
    nib.save(roi_img, out_img_223)

    # write bbox meta
    meta = {
        "case": Path(full_img_p).name.replace("_0000.nii.gz", ""),
        "z": [sl[0].start, sl[0].stop],
        "y": [sl[1].start, sl[1].stop],
        "x": [sl[2].start, sl[2].stop],
        "orig_shape": list(I.shape)
    }
    Path(meta_json_p).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_json_p, "w") as f:
        json.dump(meta, f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Crop ROIs for Datasets 222/223 from 221 coarse predictions.")
    ap.add_argument("--images221", required=True, help="Folder with full-FOV test images (Dataset221/imagesTs)")
    ap.add_argument("--pred221",   required=True, help="Folder with coarse masks (e.g., pred_221)")
    ap.add_argument("--out222",    required=True, help="Folder for Dataset222/imagesTs (cropped CTs)")
    ap.add_argument("--out223",    required=True, help="Folder for Dataset223/imagesTs (cropped CTs)")
    ap.add_argument("--outmeta",   required=True, help="Folder to save bbox JSONs")
    ap.add_argument("--pad-vox",   nargs=3, type=int, default=(12,12,12), help="Padding in voxels: z y x")
    args = ap.parse_args()

    images_dir = Path(args.images221)
    pred_dir   = Path(args.pred221)
    out222_dir = Path(args.out222)
    out223_dir = Path(args.out223)
    meta_dir   = Path(args.outmeta)

    imgs = sorted(images_dir.glob("*_0000.nii.gz"))
    if not imgs:
        raise FileNotFoundError(f"No *_0000.nii.gz found in {images_dir}")

    ok, skip = 0, 0
    for p in imgs:
        case = p.name.replace("_0000.nii.gz", "")
        mask_p = pred_dir / f"{case}.nii.gz"     # expected mask name
        if not mask_p.exists():
            print(f"[WARN] Missing mask for {case}: {mask_p}")
            skip += 1
            continue

        out_img_222 = out222_dir / f"{case}_0000.nii.gz"
        out_img_223 = out223_dir / f"{case}_0000.nii.gz"
        meta_json_p = meta_dir   / f"{case}.json"

        try:
            crop_case(str(p), str(mask_p), str(out_img_222), str(out_img_223),
                      str(meta_json_p), pad_vox=tuple(args.pad_vox))
            ok += 1
        except Exception as e:
            print(f"[ERROR] {case}: {e}")
            skip += 1

    print(f"[DONE] Cropped {ok} cases, skipped {skip}.")

if __name__ == "__main__":
    main()
