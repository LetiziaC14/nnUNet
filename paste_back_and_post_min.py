import os, json, argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

def remove_small_components(bin_mask: np.ndarray, min_size: int) -> np.ndarray:
    if bin_mask.sum() == 0:
        return bin_mask.astype(np.uint8)
    lab, n = cc_label(bin_mask.astype(np.uint8))
    out = np.zeros_like(bin_mask, dtype=np.uint8)
    for i in range(1, n + 1):
        comp = (lab == i)
        if int(comp.sum()) >= min_size:
            out[comp] = 1
    return out

def enforce_inside(child: np.ndarray, parent: np.ndarray) -> np.ndarray:
    out = np.zeros_like(child, dtype=np.uint8)
    out[(child > 0) & (parent > 0)] = 1
    return out

def paste_back_single(roi_pred_p, meta_json_p, ref_full_img_p, out_full_p,
                      kidney_id=1, tumor_id=2, cyst_id=3,
                      minvox_kidney=20000, minvox_tumor=200, minvox_cyst=50):
    # Load ROI prediction (labelmap)
    roi_nii = nib.load(roi_pred_p)
    P = np.asanyarray(roi_nii.dataobj)

    # Load bbox meta
    with open(meta_json_p, "r") as f:
        meta = json.load(f)
    # support either flat keys or nested bbox dict
    if "z" in meta:
        z0, z1 = meta["z"]; y0, y1 = meta["y"]; x0, x1 = meta["x"]
        Z, Y, X = meta["orig_shape"] if "orig_shape" in meta else meta["orig_shape_zyx"]
    else:
        z0, z1 = meta["bbox"]["z"]; y0, y1 = meta["bbox"]["y"]; x0, x1 = meta["bbox"]["x"]
        Z, Y, X = meta["orig_shape"]

    # Reference full-FOV header/affine & shape
    ref_full = nib.load(ref_full_img_p)
    full_shape = tuple(ref_full.shape[:3])  # (Z,Y,X)

    # Safety: if meta shape disagrees with reference, trust the reference shape but clamp slice
    Zr, Yr, Xr = full_shape
    z0 = max(0, min(z0, Zr)); z1 = max(0, min(z1, Zr))
    y0 = max(0, min(y0, Yr)); y1 = max(0, min(y1, Yr))
    x0 = max(0, min(x0, Xr)); x1 = max(0, min(x1, Xr))

    # Paste ROI into full volume
    full = np.zeros(full_shape, dtype=np.uint8)
    zlen, ylen, xlen = z1 - z0, y1 - y0, x1 - x0
    # Clip ROI if bbox extends beyond ROI array (robustness)
    full[z0:z1, y0:y1, x0:x1] = P[:zlen, :ylen, :xlen].astype(np.uint8)

    # ---- Post-processing ----
    kidney_bin = (full == kidney_id).astype(np.uint8)
    tumor_bin  = (full == tumor_id ).astype(np.uint8)
    cyst_bin   = (full == cyst_id  ).astype(np.uint8)

    kidney_bin = remove_small_components(kidney_bin, minvox_kidney)
    tumor_bin  = remove_small_components(tumor_bin,  minvox_tumor)
    cyst_bin   = remove_small_components(cyst_bin,   minvox_cyst)

    # Enforce anatomy: lesions inside kidney
    tumor_bin = enforce_inside(tumor_bin, kidney_bin)
    cyst_bin  = enforce_inside(cyst_bin,  kidney_bin)

    # Reassemble labelmap: 0 bg, IDs as given
    final = np.zeros_like(full, dtype=np.uint8)
    final[kidney_bin > 0] = kidney_id
    final[tumor_bin  > 0] = tumor_id
    final[cyst_bin   > 0] = cyst_id

    # Save with the full image affine/header
    Path(os.path.dirname(out_full_p)).mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(final, ref_full.affine, ref_full.header), out_full_p)

def main():
    ap = argparse.ArgumentParser(description="Paste back ROI predictions to full FOV and post-process.")
    ap.add_argument("--pred223", required=True, help="Folder with ROI predictions (e.g., pred_223/*.nii.gz)")
    ap.add_argument("--meta",    required=True, help="Folder with bbox JSONs (roi_meta/*.json)")
    ap.add_argument("--images221", required=True, help="Folder with full-FOV images used in 221 (CASE_0000.nii.gz)")
    ap.add_argument("--out",     required=True, help="Output folder for full-FOV segmentations")
    ap.add_argument("--kidney-id", type=int, default=1)
    ap.add_argument("--tumor-id",  type=int, default=2)
    ap.add_argument("--cyst-id",   type=int, default=3)
    ap.add_argument("--minvox-kidney", type=int, default=20000)
    ap.add_argument("--minvox-tumor",  type=int, default=200)
    ap.add_argument("--minvox-cyst",   type=int, default=50)
    args = ap.parse_args()

    pred_dir  = Path(args.pred223)
    meta_dir  = Path(args.meta)
    img_dir   = Path(args.images221)
    out_dir   = Path(args.out)

    # Match by CASE id: pred=CASE.nii.gz, meta=CASE.json, image=CASE_0000.nii.gz
    pred_map = {p.stem: p for p in pred_dir.glob("*.nii.gz")}
    meta_map = {p.stem: p for p in meta_dir.glob("*.json")}
    img_map  = {p.name.replace("_0000.nii.gz", ""): p for p in img_dir.glob("*_0000.nii.gz")}

    cases = sorted(set(pred_map) & set(meta_map) & set(img_map))
    if not cases:
        raise RuntimeError("No overlapping cases across pred223/meta/images221")

    print(f"[INFO] Pasting {len(cases)} cases")
    for case in cases:
        paste_back_single(
            roi_pred_p=str(pred_map[case]),
            meta_json_p=str(meta_map[case]),
            ref_full_img_p=str(img_map[case]),
            out_full_p=str(out_dir / f"{case}.nii.gz"),
            kidney_id=args.kidney_id, tumor_id=args.tumor_id, cyst_id=args.cyst_id,
            minvox_kidney=args.minvox_kidney, minvox_tumor=args.minvox_tumor, minvox_cyst=args.minvox_cyst,
        )
        print(f"[OK] {case}")
    print("[DONE]")

if __name__ == "__main__":
    main()
