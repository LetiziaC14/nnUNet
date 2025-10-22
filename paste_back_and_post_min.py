import os, json, argparse, re, warnings
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation

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


def keep_components_with_overlap(child_bin: np.ndarray,
                                 parent_bin: np.ndarray,
                                 min_overlap_vox: int = 1) -> np.ndarray:
    """
    Keep an entire connected component from child_bin if it overlaps parent_bin
    by at least `min_overlap_vox` voxels. Otherwise drop it.
    """
    if child_bin.sum() == 0 or parent_bin.sum() == 0:
        return np.zeros_like(child_bin, dtype=np.uint8)

    lab, n = cc_label(child_bin.astype(np.uint8))
    out = np.zeros_like(child_bin, dtype=np.uint8)
    parent_bool = parent_bin.astype(bool)
    for i in range(1, n + 1):
        comp = (lab == i)
        if (comp & parent_bool).sum() >= min_overlap_vox:
            out[comp] = 1
    return out

def keep_components_inside_dilated_kidney(child_bin: np.ndarray, 
                                        kidney_bin: np.ndarray,
                                        dilation_radius: int = 3) -> np.ndarray:
    """Keep only components that are mostly inside dilated kidney."""
    if child_bin.sum() == 0 or kidney_bin.sum() == 0:
        return np.zeros_like(child_bin, dtype=np.uint8)
    
    # Create expanded kidney region
    kidney_dilated = binary_dilation(kidney_bin, iterations=dilation_radius).astype(bool)
    
    lab, n = cc_label(child_bin.astype(np.uint8))
    out = np.zeros_like(child_bin, dtype=np.uint8)
    
    for i in range(1, n + 1):
        comp = (lab == i)
        # Require majority of component to be inside dilated kidney
        overlap_ratio = (comp & kidney_dilated).sum() / comp.sum()
        if overlap_ratio >= 0.1:  # 10% of component must be inside
            out[comp] = 1
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

    print("FULL unique labels (pre):", np.unique(full))
    print("counts pre:", {i: int((full==i).sum()) for i in [kidney_id, tumor_id, cyst_id]})

    # ---- Post-processing ----
    # ---- STRICT post-processing with overlap vs pre-filter kidney ----
    # Raw splits
    kidney_raw = (full == kidney_id).astype(np.uint8)
    tumor_raw  = (full == tumor_id ).astype(np.uint8)
    cyst_raw   = (full == cyst_id  ).astype(np.uint8)

    # 1) Fixed size filtering (exact thresholds you specified)
    kidney_bin = remove_small_components(kidney_raw, 20000)  # kidney >= 20,000
    tumor_bin  = remove_small_components(tumor_raw,    200)  # tumor  >= 200
    cyst_bin   = remove_small_components(cyst_raw,       50) # cyst   >= 50

    # 2) Overlap reference: pre-filter kidney (slightly dilated to be tolerant)
    kidney_ref = binary_dilation(kidney_raw, iterations=1).astype(np.uint8)
    #kidney_ref= kidney_bin

    # 3) Remove all tumor/cyst components that do NOT overlap the kidney_ref
    #tumor_bin = keep_components_with_overlap(tumor_bin, kidney_ref, min_overlap_vox=200)
    #cyst_bin  = keep_components_with_overlap(cyst_bin,  kidney_ref, min_overlap_vox=60)
    tumor_bin = keep_components_inside_dilated_kidney(tumor_bin, kidney_bin, dilation_radius=2)
    cyst_bin  = keep_components_inside_dilated_kidney(cyst_bin,  kidney_bin, dilation_radius=2)

    # 4) Reassemble with lesion-over-kidney priority
    final = np.zeros_like(full, dtype=np.uint8)
    final[kidney_bin > 0] = kidney_id
    final[tumor_bin  > 0] = tumor_id
    final[cyst_bin   > 0] = cyst_id

    # 5) Save with a label-safe header
    hdr = ref_full.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 1
    hdr["scl_inter"] = 0
    nib.save(nib.Nifti1Image(final, ref_full.affine, hdr), out_full_p)


def _extract_numeric_id(path: Path, width: int = 5):
    """
    Extract the FIRST numeric block from filename (stem+suffix),
    return zero-padded ID (default width=5). Example:
      'CASE_00489_pred.nii.gz' -> '00489'
      'img-12_0000.nii.gz'     -> '00012'
    Returns None if no digits found.
    """
    m = re.search(r'(\d{1,10})', path.name)
    if not m:
        return None
    return f"{int(m.group(1)):0{width}d}"

def _index_dir_by_id(files, expect_unique=True, width: int = 5):
    """
    Build a dict: id -> Path. If multiple files share the same id,
    keep the first and warn (unless expect_unique=False, in which case keep the first silently).
    """
    out = {}
    collisions = {}
    for p in files:
        case_id = _extract_numeric_id(p, width=width)
        if case_id is None:
            warnings.warn(f"[WARN] No numeric id in {p}")
            continue
        if case_id in out:
            collisions.setdefault(case_id, []).append(p)
            # keep the first occurrence; still warn
        else:
            out[case_id] = p
    for cid, dup in collisions.items():
        warnings.warn(f"[WARN] Multiple files for id {cid}: kept {out[cid].name}; ignored {[d.name for d in dup]}")
    return out

# ----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Paste back ROI predictions to full FOV and post-process (ID-matched across pred223/images221/meta).")
    ap.add_argument("--pred223",    required=True, help="Folder with ROI predictions (*.nii.gz)")
    ap.add_argument("--meta",       required=True, help="Folder with bbox JSONs (*.json)")
    ap.add_argument("--images221",  required=True, help="Folder with full-FOV images (e.g., *_0000.nii.gz)")
    ap.add_argument("--out",        required=True, help="Output folder for full-FOV segmentations")
    ap.add_argument("--kidney-id", type=int, default=1)
    ap.add_argument("--tumor-id",  type=int, default=2)
    ap.add_argument("--cyst-id",   type=int, default=3)
    ap.add_argument("--minvox-kidney", type=int, default=20000)
    ap.add_argument("--minvox-tumor",  type=int, default=200)
    ap.add_argument("--minvox-cyst",   type=int, default=50)
    ap.add_argument("--id-width", type=int, default=5, help="Zero-pad width for numeric id matching (default: 5)")
    args = ap.parse_args()

    pred_dir = Path(args.pred223)
    meta_dir = Path(args.meta)
    img_dir  = Path(args.images221)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate files
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    meta_files = sorted(meta_dir.glob("*.json"))

    # Images: prefer *_0000.nii.gz (221 convention), fall back to *.nii.gz if none match
    img_files = sorted(img_dir.glob("*_0000.nii.gz"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.nii.gz"))

    if not pred_files:
        raise RuntimeError(f"No predictions found in {pred_dir}")
    if not meta_files:
        raise RuntimeError(f"No meta JSONs found in {meta_dir}")
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")

    # Index by numeric ID (robust against naming differences)
    pred_map = _index_dir_by_id(pred_files, width=args.id_width)
    meta_map = _index_dir_by_id(meta_files, width=args.id_width)
    img_map  = _index_dir_by_id(img_files,  width=args.id_width)

    common_ids = sorted(set(pred_map) & set(meta_map) & set(img_map))
    missing_pred = sorted((set(meta_map) & set(img_map)) - set(pred_map))
    missing_meta = sorted((set(pred_map) & set(img_map)) - set(meta_map))
    missing_img  = sorted((set(pred_map) & set(meta_map)) - set(img_map))

    if not common_ids:
        raise RuntimeError("No overlapping cases across pred223/meta/images221 after ID-based matching")

    if missing_pred:
        warnings.warn(f"[WARN] {len(missing_pred)} ids missing in pred223: {missing_pred[:10]}{' ...' if len(missing_pred)>10 else ''}")
    if missing_meta:
        warnings.warn(f"[WARN] {len(missing_meta)} ids missing in meta: {missing_meta[:10]}{' ...' if len(missing_meta)>10 else ''}")
    if missing_img:
        warnings.warn(f"[WARN] {len(missing_img)} ids missing in images221: {missing_img[:10]}{' ...' if len(missing_img)>10 else ''}")

    print(f"[INFO] Pasting {len(common_ids)} cases")

    for cid in common_ids:
        pred_p = str(pred_map[cid])
        meta_p = str(meta_map[cid])
        img_p  = str(img_map[cid])
        out_p  = str(out_dir / f"{cid}.nii.gz")

        paste_back_single(
            roi_pred_p=pred_p,
            meta_json_p=meta_p,
            ref_full_img_p=img_p,
            out_full_p=out_p,
            kidney_id=args.kidney_id, tumor_id=args.tumor_id, cyst_id=args.cyst_id,
            minvox_kidney=args.minvox_kidney, minvox_tumor=args.minvox_tumor, minvox_cyst=args.minvox_cyst,
        )
        print(f"[OK] {cid}")

    print("[DONE]")

if __name__ == "__main__":
    main()