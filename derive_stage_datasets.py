# make_stage_datasets_from_regions.py
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import numpy as np
from shutil import copyfile
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

SRC_ID = 220
SRC_NAME = f"Dataset{SRC_ID:03d}_KiTS2023"   # your existing dataset
SRC_BASE = join(nnUNet_raw, SRC_NAME)


def load_nii(p):
    return nib.load(p)


def save_like(ref_img, data, out_path, dtype=np.uint8):
    """
    Save 'data' as NIfTI, reusing affine & header from 'ref_img'.
    """
    nii = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(nii, out_path)


def bbox_from_mask(mask):
    axes = tuple(range(mask.ndim))
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    mins = [coords[d].min() for d in axes]
    maxs = [coords[d].max() + 1 for d in axes]
    return tuple(slice(mins[d], maxs[d]) for d in axes)


def ensure_dirs(base):
    maybe_mkdir_p(join(base, "imagesTr"))
    maybe_mkdir_p(join(base, "labelsTr"))
    maybe_mkdir_p(join(base, "imagesTs"))  # needed for the 100 held-out test cases


def _split_cases():
    
    cases = subfiles(join(SRC_BASE, "labelsTr"), suffix=".nii.gz", join=False)
    cases.sort()
    if len(cases) < 489:
        raise RuntimeError(
            f"Expected at least 489 label files in {join(SRC_BASE, 'labelsTr')}, got {len(cases)}."
        )
    train_cases = cases[:389]
    test_cases = cases[389:489]
    return train_cases, test_cases


def derive_221_kidney_coarse():
    dst_id = 221
    name = "KidneyCoarse"
    dst_base = join(nnUNet_raw, f"Dataset{dst_id:03d}_{name}")
    ensure_dirs(dst_base)

    train_cases, test_cases = _split_cases()

    # --- training set: copy image and write binary kidney label ---
    for c in train_cases:
        case = c[:-7]  # strip ".nii.gz"
        img = load_nii(join(SRC_BASE, "imagesTr", case + "_0000.nii.gz"))
        lab = load_nii(join(SRC_BASE, "labelsTr", case + ".nii.gz"))
        L = lab.get_fdata()

        # regions style in SRC: kidney = union(1,2,3)
        kidney_mask = np.isin(L, [1, 2, 3]).astype(np.uint8)

        copyfile(join(SRC_BASE, "imagesTr", case + "_0000.nii.gz"),
                 join(dst_base, "imagesTr", case + "_0000.nii.gz"))
        save_like(lab, kidney_mask, join(dst_base, "labelsTr", case + ".nii.gz"))

    # --- test set: images only to imagesTs ---
    for c in test_cases:
        case = c[:-7]
        copyfile(join(SRC_BASE, "imagesTr", case + "_0000.nii.gz"),
                 join(dst_base, "imagesTs", case + "_0000.nii.gz"))

    generate_dataset_json(
        dst_base, {0: "CT"},
        labels={"background": 0, "kidney": 1},
        file_ending=".nii.gz",
        num_training_cases=len(train_cases),  # 389
        dataset_name=name, reference="derived from KiTS2023",
        release="1.0"
    )


def derive_222_kidney_fine_and_223_masses():
    # build both fine datasets in one pass to reuse the ROI crop
    dst_id_fine = 222
    name_fine = "KidneyFine"
    dst_id_mass = 223
    name_mass = "MassesTumor"
    base_fine = join(nnUNet_raw, f"Dataset{dst_id_fine:03d}_{name_fine}")
    base_mass = join(nnUNet_raw, f"Dataset{dst_id_mass:03d}_{name_mass}")
    ensure_dirs(base_fine)
    ensure_dirs(base_mass)

    train_cases, test_cases = _split_cases()

    # -----------------------
    # Training split (389)
    # -----------------------
    for c in train_cases:
        case = c[:-7]
        img = load_nii(join(SRC_BASE, "imagesTr", case + "_0000.nii.gz"))
        lab = load_nii(join(SRC_BASE, "labelsTr", case + ".nii.gz"))

        # keep image dtype consistent
        src_dtype = img.get_data_dtype()
        I = img.get_fdata(dtype=src_dtype)
        L = lab.get_fdata().astype(np.int16)

        kidney_mask = np.isin(L, [1, 2, 3]).astype(np.uint8)
        sl = bbox_from_mask(kidney_mask)
        if sl is None:
            sl = (slice(0, I.shape[0]), slice(0, I.shape[1]), slice(0, I.shape[2]))

        # crop image & labels to ROI
        I_roi = I[sl]
        L_roi = L[sl]
        kidney_roi = kidney_mask[sl]

        # --- Dataset222: KidneyFine (binary) ---
        save_like(img, I_roi, join(base_fine, "imagesTr", case + "_0000.nii.gz"), dtype=src_dtype)
        save_like(lab, kidney_roi, join(base_fine, "labelsTr", case + ".nii.gz"), dtype=np.uint8)

        # --- Dataset223: Masses/Tumor (REGIONS style) ---
        # masses = union(2,3); tumor = 2; kidney optional context
        save_like(img, I_roi, join(base_mass, "imagesTr", case + "_0000.nii.gz"), dtype=src_dtype)
        save_like(lab, L_roi, join(base_mass, "labelsTr", case + ".nii.gz"), dtype=np.int16)

    # -----------------------
    # Test split (100) → imagesTs only
    # -----------------------
    for c in test_cases:
        case = c[:-7]
        img = load_nii(join(SRC_BASE, "imagesTr", case + "_0000.nii.gz"))
        lab = load_nii(join(SRC_BASE, "labelsTr", case + ".nii.gz"))

        src_dtype = img.get_data_dtype()
        I = img.get_fdata(dtype=src_dtype)
        L = lab.get_fdata().astype(np.int16)

        kidney_mask = np.isin(L, [1, 2, 3]).astype(np.uint8)
        sl = bbox_from_mask(kidney_mask)
        if sl is None:
            sl = (slice(0, I.shape[0]), slice(0, I.shape[1]), slice(0, I.shape[2]))

        I_roi = I[sl]

        # --- Dataset222: KidneyFine (binary) ---
        save_like(img, I_roi, join(base_fine, "imagesTs", case + "_0000.nii.gz"), dtype=src_dtype)

        # --- Dataset223: Masses/Tumor (REGIONS style) ---
        save_like(img, I_roi, join(base_mass, "imagesTs", case + "_0000.nii.gz"), dtype=src_dtype)

    # KidneyFine (binary)
    generate_dataset_json(
        base_fine, {0: "CT"},
        labels={"background": 0, "kidney": 1},
        file_ending=".nii.gz",
        num_training_cases=len(train_cases),  # 389
        dataset_name=name_fine, reference="derived from KiTS2023",
        release="1.0"
    )

    # MassesTumor: keep your REGIONS style for the fine stage
    generate_dataset_json(
        base_mass, {0: "CT"},
        labels={
            "background": 0,
            "kidney": (1, 2, 3),
            "masses": (2, 3),
            "tumor": 2
        },
        regions_class_order=("kidney", "masses", "tumor"),
        file_ending=".nii.gz",
        num_training_cases=len(train_cases),  # 389
        dataset_name=name_mass, reference="derived from KiTS2023",
        release="1.0"
    )


if __name__ == "__main__":
    derive_221_kidney_coarse()
    derive_222_kidney_fine_and_223_masses()
