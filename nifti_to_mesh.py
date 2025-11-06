import os
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
import config

def _face_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else np.array([0.0, 0.0, 0.0], dtype=float)

def write_stl_ascii(path, verts, faces, solid_name="mesh"):
    with open(path, "w") as f:
        f.write(f"solid {solid_name}\n")
        for face in faces:
            v0, v1, v2 = (verts[face[0]], verts[face[1]], verts[face[2]])
            n = _face_normal(v0, v1, v2)
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")

def labels_to_export(data):
    if hasattr(config, "LABELS_INCLUDE") and config.LABELS_INCLUDE:
        return [(int(l), name) for l, name in config.LABELS_INCLUDE.items()]
    labs = sorted(l for l in np.unique(data.astype(np.int32)) if l > 0)
    return [(int(l), f"label_{int(l)}")]

def main():
    # You should define NIFTI_INPUT_DIR in config.py
    nii_dir = getattr(config, "NIFTI_INPUT_DIR", None)
    if not nii_dir or not os.path.isdir(nii_dir):
        print("Set NIFTI_INPUT_DIR in config.py to your .nii/.nii.gz folder")
        return

    os.makedirs(config.INPUT_MESH_DIR, exist_ok=True)
    nii_files = [p for p in os.listdir(nii_dir) if p.endswith(".nii") or p.endswith(".nii.gz")]
    if not nii_files:
        print(f"No NIfTI files in {nii_dir}")
        return

    for fname in nii_files:
        path = os.path.join(nii_dir, fname)
        img = nib.as_closest_canonical(nib.load(path))
        data = img.get_fdata()
        vx = nib.affines.voxel_sizes(img.affine)  # spacing (likely mm)
        base = os.path.splitext(os.path.splitext(fname)[0])[0]

        data_i = data.astype(np.int32)
        for lab, name in labels_to_export(data_i):
            mask = (data_i == lab)
            if mask.sum() < 100:
                continue
            try:
                verts, faces, _, _ = marching_cubes(mask, level=0.5, spacing=vx)
            except ValueError:
                continue
            out_path = os.path.join(config.INPUT_MESH_DIR, f"{base}_{name}.stl")
            write_stl_ascii(out_path, verts, faces, solid_name=f"{base}_{name}")
            print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()