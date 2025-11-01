import nibabel as nib
import numpy as np

img = nib.load("C:/Users/letiz/Downloads/NEWCASE_0000.nii.gz")
arr = img.get_fdata()

print("shape:", arr.shape)
print("spacing from header (zooms):", img.header.get_zooms())
print("dtype:", arr.dtype, "min:", np.min(arr), "max:", np.max(arr), "mean:", np.mean(arr), "std:", np.std(arr))
print("=== NEWCASE geometry/intensity check ===")
print("File:", img)
print("Array shape (z,y,x or x,y,z depending on how it was saved):", arr.shape)
print("Header zooms (voxel spacing as stored in file):", img.header.get_zooms())
print("dtype:", arr.dtype)
print("min:", float(np.min(arr)))
print("max:", float(np.max(arr)))
print("mean:", float(np.mean(arr)))
print("std:", float(np.std(arr)))

print("\nReference expectations from Dataset221_KidneyCoarse plan:")
print("  typical pre-resample spacing ~ [3.0, 0.78125, 0.78125] mm (after nnU-Net transpose)")
print("  fullres target spacing      ~ [1.0, 0.78125, 0.78125] mm")
print("  HU-like intensity range     ~ about -1000 .. 300")