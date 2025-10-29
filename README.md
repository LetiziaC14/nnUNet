# Kidney-segmentation
Kidney segmentation based  on the approach of "A Coarse-to-Fine Framework for the 2021 Kidney and Kidney Tumor Segmentation Challenge".

## Framework Pipeline
1) **Preprocessing**:
- Mainly handled by nnU-Net standard pipeline
- Resample to isotropic voxel spacing [0.786, 0.78125, 0.78125] using third order spline interpolation and then normalize the data.
- Data augmentation (rotation, scaling, mirroring, elastic deformations, gamma correction, etc.).


2) **Coarse segmentation**:
-	A standard nnU-Net is trained on full CT scans.
-	Output: coarse kidney mask.
-	Loss: Dice + cross-entropy (no surface loss at this stage).
-	Purpose: crop the CT to a kidney region of interest (ROI) 

3) **Fine kidney segmentation**:
-	Another nnU-Net trained only on cropped ROI. The coarse model ensures recall (does not miss kidney tissue, even at the cost of false positives). The fine model improves precision.
-	Cleaner kidney boundaries.
-	Produces refined kidney mask.ù

4) **Fine tumor & cyst segmentation**:
-	Two additional nnU-Nets: One for tumor, one for masses (tumor + cysts).
-	Both use the predicted kidney mask + ROI as input.
-	Final segmentation = combination of kidney + tumor + cyst outputs.
-	Loss function: Still mainly Dice + CE. But they also introduce Surface Loss here (assess the overlap between the predicted surface of segmentation and the real surface).

5) **Postprocessing**:
Remove tumors/cysts predicted outside kidneys and small isolated components:
-	<20k voxels for kidney
-	<200 voxels for tumor
-	<50 voxels for cyst

## How to train nnU-Net on KiTS23 dataset
1) **Download data kits23** following the intructions in the repo https://github.com/neheller/kits23.git
   
2) **Installation**:
   - create a virtual environment and install pytorch (if running on CPU use python 3.12)
   - Install requirements: cd nnunet
     pip install -e .
     
3) **Import dataset kits2023** in the current directory using the python converter file : nnuntev2 -> dataset_conversion -> Dataset220_KiTS2023.py
4) **Update parameters in nnUNet**: target spacing, postprocessing (mind nnUNet is now a submodule)
   
5) **Run derive_stage_datasets.py** to create the new datasets for the different steps
   
6) **Plan & preprocess** each stage for the different datasets :
- nnUNetv2_plan_and_preprocess -d 221 -c 3d_lowres   --verify_dataset_integrity
- nnUNetv2_plan_and_preprocess -d 222 -c 3d_fullres --verify_dataset_integrity
- nnUNetv2_plan_and_preprocess -d 223 -c 3d_fullres --verify_dataset_integrity
221,222,223 are the indexes of the different datasets
     
6) **Train** per stage:
-  nnUNetv2_train 221 3d_lowres 0 -device cuda   # or -device cpu
-  nnUNetv2_train 222 3d_fullres 0 -device cuda
-  nnUNetv2_train 223 3d_fullres 0 -device cuda
   
## How to make inference on KiTS23 test dataset**:
- Coarse prediction on full FOV
  nnUNetv2_predict -d 221 -c 3d_lowres \
 -i <path_to_fullFOV_images> \
 -o <output_folder_for_kidney_masks>
- Build the ROI from the coarse mask & crop the image (using crop_from_prediction.py)
- Run fine models on the ROI
- Paste ROI predictions back to the full image space and Post-processing (using paste_back_and_post_min.py)












