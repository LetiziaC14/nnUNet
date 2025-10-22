#!/bin/bash
set -e  

#echo "Starting preprocessing dataset221"
#nUNetv2_plan_and_preprocess -d 221 -c 3d_fullres -pl nnUNetPlans -np 1

#echo "Starting training dataset221"
#nnUNetv2_train 221 3d_lowres 0 -device cuda

#echo "Starting training dataset222"
#nnUNetv2_train 222 3d_fullres 0 -device cuda

#echo "Starting training dataset223"
#nnUNetv2_train 223 3d_fullres 0 -device cuda

#echo "Rebuild the preprocessed cache for dataset223"
#rm -rf "$nnUNet_preprocessed/Dataset223_MassesTumor"
#nnUNetv2_plan_and_preprocess -d 223 -c 3d_fullres --verify_dataset_integrity --clean

#echo "Starting validation dataset223"
#nnUNetv2_train 223 3d_fullres 0 --val

#echo "Starting prediction dataset221"
#nnUNetv2_predict -d 221 -c 3d_lowres -f 0 \
#  -i "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset221_KidneyCoarse/imagesTs" \
#  -o "$HOME/Documents/nnUNet_data/nnUNet_inference/pred_221" \
#  --save_probabilities

#echo "Starting cropping_from_prediction"
#python3 crop_from_prediction.py \
#  --images221 "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset221_KidneyCoarse/imagesTs" \
#  --pred221   "$HOME/Documents/nnUNet_data/nnUNet_inference/pred_221" \
#  --out222    "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset222_KidneyFine/imagesTs" \
#  --out223    "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset223_MassesTumor/imagesTs" \
#  --outmeta   "$HOME/Documents/nnUNet_data/nnUNet_inference/roi_meta" \
#  --pad-vox   12 12 12

#echo "Starting prediction dataset222"
#nnUNetv2_predict \
#  -d 222 -c 3d_fullres \
#  -f 0 \
#  -i "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset222_KidneyFine/imagesTs" \
#  -o "$HOME/Documents/nnUNet_data/nnUNet_inference/pred_222" \
#  --save_probabilities

#echo "Starting prediction dataset223"
#nnUNetv2_predict \
#  -d 223 -c 3d_fullres \
#  -f 0 \
#  -i "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset223_MassesTumor/imagesTs" \
#  -o "$HOME/Documents/nnUNet_data/nnUNet_inference/pred_223" \
#  --save_probabilities

echo "Starting paste_back"
python3 paste_back_and_post_min.py \
  --pred223   "$HOME/Documents/nnUNet_data/nnUNet_inference/pred_223" \
  --meta      "$HOME/Documents/nnUNet_data/nnUNet_inference/roi_meta" \
  --images221 "$HOME/Documents/nnUNet_data/nnUNet_raw/Dataset221_KidneyCoarse/imagesTs" \
  --out       "$HOME/Documents/nnUNet_data/nnUNet_inference/final_fullsize" \
  --minvox-kidney 20000 --minvox-tumor 200 --minvox-cyst 50

echo "All jobs finished at $(date)"

#chmod +x run_all.sh
#nohup bash ./run_all.sh > run_all.log 2>&1 &
