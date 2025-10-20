import json, shutil, sys, io

p = "/home/letizia14/Documents/nnUNet_data/nnUNet_preprocessed/Dataset221_KidneyCoarse/nnUNetPlans.json"

# backup
shutil.copy2(p, p + ".bak")

with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)  # fallirà se il file NON è JSON valido

cfg = data.get("configurations", {})
lr = cfg.get("3d_lowres", {})
lr.pop("next_stage", None)  # rimuove se presente

with open(p, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print("Removed configurations.3d_lowres.next_stage (if it existed).")
