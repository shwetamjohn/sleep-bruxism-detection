"""
Utility to inject in-lab sensor CWT images (Trials 1 & 3) directly into the
validation dataset folder (data/dataset/val/) alongside the CAP Sleep validation set.
"""

import os
import shutil

def add_lab_cwt_to_validation(cwt_base="outputs/lab_test/cwt_images", dataset_dir="data/dataset"):
    val_dir = os.path.join(dataset_dir, "val")
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}. Please ensure dataset exists.")
        return

    val_brux = os.path.join(val_dir, "bruxism")
    val_norm = os.path.join(val_dir, "normal")
    os.makedirs(val_brux, exist_ok=True)
    os.makedirs(val_norm, exist_ok=True)

    n_orig_b = len([f for f in os.listdir(val_brux) if not f.startswith("trial")])
    n_orig_n = len([f for f in os.listdir(val_norm) if not f.startswith("trial")])

    print("=======================================================")
    print("      MERGING IN-LAB CWT IMAGES INTO VALIDATION SET    ")
    print("=======================================================")
    print(f"Original CAP Sleep validation set: {n_orig_b} bruxism, {n_orig_n} normal")

    added_b = 0
    added_n = 0

    for trial in ["trial_1", "trial_3"]:
        trial_path = os.path.join(cwt_base, trial)
        if not os.path.exists(trial_path):
            print(f"Warning: {trial_path} not found.")
            continue

        for cls in ["bruxism", "normal"]:
            src_dir = os.path.join(trial_path, cls)
            dest_dir = os.path.join(val_dir, cls)
            for fname in os.listdir(src_dir):
                if fname.endswith(".png"):
                    shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))
                    if cls == "bruxism":
                        added_b += 1
                    else:
                        added_n += 1

    tot_b = len(os.listdir(val_brux))
    tot_n = len(os.listdir(val_norm))

    print(f"Added from In-Lab Sensor Trials:  {added_b} bruxism, {added_n} normal")
    print("-------------------------------------------------------")
    print(f"Updated Total Validation Set:      {tot_b} bruxism, {tot_n} normal ({tot_b + tot_n} total)")
    print("=======================================================\n")
    print("Now you can run:")
    print("  python3 evaluate_demo.py --checkpoint checkpoints/best_model.pth --model_name lsknet_tiny")

if __name__ == "__main__":
    add_lab_cwt_to_validation()
