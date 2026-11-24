"""
Data Preparation Pipeline for ALL-IDB Leukemia Classification
SmartCell - Phase 2 Data Preparation

Prepares separate dataset directory: D:/smartcell/dataset_all
Features:
- Preserves original ALL_IDB1 and ALL_IDB2 datasets completely untouched.
- Excludes duplicate image Im257_0.tif (duplicate of Im223_0.tif) to prevent cross-split leakage.
- Splits into 70% Train, 15% Validation, 15% Test stratified by class with random seed 42.
- Preserves class balance (Class 0: normal lymphocytes, Class 1: lymphoblasts).
- Creates standardized directory layout:
    dataset_all/
        train/
            0_normal/
            1_blast/
        val/
            0_normal/
            1_blast/
        test/
            0_normal/
            1_blast/
- Saves split manifest and summary metrics.
"""

import os
import glob
import shutil
import hashlib
import json
import re
from sklearn.model_selection import train_test_split
from collections import Counter

PROJECT_ROOT = r"D:\smartcell"
ORIGINAL_IDB2_IMG_DIR = os.path.join(PROJECT_ROOT, "ALL_IDB2", "img")
TARGET_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_all")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")

os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RANDOM_SEED = 42

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def prepare_dataset():
    print("=" * 70)
    print("PHASE 2: DATA PREPARATION FOR ALL_IDB LEUKEMIA CLASSIFICATION")
    print("=" * 70)

    # 1. Collect all original images from ALL_IDB2
    raw_files = sorted(glob.glob(os.path.join(ORIGINAL_IDB2_IMG_DIR, "*.tif")))
    print(f"Total raw images found in ALL_IDB2: {len(raw_files)}")

    # 2. Duplicate detection and deduplication
    hash_map = {}
    unique_files = []
    excluded_duplicates = []

    for f in raw_files:
        fname = os.path.basename(f)
        h = sha256_file(f)
        if h in hash_map:
            excluded_duplicates.append({
                "duplicate_file": fname,
                "original_file": hash_map[h],
                "sha256": h
            })
            print(f"  [DEDUPLICATION] Excluded duplicate file: {fname} (identical to {hash_map[h]})")
        else:
            hash_map[h] = fname
            unique_files.append(f)

    print(f"Total unique images retained: {len(unique_files)}")
    print(f"Excluded duplicates: {len(excluded_duplicates)}")

    # 3. Parse classes
    file_records = []
    for f in unique_files:
        fname = os.path.basename(f)
        match = re.match(r"^Im(\d+)_([01])\.", fname)
        if not match:
            raise ValueError(f"Unexpected filename format: {fname}")
        label = int(match.group(2))
        file_records.append({
            "path": f,
            "filename": fname,
            "label": label,
            "class_name": "normal" if label == 0 else "blast"
        })

    classes = [r["label"] for r in file_records]
    class_counts = Counter(classes)
    print(f"Class distribution in unique dataset: Class 0 (Normal)={class_counts[0]}, Class 1 (Blast)={class_counts[1]}")

    # 4. Stratified Split: 70% Train, 15% Val, 15% Test
    # First split off 15% test:
    train_val_records, test_records = train_test_split(
        file_records,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=classes
    )

    # From remaining 85%, validation is 15/85 of remaining (~0.17647) to get 15% overall:
    train_val_classes = [r["label"] for r in train_val_records]
    val_ratio_of_remaining = 0.15 / (0.70 + 0.15)
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_ratio_of_remaining,
        random_state=RANDOM_SEED,
        stratify=train_val_classes
    )

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }

    # Verify split distributions
    print("\nDataset Split Summary:")
    split_manifest = {
        "dataset_name": "ALL_IDB2_Leukemia",
        "random_seed": RANDOM_SEED,
        "deduplication": excluded_duplicates,
        "total_unique_samples": len(file_records),
        "splits": {}
    }

    for split_name, records in splits.items():
        s_classes = [r["label"] for r in records]
        c_dist = Counter(s_classes)
        print(f"  {split_name.upper():5s}: Total={len(records):3d} | Class 0 (Normal)={c_dist[0]:2d} | Class 1 (Blast)={c_dist[1]:2d}")
        
        # Prepare destination folders
        for class_subfolder in ["0_normal", "1_blast"]:
            os.makedirs(os.path.join(TARGET_DATASET_DIR, split_name, class_subfolder), exist_ok=True)

        copied_files = []
        for r in records:
            subfolder = "0_normal" if r["label"] == 0 else "1_blast"
            base_no_ext = os.path.splitext(r["filename"])[0]
            dest_filename = base_no_ext + ".png"
            dest_path = os.path.join(TARGET_DATASET_DIR, split_name, subfolder, dest_filename)
            
            # Losslessly convert TIFF to PNG to ensure full Keras image_dataset compatibility
            from PIL import Image
            with Image.open(r["path"]) as im:
                im.save(dest_path, "PNG")
                
            copied_files.append({
                "source_file": r["filename"],
                "prepared_file": dest_filename,
                "label": r["label"],
                "class_name": r["class_name"],
                "sha256": sha256_file(dest_path)
            })

        split_manifest["splits"][split_name] = {
            "total_count": len(records),
            "class_0_count": c_dist[0],
            "class_1_count": c_dist[1],
            "files": copied_files
        }

    # Save manifest
    manifest_path = os.path.join(TARGET_DATASET_DIR, "split_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(split_manifest, fp, indent=2)
    print(f"\nSaved split manifest to: {manifest_path}")

    # Also save report in results/metrics/
    prep_report_path = os.path.join(METRICS_DIR, "all_idb_data_preparation_report.json")
    with open(prep_report_path, "w") as fp:
        json.dump({
            "preparation_timestamp": "2026-11-21T22:45:00+05:30",
            "source_dataset": "D:/smartcell/ALL_IDB2/img",
            "target_dataset": TARGET_DATASET_DIR,
            "random_seed": RANDOM_SEED,
            "total_raw_images": len(raw_files),
            "excluded_duplicates": excluded_duplicates,
            "total_unique_images": len(file_records),
            "splits": {
                s: {
                    "count": split_manifest["splits"][s]["total_count"],
                    "class_0_normal": split_manifest["splits"][s]["class_0_count"],
                    "class_1_blast": split_manifest["splits"][s]["class_1_count"]
                }
                for s in splits
            }
        }, fp, indent=2)
    print(f"Saved preparation report to: {prep_report_path}")
    print("\nPhase 2: Data Preparation completed successfully!")

if __name__ == "__main__":
    prepare_dataset()
