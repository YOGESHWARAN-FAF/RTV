#!/usr/bin/env python3
"""
RTV Kaggle & Cloud Packager
===========================
Creates an optimized, lightweight zip package of RTV for Kaggle / Cloud GPUs.
Excludes massive unnecessary files (venv, .git history, raw datasets) so that
Kaggle uploads in minutes and extracts in seconds without timeouts or errors.
"""

import os
import sys
import zipfile
import argparse

RTV_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(RTV_DIR)


def build_kaggle_package(mode="recommended", output_zip=None):
    """
    mode:
      - 'recommended' (Default): Code + Assets + SMPL + Core garments (Winter Coat + Lab Coat). ~1.4 GB.
      - 'all': Code + Assets + SMPL + All 10 garment models (excluding .git / venv). ~7.5 GB.
      - 'code_only': Code + Assets + SMPL (no heavy checkpoints). ~85 MB.
    """
    if output_zip is None:
        output_zip = os.path.join(PROJECT_ROOT, "rtv_kaggle_ready.zip")

    # Excluded directories (crucial to prevent Kaggle 1-hour timeout & extraction crash)
    EXCLUDE_DIRS = {
        'venv', '.git', '.gradio', '__pycache__', 'PerGarmentDatasets',
        'rollback_backup', 'rollback_point_v2',
        'scratch', '.ipynb_checkpoints'
    }

    # Garments to include based on mode
    if mode == "recommended":
        ALLOWED_GARMENTS = {'coat_seq_vmssdp2ta_576', 'lab_03_vmsdp2ta', 'han_seq_vmssdp2ta_576'}
    elif mode == "code_only":
        ALLOWED_GARMENTS = set()
    else:  # 'all'
        ALLOWED_GARMENTS = None  # Include all in rtv_ckpts

    print("=" * 70)
    print("RTV KAGGLE CLOUD PACKAGER")
    print(f"Mode:       {mode.upper()}")
    print(f"Source:     {RTV_DIR}")
    print(f"Output:     {output_zip}")
    print("=" * 70)
    print("[1/3] Scanning files to package...")

    files_to_pack = []
    total_raw_bytes = 0

    for root, dirs, files in os.walk(RTV_DIR):
        # Exclude blacklisted directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        rel_root = os.path.relpath(root, RTV_DIR)
        parts = rel_root.split(os.sep)

        # Filter rtv_ckpts based on mode
        if parts[0] == 'rtv_ckpts' and len(parts) > 1:
            g_name = parts[1]
            if ALLOWED_GARMENTS is not None and g_name not in ALLOWED_GARMENTS:
                continue

        # Skip discriminator models (netD) which are only used in training, saving hundreds of MB
        for f in files:
            if f.endswith(('.pyc', '.tmp', '.log')) or f.startswith('.'):
                continue
            if f.endswith('latest_net_D.pth'):
                continue
            fpath = os.path.join(root, f)
            arcname = os.path.join("RTV", os.path.relpath(fpath, RTV_DIR))
            sz = os.path.getsize(fpath)
            files_to_pack.append((fpath, arcname, sz))
            total_raw_bytes += sz

    print(f"[2/3] Found {len(files_to_pack)} clean files ({total_raw_bytes / (1024**3):.2f} GB).")
    print(f"[3/3] Compressing into {output_zip}...")

    # Write zip file with progress updates
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        packed_bytes = 0
        for idx, (fpath, arcname, sz) in enumerate(files_to_pack):
            zf.write(fpath, arcname)
            packed_bytes += sz
            if idx % 50 == 0 or idx == len(files_to_pack) - 1:
                pct = (packed_bytes / max(1, total_raw_bytes)) * 100.0
                print(f"  -> Packed {idx+1}/{len(files_to_pack)} files ({pct:.1f}%)...")

    final_size_mb = os.path.getsize(output_zip) / (1024**2)
    final_size_gb = final_size_mb / 1024.0

    print("=" * 70)
    print("SUCCESSFULLY CREATED KAGGLE READY ZIP!")
    print(f"File:       {output_zip}")
    print(f"Files:      {len(files_to_pack)} (Clean, 0 Windows venv files, 0 .git bloat)")
    print(f"Zip Size:   {final_size_mb:.1f} MB ({final_size_gb:.2f} GB)")
    print("=" * 70)
    print("\nNext Step:")
    print("Upload 'rtv_kaggle_ready.zip' to Kaggle under '+ Add Data' -> 'Upload Dataset'.")
    print("Kaggle will process it in less than 30 seconds!\n")
    return output_zip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package RTV for Kaggle & Cloud GPUs")
    parser.add_argument(
        "--mode", choices=["recommended", "all", "code_only"], default="recommended",
        help="Packaging preset: 'recommended' (~1.4GB with core garments), 'all' (~7GB), 'code_only' (~85MB)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output zip file path")
    args = parser.parse_args()

    build_kaggle_package(mode=args.mode, output_zip=args.output)

