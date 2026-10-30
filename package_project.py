#!/usr/bin/env python3
"""
RTV Portable Project Packager
=============================
Creates a clean, optimized, and portable zip package of the RTV project
ready to run on another Windows or Linux/Mac system.

Excludes machine-specific bloat:
  - Windows virtual environment ('venv/') (~6.2 GB)
  - Duplicate Git LFS objects ('rtv_ckpts/.git/') (~5.6 GB)
  - Raw dataset video frames ('PerGarmentDatasets/') (~1.2 GB)
  - Old zip files and caches ('__pycache__', '.gradio', 'scratch')
  - Training discriminators ('latest_net_D.pth')
"""

import os
import sys
import time
import zipfile
import argparse

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Formats that are already compressed; storing them saves huge CPU time with negligible size difference
PRECOMPRESSED_EXTS = {'.pth', '.pkl', '.mp4', '.gif', '.png', '.jpg', '.jpeg', '.zip'}

EXCLUDE_DIRS = {
    'venv', '.git', '.gradio', '__pycache__', 'PerGarmentDatasets',
    'rollback_backup', 'rollback_point_v2', 'scratch', '.ipynb_checkpoints',
    'checkpoints'
}

CORE_GARMENTS = {
    'coat_seq_vmssdp2ta_576',  # Full-Body Winter Puffer Coat
    'han_seq_vmssdp2ta_576',   # Full-Body Hanfu Silk Dress
    'lab_03_vmsdp2ta',         # Upper-Body Lab Coat / Jacket
    'rcb_jersey_18'            # Upper-Body Custom Jersey
}


def create_portable_package(mode="recommended", output_zip=None):
    """
    mode:
      - 'recommended': Code + Assets + ROMP + Core Garments (Winter Coat, Hanfu, Lab Coat, RCB Jersey). ~2.1 GB.
      - 'all': Code + Assets + ROMP + All 10 Pretrained Garment Models. ~4.9 GB.
      - 'code_only': Code + Assets + ROMP + SMPL definitions (0 heavy garment models). ~85 MB.
    """
    if output_zip is None:
        output_zip = os.path.join(ROOT_DIR, f"RTV_portable_{mode}.zip")

    if mode == "recommended":
        allowed_garments = CORE_GARMENTS
        mode_desc = "Recommended Core Garments (~2.1 GB)"
    elif mode == "all":
        allowed_garments = None  # All garments
        mode_desc = "All 10 Outfits (~4.9 GB)"
    elif mode == "code_only":
        allowed_garments = set()
        mode_desc = "Code & Assets Only (~85 MB)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("=" * 75)
    print("      REAL-TIME VIRTUAL TRY-ON (RTV) - PORTABLE PACKAGER")
    print("=" * 75)
    print(f"Preset Mode:  {mode.upper()} - {mode_desc}")
    print(f"Source:       {ROOT_DIR}")
    print(f"Output File:  {output_zip}")
    print("=" * 75)
    print("[1/3] Scanning project files...")

    files_to_pack = []
    total_raw_bytes = 0

    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter blacklisted directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        rel_root = os.path.relpath(root, ROOT_DIR)
        parts = rel_root.split(os.sep)

        # Filter garment checkpoints in RTV/rtv_ckpts/<garment_name>
        if len(parts) >= 2 and parts[0] == 'RTV' and parts[1] == 'rtv_ckpts':
            if len(parts) >= 3:
                garment_name = parts[2]
                if allowed_garments is not None and garment_name not in allowed_garments:
                    continue

        for f in files:
            # Exclude unwanted extensions and files
            if f.endswith(('.pyc', '.tmp', '.log', '.zip')) or f.startswith('.'):
                continue
            if f.endswith('latest_net_D.pth'):
                continue

            fpath = os.path.join(root, f)
            arcname = os.path.relpath(fpath, ROOT_DIR)
            sz = os.path.getsize(fpath)
            files_to_pack.append((fpath, arcname, sz))
            total_raw_bytes += sz

    total_mb = total_raw_bytes / (1024 ** 2)
    total_gb = total_mb / 1024.0
    print(f"[2/3] Collected {len(files_to_pack)} clean files ({total_mb:.1f} MB / {total_gb:.2f} GB raw).")
    print(f"[3/3] Packing into '{os.path.basename(output_zip)}'...")

    start_time = time.time()
    packed_bytes = 0

    # Ensure output directory exists and file is overwritten cleanly
    if os.path.exists(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, 'w', allowZip64=True) as zf:
        for idx, (fpath, arcname, sz) in enumerate(files_to_pack):
            _, ext = os.path.splitext(fpath.lower())
            # Use ZIP_STORED for heavy already-compressed files (massive speedup)
            # Use ZIP_DEFLATED for code, text, configs
            compress_type = zipfile.ZIP_STORED if ext in PRECOMPRESSED_EXTS else zipfile.ZIP_DEFLATED
            zf.write(fpath, arcname, compress_type=compress_type)

            packed_bytes += sz
            if idx % 50 == 0 or idx == len(files_to_pack) - 1:
                pct = (packed_bytes / max(1, total_raw_bytes)) * 100.0
                print(f"  -> Packed {idx + 1:4d}/{len(files_to_pack)} files ({pct:5.1f}%)...")

    elapsed = time.time() - start_time
    zip_size_mb = os.path.getsize(output_zip) / (1024 ** 2)
    zip_size_gb = zip_size_mb / 1024.0

    print("=" * 75)
    print("                 PORTABLE PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 75)
    print(f"Saved to:     {output_zip}")
    print(f"Total Files:  {len(files_to_pack)}")
    print(f"Package Size: {zip_size_mb:.1f} MB ({zip_size_gb:.2f} GB)")
    print(f"Time Taken:   {elapsed:.1f} seconds")
    print("=" * 75)
    print("\nNext Steps on the Target System:")
    print("1. Copy & extract this zip on the destination machine.")
    print("2. Run 'setup_new_system.bat' to automatically configure Python & PyTorch.")
    print("3. Run 'run_demo.bat' or 'run_web_ui.bat' to start the application!\n")
    return output_zip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package RTV for transfer to another system")
    parser.add_argument(
        "--mode", choices=["recommended", "all", "code_only"], default="recommended",
        help="Packaging preset: 'recommended' (~2.1GB), 'all' (~4.9GB), 'code_only' (~85MB)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output zip file path")
    args = parser.parse_args()

    create_portable_package(mode=args.mode, output_zip=args.output)

