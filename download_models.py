#!/usr/bin/env python3
"""
RTV Pretrained Model Downloader
===============================
Downloads official pretrained garment neural models from Hugging Face:
https://huggingface.co/wuzaiqiang/rtv_ckpts

Usage:
    python download_models.py --mode recommended   # Downloads core outfits (Coat, Hanfu, Lab Coat)
    python download_models.py --mode all           # Downloads all 8 official garments
"""

import os
import sys
import time
import argparse
import urllib.request

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CKPTS_DIR = os.path.join(ROOT_DIR, "RTV", "rtv_ckpts")

HF_BASE_URL = "https://huggingface.co/wuzaiqiang/rtv_ckpts/resolve/main"

OFFICIAL_GARMENTS = {
    'coat_seq_vmssdp2ta_576': 'Full-Body Winter Puffer Coat (RNN)',
    'han_seq_vmssdp2ta_576':  'Full-Body Hanfu Silk Dress (RNN)',
    'lab_03_vmsdp2ta':        'Upper-Body Lab Coat / Jacket',
    'lab_04_vmsdp2ta':        'Upper-Body Loose Hoodie',
    'lab_07_vmsdp2ta':        'Upper-Body Long-Sleeve Knitwear',
    'jin_17_vmsdp2ta':        'Upper-Body Casual Shirt',
    'jin_18_vmsdp2ta':        'Upper-Body Polo Shirt',
    'jin_22_vmsdp2ta':        'Upper-Body Short Sleeve',
}

CORE_GARMENTS = ['coat_seq_vmssdp2ta_576', 'han_seq_vmssdp2ta_576', 'lab_03_vmsdp2ta']


def download_file_with_progress(url, dest_path, desc):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        sz = os.path.getsize(dest_path) / (1024 ** 2)
        if sz > 100:  # Already downloaded
            print(f"[EXISTS] {desc} ({sz:.1f} MB) is already present. Skipping.")
            return

    print(f"\nDownloading: {desc}")
    print(f"Source:      {url}")
    print(f"Target:      {dest_path}")

    start_time = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'RTV-Downloader/1.0'})

    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.getheader('Content-Length', 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB buffer

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    elapsed = time.time() - start_time
                    speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        mb_done = downloaded / (1024 ** 2)
                        mb_tot = total_size / (1024 ** 2)
                        sys.stdout.write(f"\r  -> [{pct:5.1f}%] {mb_done:.1f}/{mb_tot:.1f} MB ({speed:.2f} MB/s)")
                    else:
                        mb_done = downloaded / (1024 ** 2)
                        sys.stdout.write(f"\r  -> {mb_done:.1f} MB downloaded ({speed:.2f} MB/s)")
                    sys.stdout.flush()

        print(f"\n[OK] Downloaded {desc} successfully in {time.time() - start_time:.1f}s.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {desc}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)


def main():
    parser = argparse.ArgumentParser(description="Download RTV pretrained models from Hugging Face")
    parser.add_argument("--mode", choices=["recommended", "all"], default="recommended",
                        help="Choose 'recommended' (~2.1GB) or 'all' (~5.6GB)")
    args = parser.parse_args()

    targets = CORE_GARMENTS if args.mode == "recommended" else list(OFFICIAL_GARMENTS.keys())

    print("=" * 70)
    print("      REAL-TIME VIRTUAL TRY-ON (RTV) - MODEL DOWNLOADER")
    print("=" * 70)
    print(f"Mode:    {args.mode.upper()}")
    print(f"Models:  {len(targets)} garments to check/download")
    print(f"Target:  {CKPTS_DIR}")
    print("=" * 70)

    for garment in targets:
        url = f"{HF_BASE_URL}/{garment}/latest_net_G.pth"
        dest = os.path.join(CKPTS_DIR, garment, "latest_net_G.pth")
        desc = f"{OFFICIAL_GARMENTS.get(garment, garment)}"
        download_file_with_progress(url, dest, desc)

    print("\n" + "=" * 70)
    print("All requested models are ready to run!")
    print("Run 'run_demo.bat' or 'run_web_ui.bat' to start trying on clothes.")
    print("=" * 70)


if __name__ == "__main__":
    main()

