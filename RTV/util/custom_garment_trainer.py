import os
import sys
import time
import json
import shutil
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from SMPL.smpl_regressor import SMPL_Regressor
from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.densepose_util import IUV2SDP
try:
    from Datasets.upperbody_garment.upperbody_garment import UpperBodyGarment
except ImportError:
    UpperBodyGarment = None
from options.train_options import TrainOptions
from model.pix2pixHD.models import create_model

class CustomGarmentTrainer:
    """
    Production-grade Custom Garment Pipeline for RTV:
    1. Dataset Generation: SMPL 3D Mesh + DensePose + Automatic or Custom Garment Masks.
    2. Neural Training: Pix2PixHD_RGBA with AMP FP16 on RTX 2050 Tensor Cores.
    3. One-Click Checkpoint Deployment directly to rtv_ckpts/.
    4. Instant Static Image & Video Try-On Testing.
    """
    def __init__(self, checkpoints_dir="./rtv_ckpts", datasets_dir="./PerGarmentDatasets"):
        self.checkpoints_dir = os.path.abspath(checkpoints_dir)
        self.datasets_dir = os.path.abspath(datasets_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

    @staticmethod
    def _call_cb(cb, *args):
        if not cb:
            return
        try:
            cb(*args)
        except TypeError:
            try:
                if len(args) >= 5:
                    cb(args[0], args[1], args[4])
                elif len(args) >= 1:
                    cb(args[-1])
            except Exception:
                pass

    def import_checkpoint(self, pth_or_dir_path, garment_name, preview_path=None):
        """
        Directly imports an existing .pth model or directory into the RTV wardrobe.
        """
        clean_name = garment_name.strip().replace(" ", "_").lower()
        target_dir = os.path.join(self.checkpoints_dir, clean_name)
        os.makedirs(target_dir, exist_ok=True)

        target_pth = os.path.join(target_dir, "latest_net_G.pth")
        if os.path.isdir(pth_or_dir_path):
            cand = os.path.join(pth_or_dir_path, "latest_net_G.pth")
            if not os.path.exists(cand):
                # Search for any .pth file in the dir
                pths = [f for f in os.listdir(pth_or_dir_path) if f.endswith(".pth")]
                if not pths:
                    raise FileNotFoundError(f"No .pth checkpoint found inside {pth_or_dir_path}")
                cand = os.path.join(pth_or_dir_path, pths[0])
            shutil.copy2(cand, target_pth)
        elif os.path.isfile(pth_or_dir_path):
            shutil.copy2(pth_or_dir_path, target_pth)
        else:
            raise FileNotFoundError(f"Checkpoint source not found: {pth_or_dir_path}")

        # Save thumbnail preview
        thumb_dest = os.path.join(target_dir, "preview.jpg")
        assets_thumb = os.path.abspath(f"./assets/garment_images/{clean_name}_preview.jpg")
        os.makedirs(os.path.dirname(assets_thumb), exist_ok=True)

        if preview_path and os.path.exists(preview_path):
            shutil.copy2(preview_path, thumb_dest)
            shutil.copy2(preview_path, assets_thumb)
        else:
            # Create a clean placeholder thumbnail
            thumb = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.rectangle(thumb, (5, 5), (195, 195), (0, 200, 100), 2)
            cv2.putText(thumb, clean_name[:12], (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imwrite(thumb_dest, thumb)
            cv2.imwrite(assets_thumb, thumb)

        print(f"[RTV Trainer] Successfully imported garment '{clean_name}' to {target_dir}")
        return clean_name, target_dir

    def find_best_pretrained_checkpoint(self, exclude_name=None):
        """
        Finds the best existing high-quality pretrained checkpoint in rtv_ckpts/
        to initialize the Pix2PixHD Generator for rapid Transfer Learning fine-tuning.
        """
        preferred_bases = [
            "jin_22_vmsdp2ta",
            "lab_03_vmsdp2ta",
            "jin_18_vmsdp2ta",
            "jin_17_vmsdp2ta",
            "lab_04_vmsdp2ta",
            "lab_07_vmsdp2ta",
            "coat_seq_vmssdp2ta_576",
            "han_seq_vmssdp2ta_576"
        ]
        clean_exclude = (exclude_name or "").strip().lower()

        for base in preferred_bases:
            if clean_exclude and base == clean_exclude:
                continue
            cand = os.path.join(self.checkpoints_dir, base)
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "latest_net_G.pth")):
                return cand

        # Fallback: scan any directory in checkpoints_dir with latest_net_G.pth
        if os.path.exists(self.checkpoints_dir):
            for d in os.listdir(self.checkpoints_dir):
                if d.startswith(".") or (clean_exclude and d == clean_exclude):
                    continue
                cand = os.path.join(self.checkpoints_dir, d)
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "latest_net_G.pth")):
                    return cand
        return None

    def generate_dataset_from_multiview(self, views_dict, dataset_name="custom_garment",
                                        samples_per_view=None, progress_cb=None, force_rebuild=False,
                                        mode="turbo"):
        """
        Builds a rich, balanced 360-degree training dataset from 4 cardinal angle photos:
        Front, Back, Right, and Left.
        Synthesizes realistic geometric and photometric perturbations around each view to expand
        the 4 stills into balanced training pairs for Pix2PixHD training.

        Args:
            views_dict: Dict with keys in ('front', 'back', 'right', 'left') and filepaths or cv2 numpy images as values.
            dataset_name: Name of dataset directory in ./PerGarmentDatasets/<name>
            samples_per_view: Number of augmented samples to synthesize per provided view.
                              If None, dynamically selected based on mode:
                              - 'turbo': 10 samples/view (40 pairs total, ~1 min training)
                              - 'balanced': 20 samples/view (80 pairs total)
                              - 'deep': 30 samples/view (120 pairs total)
            progress_cb: Optional callback(current, total, message)
            force_rebuild: If False and valid dataset already exists, skips Phase 1 synthesis
            mode: 'turbo', 'balanced', or 'deep'
        """
        clean_name = dataset_name.strip().replace(" ", "_").lower()
        target_path = os.path.join(self.datasets_dir, clean_name)
        os.makedirs(target_path, exist_ok=True)

        if samples_per_view is None:
            if mode == "turbo":
                samples_per_view = 10
            elif mode == "balanced":
                samples_per_view = 20
            else:
                samples_per_view = 30

        # Collect valid views
        valid_views = []
        for key in ["front", "back", "right", "left"]:
            val = views_dict.get(key)
            if val is None:
                continue
            img = None
            if isinstance(val, str) and os.path.isfile(val):
                img = cv2.imread(val)
            elif isinstance(val, np.ndarray):
                img = val.copy()
            if img is not None:
                valid_views.append((key, img))

        if not valid_views:
            raise ValueError("No valid view images provided. Please provide at least the Front view photo.")

        total_views = len(valid_views)
        total_expected = total_views * samples_per_view

        # Check if existing complete dataset already exists to save time
        info_path = os.path.join(target_path, "dataset_info.json")
        if not force_rebuild and os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    cached_info = json.load(f)
                cached_count = cached_info.get("samples", 0)
                last_sample = os.path.join(target_path, f"{str(cached_count-1).zfill(5)}_garment.jpg")
                if cached_count >= total_expected and os.path.exists(last_sample):
                    if progress_cb:
                        progress_cb(cached_count, cached_count, f"Found existing verified 360° dataset ({cached_count} pairs). Skipping Phase 1 synthesis.")
                    return target_path, cached_count
            except Exception:
                pass

        view_names_str = ", ".join(k.upper() for k, _ in valid_views)
        if progress_cb:
            progress_cb(0, total_expected, f"Loaded {total_views} views ({view_names_str}). Initializing 3D SMPL and DensePose...")

        # Save front view (or first available view) as garment preview thumbnail
        front_img = None
        for k, im in valid_views:
            if k == "front":
                front_img = im
                break
        if front_img is None:
            front_img = valid_views[0][1]

        if front_img is not None:
            target_dir = os.path.join(self.checkpoints_dir, clean_name)
            os.makedirs(target_dir, exist_ok=True)
            thumb_dest = os.path.join(target_dir, "preview.jpg")
            assets_thumb = os.path.abspath(f"./assets/garment_images/{clean_name}_preview.jpg")
            os.makedirs(os.path.dirname(assets_thumb), exist_ok=True)
            th_h, th_w = front_img.shape[:2]
            crop_size = min(th_h, th_w)
            cy, cx = th_h // 2, th_w // 2
            thumb_sq = front_img[max(0, cy - crop_size//2):cy + crop_size//2, max(0, cx - crop_size//2):cx + crop_size//2]
            if thumb_sq.size > 0:
                thumb_sq = cv2.resize(thumb_sq, (200, 200))
                cv2.imwrite(thumb_dest, thumb_sq)
                cv2.imwrite(assets_thumb, thumb_sq)

        # Initialize vision backends
        smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
        upper_body = UpperBodySMPL()
        densepose_extractor = DensePoseExtractor()

        resolution = 1024
        processed_count = 0
        ref_height, ref_width = valid_views[0][1].shape[:2]

        for v_idx, (view_name, raw_image) in enumerate(valid_views):
            height, width = raw_image.shape[:2]
            new_height = 1024
            new_width = new_height * width // height
            resized_image = cv2.resize(raw_image, (new_width, new_height))

            # 1. 3D Body Mesh Regression
            smpl_param, trans2roi, _ = smpl_regressor.forward(raw_image, True, size=1.45, roi_img_size=resolution)
            if smpl_param is None:
                smpl_param, trans2roi, _ = smpl_regressor.forward(raw_image, True, size=1.65, roi_img_size=resolution)
                if smpl_param is None:
                    print(f"[RTV MultiView] Warning: Pose detection failed for {view_name} view.")
                    continue

            vertices = smpl_regressor.get_raw_verts(smpl_param)
            vertices = torch.from_numpy(vertices).unsqueeze(0)
            raw_vm = upper_body.render(vertices[0], height=new_height, width=new_width)

            # 2. DensePose IUV Surface Coordinate Mapping
            raw_IUV = densepose_extractor.get_IUV(resized_image, isRGB=False)
            if raw_IUV is None:
                print(f"[RTV MultiView] Warning: DensePose failed for {view_name} view.")
                continue

            # 3. Garment Segmentation Mask
            dp_torso = np.isin(raw_IUV[:, :, 0], [1, 2, 15, 16, 17, 18])
            dp_torso_full = cv2.resize(dp_torso.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
            raw_mask = cv2.dilate(dp_torso_full, kernel)

            # Isolate garment ROI
            raw_garment = raw_image.copy()
            raw_garment[raw_mask < 80] = 0
            roi_garment_img = cv2.warpAffine(raw_garment, trans2roi, (resolution, resolution),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Base sample
            seq_str = str(processed_count).zfill(5)
            cv2.imwrite(os.path.join(target_path, f"{seq_str}_garment.jpg"), roi_garment_img)
            cv2.imwrite(os.path.join(target_path, f"{seq_str}_vm.jpg"), raw_vm)
            cv2.imwrite(os.path.join(target_path, f"{seq_str}_mask.png"), raw_mask)
            np.save(os.path.join(target_path, f"{seq_str}_iuv.npy"), raw_IUV)
            np.save(os.path.join(target_path, f"{seq_str}_trans2roi.npy"), trans2roi)
            processed_count += 1

            if progress_cb:
                progress_cb(processed_count, total_expected, f"Processed base {view_name.upper()} view (Sample {processed_count}/{total_expected})")

            # 4. Generate Perturbed Geometric & Photometric Variations
            n_aug = samples_per_view - 1
            for a_idx in range(n_aug):
                theta = float(np.random.uniform(-5.0, 5.0))
                scale = float(np.random.uniform(0.95, 1.05))
                dx = float(np.random.uniform(-14.0, 14.0))
                dy = float(np.random.uniform(-14.0, 14.0))

                # 2D affine matrix for full image
                M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), theta, scale)
                M[0, 2] += dx
                M[1, 2] += dy

                p_img = cv2.warpAffine(raw_image, M, (width, height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                p_mask = cv2.warpAffine(raw_mask, M, (width, height),
                                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # 2D affine matrix for resized VM / IUV
                M_res = cv2.getRotationMatrix2D((new_width / 2.0, new_height / 2.0), theta, scale)
                M_res[0, 2] += dx * (new_width / float(width))
                M_res[1, 2] += dy * (new_height / float(height))

                p_vm = cv2.warpAffine(raw_vm, M_res, (new_width, new_height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                p_iuv = cv2.warpAffine(raw_IUV, M_res, (new_width, new_height),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # Photometric jitter on garment texture
                p_garment = p_img.copy()
                p_garment[p_mask < 80] = 0

                alpha_contrast = float(np.random.uniform(0.92, 1.08))
                beta_brightness = float(np.random.uniform(-10, 10))
                p_garment = np.clip(alpha_contrast * p_garment + beta_brightness, 0, 255).astype(np.uint8)
                p_garment[p_mask < 80] = 0

                p_roi = cv2.warpAffine(p_garment, trans2roi, (resolution, resolution),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                seq_str = str(processed_count).zfill(5)
                cv2.imwrite(os.path.join(target_path, f"{seq_str}_garment.jpg"), p_roi)
                cv2.imwrite(os.path.join(target_path, f"{seq_str}_vm.jpg"), p_vm)
                cv2.imwrite(os.path.join(target_path, f"{seq_str}_mask.png"), p_mask)
                np.save(os.path.join(target_path, f"{seq_str}_iuv.npy"), p_iuv)
                np.save(os.path.join(target_path, f"{seq_str}_trans2roi.npy"), trans2roi)
                processed_count += 1

                if progress_cb and (processed_count % 5 == 0 or processed_count == total_expected):
                    progress_cb(processed_count, total_expected, f"Synthesized {view_name.upper()} variation ({processed_count}/{total_expected})")

        dataset_info = {
            "height": ref_height,
            "width": ref_width,
            "samples": processed_count,
            "multiview": True,
            "views": [k for k, _ in valid_views]
        }
        with open(os.path.join(target_path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f)

        if progress_cb:
            progress_cb(total_expected, total_expected, f"Multi-view dataset complete! ({processed_count} training pairs synthesized)")

        return target_path, processed_count

    def generate_dataset_from_source(self, source_path, mask_dir=None, dataset_name="custom_garment",
                                    max_frames=120, progress_cb=None):
        """
        Builds a full per-garment training dataset from a video file or folder of images.
        """
        clean_name = dataset_name.strip().replace(" ", "_").lower()
        target_path = os.path.join(self.datasets_dir, clean_name)
        os.makedirs(target_path, exist_ok=True)

        # Check if verified dataset already exists
        info_path = os.path.join(target_path, "dataset_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    cached_info = json.load(f)
                cached_count = cached_info.get("samples", 0)
                last_sample = os.path.join(target_path, f"{str(cached_count-1).zfill(5)}_garment.jpg")
                if cached_count > 0 and os.path.exists(last_sample):
                    if progress_cb:
                        progress_cb(cached_count, cached_count, f"Found existing verified dataset ({cached_count} pairs). Skipping extraction.")
                    return target_path, cached_count
            except Exception:
                pass

        # Collect source frames
        frames = []
        if os.path.isfile(source_path):
            cap = cv2.VideoCapture(source_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_vid_frames > 0:
                sample_indices = np.linspace(0, total_vid_frames - 1, min(total_vid_frames, max_frames), dtype=int)
                sample_set = set(sample_indices)
                curr_f = 0
                while cap.isOpened() and len(frames) < len(sample_indices):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if curr_f in sample_set:
                        frames.append(frame)
                    curr_f += 1
            else:
                while cap.isOpened() and len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
            cap.release()
        elif os.path.isdir(source_path):
            img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            files = sorted([os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(img_exts)])
            for f in files[:max_frames]:
                img = cv2.imread(f)
                if img is not None:
                    frames.append(img)
        else:
            raise FileNotFoundError(f"Source media not found: {source_path}")

        if not frames:
            raise ValueError(f"No valid frames found in source: {source_path}")

        total_frames = len(frames)
        if progress_cb:
            progress_cb(0, total_frames, f"Loaded {total_frames} frames evenly sampled across video. Initializing 3D SMPL and DensePose...")

        # Initialize vision backends
        smpl_regressor = SMPL_Regressor(use_bev=True, fix_body=True)
        upper_body = UpperBodySMPL()
        densepose_extractor = DensePoseExtractor()

        resolution = 1024
        height, width = frames[0].shape[:2]
        processed_count = 0

        # Mask files if supplied
        mask_files = []
        if mask_dir and os.path.isdir(mask_dir):
            mask_exts = ('.png', '.jpg', '.jpeg')
            mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(mask_exts)])

        for idx, raw_image in enumerate(frames):
            new_height = 1024
            new_width = new_height * width // height
            resized_image = cv2.resize(raw_image, (new_width, new_height))

            # 1. 3D Body Mesh Regression
            smpl_param, trans2roi, _ = smpl_regressor.forward(raw_image, True, size=1.45, roi_img_size=resolution)
            if smpl_param is None:
                continue

            vertices = smpl_regressor.get_raw_verts(smpl_param)
            vertices = torch.from_numpy(vertices).unsqueeze(0)
            raw_vm = upper_body.render(vertices[0], height=new_height, width=new_width)

            # 2. DensePose IUV Surface Coordinate Mapping
            raw_IUV = densepose_extractor.get_IUV(resized_image, isRGB=False)
            if raw_IUV is None:
                continue

            # 3. Garment Segmentation Mask
            if idx < len(mask_files) and os.path.exists(mask_files[idx]):
                m_read = cv2.imread(mask_files[idx], cv2.IMREAD_UNCHANGED)
                if m_read.ndim == 3 and m_read.shape[2] == 4:
                    raw_mask = m_read[:, :, 3]
                elif m_read.ndim == 3:
                    raw_mask = cv2.cvtColor(m_read, cv2.COLOR_BGR2GRAY)
                else:
                    raw_mask = m_read
                raw_mask = cv2.resize(raw_mask, (width, height))
            else:
                # Automated foreground clothing mask extraction from DensePose torso & upper arms (short sleeves)
                dp_torso = np.isin(raw_IUV[:, :, 0], [1, 2, 15, 16, 17, 18])
                dp_torso_full = cv2.resize(dp_torso.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
                raw_mask = cv2.dilate(dp_torso_full, kernel)

            # Isolate garment ROI
            raw_garment = raw_image.copy()
            raw_garment[raw_mask < 80] = 0
            roi_garment_img = cv2.warpAffine(raw_garment, trans2roi, (resolution, resolution),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            seq_str = str(processed_count).zfill(5)
            garment_path = os.path.join(target_path, f"{seq_str}_garment.jpg")
            vm_path = os.path.join(target_path, f"{seq_str}_vm.jpg")
            mask_path = os.path.join(target_path, f"{seq_str}_mask.png")
            iuv_path = os.path.join(target_path, f"{seq_str}_iuv.npy")
            trans2roi_path = os.path.join(target_path, f"{seq_str}_trans2roi.npy")

            cv2.imwrite(garment_path, roi_garment_img)
            cv2.imwrite(vm_path, raw_vm)
            cv2.imwrite(mask_path, raw_mask)
            np.save(iuv_path, raw_IUV)
            np.save(trans2roi_path, trans2roi)
            processed_count += 1

            if progress_cb and (idx % 5 == 0 or idx == total_frames - 1):
                progress_cb(idx + 1, total_frames, f"Processed {idx+1}/{total_frames} frames ({processed_count} valid pairs)")

        dataset_info = {"height": height, "width": width, "samples": processed_count}
        with open(os.path.join(target_path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f)

        if progress_cb:
            progress_cb(total_frames, total_frames, f"Dataset generation complete! ({processed_count} training pairs saved)")

        return target_path, processed_count

    def train_garment(self, dataset_name, garment_name=None, num_epochs=None, batch_size=None,
                      img_size=512, lr=0.0002, progress_cb=None, epoch_cb=None, stop_event=None,
                      epochs=None, mode="turbo", pretrained_base=None, **kwargs):
        """
        Trains the Pix2PixHD_RGBA neural generator using PyTorch AMP FP16 on the RTX 2050.
        Supports ⚡ Turbo Mode fine-tuning from high-quality pretrained bases in ~1–2 minutes.
        Auto-deploys the resulting checkpoint directly into rtv_ckpts/.
        """
        # Resolve epochs & batch size based on mode
        if epochs is not None:
            num_epochs = epochs
        elif "epochs" in kwargs and kwargs["epochs"] is not None:
            num_epochs = kwargs["epochs"]

        if num_epochs is None:
            if mode == "turbo":
                num_epochs = 3
            elif mode == "balanced":
                num_epochs = 6
            else:
                num_epochs = 15

        if batch_size is None:
            batch_size = 2

        if "epoch_cb" in kwargs and kwargs["epoch_cb"] is not None:
            epoch_cb = kwargs["epoch_cb"]

        if os.path.exists(dataset_name) and os.path.isdir(dataset_name):
            dataset_path = os.path.abspath(dataset_name)
            clean_ds_name = os.path.basename(dataset_path)
        else:
            clean_ds_name = dataset_name.strip().replace(" ", "_").lower()
            dataset_path = os.path.join(self.datasets_dir, clean_ds_name)

        clean_garment_name = (garment_name or clean_ds_name).strip().replace(" ", "_").lower()

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        # Resolve pretrained base checkpoint for rapid transfer learning
        if pretrained_base is None and mode in ("turbo", "balanced", "fine_tune"):
            pretrained_base = self.find_best_pretrained_checkpoint(exclude_name=clean_garment_name)

        # Configure training options
        opt = TrainOptions().parse(save=False, use_default=True, show_info=False)
        opt.name = clean_garment_name
        opt.dataset_path = dataset_path
        opt.model = "pix2pixHD_RGBA"
        opt.input_nc = 6
        opt.output_nc = 4
        opt.batchSize = batch_size
        opt.img_size = img_size
        opt.niter = max(1, num_epochs // 2)
        opt.niter_decay = max(1, num_epochs - opt.niter)
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        opt.checkpoints_dir = os.path.abspath("./checkpoints")
        os.makedirs(os.path.join(opt.checkpoints_dir, clean_garment_name), exist_ok=True)
        opt.print_freq = 10
        opt.display_freq = 999999
        opt.save_latest_freq = 200

        # Enable transfer learning fine-tuning if pretrained base found
        if pretrained_base and os.path.exists(pretrained_base):
            opt.load_pretrain = os.path.abspath(pretrained_base)
            base_label = os.path.basename(opt.load_pretrain)
            print(f"[RTV Turbo Mode] Initializing Generator with pretrained base: '{base_label}'", flush=True)
            self._call_cb(progress_cb, 0, num_epochs, 0, 0.0, f"Turbo Mode: Fine-tuning from pretrained '{base_label}' ({num_epochs} epochs, ~90s)...")
        else:
            opt.load_pretrain = ""
            print(f"[RTV Trainer] Training model '{clean_garment_name}' from scratch ({num_epochs} epochs)...", flush=True)

        dataset = UpperBodyGarment(dataset_path, img_size=opt.img_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batchSize, shuffle=True, num_workers=0, drop_last=False
        )

        model = create_model(opt)
        total_epochs = opt.niter + opt.niter_decay
        total_steps = 0

        # Enable CUDA AMP scaler for Tensor Core acceleration on RTX 2050
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        init_msg = f"Turbo Mode Training active! (Epochs: {total_epochs}, Batch: {opt.batchSize}, Samples: {len(dataset)}, AMP FP16)"
        print(f"[RTV Trainer] {init_msg}", flush=True)
        self._call_cb(progress_cb, 0, total_epochs, 0, 0.0, init_msg)

        for epoch in range(1, total_epochs + 1):
            if stop_event and stop_event.is_set():
                self._call_cb(progress_cb, epoch, total_epochs, total_steps, 0.0, "Training stopped by user.")
                return None

            epoch_loss = 0.0
            num_batches = 0

            for i, data in enumerate(dataloader):
                if stop_event and stop_event.is_set():
                    break

                garment_img, vm_img, dp_img, garment_mask = data
                input_img = torch.cat([vm_img, dp_img], 1)
                gt_image = torch.cat([garment_img, garment_mask], 1)

                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    gt_image = gt_image.cuda()

                model.module.optimizer_G.zero_grad()
                model.module.optimizer_D.zero_grad()

                if scaler:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        losses, _ = model(input_img, gt_image, infer=False)
                        loss_G = (losses[0] + losses[1] + losses[2]).mean()
                        loss_D = ((losses[3] + losses[4]) * 0.5).mean()

                    scaler.scale(loss_D).backward(retain_graph=True)
                    scaler.scale(loss_G).backward()
                    scaler.step(model.module.optimizer_D)
                    scaler.step(model.module.optimizer_G)
                    scaler.update()
                else:
                    losses, _ = model(input_img, gt_image, infer=False)
                    loss_G = (losses[0] + losses[1] + losses[2]).mean()
                    loss_D = ((losses[3] + losses[4]) * 0.5).mean()
                    loss_D.backward(retain_graph=True)
                    loss_G.backward()
                    model.module.optimizer_D.step()
                    model.module.optimizer_G.step()

                total_steps += opt.batchSize
                step_loss = float(loss_G.item())
                epoch_loss += step_loss
                num_batches += 1

                if (i + 1) % 5 == 0 or (i + 1) == len(dataloader):
                    batch_msg = f"Epoch {epoch}/{total_epochs} [{i+1}/{len(dataloader)}] - Loss: {step_loss:.4f}"
                    print(f"[RTV Trainer] {batch_msg}", flush=True)
                    self._call_cb(progress_cb, epoch, total_epochs, total_steps, step_loss, batch_msg)

            avg_loss = epoch_loss / max(1, num_batches)
            msg = f"Epoch {epoch}/{total_epochs} completed | Generator Loss: {avg_loss:.4f}"
            print(f"[RTV Trainer] {msg}", flush=True)

            self._call_cb(progress_cb, epoch, total_epochs, total_steps, avg_loss, msg)
            self._call_cb(epoch_cb, epoch, total_epochs, total_steps, avg_loss, msg)


        # Save model
        model.module.save('latest')
        saved_pth = os.path.join(opt.checkpoints_dir, clean_garment_name, 'latest_net_G.pth')

        # Auto-deploy to rtv_ckpts/
        deployed_dir = os.path.join(self.checkpoints_dir, clean_garment_name)
        os.makedirs(deployed_dir, exist_ok=True)
        deployed_pth = os.path.join(deployed_dir, 'latest_net_G.pth')
        if os.path.exists(saved_pth):
            try:
                shutil.copy2(saved_pth, deployed_pth)
            except OSError:
                if os.path.exists(deployed_pth):
                    try:
                        os.remove(deployed_pth)
                    except Exception:
                        pass
                shutil.move(saved_pth, deployed_pth)

        # Auto-create thumbnail from first garment image
        first_garment = os.path.join(dataset_path, "00000_garment.jpg")
        thumb_dest = os.path.join(deployed_dir, "preview.jpg")
        assets_thumb = os.path.abspath(f"./assets/garment_images/{clean_garment_name}_preview.jpg")
        if os.path.exists(first_garment):
            g_img = cv2.imread(first_garment)
            if g_img is not None:
                gray = cv2.cvtColor(g_img, cv2.COLOR_BGR2GRAY)
                coords = cv2.findNonZero(gray)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    crop = g_img[max(0, y):min(g_img.shape[0], y+h), max(0, x):min(g_img.shape[1], x+w)]
                    if crop.size > 0:
                        thumb = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)
                    else:
                        thumb = cv2.resize(g_img, (200, 200))
                else:
                    thumb = cv2.resize(g_img, (200, 200))
                cv2.imwrite(thumb_dest, thumb)
                cv2.imwrite(assets_thumb, thumb)

        print(f"[RTV Trainer] Model successfully trained & deployed to {deployed_pth}")
        return clean_garment_name, deployed_pth
