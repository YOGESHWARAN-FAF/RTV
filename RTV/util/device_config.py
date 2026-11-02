import os
import torch

_INITIALIZED = False
_DEVICE_POSE = None
_DEVICE_GEN = None
_DEVICE_DP_STR = None

def init_rtv_devices():
    """
    Intelligently select PyTorch devices for Pose Estimation and Garment Generation.
    
    Prevents CUDA Out of Memory (OOM) by verifying that GPU 1 has at least 8 GB VRAM
    and has not been disabled before assigning any neural network to it.
    
    On systems where GPU 0 is a massive accelerator (e.g. 48GB NVIDIA L40S)
    and GPU 1 is a small auxiliary card (e.g. 4GB T400), ALL models are safely routed
    to GPU 0, utilizing the high-speed Tensor Cores and eliminating VRAM exhaustion.
    """
    global _INITIALIZED, _DEVICE_POSE, _DEVICE_GEN, _DEVICE_DP_STR
    
    # Enable expandable segments to avoid PyTorch CUDA memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    if not torch.cuda.is_available():
        _DEVICE_POSE = torch.device('cpu')
        _DEVICE_GEN = torch.device('cpu')
        _DEVICE_DP_STR = 'cpu'
        _INITIALIZED = True
        return _DEVICE_POSE, _DEVICE_GEN, _DEVICE_DP_STR

    cnt = torch.cuda.device_count()
    primary_idx = int(os.environ.get("RTV_GPU", "0"))
    if primary_idx >= cnt or primary_idx < 0:
        primary_idx = 0
    primary_dev_str = f"cuda:{primary_idx}"

    # Default to single GPU on primary device
    _DEVICE_POSE = torch.device(primary_dev_str)
    _DEVICE_GEN = torch.device(primary_dev_str)
    _DEVICE_DP_STR = primary_dev_str

    # Check whether dual GPU was explicitly requested
    enable_dual = os.environ.get("RTV_DUAL_GPU", "0").lower() in ("1", "true", "yes")

    if cnt >= 2:
        try:
            prop0 = torch.cuda.get_device_properties(0)
            prop1 = torch.cuda.get_device_properties(1)
            mem1_gb = prop1.total_memory / (1024 ** 3)
            mem0_gb = prop0.total_memory / (1024 ** 3)
            
            # If user explicitly forced single GPU or GPU 1 has < 8GB VRAM (e.g. 4GB auxiliary card):
            if not enable_dual or mem1_gb < 7.5:
                _DEVICE_POSE = torch.device(primary_dev_str)
                _DEVICE_GEN = torch.device(primary_dev_str)
                _DEVICE_DP_STR = primary_dev_str
                if not _INITIALIZED:
                    if mem1_gb < 7.5:
                        print(f"[RTV-Device] Notice: GPU 1 ({prop1.name}) has {mem1_gb:.1f} GB VRAM (< 8.0 GB required).")
                        print(f"[RTV-Device] Unified Pipeline: Routing all neural models to GPU 0 ({prop0.name}, {mem0_gb:.1f} GB) to prevent CUDA OOM.")
                    else:
                        print(f"[RTV-Device] Unified Single-GPU Mode active on {prop0.name} ({mem0_gb:.1f} GB VRAM).")
            else:
                # Both GPUs have >= 8GB VRAM and dual GPU requested
                _DEVICE_POSE = torch.device('cuda:0')
                _DEVICE_GEN = torch.device('cuda:1')
                _DEVICE_DP_STR = 'cuda:1'
                if not _INITIALIZED:
                    print(f"[RTV-Device] Dual-GPU Mode active: Pose -> GPU 0 ({prop0.name}, {mem0_gb:.1f} GB), Garment Gen -> GPU 1 ({prop1.name}, {mem1_gb:.1f} GB)")
        except Exception as e:
            if not _INITIALIZED:
                print(f"[RTV-Device] GPU memory inspection note: {e}. Defaulting to {primary_dev_str}.")
    else:
        if not _INITIALIZED:
            prop0 = torch.cuda.get_device_properties(0)
            mem0_gb = prop0.total_memory / (1024 ** 3)
            print(f"[RTV-Device] Single-GPU active: {prop0.name} ({mem0_gb:.1f} GB VRAM)")

    _INITIALIZED = True
    return _DEVICE_POSE, _DEVICE_GEN, _DEVICE_DP_STR


def get_pose_device():
    if not _INITIALIZED:
        init_rtv_devices()
    return _DEVICE_POSE


def get_gen_device():
    if not _INITIALIZED:
        init_rtv_devices()
    return _DEVICE_GEN


def get_densepose_device_str():
    if not _INITIALIZED:
        init_rtv_devices()
    return _DEVICE_DP_STR
