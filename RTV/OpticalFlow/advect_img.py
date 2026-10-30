import torch
from torch.nn import functional as F
import  numpy as np
import cv2


def advect_tensor_image(img_tensor, flow):
    h=img_tensor.shape[-2]
    w=img_tensor.shape[-1]
    sample_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)
    sample_w=torch.linspace(-1,1,w).repeat(h,1)
    grid = torch.cat([sample_w.unsqueeze(2),sample_h.unsqueeze(2)],dim=2)
    grid = grid.unsqueeze(0)  # 1XHXWX2
    raw_grid = grid.copy()
    # flow: 1X2XHXW
    flow = flow.permute(0,2,3,1)
    #flow[:,:,:,0] = flow[:,:,:,0]/h
    #flow[:, :, :, 1] = flow[:, :, :, 1] / w

    grid = grid-flow*20

    advected = F.grid_sample(img_tensor,grid=grid,mode='bilinear')
    return advected

def advect_numpy_image2(np_image:np.ndarray, flow):
    raw_dim = np_image.ndim
    raw_dtype = np_image.dtype
    if np_image.ndim ==2:
        np_image=np.expand_dims(2)
    np_image = np_image.astype(np.float32)
    img_tensor = torch.from_numpy(np_image).permute(2,0,1).unsqueeze(0)
    advected_tensor = advect_tensor_image(img_tensor,flow)
    advected_img = advected_tensor.squeeze(0).permute(1,2,0).numpy().astype(raw_dtype)
    if raw_dim ==2:
        advected_img=advected_img[:,:,0]
    return advected_img

def advect_numpy_image(np_image:np.ndarray, flow):
    raw_dim = np_image.ndim
    raw_dtype = np_image.dtype
    if np_image.ndim ==2:
        np_image=np.expand_dims(2)
    np_image = np_image.astype(np.uint8)
    h=np_image.shape[0]
    w=np_image.shape[1]
    flow = flow[0].permute(1,2,0).numpy()*1
    flow=-flow
    #print(flow[:, :, 0].shape)
    #print(np.arange(w).shape)
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    nextImg = cv2.remap(np_image, flow, None, cv2.INTER_LINEAR)
    if raw_dim==2:
        nextImg = nextImg[:,:,0]
    nextImg = nextImg.astype(raw_dtype)
    return nextImg

def mask_smooth(masks,flows):
    assert  len(masks)==len(flows)+1,"different length"
    smooth_mask_images = []
    for i in range(len(masks)):
        prev = None
        post = None
        for k in range(i - 1, i + 1):
            if k < 0:
                continue
            if k < i:
                prev = advect_numpy_image(masks[k], flows[k])
            if k > i:
                post = advect_numpy_image(masks[k], -flows[k])
        smoothed = masks[i].astype(np.float32)
        count = 1
        if prev is not None:
            smoothed += prev.astype(np.float32)
            count = count + 1
        if post is not None:
            smoothed += post.astype(np.float32)
            count = count + 1
        smoothed = (smoothed / count).astype(np.uint8)
        smooth_mask_images.append(smoothed)
    return smooth_mask_images
