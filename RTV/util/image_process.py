import cv2
import numpy as np
import torch


def blur_image_by_mask(image, mask, kernel_size=23):
    mask=mask.astype(np.uint8)


    # 对图像和掩码进行位与运算，得到只有目标区域的图像
    masked = image.copy()#cv2.bitwise_and(image, image, mask=mask)

    # 对这个图像进行高斯模糊
    blurred = cv2.GaussianBlur(masked, (kernel_size, kernel_size), 0)

    # 用位或运算将模糊后的图像和原图像合并
    #final = cv2.bitwise_or(image, blurred)
    final = image.copy()
    final[mask>0] = blurred[mask>0]
    return final

def blur_image(image, center, radius):
    # 创建一个和图像大小相同的掩码
    mask = np.zeros(image.shape[:2], dtype="uint8")
    center = center.astype(int)

    # 在掩码上画出想要模糊的区域，例如一个圆形
    #cv2.circle(mask, (center[0], center[1]), int(radius), 255, -1)
    cv2.ellipse(mask, (center[0], center[1]), (int(radius), int(radius*1.5)), 0,0,360, 255, -1)


    # 对图像和掩码进行位与运算，得到只有目标区域的图像
    masked = image.copy()#cv2.bitwise_and(image, image, mask=mask)

    # 对这个图像进行高斯模糊
    blurred = cv2.GaussianBlur(masked, (23, 23), 0)

    # 用位或运算将模糊后的图像和原图像合并
    #final = cv2.bitwise_or(image, blurred)
    final = image.copy()
    final[mask>0] = blurred[mask>0]
    return final

import torchvision.transforms as transforms

def poisson_blend_np(src:np.ndarray,dst:np.ndarray,mask:np.ndarray)->np.ndarray:
    """
        * inputs:
            - input (numpy array, required) shape = (H, W, 3).
            - target (numpy array, required) shape = (H, W, 3).

            - mask (np.bool, required) shape = (H, W).
                    Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
        * returns:
                    Output image numpy of shape (H, W, 3) inpainted with poisson image editing method.
        """
    mask=np.expand_dims(mask,2)
    mask=(np.concatenate([mask,mask,mask],2)*255).astype(np.uint8)
    dstimg = dst
    srcimg = src
    msk=mask
    # compute mask's center
    xs, ys = [], []
    for j in range(msk.shape[0]):
        for k in range(msk.shape[1]):
            if msk[j, k, 0] == 255:
                ys.append(j)
                xs.append(k)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
    dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
    out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
    return out



def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 3, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = input.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(input[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret