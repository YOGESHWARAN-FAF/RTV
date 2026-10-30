import cv2
import numpy as np
import torch

# https://www.analyticsvidhya.com/blog/2021/07/colour-quantization-using-k-means-clustering-and-opencv/
background = [0.0, 0.0, 0.0]
sleeve_g = [17, 229, 102]
sleeve_b = [3, 62, 229]
body_1 = [24, 227, 220]
body_2 = [206, 228, 199]
body_3 = [172, 90, 229]
body_4 = [94, 131, 229]
body_5 = [127, 229, 68]
body_6 = [229, 112, 182]
back_right_top = [229, 33, 89]
back_right_bottom = [226, 228, 48]
back_left_top = [229, 114, 71]
back_left_bottom = [229, 34, 81]
quantized_colors = [background, sleeve_g, sleeve_b, body_1, body_2, body_3, body_4, body_5, body_6, back_right_top,
                    back_right_bottom, back_left_top, back_left_bottom]


def image_quantization(img: np.array) -> np.array:
    img = img2tensor(img)
    img = img * 255.0
    color_list = torch.from_numpy(np.array(quantized_colors, dtype=np.float32))
    color_tensor = color_list.unsqueeze(2).unsqueeze(3).repeat(1, 1,
                                                               1000,
                                                               1000)
    img_tensor = img.repeat(len(quantized_colors), 1, 1, 1)
    diff = img_tensor - color_tensor  # ncolorx3XHxW
    diff = diff.square().sum(1)  # ncolorxHxW
    vs, indices = torch.min(diff, 0)  # HxW
    # print(vs.shape)
    # print(indices.shape)
    img = img.permute(2, 3, 0, 1).squeeze()
    for i in range(len(quantized_colors)):
        img[indices == i] = color_list[i]
    img = img.permute(2, 0, 1).unsqueeze(0)
    final_img = img/255.0
    final_img = tensor2img(final_img)
    return final_img


def quantimage(image, k):
    image = image * 255
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    final_img = final_img / 255
    return final_img


# [B x 3 x H x W]
def img2tensor(img):
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img


def tensor2img(img):
    img = img.squeeze(0).permute(1, 2, 0)
    return img.numpy()


def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def measurement_quantization(path):
    img = cv2.imread(path)

    img = img.astype(np.float32)
    img = img[:, :, [2, 1, 0]]
    img /= 255.0
    img[0:100, :, :] = 0
    img = torch.from_numpy(img)
    color_norm = img.square().sum(2).sqrt()
    img[color_norm < 0.1] = 0
    raw_img = img
    color_norm = img.abs().sum(2) / 2

    img = tensor2img(rgb2hsv_torch(img2tensor(img.numpy())))
    img[:, :, 2] = 0.9

    img[color_norm < 0.1] = raw_img[color_norm < 0.1]
    img = tensor2img(hsv2rgb_torch(img2tensor(img)))

    # img = quantimage(img, 11)
    img = image_quantization(img)
    img = img2tensor(img)
    return img


