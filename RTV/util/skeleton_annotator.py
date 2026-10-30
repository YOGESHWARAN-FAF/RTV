import sys
import tools.init_paths
from core.inference import get_max_preds
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch


def GetJointCoord(heatmaps: torch.Tensor):
    preds, maxvals = get_max_preds(heatmaps.cpu().detach().numpy())
    xs = []
    ys = []
    colors = []
    top = np.array((0, 0, 1.0))
    bottom = np.array((1.0, 0, 0))
    for i in range(len(maxvals[0])):
        xs.append(preds[0][i][0])
        ys.append(preds[0][i][1])
        colors.append(maxvals[0][i] * top + (1 - maxvals[0][i]) * bottom)
        colors[i] = np.multiply(colors[i], 255).astype(np.uint8)
    return xs, ys, colors


def SkeletonAnnotation(image: np.array, heatmaps: torch.Tensor, color=(255, 255, 255), one_channel=True) -> np.array:
    xs, ys, _ = GetJointCoord(heatmaps)
    thorax = [xs[7], ys[7]]
    rShoulder = [xs[12], ys[12]]
    lShoulder = [xs[13], ys[13]]
    lElbow = [xs[14], ys[14]]
    rElbow = [xs[11], ys[11]]
    lHip = [xs[3], ys[3]]
    rHip = [xs[2], ys[2]]
    pelvis = [xs[6], ys[6]]
    joints_2d = [thorax, rShoulder, lShoulder, lElbow, rElbow, lHip, rHip, pelvis]
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    def connect(i, j, color=(255, 255, 255)):
        draw.line(joints_2d[i] + joints_2d[j], fill=color, width=10)

    if one_channel:
        connect(0, 1)
        connect(0, 2)
        connect(2, 3)
        connect(1, 4)
        connect(0, 7)
        connect(5, 7)
        connect(6, 7)
    else:
        connect(0, 1, (255, 0, 0))
        connect(0, 2, (0, 255, 0))
        connect(2, 3, (0, 0, 255))
        connect(1, 4, (0, 255, 255))
        connect(0, 7, (255, 0, 255))
        connect(5, 7, (255, 255, 0))
        connect(6, 7, (125, 125, 255))
    for x, y in joints_2d:
        r = 5
        draw.ellipse(((x - r, y - r), (x + r, y + r)), fill=(255, 255, 255))
    image = np.array(img)
    return image


def SkeletonTensor(heatmaps: torch.Tensor) -> torch.Tensor:
    xs, ys, _ = GetJointCoord(heatmaps)
    thorax = [xs[7], ys[7]]
    rShoulder = [xs[12], ys[12]]
    lShoulder = [xs[13], ys[13]]
    lElbow = [xs[14], ys[14]]
    rElbow = [xs[11], ys[11]]
    lHip = [xs[3], ys[3]]
    rHip = [xs[2], ys[2]]
    pelvis = [xs[6], ys[6]]
    joints_2d = [thorax, rShoulder, lShoulder, lElbow, rElbow, lHip, rHip, pelvis]
    image = np.zeros((512, 512, 3), np.uint8)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    def connect(i, j, color=(255, 255, 255)):
        draw.line(joints_2d[i] + joints_2d[j], fill=color, width=10)

    connect(0, 1)
    connect(0, 2)
    connect(2, 3)
    connect(1, 4)
    connect(0, 7)
    connect(5, 7)
    connect(6, 7)

    for x, y in joints_2d:
        r = 5
        draw.ellipse(((x - r, y - r), (x + r, y + r)), fill=(255, 255, 255))
    image = np.array(img)
    image = image.astype(np.float32)
    image = image/255
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    image = image[[0]]
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    return image
