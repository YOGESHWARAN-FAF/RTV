import torchvision.transforms as T
import torch
import torchvision.transforms.functional as F
from torch.nn import functional as FF
import numpy as np
import cv2

def compute_trans(height,width,ret):
    angle, translate, scale, shear = ret
    center = [0, 0]  # [width * 0.5, height * 0.5]
    matrix = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
    matrix = torch.tensor(matrix).float()
    matrix = matrix.reshape(2, 3)
    matrix[0, 2] /= (height // 2)
    matrix[1, 2] /= (width // 2)

    return matrix

def compute_inv_trans(height,width,ret):
    matrix = compute_trans(height,width,ret)
    inv_R = torch.inverse(matrix[:, :2])
    t = matrix[:, 2]

    inv_matrix = torch.zeros(2, 3)
    inv_matrix[:, :2] = inv_R
    inv_matrix[:, 2] = -torch.mv(inv_R, t)
    return inv_matrix

class RandomAffineBatch(T.RandomAffine):
    def __int__(self, *args):
        super(RandomAffineBatch, self).__init__(*args)

    def forward(self, imgs):
        channels, height, width = F.get_dimensions(imgs[0])
        img_size = [width, height]  # flip for keeping BC on get_params call
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        results = []
        for img in imgs:
            fill = self.fill
            channels, height, width = F.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            results.append(F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center))
        return results

    def forward_with_trans(self, imgs):
        channels, height, width = F.get_dimensions(imgs[0])
        img_size = [width, height]  # flip for keeping BC on get_params call
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        results = []
        for img in imgs:
            fill = self.fill
            channels, height, width = F.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            results.append(F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center))
        trans = compute_trans(height, width, ret)
        if torch.cuda.is_available():
            trans = trans.cuda()
        return results, trans

    def forward_with_inv_trans(self, imgs):
        channels, height, width = F.get_dimensions(imgs[0])
        img_size = [width, height]  # flip for keeping BC on get_params call
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        results = []
        for img in imgs:
            fill = self.fill
            channels, height, width = F.get_dimensions(img)
            if isinstance(img, torch.Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * channels
                else:
                    fill = [float(f) for f in fill]
            results.append(F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center))
        inv_trans = compute_inv_trans(height,width,ret)
        if torch.cuda.is_available():
            inv_trans = inv_trans.cuda()
        return results, inv_trans



class RandomAffineBatchNumpy:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, imgs):
        if isinstance(imgs, list):
            img_list=imgs
        else:
            img_list=[imgs,]
        h,w=img_list[0].shape[:2]
        assert h==w
        img_size = h
        random_matrix = RandomAffineMatrix(degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear, img_size=img_size)
        trans=random_matrix()
        result_list=[]
        for img in img_list:
            result = cv2.warpAffine(img, trans, (img_size,img_size),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
            result_list.append(result)
        if len(result_list)==1:
            return result_list[0]
        else:
            return result_list

class RandomAffineMatrix:
    def __init__(self, degrees, translate=None, scale=None, shear=None, img_size=1024):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.img_size=img_size

    def __call__(self):
        trans=np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        R_hat,t_hat=self.get_random_affine_params(self.degrees, self.translate, self.scale, self.shear)
        new_trans=self.deform(R_hat,t_hat,trans)
        return new_trans

    def batch_forward(self, trans_list):
        R_hat,t_hat=self.get_random_affine_params(self.degrees, self.translate, self.scale, self.shear)
        new_list =[]
        for trans in trans_list:
            new_trans=self.deform(R_hat,t_hat,trans)
            new_list.append(new_trans)
        return new_list

    def deform(self,R_hat,t_hat,trans):
        c = np.array([self.img_size/2, self.img_size/2])
        R = trans[:, :2]
        t = trans[:, 2]
        R_new = np.dot(R_hat, R)
        t_new = np.dot(R_hat, t - c) + t_hat + c
        new_trans = np.concatenate((R_new, t_new[:, None]), axis=1)
        return new_trans

    def get_random_affine_params(self, degrees, translate=None, scale=None, shear=None):
        # Random rotation angle
        angle = np.random.uniform(-degrees, degrees)
        angle_rad = np.deg2rad(angle)
        img_size = self.img_size

        # Random translation
        if translate is not None:
            max_dx = translate[0] * img_size
            max_dy = translate[1] * img_size
            tx = np.random.uniform(-max_dx, max_dx)
            ty = np.random.uniform(-max_dy, max_dy)
        else:
            tx, ty = 0, 0

        # Random scaling
        if scale is not None:
            scale_factor = np.random.uniform(scale[0], scale[1])
        else:
            scale_factor = 1.0

        # Random shear
        if shear is not None:
            shear_x = np.random.uniform(shear[0], shear[1])
            shear_y = np.random.uniform(shear[2], shear[3]) if len(shear) > 2 else 0
        else:
            shear_x, shear_y = 0, 0

        # Compute the affine transformation matrix
        cos_theta = np.cos(angle_rad) * scale_factor
        sin_theta = np.sin(angle_rad) * scale_factor
        shear_x_rad = np.deg2rad(shear_x)
        shear_y_rad = np.deg2rad(shear_y)

        # Create the affine transformation matrix
        M = np.array([
            [cos_theta + np.tan(shear_y_rad) * sin_theta, -sin_theta + np.tan(shear_y_rad) * cos_theta],
            [sin_theta + np.tan(shear_x_rad) * cos_theta, cos_theta + np.tan(shear_x_rad) * sin_theta]
        ])

        # Translation vector
        t = np.array([tx, ty])

        return M, t
