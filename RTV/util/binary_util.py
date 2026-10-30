import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import interpolate

def multiple_random_dilation(image,n):
    isbool=False
    if image.dtype==bool:
        isbool=True
        image=image.astype(np.uint8)*255
    k=5#size of kernel
    smooth_field=get_smooth_2d()
    smooth_field[image>0]=1
    smooth_field=np.expand_dims(smooth_field,2)
    smooth_field = np.expand_dims(smooth_field, 3)

    smooth_field=smooth_field[:,:,[0]*(2*k+1),:]
    smooth_field = smooth_field[:, :, :,[0] * (2 * k + 1)]
    for _ in range(n):
        image=random_dilate(image,k,weight=smooth_field)
    if isbool:
        image=(image/255).astype(bool)
    return image

def random_dilate(image,k, weight):
    image=image.astype(np.float32)/255
    kernel=make_kernel(k)
    pad_image = np.pad(image, k, 'edge')
    areas = np.lib.stride_tricks.as_strided(pad_image, image.shape + (2*k+1, 2*k+1), pad_image.strides * 2)
    areas=areas*kernel
    areas=areas*weight
    result=(np.mean(areas, axis=(2, 3))>0.2)|(image>0)
    return result.astype(np.uint8)*255


def dilate(image, boundary='edge'):
    k=10
    kernel=make_kernel(k)
    pad_image = np.pad(image, k, boundary)
    areas = np.lib.stride_tricks.as_strided(pad_image, image.shape + (2*k+1, 2*k+1), pad_image.strides * 2)
    areas=areas*kernel
    return np.max(areas, axis=(2, 3))

def make_kernel(k):
    kernel = np.ones((2*k+1,2*k+1),np.uint8)
    for i in range(2*k+1):
        for j in range(2*k+1):
            x=i-(k)
            y=j-(k)
            if x*x+y*y>(k+0.5)*(k+0.5):
                kernel[i,j]=0
    return kernel

def get_smooth_2d():
    # Define the size of the image
    width, height = 8, 8
    x = np.array(range(width))
    y = np.array(range(height))
    # Generate a random 2D array
    random_image = np.random.rand(height, width)
    f = interpolate.interp2d(x, y, random_image, kind='linear')

    xnew = np.linspace(0, width, 512)
    ynew = np.linspace(0, height, 512)
    random_image = f(xnew, ynew)

    # Apply Gaussian filter for smoothness
    smooth_image = gaussian_filter(random_image, sigma=32)
    return smooth_image