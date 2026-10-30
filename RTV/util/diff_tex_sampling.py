import torch
import numpy as np


def diff_tex_sampling(uv_map: np.ndarray, tex: torch.Tensor) -> torch.Tensor:
    assert tex.shape[2] == 3
    height = uv_map.shape[0]
    width = uv_map.shape[1]
    grid = torch.from_numpy(uv_map.astype(np.float32)/255.0).unsqueeze(0)#B Hout Wout 2
    grid = grid*2.0 -1.0
    input = tex.permute(2,0,1).unsqueeze(0)
    #print(input.shape)
    #print(grid.shape)

    output = torch.nn.functional.grid_sample(input.clone().cuda(), grid.cuda(), mode='bilinear', padding_mode='zeros', align_corners=False)
    return output


if __name__ == '__main__':
    uv_map = np.ones([512,512,2],np.uint8)
    tex = torch.ones((256,256,3))
    img = diff_tex_sampling(uv_map,tex)
    print(img.shape)

