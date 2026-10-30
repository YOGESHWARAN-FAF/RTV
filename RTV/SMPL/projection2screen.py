import numpy as np
import glm

def projection2screencoord(v,view,projection,height,width,hw=False):

    view = view.transpose()
    projection = projection.transpose()
    transform = projection @ view
    point = np.array([v[0], v[1], v[2], 1.0], dtype=v.dtype)
    projected_point = transform @ np.expand_dims(point, 1)
    #print(projection)

    projected_point /= projected_point[3]
    x, y = projected_point[0], projected_point[1]
    # print('---------')
    # print(x, y)
    #print(x)
    id_x = ((x + 1) / 2) * width
    id_y = ((1 - y) / 2) * height
    #print(id_x.shape)
    if not hw:
        return np.array([id_x[0], id_y[0]], v.dtype)
    else:
        return np.array([id_y[0], id_x[0]], v.dtype)

if __name__ == '__main__':
    width=600
    height=800
    view = glm.mat4(1.0)
    projection = glm.perspective(np.pi / 3, width / height, 0.1, 1000.0)
    v=np.array([1/np.sqrt(3),-1/np.sqrt(3),-1],np.float32)
    print(projection2screencoord(v,np.array(view.to_list()), np.array(projection.to_list()), height, width,True))