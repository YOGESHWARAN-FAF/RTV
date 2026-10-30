import pyrender
import numpy as np

class PerspectiveCamera:
    def __init__(self,model,view,projection):
        self.model = model.transpose()
        self.view = view.transpose()
        self.projection = projection.transpose()
        self.transform = self.projection@self.view@self.model#np.matmul(self.projection,self.view,self.model)



    def projection_to_screen(self, v, width=512, height=512):
        point = np.array([v[0], v[1], v[2], 1.0], dtype=v.dtype)
        projected_point = self.transform@np.expand_dims(point,1)

        projected_point /= projected_point[3]
        x, y = projected_point[0], projected_point[1]
        #print('---------')
        #print(x, y)
        id_x = int(((x + 1) / 2) * width)
        id_y = int(((1 - y) / 2) * height)
        #print(id_x, id_y)
        return id_x, id_y


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation
        self.P = self.get_projection_matrix()

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

    def projection_to_screen(self, v, width=512, height=512):
        point = np.array([v[0], v[1], v[2], 1.0], dtype=v.dtype)
        projected_point = self.P.dot(point)
        projected_point /= projected_point[3]
        x, y = projected_point[0], projected_point[1]

        id_x = int(((x + 1) / 2) * width)
        id_y = int(((1 - y) / 2) * height)

        return id_x, id_y
