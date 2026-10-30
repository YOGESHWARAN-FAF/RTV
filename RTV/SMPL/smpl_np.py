import pickle
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
#import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import torch
from util.obj_io import save_obj
from .amputated_smpl import AmputatedSMPL
from SMPL.tshirt_smpl.TshirtModelVis import TshirtModelVis, TshirtModelHashTexVis, TshirtModelPhongVis
import glm
import cv2
from .projection2screen import projection2screencoord


class SMPLModel():
    def __init__(self, model_path='./SMPL/model.pkl'):
        """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.update()
        self.amputated_smpl = AmputatedSMPL()
        self.renderer = None
        self.tshirt = TshirtModelVis()
        self.tshirt_hashtex = TshirtModelHashTexVis()
        self.tshirt_phong = TshirtModelPhongVis()

    def set_params(self, pose=None, beta=None, trans=None):
        """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
    Called automatically when parameters are updated.

    """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        self.J = self.J_regressor.dot(self.verts)

    def rodrigues(self, r):
        """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
        #self.pose = np.zeros(self.pose_shape)
        #self.beta = np.zeros(self.beta_shape)
        #self.trans = np.zeros(self.trans_shape)
        self.update()
        save_obj(path, self.verts, self.faces)
        print(self.verts)

    def save_A_to_obj(self, path):

        #self.pose = np.zeros(self.pose_shape)
        #self.beta = np.zeros(self.beta_shape)
        #self.trans = np.zeros(self.trans_shape)
        rr = R.from_euler('XYZ', [0, 0, -45], degrees=True)
        rr = rr.as_rotvec()
        rr2 = R.from_euler('XYZ', [0, 0, 45], degrees=True)
        rr2 = rr2.as_rotvec()
        self.pose[17, 0] = rr2[0]
        self.pose[17, 1] = rr2[1]
        self.pose[17, 2] = rr2[2]
        self.pose[16, 0] = rr[0]
        self.pose[16, 1] = rr[1]
        self.pose[16, 2] = rr[2]
        self.update()
        save_obj(path, self.verts, self.faces)
        print(self.verts)

    def save_shapes(self):
        for i in range(10):
            save_obj('./shape'+str(i).zfill(2)+'.obj',self.shapedirs[:,:,i]+self.v_template,self.faces)






    def render_amputated_align(self, smpl_param,height=512,width=512):

        triangles = smpl_param['smpl_face'].cpu().numpy().astype(np.int32)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        vertices = smpl_param['verts_camed_org'][depth_order].cpu().numpy()
        verts_tran = (smpl_param['verts'] + cam_trans.unsqueeze(1))[depth_order]
        vertices[:, :, 2] = vertices[:, :, 2] * -1
        verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
        model = glm.mat4(1)
        view = glm.mat4(1)
        projection = glm.perspective(np.pi / 3, width / height, 0.1, 1000.0)

        colorimg = self.amputated_smpl.render_base(verts_tran[0].cpu().numpy(), model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()),height=height,width=width)
        Joints = self.J_regressor.dot(verts_tran)


        return colorimg

    def render_amputated_centered(self, smpl_param,raw_height,raw_width,height=512,width=512,scale_factor=1.0):

        triangles = smpl_param['smpl_face'].cpu().numpy().astype(np.int32)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        vertices = smpl_param['verts_camed_org'][depth_order].cpu().numpy()
        verts_tran = (smpl_param['verts'] + cam_trans.unsqueeze(1))[depth_order]
        vertices[:, :, 2] = vertices[:, :, 2] * -1
        verts_tran[:, :, 2] = verts_tran[:, :, 2] * -1
        verts_tran = verts_tran[0].cpu().numpy()

        Joints = self.J_regressor.dot(verts_tran)
        center = Joints[6]
        #verts_tran = verts_tran - center#centralized
        dist = abs(center[2])
        #verts_tran[:,2] = -10
        model = glm.mat4(1)
        view = glm.translate(glm.vec3(-center[0], -center[1]+0.05, -center[2]-3.75))
        #projection = glm.perspective(np.pi / 3, width / height, 0.1, 1000.0)
        projection = glm.perspective(np.pi / 12, width / height, 0.1, 1000.0)

        colorimg = self.amputated_smpl.render(verts_tran, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()),height=height,width=width)
        trans2align = self.get_transform_center2align(Joints, raw_height, raw_width,scale_factor)

        return colorimg, trans2align

    def get_transform_center2align(self,Joints,height,width,scale_factor=1.0):
        center = Joints[6]*0.6 + Joints[3]*0.4
        p0 = np.array([0.5,0.5,0],np.float32) +center
        p1 = np.array([-0.5, 0.5, 0], np.float32) +center
        p2 = np.array([-0.5, -0.5, 0], np.float32) +center
        tp0 = np.array([0.5, 0.5, 0], np.float32)*scale_factor + center
        tp1 = np.array([-0.5, 0.5, 0], np.float32)*scale_factor + center
        tp2 = np.array([-0.5, -0.5, 0], np.float32)*scale_factor + center
        h=512
        w=512
        view = glm.translate(glm.vec3(-center[0], center[1] - 0.05, -center[2] - 3.75))
        # projection = glm.perspective(np.pi / 3, width / height, 0.1, 1000.0)
        projection = glm.perspective(np.pi / 12, w / h, 0.1, 1000.0)
        src = np.zeros([3,2],np.float32)
        src[0,:] = projection2screencoord(p0,np.array(view.to_list()),np.array(projection.to_list()),h,w)
        src[1, :] = projection2screencoord(p1, np.array(view.to_list()), np.array(projection.to_list()), h, w)
        src[2, :] = projection2screencoord(p2, np.array(view.to_list()), np.array(projection.to_list()), h, w)

        view = glm.mat4(1)
        projection = glm.perspective(np.pi / 3, width / height, 0.1, 1000.0)
        dst = np.zeros([3, 2], np.float32)
        dst[0, :] = projection2screencoord(tp0, np.array(view.to_list()), np.array(projection.to_list()), height, width)
        dst[1, :] = projection2screencoord(tp1, np.array(view.to_list()), np.array(projection.to_list()), height, width)
        dst[2, :] = projection2screencoord(tp2, np.array(view.to_list()), np.array(projection.to_list()), height, width)
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def render_amputated_by_param(self, smpl_param,height=512,width=512):

        theta = smpl_param['smpl_thetas'][0].reshape(24, 3)
        beta = smpl_param['smpl_betas'][0]
        self.pose = theta
        self.beta = beta
        # self.trans = -smpl_param['cam'][0]
        self.update()

        self.verts[:, 2] = -self.verts[:, 2]

        ### initialize camera
        # sx, sy, tx, ty = [1.0, 1.0, 0.0, -0.2]
        # sx, sy, tx, ty = [1.6, 1.75, 0, 0.02]
        depth = smpl_param['cam_trans'][2]

        sx, sy, tx, ty = [1.5 / depth, 1.5 / depth, smpl_param['cam_trans'][0], smpl_param['cam_trans'][1]]


        tx, ty, tz = smpl_param['cam_trans']

        scale = 1.0
        model = glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(tx, 1.3 * ty, 0)) * glm.translate(
            glm.vec3(0, 0, -0.95 * tz))
        #glm::perspective(aspect, (GLfloat)WIDTH/(GLfloat)HEIGHT, 0.1f, 100.0f)
        projection = glm.perspective(np.pi / 3, width/height, 0.1, 1000.0)

        colorimg = self.amputated_smpl.render(self.verts, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()),height=height,width=width)

        return colorimg



    def render_fixed_amputated_by_paramliu(self, smpl_param):

        theta = smpl_param['smpl_thetas'][0].reshape(24, 3)
        beta = smpl_param['smpl_betas'][0]

        romp_pose = smpl_param['body_pose'][0,:]
        global_pose = smpl_param['global_orient'][0,:]
        self.pose=np.zeros(self.pose_shape)
        self.pose[1:,:]=romp_pose.reshape([23,3])
        self.pose[0,1] = global_pose[2]
        #self.pose = theta
        self.beta = beta

        # self.trans = -smpl_param['cam'][0]
        self.update()

        self.verts[:, 2] = -self.verts[:, 2]

        scale = 2.6
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 1.1, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        #view = glm.translate(glm.vec3(0, 0, -10))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)

        colorimg = self.amputated_smpl.render(self.verts, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()))


        return colorimg

    def render_amputated_from_mannequin(self, path):
        self.pose = mannequin2SMPL(path)
        self.update()

        ### initialize camer

        scale = 2.6
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        colorimg = self.amputated_smpl.render(self.verts, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()))

        return colorimg

    def render_thin_amputated_from_mannequin(self, path):
        self.pose = mannequin2SMPL(path)
        #self.beta[0] = 1.2
        self.beta[1] = 1.2
        self.update()

        ### initialize camer

        scale = 2.6
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        colorimg = self.amputated_smpl.render(self.verts, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()))

        return colorimg

    def render_tshirt_from_mannequin(self, path, semantic=False):
        pose = mannequin2SMPL(path)
        self.tshirt.set_params(pose=pose)
        ### initialize camer

        scale = 2.4
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        colorimg = self.tshirt.render(model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()), semantic=semantic)

        return colorimg

    def render_tshirt_hashtex_from_mannequin(self, path, ratio=None, pose=None):
        if pose is None:
            pose = mannequin2SMPL(path)
        self.tshirt_hashtex.set_params(pose=pose)
        ### initialize camer

        '''
        scale = 2.4
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)
        '''
        scale = 2.4
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -5.0))  * glm.translate(glm.vec3(0, -0.05, 0))
        projection = glm.perspective(np.pi / 6, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        colorimg = self.tshirt_hashtex.render(model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()),ratio=ratio)

        return colorimg

    def render_tshirt_phong_from_mannequin(self, path, pose=None):
        if pose is None:
            pose = mannequin2SMPL(path)
        self.tshirt_phong.set_params(pose=pose)
        ### initialize camer

        '''
        scale = 2.4
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)
        '''
        scale = 2.4
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -5.0))  * glm.translate(glm.vec3(0, -0.05, 0))
        projection = glm.perspective(np.pi / 6, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        colorimg = self.tshirt_phong.render(model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()))

        return colorimg

    def render_uv_thin_amputated_from_mannequin(self, path):
        self.pose = mannequin2SMPL(path)
        #self.beta[0] = 1.2
        self.beta[1] = 1.2
        self.update()

        ### initialize camer

        scale = 2.6
        model = glm.translate(glm.vec3(0, 0.15, 0)) * glm.scale(glm.vec3(scale, scale, scale))
        view = glm.translate(glm.vec3(0, 0, -10))  # * glm.rotate(np.pi, (0, 1.0, 0))
        projection = glm.perspective(np.pi / 12, 1.0, 0.1, 1000.0)

        self.verts[:, 1] = -self.verts[:, 1]
        uvimg = self.amputated_smpl.render_uv(self.verts, model=np.array(model.to_list()),
                                              view=np.array(view.to_list()), proj=np.array(projection.to_list()))

        return uvimg



    def get_pose_from_mannequin(self, path):
        return mannequin2SMPL(path)



    def get_n0_a(self, file_name):
        args = file_name.split('/')[-1].split('_')

        n0 = float(args[0])
        n1 = float(args[1])
        n2 = float(args[2])
        n3 = float(args[3])
        n4 = float(args[4])

        a_lp = 60 - float(n1) * 150 / 1000
        a_lr = 75 - float(n2) * 84 / 1000
        a_rp = 60 - float(n3) * 150 / 1000
        a_rr = -75 + float(n4) * 84 / 1000
        a_global1 = n0 * 120.0 / 1524 - 80.157
        a_global2 = 160.0 * (n0 - 256.0) / (1780 - 256) - 80
        a_global = (a_global1 + a_global2) / 2

        return n0, a_global

    def draw_2D_joints(self, image, camera, one_channel=False):
        thorax = self.J[12] * 0.5 + self.J[9] * 0.5
        rShoulder = self.J[17]*0.9 + self.J[16]*0.1
        lShoulder = self.J[16]*0.9 + self.J[17]*0.1
        lElbow = self.J[18]
        rElbow = self.J[19]
        lHip = self.J[2] + (self.J[1] - self.J[2]) * 1.2  # + (self.J[0] - self.J[3]) * 0.3
        rHip = self.J[1] + (self.J[2] - self.J[1]) * 1.2  # + (self.J[0] - self.J[3]) * 0.3
        pelvis = (self.J[1] + self.J[2]) * 0.5
        joints = [thorax, rShoulder, lShoulder, lElbow, rElbow, lHip, rHip, pelvis]
        joints_2d = [camera.projection_to_screen(joint, 512, 512) for joint in joints]
        image = np.zeros(image.shape,np.uint8)
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

        return image, joints_2d


# parse file url for SMPL pose
def mannequin2SMPL(file_name):
    args = file_name.split('/')[-1].split('_')

    n0 = float(args[0])
    n1 = float(args[1])
    n2 = float(args[2])
    n3 = float(args[3])
    n4 = float(args[4])
    #front-view A pose: n0=1018 , n1=400, n2=357, n3=400, n4=357

    a_lp = 60 - float(n1) * 150 / 1000
    a_lr = 75 - float(n2) * 84 / 1000
    a_rp = 60 - float(n3) * 150 / 1000
    a_rr = -75 + float(n4) * 84 / 1000
    a_global1 = n0 * 120.0 / 1524 - 80.157
    a_global2 = 160.0 * (n0 - 256.0) / (1780 - 256) - 80
    a_global = a_global1  # (a_global1 + a_global2) / 2
    alpha_min = -70.0
    alpha_max = 70.0
    a_global = (n0 - 256) * (alpha_max - alpha_min) / (1780 - 256) + alpha_min

    pose = np.zeros([24, 3], dtype=np.float64)  # smpl.pose_shape

    #rr = R.from_euler('XYZ', [a_rp, 0, -a_rr], degrees=True)
    rr = R.from_euler('XZX', [a_rp, -a_rr, -a_rp], degrees=True)
    rr = rr.as_rotvec()
    pose[17, 0] = rr[0]
    pose[17, 1] = rr[1]
    pose[17, 2] = rr[2]

    #rl = R.from_euler('XYZ', [a_lp, 0, -a_lr], degrees=True)
    rl = R.from_euler('XZX', [a_lp, -a_lr, -a_lp], degrees=True)
    rl = rl.as_rotvec()
    pose[16, 0] = rl[0]
    pose[16, 1] = rl[1]
    pose[16, 2] = rl[2]

    rg = R.from_euler('XYZ', [0, a_global, 0], degrees=True)
    rg = rg.as_rotvec()
    # pose[0, 0] = np.pi
    pose[0, 0] = 0
    pose[0, 1] = rg[1]
    pose[0, 2] = 0

    return pose
