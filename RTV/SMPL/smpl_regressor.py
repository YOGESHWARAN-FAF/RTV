import cv2
import time
import romp
from romp import ROMP
from romp.post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from romp.utils import convert_cam_to_3d_trans, convert_tensor2numpy
from SMPL.smpl_np_base import SMPLModel
from SMPL.projection2screen import projection2screencoord
import torch
import glm
import numpy as np
import bev
from bev import BEV
from SMPL.my_bev import MyBEV
from scipy.spatial.transform import Rotation as R
from util.image_process import blur_image






# settings = romp.main.default_settings
# settings is just a argparse Namespace. To change it, for instance, you can change mode via
# settings.mode='video'
# romp_model = romp.ROMP(settings)
# outputs = romp_model(cv2.imread('./humanbody.png')) # please note that we take the input image in BGR format (cv2.imread).
# dict_keys=list(outputs.keys())
# print(outputs)
class MyROMP(ROMP):
    def __init__(self, romp_settings):
        super(MyROMP, self).__init__(romp_settings)

    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.single_image_forward(image)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
        outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])

        outputs = self.smpl_parser(outputs, root_align=self.settings.root_align)
        outputs.update(body_mesh_projection2image(outputs['joints'], outputs['cam'], vertices=outputs['verts'],
                                                  input2org_offsets=image_pad_info))
        # print(outputs.keys())
        outputs = self.smpl_parser.forward(outputs)

        return outputs  # convert_tensor2numpy(outputs)


def decompose_affine(M):
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    sx = float(np.hypot(M[0, 0], M[0, 1]))
    sy = float(np.hypot(M[1, 0], M[1, 1]))
    s = (sx + sy) * 0.5
    theta = float(np.arctan2(M[1, 0], M[0, 0]))
    return np.array([tx, ty, s, theta], dtype=np.float32)

def compose_affine(params):
    tx, ty, s, theta = params
    cos_t = np.cos(theta) * s
    sin_t = np.sin(theta) * s
    return np.array([[cos_t, -sin_t, tx],
                     [sin_t,  cos_t, ty]], dtype=np.float32)


# https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md
class SMPL_Regressor:
    def __init__(self,use_bev=True, fix_body=False):

        self.regressor_model = self.create_romp_model() if not use_bev else self.create_bev_model(fix_body)
        self.smpl = SMPLModel()
        #self.smpl.beta[0]+=1.2
        self.smpl.beta[1]+=1.2#todo:???
        self.smpl.update()

        # Industry-standard 1-Euro temporal filters for 0-jitter pose and box tracking
        from SMPL.one_euro_filter import OneEuroFilter
        self.cam_trans_filter = OneEuroFilter(min_cutoff=0.6, beta=0.015)
        self.thetas_filter = OneEuroFilter(min_cutoff=0.8, beta=0.020)
        self.trans_filter = OneEuroFilter(min_cutoff=0.7, beta=0.020)
        self.pj2d_filter = OneEuroFilter(min_cutoff=0.8, beta=0.025)
        self.tracked_subject_center = None

    def reset_tracking(self):
        """Reset subject tracking and temporal filter memory."""
        self.tracked_subject_center = None
        for f in [self.cam_trans_filter, self.thetas_filter, self.trans_filter, self.pj2d_filter]:
            if hasattr(f, 'x_prev'):
                f.x_prev = None
            if hasattr(f, 'dx_prev'):
                f.dx_prev = None

    def create_bev_model(self,fix_body=False):
        settings = bev.main.default_settings
        settings.mode = 'video'
        settings.render_mesh = False
        bev_model = MyBEV(settings,fix_body=fix_body)
        return bev_model

    def create_romp_model(self):
        settings = romp.main.default_settings
        settings.temporal_optimize = False
        settings.calc_smpl = True
        settings.render_mesh = False
        settings.smooth_coeff = 3.0
        # settings is just a argparse Namespace. To change it, for instance, you can change mode via
        settings.mode = 'video'
        romp_model = MyROMP(settings)
        return romp_model

    def get_rotation_angle(self, smpl_param):
        if smpl_param is None or 'cam_trans' not in smpl_param or len(smpl_param['cam_trans']) == 0:
            return None
        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(thetas) else 0
        rot_vec = R.from_rotvec(thetas[best_idx, 0:3].cpu().numpy())
        angles = rot_vec.as_euler('XYZ', degrees=False)
        return angles[1]

    def get_thetas(self, smpl_param):
        if smpl_param is None or 'cam_trans' not in smpl_param or len(smpl_param['cam_trans']) == 0:
            return None
        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(thetas) else 0
        return thetas[best_idx].cpu().numpy()

    def get_thin_verts(self, smpl_param):
        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        verts = smpl_param['verts']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(verts) else 0
        verts_tran = verts[best_idx].cpu().numpy()
        theta = thetas[best_idx].cpu().numpy()
        self.smpl.pose = theta
        self.smpl.update()
        v = self.smpl.verts
        v = v - v.mean(0) + verts_tran.mean(0)
        return v

    def get_tshirt_verts(self, smpl_param):
        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        verts = smpl_param['verts']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(verts) else 0
        verts_tran = verts[best_idx].cpu().numpy()
        theta = thetas[best_idx].cpu().numpy().reshape(24, 3)
        theta[[18, 19], :] = 0
        theta = theta.reshape(72)
        self.smpl.pose = theta
        self.smpl.update()
        self.smpl.tshirt.set_params(pose=theta.reshape(24, 3))
        tshirt_verts = self.smpl.tshirt.verts
        tshirt_verts[:, 1] += 0.25
        return tshirt_verts, self.smpl.tshirt.uv, self.smpl.tshirt.faces

    @classmethod
    def get_raw_verts(cls, smpl_param, person_idx=None):
        if smpl_param is None or 'cam_trans' not in smpl_param or len(smpl_param['cam_trans']) == 0:
            return np.zeros((6890, 3), dtype=np.float32)
        thetas = smpl_param['smpl_thetas']
        cam_trans = smpl_param['cam_trans']
        verts = smpl_param['verts']
        if person_idx is None:
            if 'best_person_idx' in smpl_param:
                best_idx = int(smpl_param['best_person_idx'])
            else:
                depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
                best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(verts) else 0
        else:
            best_idx = int(person_idx)
        best_idx = min(best_idx, len(verts) - 1)
        verts_tran = verts[best_idx].clone()
        verts_tran += cam_trans[best_idx]
        verts_tran[:, 2] *= -1
        return verts_tran.cpu().numpy()

    def forward(self, img, roi=False, size=1.2, roi_img_size=512):
        outputs = self.regressor_model.forward(img)
        if outputs is None or 'cam_trans' not in outputs or len(outputs['cam_trans']) == 0:
            return (None, None, None) if roi else None

        num_people = min(len(outputs['cam_trans']), len(outputs.get('verts', outputs['cam_trans'])))
        if num_people == 0:
            return (None, None, None) if roi else None

        pj2d_raw = outputs['pj2d_org']
        cam_trans_raw = outputs['cam_trans']
        h, w = img.shape[:2]
        frame_cx = w * 0.5

        # Persistent Subject Tracking: lock onto primary user and maintain temporal consistency
        best_idx = 0
        if self.tracked_subject_center is not None and num_people > 1:
            best_dist = float('inf')
            for i in range(num_people):
                pj = pj2d_raw[i].cpu().numpy()
                cx = float(pj[:, 0].mean())
                cy = float(pj[:, 1].mean())
                dist = np.hypot(cx - self.tracked_subject_center[0], cy - self.tracked_subject_center[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_dist < 350.0:
                pj_sel = pj2d_raw[best_idx].cpu().numpy()
                new_cx = float(pj_sel[:, 0].mean())
                new_cy = float(pj_sel[:, 1].mean())
                self.tracked_subject_center = [
                    self.tracked_subject_center[0] * 0.85 + new_cx * 0.15,
                    self.tracked_subject_center[1] * 0.85 + new_cy * 0.15
                ]
            else:
                self.tracked_subject_center = None

        if self.tracked_subject_center is None:
            # Initial Subject Selection: prefer central person with closest depth
            best_score = float('inf')
            best_idx = 0
            for i in range(num_people):
                pj = pj2d_raw[i].cpu().numpy()
                cx = float(pj[:, 0].mean())
                cy = float(pj[:, 1].mean())
                dist_center = abs(cx - frame_cx)
                depth_penalty = float(cam_trans_raw[i, 2].item()) * 20.0
                score = dist_center + depth_penalty
                if score < best_score:
                    best_score = score
                    best_idx = i
            pj_sel = pj2d_raw[best_idx].cpu().numpy()
            self.tracked_subject_center = [float(pj_sel[:, 0].mean()), float(pj_sel[:, 1].mean())]

        outputs['best_person_idx'] = best_idx

        # Subject-specific temporal 1-Euro filtering (applied ONLY to tracked person)
        t_now = time.time()
        if 'cam_trans' in outputs:
            raw_ct = outputs['cam_trans'][best_idx:best_idx+1].cpu().numpy()
            filtered_ct = self.cam_trans_filter(raw_ct, t_now)
            outputs['cam_trans'][best_idx:best_idx+1] = torch.from_numpy(filtered_ct).to(outputs['cam_trans'].device)
        if 'smpl_thetas' in outputs:
            raw_th = outputs['smpl_thetas'][best_idx:best_idx+1].cpu().numpy()
            filtered_th = self.thetas_filter(raw_th, t_now)
            outputs['smpl_thetas'][best_idx:best_idx+1] = torch.from_numpy(filtered_th).to(outputs['smpl_thetas'].device)
        if 'pj2d_org' in outputs:
            raw_pj = outputs['pj2d_org'][best_idx:best_idx+1].cpu().numpy()
            filtered_pj = self.pj2d_filter(raw_pj, t_now)
            outputs['pj2d_org'][best_idx:best_idx+1] = torch.from_numpy(filtered_pj).to(outputs['pj2d_org'].device)

        Joints = outputs['pj2d_org'][best_idx].cpu().numpy()
        outputs['target_center'] = (Joints[12] + Joints[9]) * 0.5

        if roi:
            trans2roi, inv_trans2roi = self.get_trans2roi(Joints, s=size, img_size=roi_img_size)
            return outputs, trans2roi, inv_trans2roi
        else:
            return outputs

    def get_face_region(self, outputs):
        cam_trans = outputs['cam_trans']
        pj2d = outputs['pj2d_org']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(pj2d) else 0
        Joints = pj2d[best_idx].cpu().numpy()
        face_center = Joints[15] + (Joints[15] - Joints[12]) * 1.3
        face_radius = np.linalg.norm((Joints[15] + Joints[12]) * 0.5 - face_center) * 0.6
        return face_center, face_radius

    def blur_face(self,raw_img,smpl_param):
        face_center, face_radius = self.get_face_region(smpl_param)
        raw_image = blur_image(raw_img.copy(), face_center, face_radius)
        return raw_image


    def get_trans2roi(self, Joints, s=1.2, img_size=512):
        src = np.zeros([3, 2], np.float32)
        # Center accurately at sternum: between neck (12) and thorax (9), horizontally aligned with shoulders (16, 17)
        sh_center_x = (Joints[16][0] + Joints[17][0]) * 0.5
        center_x = sh_center_x * 0.70 + Joints[12][0] * 0.30
        center_y = Joints[12][1] * 0.35 + Joints[9][1] * 0.65
        center = np.array([center_x, center_y], dtype=np.float32)

        # Robust invariant upper body scale (immune to off-screen pelvis or arm positions)
        sh_width = np.linalg.norm(Joints[16] - Joints[17])
        head_neck = np.linalg.norm(Joints[15] - Joints[12])
        neck_spine = np.linalg.norm(Joints[12] - Joints[9])

        # Primary scale based on upper torso anatomy
        base_size = max(sh_width * 1.5, neck_spine * 3.2, head_neck * 2.8)

        # Check pelvis only if clearly valid and within reasonable human proportions
        head_pelvis = np.linalg.norm(Joints[15] - Joints[0])
        if 0.8 * base_size < head_pelvis < 1.8 * base_size:
            size = (head_pelvis * 0.6 + base_size * 0.4) * s
        else:
            size = base_size * s

        # Accommodate spread arms in upperbody framing so sleeves are never clipped
        if len(Joints) > 21:
            elbow_l, elbow_r = Joints[18], Joints[19]
            wrist_l, wrist_r = Joints[20], Joints[21]
            max_x = max(wrist_l[0], wrist_r[0], elbow_l[0], elbow_r[0], Joints[16][0], Joints[17][0])
            min_x = min(wrist_l[0], wrist_r[0], elbow_l[0], elbow_r[0], Joints[16][0], Joints[17][0])
            lateral_half = max(abs(max_x - center_x), abs(center_x - min_x))
            size = max(size, lateral_half * 1.15)

        src[0, :] = center + np.array([-size, size], np.float32)
        src[1, :] = center + np.array([-size, -size], np.float32)
        src[2, :] = center + np.array([size, -size], np.float32)

        dst = np.zeros([3, 2], np.float32)
        dst[0, :] = np.array([0, img_size - 1], np.float32)
        dst[1, :] = np.array([0, 0], np.float32)
        dst[2, :] = np.array([img_size - 1, 0], np.float32)
        raw_trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        params = decompose_affine(raw_trans)
        filtered_params = self.trans_filter(params, time.time())
        trans = compose_affine(filtered_params)
        inv_trans = cv2.invertAffineTransform(trans)
        return trans, inv_trans

    def get_fullbody_trans2roi(self, smpl_param, s=1.38, new_h=1024, new_w=768):
        if smpl_param is None or 'cam_trans' not in smpl_param or len(smpl_param['cam_trans']) == 0:
            return np.eye(2, 3, dtype=np.float32), np.eye(2, 3, dtype=np.float32)
        cam_trans = smpl_param['cam_trans']
        pj2d = smpl_param['pj2d_org']
        if 'best_person_idx' in smpl_param:
            best_idx = int(smpl_param['best_person_idx'])
        else:
            depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
            best_idx = int(depth_order[0]) if len(depth_order) > 0 and depth_order[0] < len(pj2d) else 0
        best_idx = min(best_idx, len(pj2d) - 1)
        Joints = pj2d[best_idx].cpu().numpy()

        src = np.zeros([3, 2], np.float32)

        neck = Joints[12]
        spine = Joints[9]
        pelvis = Joints[0]
        sh_l, sh_r = Joints[16], Joints[17]
        shoulder_width = np.linalg.norm(sh_l - sh_r)
        neck_spine = np.linalg.norm(neck - spine)
        torso_height = np.linalg.norm(neck - pelvis)

        # Check full body ankle span: Joints 7 and 8 are L_Ankle and R_Ankle
        ankles_mid = (Joints[7] + Joints[8]) * 0.5
        head_ankle_dist = np.linalg.norm(Joints[15] - ankles_mid)
        head_height = np.linalg.norm(Joints[15] - neck)

        # Detect arm elevation using elbows (18, 19) and wrists (20, 21)
        arm_elevation = 0.0
        if len(Joints) > 21:
            elbow_l, elbow_r = Joints[18], Joints[19]
            wrist_l, wrist_r = Joints[20], Joints[21]
            min_arm_y = min(elbow_l[1], elbow_r[1], wrist_l[1], wrist_r[1])
            shoulder_y = min(sh_l[1], sh_r[1])
            if min_arm_y < shoulder_y:
                # Smooth continuous elevation expansion (max 15%, not abrupt 30% jump)
                arm_elevation = min(0.15, max(0.0, (shoulder_y - min_arm_y) / max(1.0, shoulder_width * 2.0)))

        # Always center horizontally between shoulders for 100% accurate left/right tracking
        center_x = (sh_l[0] + sh_r[0]) * 0.5 * 0.75 + neck[0] * 0.25

        # Continuous blend between standing full-body and seated/upper-body framing
        # Eliminates discrete step jumping and shaking when ankles move near visibility boundary
        torso_est = max(torso_height, neck_spine * 2.5, shoulder_width * 1.5)
        size_standing = head_ankle_dist * s
        center_y_standing = (Joints[15][1] + ankles_mid[1]) * 0.5

        size_seated = torso_est * 2.0 * s
        center_y_seated = neck[1] + torso_est * 0.60

        ankles_valid = (Joints[7][1] > pelvis[1] + torso_height * 0.4) and (Joints[8][1] > pelvis[1] + torso_height * 0.4) and not (Joints[7][0] == 0 and Joints[7][1] == 0)
        if not ankles_valid:
            w_standing = 0.0
        else:
            is_standing_score = min(head_ankle_dist / max(1.0, head_height * 4.5), head_ankle_dist / max(1.0, torso_height * 1.8))
            w_standing = float(np.clip((is_standing_score - 0.90) / 0.25, 0.0, 1.0))

        size = size_standing * w_standing + size_seated * (1.0 - w_standing)
        center_y = center_y_standing * w_standing + center_y_seated * (1.0 - w_standing)

        # Accommodate arm elevation by expanding size isotropically (preserves strict 3:4 aspect ratio)
        if arm_elevation > 0.0:
            size = size * (1.0 + arm_elevation * 0.25)

        # Accommodate full lateral arm span across wrists, elbows, and shoulders
        # Guarantees sleeves are NEVER clipped or amputated when opening arms
        aspect_ratio = float(new_w) / float(new_h)
        if len(Joints) > 21:
            elbow_l, elbow_r = Joints[18], Joints[19]
            wrist_l, wrist_r = Joints[20], Joints[21]
            max_x = max(wrist_l[0], wrist_r[0], elbow_l[0], elbow_r[0], sh_l[0], sh_r[0])
            min_x = min(wrist_l[0], wrist_r[0], elbow_l[0], elbow_r[0], sh_l[0], sh_r[0])
            lateral_half_span = max(abs(max_x - center_x), abs(center_x - min_x))
            required_half_w = lateral_half_span * 1.18
            size_for_lateral = (required_half_w / aspect_ratio) * 2.0
            size = max(size, size_for_lateral)

            # Accommodate raised arm elevation (overhead gestures)
            head_top = min(Joints[15][1], min_arm_y)
            if head_top < center_y - (size / 2.0):
                needed_half_h = abs(center_y - head_top) * 1.15
                size = max(size, needed_half_h * 2.0)

        center = np.array([center_x, center_y], dtype=np.float32)

        half_h = size / 2.0
        half_w = half_h * (float(new_w) / float(new_h))
        src[0, :] = center + np.array([-half_w, half_h], np.float32)
        src[1, :] = center + np.array([-half_w, -half_h], np.float32)
        src[2, :] = center + np.array([half_w, -half_h], np.float32)

        dst = np.zeros([3, 2], np.float32)
        dst[0, :] = np.array([0, new_h - 1], np.float32)
        dst[1, :] = np.array([0, 0], np.float32)
        dst[2, :] = np.array([new_w - 1, 0], np.float32)
        raw_trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        params = decompose_affine(raw_trans)
        filtered_params = self.trans_filter(params, time.time())
        trans = compose_affine(filtered_params)
        inv_trans = cv2.invertAffineTransform(trans)
        return trans, inv_trans
