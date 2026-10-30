import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp, RotationSpline
from scipy.linalg import polar
import numpy as np

class OfflineFilter:
    def __init__(self):
        self.prev_param = None
        self.current_param = None
        self.param_list = []

    def __len__(self):
        return len(self.param_list)

    def append(self,smpl_param):
        if smpl_param is None:
            if self.prev_param is None:
                return
            self.current_param = self.prev_param
        else:
            self.prev_param = self.current_param
            self.current_param = smpl_param
        self.param_list.append(self.current_param)

    def offline_smooth(self):
        length = len(self.param_list)
        pos_list = []
        for i in range(length):
            #print(np.array(self.param_list[i]['cam_trans'][0],dtype=np.float32))
            pos_list.append(np.array(self.param_list[i]['cam_trans'][0],dtype=np.float32))
        avg_pos_list = []
        radius = 10
        for i in range(length):
            count = 0
            sum = np.zeros(3,np.float32)
            for j in range(i-radius,i+radius+1):
                if length>j>=0:
                    sum+=pos_list[j]
                    count+=1
            avg_pos_list.append(sum/count)
        for i in range(length):
            self.param_list[i]['cam_trans'] = avg_pos_list[i]

    def get(self, idx):
        return self.param_list[idx]

    def get_trans(self, idx):
        return self.param_list[idx]['cam_trans']

class PoseFilter:
    def __init__(self):
        self.prev_param = None
        self.current_param = None
        self.trans_avg = Average(2)

    def append(self, smpl_param):
        if smpl_param is None:
            self.current_param = self.prev_param
        else:
            self.prev_param = self.current_param
            self.current_param = smpl_param
        self.trans_avg.append(self.current_param['cam_trans'][0])

    def blend(self, p1, p2):
        alpha = 0.05
        result = copy.deepcopy(p1)
        result['cam_trans'] = alpha * p1['cam_trans'] + (1 - alpha) * p2['cam_trans']
        result['smpl_betas'] = alpha * p1['smpl_betas'] + (1 - alpha) * p2['smpl_betas']
        alpha2 = 0.2
        #result['smpl_thetas'] = self.slerp_batch(p1, p2, alpha2)
        result['smpl_thetas']=p1['smpl_thetas']#todo
        return result

    def slerp_batch(self, v1, v2, t):
        v1 = v1['smpl_thetas'][0].reshape(24, 3)
        v2 = v2['smpl_thetas'][0].reshape(24, 3)


        rs = []
        for i in range(24):
            r1 = R.from_euler('XYZ', v1[i], degrees=False).as_matrix()
            r2 = R.from_euler('XYZ', v2[i], degrees=False).as_matrix()
            res = r1*t + r2*(1-t)
            #u, _ = polar(res)

            r = R.from_matrix(res).as_euler('XYZ',degrees=False)
            r = np.expand_dims(r,0)
            rs.append(r)
        result = np.concatenate(rs,axis=0).reshape(1, 72)
        return result

        '''
        
       
        # a /= np.sqrt((a ** 2).sum(-1))[..., np.newaxis]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        # if np.linalg.norm(v1 - v2) < 0.0001:
        #    return v1
        # 解决插值绕远路的问题：https://blog.csdn.net/weixin_46477226/article/details/121258542
        r2[np.sum(r1 * r2, 1) < 0] *= -1
        alpha0 = np.arccos(np.sum(r1 * r2, 1))

        alpha = alpha0.copy()
        alpha[alpha0 < 0.0001] = 0.0001
        alpha = alpha.repeat(4).reshape(24, 4)

        result = np.zeros([24, 4])
        result[alpha0 >= 0.0001] = np.sin(t * alpha[alpha0 >= 0.0001]) / np.sin(alpha[alpha0 >= 0.0001]) * r1[
            alpha0 >= 0.0001] + np.sin((1 - t) * alpha[alpha0 >= 0.0001]) / np.sin(alpha[alpha0 >= 0.0001]) * r2[
                                       alpha0 >= 0.0001]
        result[alpha0 < 0.0001] = r1[alpha0 < 0.0001] * t + r2[alpha0 < 0.0001] * (1 - t)
        result /= np.sqrt((result ** 2).sum(-1))[..., np.newaxis]

        return R.from_quat(result).as_euler('XYZ', degrees=False).reshape(1, 72)
        '''


    def get(self):
        result = copy.deepcopy(self.current_param)
        result['cam_trans'] = self.trans_avg.get()
        return result

class Average:
    def __init__(self,size=3):
        self.size=size
        self.data = []

    def append(self,v):
        if len(self.data) >= self.size:
            self.data.pop(0)
        self.data.append(v)

    def get(self):
        sum = self.data[0]
        for i in range(len(self.data)):
            if i==0:
                continue
            sum = sum+self.data[i]
        return sum/len(self.data)


def slerp(v1, v2, t):
    t = 1 - t
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    if np.linalg.norm(v1 - v2) < 0.0001:
        return v1
    # 解决插值绕远路的问题：https://blog.csdn.net/weixin_46477226/article/details/121258542
    if np.dot(v1, v2) < 0:
        v1 = -v1
    alpha = np.arccos(np.dot(v1, v2))
    result = np.sin((1 - t) * alpha) / np.sin(alpha) * v1 + np.sin(t * alpha) / np.sin(alpha) * v2
    result /= np.linalg.norm(result)
    return result
