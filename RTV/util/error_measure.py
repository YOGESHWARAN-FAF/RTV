import numpy as np

from util.json_writer import json_wrtiter


class ErrorMeasure:
    def __init__(self):
        self.err_list = []
        self.err0_list = []
        self.err1_list = []
        self.err2_list = []
        self.err3_list = []
        self.err4_list = []
        self.err5_list = []

    def append(self, pred, truth):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if truth.dim() == 1:
            truth = truth.unsqueeze(0)
        self.err_list.append(self.get_err(pred, truth))
        self.err0_list.append(self.get_err(pred[:, 0], truth[:, 0]))
        self.err1_list.append(self.get_err(pred[:, 1], truth[:, 1]))
        self.err2_list.append(self.get_err(pred[:, 2], truth[:, 2]))
        self.err3_list.append(self.get_err(pred[:, 3], truth[:, 3]))
        self.err4_list.append(self.get_err(pred[:, 4], truth[:, 4]))
        self.err5_list.append(self.get_err(pred[:, 5], truth[:, 5]))

    def get_err(self, a, b):
        return (a - b).detach().cpu().square().mean().item()

    def save_to_json(self, path):
        # rotational angle,  left arm roll, right arm roll, left arm pitch, right arm pitch, and body size
        data_dict = {
            "total_error": np.mean(self.err_list),
            "rotational_angle_error": np.mean(self.err0_list),
            "left_arm_roll_error": np.mean(self.err1_list),
            "right_arm_roll_error": np.mean(self.err2_list),
            "left_arm_pitch_error": np.mean(self.err3_list),
            "right_arm_pitch_error": np.mean(self.err4_list),
            "body_size_error": np.mean(self.err5_list),
        }
        json_wrtiter(data_dict, path)
