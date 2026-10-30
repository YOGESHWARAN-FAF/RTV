import numpy as np

class TransSmooth:
    def __init__(self):
        self.trans_list = []
        self.smooth_trans_list = []

    def __len__(self):
        return len(self.trans_list)

    def append(self,trans):
        self.trans_list.append(trans)

    def offline_smooth(self):
        length = len(self.trans_list)
        radius = 6
        for i in range(length):
            count = 0
            sum = np.zeros((2,3))
            for j in range(i - radius, i + radius + 1):
                if length > j >= 0:
                    sum += self.trans_list[j]
                    count += 1
            self.smooth_trans_list.append(sum / count)

    def get(self,i):
        return self.smooth_trans_list[i]

