import cv2
import numpy as np
# from util.garment_heatmap import HeatmapGenerator
import torch
import torchvision.transforms as transforms
import ffmpeg
#from OpticalFlow.optical_flow import OpticalFlow

class VideoLoader:
    def __init__(self, path):
        self.path = path
        self.frames = self.load_video()
        self.min_h = 0
        self.min_w = 0
        self.max_h = self.frameHeight
        self.max_w = self.frameWidth
        self.crop2square()
        self.l = 0
        self.r = 0
        self.u = 0
        self.d = 0
        if self.frameHeight > self.frameWidth:
            self.l = (self.frameHeight - self.frameWidth) // 2
            self.r = self.l
        # self.heatmap_gen = HeatmapGenerator()
        self.post_transform = transforms.Resize((512, 512))
        self.opt_flow = None
        #self.optical_flow = OpticalFlow()



    def crop2square(self):
        if self.frameWidth > self.frameHeight:
            offset = (self.frameWidth - self.frameHeight) // 2
            self.min_w = offset
            self.max_w = offset + self.frameHeight

    def __getitem__(self, idx):
        im = self.get_image(idx)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resize = transforms.Resize((384, 288))

        all_transforms = transforms.Compose([normalize, resize])

        # with torch.no_grad():
        #    heatmaps = self.heatmap_gen.model(all_transforms(im))

        # heatmaps = self.post_transform(heatmaps)
        return im  # , heatmaps

    def __len__(self):
        return self.frames.shape[0]

    def set_bbox(self, min_h, min_w, max_h, max_w):
        self.min_h = min_h
        self.min_w = min_w
        self.max_h = max_h
        self.max_w = max_w

    def set_padding(self, l, r, u, d):
        self.l = l
        self.r = r
        self.u = u
        self.d = d

    def get_image(self, idx):
        frame = self.get_numpy_image(idx)
        img = torch.from_numpy(frame) / 255.0
        img = img.permute(2, 0, 1)  # CHW, BGR
        if torch.cuda.is_available():
            img = img.cuda()
        img = img.unsqueeze(0)
        img = self.post_transform(img)
        return img

    def get_numpy_image(self, idx):
        frame = self.frames[idx]
        frame = frame[self.min_h:self.max_h, self.min_w:self.max_w, :]
        if self.l > 0:
            left = np.zeros((frame.shape[0], self.l, frame.shape[2]), np.uint8)
            frame = np.concatenate((left, frame), 1)
        if self.r > 0:
            right = np.zeros((frame.shape[0], self.r, frame.shape[2]), np.uint8)
            frame = np.concatenate((frame, right), 1)
        if self.u > 0:
            up = np.zeros((self.u, frame.shape[1], frame.shape[2]), np.uint8)
            frame = np.concatenate((up, frame), 0)
        if self.d > 0:
            down = np.zeros((self.d, frame.shape[1], frame.shape[2]), np.uint8)
            frame = np.concatenate((frame, down), 0)
        frame = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        return frame

    def get_raw_numpy_image(self, idx):
        frame = self.frames[idx]
        return frame

    def get_heatmap(self, idx):
        _, heatmaps = self.__getitem__(idx)
        return heatmaps

    def get_motor(self, idx):
        return torch.zeros(6).cuda() if torch.cuda.is_available() else torch.zeros(6)

    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)
        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotate_code = None
        rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
        return round(int(rotate) / 90.0) * 90

    def load_video(self):
        # rotateCode = self.check_rotation(self.path)
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened(), self.path+":video load failed!"
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_height=2048#todo:
        need_resize=False
        if self.frameHeight>max_height:
            need_resize=True
            tmp=self.frameHeight
            self.frameHeight=max_height
            self.frameWidth=int(self.frameWidth*self.frameHeight/tmp)

        fc = 0
        ret = True

        frame_list = []

        while (fc < self.frameCount and ret):
            ret, temp = cap.read()
            if temp is None:
                break
            buff = np.empty((1, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
            # print(fc,temp.shape)
            buff = temp
            if need_resize:
                buff=cv2.resize(buff,(self.frameWidth,self.frameHeight))
            buff = np.expand_dims(buff, 0)

            frame_list.append(buff)
            fc += 1
        frames = np.concatenate(frame_list, 0)
        #n, h, w = frames.shape

        cap.release()
        return frames

    '''
        def compute_opt_flow(self):
        print("Start computing optical flow")
        opt_flow_list = []
        for i in range(self.__len__()-1):
            with torch.no_grad():
                opt_flow = self.optical_flow(self.frames[i],self.frames[i+1]).cpu()
            opt_flow_list.append(opt_flow)
        self.opt_flow=opt_flow_list
        print("Finish computing optical flow:",self.opt_flow.__len__())
    '''


if __name__ == '__main__':
    path = './videos/garment_test.mov'
    video_loader = VideoLoader(path)
    print(video_loader.frames.shape)
    print(len(video_loader))
    import matplotlib.pyplot as plt

    video_loader.set_bbox(0, 180, 720, 1280 - 180)
    plt.imshow(video_loader[200])
    plt.show()
