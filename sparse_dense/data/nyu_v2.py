import os
import torch
import torch.utils.data as data
import numpy as np
from torchvision.datasets.folder import default_loader
import cv2
from PIL import Image

def _make_dataset(dir, gridSize):
    images = []
    s1s = []
    s2s = []
    gt_depths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            image_path = os.path.join(root, fname)
            if os.path.isfile(image_path ) and 'colors.png' in fname:
                depth_fname = fname.replace("colors.", "depth.")
                depth_path = os.path.join(root, depth_fname)
                s1_fname = fname.replace("colors.", "s1_"+str(gridSize)+".")
                s1_path = os.path.join(root, s1_fname)
                s2_fname = fname.replace("colors.", "s2_"+str(gridSize)+".")
                s2_path = os.path.join(root, s2_fname)
                if os.path.isfile(depth_path) and os.path.isfile(s1_path) and os.path.isfile(s2_path):
                    images.append(image_path)
                    gt_depths.append(depth_path)
                    s1s.append(s1_path)
                    s2s.append(s2_path)
    return images, gt_depths, s1s, s2s

class NYU_V2(data.Dataset):

    def __init__(self, root, gridSize,
                 transform=None,
                 download=False,
                 loader=default_loader):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.gridSize = gridSize

        if download:
            self.download()

        self.imgs, self.depths, self.s1s, self.s2s = _make_dataset(os.path.join(self.root), self.gridSize)

    def __getitem__(self, index):
        path2img = self.imgs[index]
        path2depth = self.depths[index]
        path2s1 = self.s1s[index]
        path2s2 = self.s2s[index]
        img = cv2.imread(path2img)
        s1 = cv2.imread(path2s1, -1)
        s1_meters = float(1.0e-3)*s1
        s2 = cv2.imread(path2s2, -1)
        s2_float = float(1.0)*s2
        gt_depth = cv2.imread(path2depth, -1)
        gt_depth_meters = 1e-3*gt_depth

        rgbs1s2 = None
        if self.transform is not None:
            img = self.transform(img)
            s1_meters = self.transform(s1_meters[:,:,None].astype(np.float32))
            s2_float = self.transform(s2_float[:,:,None].astype(np.float32))
            gt_depth_meters = self.transform(gt_depth_meters[:,:,None].astype(np.float32))
            rgbs1s2 = torch.cat((img, s1_meters, s2_float), 0)

        return rgbs1s2, gt_depth_meters

    def __len__(self):
        return len(self.imgs)

    def download(self):
        # TODO: please download the dataset from
        raise NotImplementedError
