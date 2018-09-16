import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
import png
import cv2
import itertools

from scipy import ndimage as ndi
from skimage.morphology import watershed

def ComputeFeatures(img):

    # featureExtractor = cv2.ORB_create(nfeatures=512, fastThreshold=10)
    # kps = featureExtractor.detect(img, None)

    # mask_features = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    # featureExtractor = cv2.xfeatures2d.FREAK_create()
    # (kps, dess) = featureExtractor.detectAndCompute(img, mask_features)

    minHessian = 400
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    kps = detector.detect(img)

    return kps

def process_s1_s2(img, depth, img_path, gridSize=0):
    print("Starting computation of S1 and S2 for " + img_path)

    #compute features
    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    if gridSize == 0:
        kps = ComputeFeatures(img)
        for kp in kps:
            mask_img[int(kp.pt[1]), int(kp.pt[0])] = 255

        features_rgb = img
        for kp in kps:
            cv2.drawKeypoints(img, kps, features_rgb)
        feature_path = img_path.replace("colors.", "features.")
        cv2.imwrite(feature_path, features_rgb)
    else:
        for x in range(0, depth.shape[0], int(depth.shape[0]/gridSize)):
            for y in range(0, depth.shape[1], int(depth.shape[1]/gridSize)):
                mask_img[x, y] = 255

    mask_depth = np.zeros((depth.shape[0], depth.shape[1]))
    s1 = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint16)

    for x in range(depth.shape[0]):
        for y in range(depth.shape[1]):
            feat_val = mask_img[x,y]
            depth_val = depth[x,y]
            if feat_val > 0 and depth_val == 0:
                nearest = np.zeros((2))
                nearest_dist = 10000000
                new_depth_val = 0
                for xx in range(depth.shape[0]):
                    for yy in range(depth.shape[1]):
                        new_depth_val = depth[xx,yy]
                        if new_depth_val > 0:
                            dist = np.linalg.norm(np.asarray([xx-x, yy-y]))
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest = np.asarray([xx, yy])
                if new_depth_val > 0:
                    mask_img[x,y] = 0
                    mask_img[nearest[0], nearest[1]] = 255
                    mask_depth[nearest[0], nearest[1]] = depth_val
            elif feat_val > 0:
                mask_depth[x,y] = depth_val;

    inverse_mask= 255 - mask_img
    s2 = cv2.distanceTransform(inverse_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    seedMask = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.int32)
    seedDepth = []
    cc_idx = 0
    for x in range(mask_img.shape[0]):
        for y in range(mask_img.shape[1]):
            feat_val = mask_img[x, y]
            if feat_val > 0:
                seedMask[x,y] = cc_idx+1
                seedDepth.append(depth[x,y])
                cc_idx += 1

    seedMask = watershed(s2, seedMask, mask=None)

    s1 = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint16)
    for x in range(depth.shape[0]):
        for y in range(depth.shape[1]):
            seed_idx = seedMask[x,y]
            if seed_idx > 0:
                s1[x,y] = seedDepth[seed_idx-1]

    s1_path = img_path.replace("colors.", "s1_" + str(gridSize) + ".")
    cv2.imwrite(s1_path, s1)

    s2_int16 = s2.astype(np.uint16)
    s2_path = img_path.replace("colors.", "s2_" + str(gridSize) + ".")
    cv2.imwrite(s2_path, s2_int16)

    scaleAbsS1 = cv2.convertScaleAbs(s1, alpha=255./s1.max(), beta=0.)
    watershedMask = cv2.applyColorMap(scaleAbsS1, cv2.COLORMAP_JET)
    watershed_path = img_path.replace("colors.", "watershed_" + str(gridSize) + ".")
    cv2.imwrite(watershed_path, watershedMask)

    scaleAbsS2 = cv2.convertScaleAbs(s2_int16, alpha=255./s2_int16.max(), beta=0.)
    s2_visualise_path = img_path.replace("colors.", "s2_visualize_" + str(gridSize) + ".")
    cv2.imwrite(s2_visualise_path, scaleAbsS2)

    print("Computed S1 and S2 for " + img_path)


def create_s1_s2(dir):
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            image_path = os.path.join(root, fname)
            if os.path.isfile(image_path ) and 'colors.png' in fname:
                depth_fname = fname.replace("colors.", "depth.")
                depth_path = os.path.join(root, depth_fname)
                if os.path.isfile(depth_path):
                    img = cv2.imread(image_path)
                    depth = cv2.imread(depth_path, -1)

                    process_s1_s2(img, depth, image_path)
                    # process_s1_s2(img, depth, image_path, 32)
                    process_s1_s2(img, depth, image_path, 64)
                    # process_s1_s2(img, depth, image_path, 128)


create_s1_s2('./NYU_V2/training')
