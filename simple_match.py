import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image , rbd
from lightglue.viz2d import plot_images , plot_matches
from Masking import masking

def match(img0,img1,basedir = "",if_mask = False,mode = "SUPERPOINT"):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (mode == "SUPERPOINT"):
        # SuperPoint+LightGlue
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
        matcher = LightGlue(features='superpoint',threshold=0.2).eval().to(device)  # load the matcher
        # matcher.compile(mode='reduce-overhead')

    elif (mode == "DISK"):
        # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
        extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
        matcher = LightGlue(features='disk',threshold=0.2).eval().to(device)  # load the matcher
        # matcher.compile(mode='reduce-overhead')

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image(img0).to(device)
    image1 = load_image(img1).to(device)

    transform = transforms.Compose([transforms.Resize((1024, 1024))])
    if(image0.size(1)>1200):
        image0 = transform(image0)

    if(image1.size(1)>1200):
        image1 = transform(image1)

    image0 = image0[:,:1024,:1024]
    image1 = image1[:,:1024,:1024]
    # image0 = transform(image0)
    # image1 = transform(image1)
    # print(image0.size())
    # print(image1.size())

    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # mask process
    if(if_mask):
        mask = cv2.imread(os.path.join(basedir,"descentimgs","mask.tif"),0)[:1024,:1024]
        feats0 = masking(feats0,mask)
        feats1 = masking(feats1,mask)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    # print(matches)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    matched_img = draw_matches(image0, image1, points0, points1)

    del image0,image1,extractor,matcher

    return matched_img , points0 , points1

def draw_matches(img0, img1, points0, points1):

    plot_images([img0,img1])
    plot_matches(points0,points1,color="lime",lw=0.2)
    fig = plt.gcf()
    # plt.show()
    fig.canvas.draw()
    matched_image = np.array(fig.canvas.renderer.buffer_rgba())
    # cv2.imshow("test",matched_image)
    return matched_image

if __name__ == "__main__":
    match("../ce4/images+bestbaseimg/bestbaseimg/ce4split_03_4.jpg","../ce4/images+bestbaseimg/descentimages/ce4_tiny_02760.jpg")