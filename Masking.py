import cv2
import matplotlib.pyplot as plt
import torch

def masking(feats, mask, keys=['keypoints', 'keypoint_scores', 'descriptors']):
    index = 0
    new_feats = {}
    for key in keys:
        feats[key] = feats[key].detach().cpu().tolist()
        new_feats[key] = []
    
    for index in range(len(feats[keys[0]][0])):
        y, x = feats[keys[0]][0][index]
        if mask[int(x), int(y)] == 255:
            for key in keys:
                new_feats[key].append(feats[key][0][index])
    
    for key in keys:
        new_feats[key] = torch.tensor(new_feats[key]).unsqueeze(0).cuda()
    
    del feats
    return new_feats
