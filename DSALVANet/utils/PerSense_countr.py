import cv2
import numpy as np
import torch
import matplotlib.pyplot  as plt
import math
import os
from sys import argv

def apply_scoremap(image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=np.float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def vis_countr(src_img, boxes, pre_cnt, test_idx, density_pred):

        
        gray = cv2.cvtColor((density_pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        output = apply_scoremap(src_img, density_pred)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        for box in boxes:
            y1, x1, y2, x2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # cv2.putText(output, "Result:{0}".format(int(pre_cnt)), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
        return output, pre_cnt, gray