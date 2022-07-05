#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import math

class ValidMask(object):
    BEV_W = 768
    BEV_H = 896
    FOV_L = 30
    FOV_R = 30
    FOV_C = 0
    MSK = None #np.array, 1 for valid, 0 for invalid

    def __init__(self):
        pass

    def generate(self, h=896, w=768, fov_l=30, fov_r=30):
        self.BEV_H = h
        self.BEV_W = w
        self.FOV_L = fov_l
        self.FOV_R = fov_r

        # suppose we have fov center straight towards right
        scale_l = math.tan(self.FOV_L*math.pi/180.0)
        scale_r = math.tan(self.FOV_R*math.pi/180.0)
        self.MSK = np.ones((self.BEV_H, self.BEV_W), dtype=np.float32) #np.float32 or np.uint8
        ori_x = 0.0
        ori_y = self.BEV_H / 2.0
        for y in range(self.BEV_H):
            for x in range(self.BEV_W):
                left_edge = ori_y - math.fabs(x-ori_x) * scale_l
                right_edge = ori_y + math.fabs(x-ori_x) * scale_r
                # let left_edge be solid, right_edge be dash
                if y < left_edge:
                    self.MSK[y, x] = 0
                elif y >= right_edge:
                    self.MSK[y, x] = 0
        return self.MSK

    def save(self):
        self.MSK *= 255
        cv2.imwrite("valid_mask.png", self.MSK)

if __name__ == '__main__':
    ValidMsk = ValidMask()
    ValidMsk.generate(h=896, w=768, fov_l=40, fov_r=20)
    ValidMsk.save()