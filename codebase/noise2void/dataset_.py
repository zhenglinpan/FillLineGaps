import sys
sys.path.append('../..')
from dataset import AnimeSketch

import copy as cp
import cv2
import numpy as np
from numpy.random import randint

class AnimeSketchNoise2Void(AnimeSketch):   # inherit from AnimeSketch
    def __init__(self, dataroot, patch_size, kernel):
        super().__init__(dataroot, patch_size)
        self.patch_size = patch_size
        self.kernel = kernel
        
    def __getitem__(self, index):
        file = self.filelist[index]
        sketch = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        patch, attacked = self.crop(sketch, self.patch_size)
        if attacked:
            patch_blind, mask = patch, np.zeros_like(patch)
        else:
            patch_blind, bspot_position, attacked = self.blind_spot(patch, kernel=self.kernel)
        
        if attacked:
            patch_blind, mask = patch, np.zeros_like(patch)
        else:
            mask = np.zeros_like(patch_blind)
            mask[bspot_position[0]: bspot_position[0]+self.kernel, bspot_position[1]: bspot_position[1]+self.kernel] = 1.0
            
        return {"clear": patch[None, ...], "blind": patch_blind[None, ...], "mask": mask[None, ...], "attack": attacked}
        
    def blind_spot(self, mat, kernel):
        dst = cp.deepcopy(mat)
        k = kernel  # size of the blind spot
        h, w = dst.shape
        bcx, bcy = randint(0, h - k), randint(0, w - k)   # blind spot centre
       
        budget = 100
        while budget:   # repeate until caught lines
            if np.sum(dst[bcx: bcx + k, bcy: bcy + k]) == k * k * 255:  # no lines caught
                bcx, bcy = randint(0, h - k), randint(0, w - k)
            else:
                dst[bcx: bcx + k, bcy: bcy + k] = 255
                return dst, (bcx, bcy), False   # successfully break a line, no attack
            budget -= 1
        
        return dst, (bcx, bcy), True    # failed, attacking