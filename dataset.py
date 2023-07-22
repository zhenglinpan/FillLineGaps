import os

from torch.utils.data import Dataset

from glob import glob
import cv2
import numpy as np

class AnimeSketch(Dataset):
    def __init__(self, dataroot, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.filelist = glob(dataroot + '/*.png')[:100]
        print(f"{len(self.filelist)} images found.")

    def __getitem__(self, index):
        file = self.filelist[index]
        sketch = cv2.imread(file, cv2.COLOR_BGR2GRAY)   # only b&w is acceptable, so no red or blue lines
        try:
            sketch_patch, attacked = self.crop(sketch, self.patch_size)
            if attacked:
                print("========being attacked==========")
        except:
            print(f"Exception on =============> {file}")
        
        opened, noise_mask = self.add_noise(sketch_patch)
        line_mask = (255 - sketch_patch) // 255
        
        # Ckeck Efficacy
        # cv2.imwrite('./imgs/sketch_patch.png', sketch_patch)
        # cv2.imwrite('./imgs/opened.png', opened)
        # cv2.imwrite('./imgs/line_mask.png', (line_mask * 255).astype(np.uint8))
        # cv2.imwrite('./imgs/noise_mask.png', (noise_mask * 255).astype(np.uint8))
        
        return {'gt': sketch_patch[None, ...], 'opened': opened[None, ...], 'mask': line_mask[None, ...], 'attack': attacked}
    
    def __len__(self):
        return len(self.filelist)
    
    def crop(self, mat, patch_size):
        h, w = mat.shape
        assert patch_size <= h
        if patch_size == h:
            return mat
        
        attacked= True
        budget = 100
        while budget:
            x, y = np.random.randint(0, h-patch_size), np.random.randint(0, w-patch_size)
            cropped = mat[x: x + patch_size, y: y + patch_size]
            if np.sum(cropped) < patch_size * patch_size * 255:
                attacked = False
                return cropped, attacked
            budget -= 1
        return cropped, attacked
        
    
    def add_noise(self, mat, blur_kernels=[5, 3], alphas=[10, 10]):
        """
            mat: size([64, 64])
            :blur_kernels para: sizes of square blur kernel,
            :alpha para: how many times will the blurs happen for each kernel
        """
        inv_mat = 255 - mat
        mask = np.ones_like(mat)
        for bk_size, alpha in zip(blur_kernels, alphas):
            for _ in range(alpha):
                x = np.random.randint(0, mat.shape[0] - bk_size)
                y = np.random.randint(0, mat.shape[1] - bk_size)
                
                mask[x : x + bk_size, y : y + bk_size] = 0
            
        return 255 - (inv_mat * mask), mask
            
        