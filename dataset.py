import os

from torch.utils.data import Dataset

from glob import glob
import cv2
import numpy as np

class AnimeSketch(Dataset):
    def __init__(self, dataroot, patch_size, mode='train'):
        super().__init__()
        self.patch_size = patch_size
        if mode == 'train':
            self.filelist = glob(dataroot + '/*.png')[:300]
        else:
            self.filelist = glob(dataroot + '/*.png')[300:]
        print(f"{len(self.filelist)} images found.")

    def __getitem__(self, index):
        file = self.filelist[index]
        sketch = cv2.imread(file, cv2.IMREAD_GRAYSCALE)   # only b&w is considered, no red or blue lines
        patch, attacked = self.crop_lines(sketch, self.patch_size)
        if attacked:
            print("========being attacked==========")
        
        # opened, noise_mask = self.add_blobs(patch)
        opened, mask = self.add_lines(patch)
        
        # cv2.imwrite('./imgs/patch.png', patch)
        # cv2.imwrite('./imgs/opened.png', opened)
        # cv2.imwrite('./imgs/mask.png', (mask * 255).astype(np.uint8))
        
        return {'gt': patch[None, ...] / 255, 
                'opened': opened[None, ...] / 255, 
                'mask': mask[None, ...] / 255, 
                'attack': attacked
                }
    
    def __len__(self):
        return len(self.filelist)
    
    def crop_lines(self, mat, patch_size):
        """
            Returns a patch that contains black strokes.
        """
        h, w = mat.shape
        assert patch_size <= h
        if patch_size == h:
            return mat
        
        attacked= True
        budget = 200
        while budget:
            x, y = np.random.randint(0, h-patch_size), np.random.randint(0, w-patch_size)
            cropped = mat[x: x + patch_size, y: y + patch_size]
            if patch_size * patch_size * 255 * 0.9 < np.sum(cropped) < patch_size * patch_size * 255 * 0.97:
                attacked = False
                return cropped, attacked
            budget -= 1
        return cropped, attacked
    
    def add_blobs(self, mat, kernel_size=[5, 3], kernel_nums=[10, 10]):
        """
            Simulate opening of strokes by adding white blobs as noise.
            mat: size([64, 64])
            :kernel_size para: sizes of square blur kernel,
            :kernel_nums para: how many times will the blurs happen for each kernel
        """
        mask = np.zeros_like(mat)
        for bk_size, num in zip(kernel_size, kernel_nums):
            for _ in range(num):
                x = np.random.randint(0, mat.shape[0] - bk_size)
                y = np.random.randint(0, mat.shape[1] - bk_size)
                
                mask[x : x + bk_size, y : y + bk_size] = 255
            
        return cv2.bitwise_or(mat, mask), mask // 255
    
    def add_lines(self, mat, line_widths=[1, 2], line_numbers=[10, 3], mode="strike"):
        """
            Simulate openings of strokes by adding white lines as noise.
        """
        assert len(line_widths) == len(line_numbers)
        h, w = mat.shape
        
        dst = mat.copy()
        if mode == "strike":    ### lines go through the entile image
            budget = 100
            while np.sum(dst - mat) == 0 and budget > 0:
                budget -= 1
                mask = np.zeros_like(mat)
                for lw, ln in zip(line_widths, line_numbers):
                    for _ in range(ln):
                        edge = np.random.randint(4)
                        if edge == 0:  # Top edge
                            start_x, start_y = np.random.randint(0, w), 0
                            end_x, end_y = np.random.randint(0, w), h
                        elif edge == 1:  # Right edge
                            start_x, start_y = w, np.random.randint(0, h)
                            end_x, end_y = 0, np.random.randint(0, h)
                        elif edge == 2:  # Bottom edge
                            start_x, start_y = np.random.randint(0, w), h
                            end_x, end_y = np.random.randint(0, w), 0
                        else:  # Left edge
                            start_x, start_y = 0, np.random.randint(0, h)
                            end_x, end_y = w, np.random.randint(0, h)
                        # Draw the line on the mask
                        cv2.line(mask, (start_x, start_y), (end_x, end_y), (255,), lw)
                dst = cv2.bitwise_or(mat, mask)
        elif mode == "free":    ### some pieces of lines scatter randomly, need more line_numbers
            for lw, ln in zip(line_widths, line_numbers):
                for _ in range(ln):
                    start_x, start_y = np.random.randint(0, w), np.random.randint(0, h)
                    end_x, end_y = np.random.randint(0, w), np.random.randint(0, h)
                    
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), (255,), lw)
                dst = cv2.bitwise_or(mat, mask)
                            
        return dst, mask // 255