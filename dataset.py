import os

from torch.utils.data import Dataset

from glob import glob
import cv2
import numpy as np

class AnimeSketch(Dataset):
    def __init__(self, dataroot, patch_size=256, mode='train', color=False):
        super().__init__()
        self.patch_size = patch_size
        self.color = color
        if mode == 'train':
            self.filelist = glob(dataroot + '/*.png')[:300]
        else:
            self.filelist = glob(dataroot + '/*.png')[300:]
        print(f"{len(self.filelist)} images found at {dataroot}.")

    def __getitem__(self, index):
        file = self.filelist[index]
        
        if self.color:
            sketch = cv2.imread(file)
        else:
            sketch = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            sketch = cv2.threshold(sketch, 240, 255, cv2.THRESH_BINARY)[1]
            
        patch, attacked = self.crop_patch(sketch, self.patch_size)
        
        opened, mask = self.add_lines(patch)
        
        # cv2.imwrite('./imgs/patch.png', patch)
        # cv2.imwrite('./imgs/opened.png', opened)
        # cv2.imwrite('./imgs/mask.png', (mask * 255).astype(np.uint8))
        
        if self.color:
            gt = patch.transpose(2, 0, 1) / 255   # [3, 256, 256]
            opened = opened.transpose(2, 0, 1) / 255
        else:
            gt = patch[None, ...] / 255     # [1, 256, 256]
            opened = opened[None, ...] / 255
        mask = mask[None, ...] / 255

        return {'gt': gt, 'opened': opened, 'mask': mask, 'attack': attacked}
    
    def __len__(self):
        return len(self.filelist)
    
    def crop_patch(self, mat, patch_size):
        """
            Returns a patch that contains strokes.
        """
        
        if self.color:
            h, w, _ = mat.shape
        else:
            h, w = mat.shape
        
        assert patch_size <= h
        if patch_size == h:
            return mat
        
        attacked= True
        budget = 200
        while budget:
            x, y = np.random.randint(0, h-patch_size), np.random.randint(0, w-patch_size)
            cropped = mat[x: x + patch_size, y: y + patch_size]
            if self.color:
                condition = patch_size * patch_size * 255 * 0.9 * 3 < np.sum(cropped) < patch_size * patch_size * 255 * 0.97 * 3
            else:
                condition = patch_size * patch_size * 255 * 0.9 < np.sum(cropped) < patch_size * patch_size * 255 * 0.97
            
            if condition:
                attacked = False
                return cropped, attacked
            budget -= 1
        
        return cropped, attacked
    
    def add_lines(self, mat, line_widths=[1, 2], line_numbers=[10, 0], mode="strike"):
        """
            Simulate openings of strokes by adding white lines as noise.
        """
        assert len(line_widths) == len(line_numbers)
        if self.color:
            h, w, ch = mat.shape
        else:
            h, w = mat.shape
        
        dst = mat.copy()
        if mode == "strike":    ### lines go through the entile image
            budget = 100
            while np.sum(dst - mat) == 0 and budget > 0:
                budget -= 1
                mask = np.zeros((h, w))
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
                if self.color:
                    for c in range(ch):
                        dst[..., c] = cv2.bitwise_or(np.array(mat[..., c]).astype(np.uint8), np.array(mask).astype(np.uint8))
                else:
                    dst = cv2.bitwise_or(np.array(mat).astype(np.uint8), np.array(mask).astype(np.uint8))
        elif mode == "free":    ### some pieces of lines scatter randomly, need more line_numbers
            for lw, ln in zip(line_widths, line_numbers):
                for _ in range(ln):
                    start_x, start_y = np.random.randint(0, w), np.random.randint(0, h)
                    end_x, end_y = np.random.randint(0, w), np.random.randint(0, h)
                    
                    cv2.line(mask, (start_x, start_y), (end_x, end_y), (255,), lw)
                dst = cv2.bitwise_or(mat, mask)
                            
        return dst, mask // 255