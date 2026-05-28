import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'C:/Users/User/.gemini/antigravity-ide/brain/673b7c94-8e7c-44a1-a0f1-b2b9489d2d81/.tempmediaStorage/media_673b7c94-8e7c-44a1-a0f1-b2b9489d2d81_1779872061286.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    # Save a visualization of red and blue masks
    red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
    blue_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
    
    vis = np.zeros_like(img)
    vis[red_mask] = [0, 0, 255] # Red
    vis[blue_mask] = [255, 0, 0] # Blue
    
    cv2.imwrite("outputs/pixel_masks_vis.png", vis)
    print("Saved pixel masks visualization to outputs/pixel_masks_vis.png")
