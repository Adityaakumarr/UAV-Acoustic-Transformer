import cv2
import numpy as np

img_path = 'C:/Users/User/.gemini/antigravity-ide/brain/673b7c94-8e7c-44a1-a0f1-b2b9489d2d81/.tempmediaStorage/media_673b7c94-8e7c-44a1-a0f1-b2b9489d2d81_1779872061286.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    # BGR format in OpenCV
    # Red pixels: B is low, G is low, R is high
    red_mask = (img[:, :, 2] > 200) & (img[:, :, 1] < 50) & (img[:, :, 0] < 50)
    # Blue pixels: B is high, G is low, R is low
    blue_mask = (img[:, :, 0] > 200) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
    
    red_coords = np.argwhere(red_mask)
    blue_coords = np.argwhere(blue_mask)
    
    print(f"Number of red pixels: {len(red_coords)}")
    print(f"Number of blue pixels: {len(blue_coords)}")
    
    if len(red_coords) > 0:
        print(f"Red pixels bounding box (y_min, x_min) to (y_max, x_max):")
        print(f"  ({red_coords[:, 0].min()}, {red_coords[:, 1].min()}) to ({red_coords[:, 0].max()}, {red_coords[:, 1].max()})")
    if len(blue_coords) > 0:
        print(f"Blue pixels bounding box (y_min, x_min) to (y_max, x_max):")
        print(f"  ({blue_coords[:, 0].min()}, {blue_coords[:, 1].min()}) to ({blue_coords[:, 0].max()}, {blue_coords[:, 1].max()})")
