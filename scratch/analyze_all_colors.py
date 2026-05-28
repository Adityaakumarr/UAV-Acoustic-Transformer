import cv2
import numpy as np

img_path = 'C:/Users/User/.gemini/antigravity-ide/brain/673b7c94-8e7c-44a1-a0f1-b2b9489d2d81/.tempmediaStorage/media_673b7c94-8e7c-44a1-a0f1-b2b9489d2d81_1779872061286.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red mask (red has two ranges in Hue)
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    
    # Blue mask
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    red_coords = np.argwhere(red_mask)
    blue_coords = np.argwhere(blue_mask)
    
    print(f"HSV Red pixels count: {len(red_coords)}")
    print(f"HSV Blue pixels count: {len(blue_coords)}")
    
    if len(red_coords) > 0:
        print(f"HSV Red pixels bounding box:")
        print(f"  ({red_coords[:, 0].min()}, {red_coords[:, 1].min()}) to ({red_coords[:, 0].max()}, {red_coords[:, 1].max()})")
    if len(blue_coords) > 0:
        print(f"HSV Blue pixels bounding box:")
        print(f"  ({blue_coords[:, 0].min()}, {blue_coords[:, 1].min()}) to ({blue_coords[:, 0].max()}, {blue_coords[:, 1].max()})")
        
    # Let's save these masks as images to visually check
    cv2.imwrite("outputs/hsv_red_mask.png", red_mask)
    cv2.imwrite("outputs/hsv_blue_mask.png", blue_mask)
