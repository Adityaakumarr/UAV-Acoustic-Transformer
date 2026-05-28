import cv2
import numpy as np

img_path = 'outputs/subplots/subplot_1_left.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load Subplot 1")
else:
    print(f"Loaded Subplot 1 shape: {img.shape}")
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
    
    print(f"Red pixels: {len(red_coords)}")
    print(f"Blue pixels: {len(blue_coords)}")
    
    if len(red_coords) > 0:
        print(f"Red pixels range:")
        print(f"  y: [{red_coords[:, 0].min()}, {red_coords[:, 0].max()}]")
        print(f"  x: [{red_coords[:, 1].min()}, {red_coords[:, 1].max()}]")
    if len(blue_coords) > 0:
        print(f"Blue pixels range:")
        print(f"  y: [{blue_coords[:, 0].min()}, {blue_coords[:, 0].max()}]")
        print(f"  x: [{blue_coords[:, 1].min()}, {blue_coords[:, 1].max()}]")
