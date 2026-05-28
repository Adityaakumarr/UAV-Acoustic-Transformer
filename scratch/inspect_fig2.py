import cv2
import numpy as np

img_path = 'outputs/pdf_images/page_4_img_3_Im2.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Could not load Fig 2 image")
else:
    print(f"Loaded Fig 2 image shape: {img.shape}")
    
    # Convert to RGB to analyze colors
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV to find red and blue pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    red_coords = np.argwhere(red_mask)
    blue_coords = np.argwhere(blue_mask)
    
    print(f"Red pixels count: {len(red_coords)}")
    print(f"Blue pixels count: {len(blue_coords)}")
    
    if len(red_coords) > 0:
        print(f"Red pixels bounding box:")
        print(f"  ({red_coords[:, 0].min()}, {red_coords[:, 1].min()}) to ({red_coords[:, 0].max()}, {red_coords[:, 1].max()})")
    if len(blue_coords) > 0:
        print(f"Blue pixels bounding box:")
        print(f"  ({blue_coords[:, 0].min()}, {blue_coords[:, 1].min()}) to ({blue_coords[:, 0].max()}, {blue_coords[:, 1].max()})")
