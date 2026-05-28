import cv2
import numpy as np

img_path = 'outputs/subplots/subplot_1_left.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load Subplot 1")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red mask (red has two ranges in Hue)
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    # Find coordinates of red and blue pixels
    red_coords = np.argwhere(red_mask)
    blue_coords = np.argwhere(blue_mask)
    
    # For each blue pixel, let's find the distance to the nearest red pixel
    distances = []
    for b in blue_coords:
        # Distance to all red pixels
        dists = np.linalg.norm(red_coords - b, axis=1)
        distances.append(np.min(dists))
        
    print(f"Distance from blue pixels to nearest red pixel in the paper's figure:")
    print(f"  Mean distance: {np.mean(distances):.2f} pixels")
    print(f"  Max distance: {np.max(distances):.2f} pixels")
    print(f"  Min distance: {np.min(distances):.2f} pixels")
    print(f"  Std dev: {np.std(distances):.2f} pixels")
