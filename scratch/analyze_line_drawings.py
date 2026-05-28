import cv2
import numpy as np

img_path = 'outputs/subplots/subplot_1_left.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    
    # Find connected components for red and blue masks
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_mask)
    num_labels_blue, labels_blue, stats_blue, centroids_blue = cv2.connectedComponentsWithStats(blue_mask)
    
    print(f"Red components count (excluding background): {num_labels_red - 1}")
    print(f"Blue components count (excluding background): {num_labels_blue - 1}")
    
    # Print size of the largest components
    if num_labels_red > 1:
        sizes_red = stats_red[1:, cv2.CC_STAT_AREA]
        print(f"Red component sizes (top 5): {sorted(sizes_red, reverse=True)[:5]}")
    if num_labels_blue > 1:
        sizes_blue = stats_blue[1:, cv2.CC_STAT_AREA]
        print(f"Blue component sizes (top 5): {sorted(sizes_blue, reverse=True)[:5]}")
