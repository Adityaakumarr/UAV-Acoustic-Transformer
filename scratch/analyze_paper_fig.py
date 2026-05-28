import cv2
import numpy as np

img_path = 'C:/Users/User/.gemini/antigravity-ide/brain/673b7c94-8e7c-44a1-a0f1-b2b9489d2d81/.tempmediaStorage/media_673b7c94-8e7c-44a1-a0f1-b2b9489d2d81_1779872061286.png'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    print(f"Loaded image shape: {img.shape}")
    # Convert to RGB to analyze colors
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Let's count unique RGB values (excluding white background)
    flat = img_rgb.reshape(-1, 3)
    non_white = flat[np.any(flat < 240, axis=1)]
    
    unique_colors, counts = np.unique(non_white, axis=0, return_counts=True)
    # Sort by count
    idx = np.argsort(-counts)
    print("Top unique non-white colors in the paper's image:")
    for i in range(min(15, len(idx))):
        color = unique_colors[idx[i]]
        count = counts[idx[i]]
        print(f"  RGB: {color}, Count: {count}")
