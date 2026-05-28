import cv2
import numpy as np

img_path = 'outputs/pdf_images/page_4_img_3_Im2.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    # Convert to grayscale to find structure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Let's see the column-wise sum of non-white pixels to see the splits between subplots
    non_white_cols = np.sum(gray < 250, axis=0)
    
    # Print out where the columns are heavily populated
    print("Column-wise non-white pixel sums (every 100 pixels):")
    for x in range(0, img.shape[1], 100):
        chunk_sum = np.sum(non_white_cols[x:x+100])
        print(f"  Col {x} to {x+100}: {chunk_sum}")
