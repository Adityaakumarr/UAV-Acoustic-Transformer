import cv2
import os

img_path = 'outputs/pdf_images/page_4_img_3_Im2.jpg'
img = cv2.imread(img_path)

if img is None:
    print("Could not load image")
else:
    # Bounding boxes for columns:
    # Subplot 1: x in [0, 530]
    # Subplot 2: x in [530, 1070]
    # Subplot 3: x in [1070, 1600]
    
    os.makedirs("outputs/subplots", exist_ok=True)
    
    sub1 = img[:, 0:530]
    sub2 = img[:, 530:1070]
    sub3 = img[:, 1070:1600]
    
    cv2.imwrite("outputs/subplots/subplot_1_left.png", sub1)
    cv2.imwrite("outputs/subplots/subplot_2_middle.png", sub2)
    cv2.imwrite("outputs/subplots/subplot_3_right.png", sub3)
    
    print("Saved subplots to outputs/subplots/")
