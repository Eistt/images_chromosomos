import cv2
import numpy as np

def apply_augmentations(image):
    aug_image = image.copy()
    
    kernel_size = 3
    aug_image = cv2.GaussianBlur(aug_image, (kernel_size, kernel_size), 0)
    
    hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s.astype(np.int16)
    s += 3
    s = np.clip(s, 0, 255).astype(np.uint8)
    
    hsv = cv2.merge([h, s, v])
    aug_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return aug_image