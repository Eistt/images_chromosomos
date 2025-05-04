import os
import cv2
from model_detector import detect_objects_yolov8
from model_detector import image_files, photo_dir
from augmentation import apply_augmentations  


def image_processing():
    show_image = image_files[665]  # foto
    image_path = os.path.join(photo_dir, show_image)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1280, 1280))
    
    if image is None:
        print(f"No image found at {image_path}")
        return
    

    augmented_image = apply_augmentations(image)
    annotated_image, class_avg_confidences = detect_objects_yolov8(augmented_image)
    
    print("Average of chromosome type:")
    for class_name, avg_conf in class_avg_confidences.items():
        print(f"{class_name}: {avg_conf:.2f}")
    
    cv2.imshow("Counted Chromosomes", annotated_image)
    cv2.waitKey(0)

image_processing()











