import cv2
import os
from ultralytics import YOLO

# Define your directories
photo_dir = "dataset/24_chromosomes_object/JEPG"  # Replace with the actual path to your image directory

# Load your model
model = YOLO('/Users/aistebalkeviciute/imagesAI/stocks/baigiamasis_projektas/runs/detect/train11/weights/best.pt')  # Provide the correct path to best.pt

image_files = [f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  # Adjust extensions as needed

def detect_objects_yolov8(image):
    # Use the model to make predictions on the image
    results = model(image)  # Perform inference
    annotated_image = results[0].plot()  # Accessing the first result and using the plot method
    return annotated_image

def image_processing():
    show_image = image_files[6]  # Replace with the desired index (or loop over the list)
    image_path = os.path.join(photo_dir, show_image)

    image = cv2.imread(image_path)  # Read image
    if image is None:
        print(f"No image found at {image_path}")
        return
    
    # Detect objects using YOLOv8 model
    image = detect_objects_yolov8(image)

    # Show the processed image
    cv2.imshow("Chromosome detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_processing()