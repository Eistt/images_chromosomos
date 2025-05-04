from barplot import plot_bar_chart
import cv2
import numpy as np
from ultralytics import YOLO
import os


#is iterneto nuotraukos
# photo_dir ="/Users/aistebalkeviciute/Downloads/images_chromosomos"

# uzsiloadinam model ir foto direktorija

photo_dir = "dataset/single_chromosomes_object/JEPG"
model = YOLO('/Users/aistebalkeviciute/imagesAI/stocks/baigiamasis_projektas/runs/detect/train11/weights/best.pt')

# nuotraukas kad priimtu visais formatais
image_files = [f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]


def detect_objects_yolov8(image):
    results = model(image, conf=0.6) #confidence scoras
    result = results[0] #pirma foto is direktorijos
    count = len(result.boxes) if result.boxes is not None else 0 #skaiciuoja aptiktus objektus, jei nieko, grazina 0
    
    class_counts = {} #aptiktos chromosomos
    class_confidences = {}   #aptikti conf
    
    # kiekvienos chromosomos klase ir confidence
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id] if model.names else f'Class {class_id}'
            confidence = float(box.conf)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_name not in class_confidences:
                class_confidences[class_name] = []
            class_confidences[class_name].append(confidence)
    
    # vidutinis confidence klasei
    class_avg_confidences = {}
    for class_name, confidences in class_confidences.items():
        class_avg_confidences[class_name] = sum(confidences) / len(confidences)
    
    # overall confidence
    if class_avg_confidences:
        overall_avg_confidence = sum(class_avg_confidences.values()) / len(class_avg_confidences)
        print(f"Overall confidence of {count} detected chromosomes: {overall_avg_confidence:.2f}")
    else:
        overall_avg_confidence = 0
    
    annotated_image = result.plot()
    text = f'Count: {count}'
    
    cv2.putText(annotated_image, text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (55, 25, 12), 2)
    chart_image = plot_bar_chart(class_counts, annotated_image.shape[0])
    combined_image = np.hstack((annotated_image, chart_image))
    
    return combined_image, class_avg_confidences


