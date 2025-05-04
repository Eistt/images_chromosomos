import os
import cv2
import xml.etree.ElementTree as ET
from main import detect_objects_yolov8


photo_dir = "dataset/single_chromosomes_object/JEPG"  
xml_dir = "dataset/single_chromosomes_object/anntations"  


image_files = [f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

def drawing_annotation(image, xml_file):
    load= ET.parse(xml_file) #loads xml file
    get= load.getroot() #gets element
    
    for object in get.findall("object"):#iteruoja per <object>
     box= object.find("bndbox") #iteruoja per bndbox
     if box is not None:
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        
        label = object.find("name").text  # Get class label
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image


def image_processing():
    show_image = image_files[400] #cia keisti kuria foto
    image_path = os.path.join(photo_dir, show_image)

    base_xml = os.path.splitext(show_image)[0]  # panaikina xml dali
    xml_file = os.path.join(xml_dir, base_xml + ".xml")

    if os.path.exists(xml_file):  # jei egzisutuoja 
        image = cv2.imread(image_path)  # perskaito
        if image is None:
            print(f"No img found {image_path}")
            return  
        
        image = drawing_annotation(image, xml_file)# nupiesia anotacija
        # image= detect_objects_yolov8(image)

        cv2.imshow("chromosomos", image)
        cv2.waitKey(0)  

image_processing()








