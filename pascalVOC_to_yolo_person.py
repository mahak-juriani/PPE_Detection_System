import xml.etree.ElementTree as ET
import os

def voc_to_yolo(voc_file, output_dir, class_mapping):
    tree = ET.parse(voc_file)
    root = tree.getroot()
    
    # Get image dimensions
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    # Prepare the YOLO format annotation file
    base_filename = os.path.splitext(os.path.basename(voc_file))[0]
    yolo_file = os.path.join(output_dir, f"{base_filename}.txt")
    
    with open(yolo_file, 'w') as out_file:
        for obj in root.findall('object'):
            # Read object class
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue
            class_id = class_mapping[class_name]
            
            # Read bounding box
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            obj_width = (xmax - xmin) / width
            obj_height = (ymax - ymin) / height
            
            # Write to file
            out_file.write(f"{class_id} {x_center} {y_center} {obj_width} {obj_height}\n")

def convert_all_xml_to_yolo_person(input_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            voc_file = os.path.join(input_dir, filename)
            voc_to_yolo(voc_file, output_dir, class_mapping)

# usage

input_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels/voc'
output_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels'

class_mapping = {
    'person': 0
}

convert_all_xml_to_yolo_person(input_dir, output_dir, class_mapping)
