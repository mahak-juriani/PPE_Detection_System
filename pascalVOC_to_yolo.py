import xml.etree.ElementTree as ET
import os
import argparse

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

def convert_all_xml_to_yolo(input_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            voc_file = os.path.join(input_dir, filename)
            voc_to_yolo(voc_file, output_dir, class_mapping)

# # usage
# # TODO: argparse
# input_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels/voc'
# output_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels/yolo'

# class_mapping = {
#     'person': 0,
#     'hard-hat': 1,
#     'gloves': 2,
#     'mask': 3,
#     'glasses': 4,
#     'boots': 5,
#     'vest': 6,
#     'ppe-suit': 7,
#     'ear-protector': 8,
#     'safety-harness': 9
# }

# convert_all_xml_to_yolo(input_dir, output_dir, class_mapping)

def main():
    parser = argparse.ArgumentParser(description="Convert VOC format annotations to YOLO format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing VOC XML files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save YOLO format annotations.")
    parser.add_argument('--class_mapping', type=str, required=True, help="Path to JSON file containing class mapping.")
    
    args = parser.parse_args()
    
    # Load class mapping from JSON file
    import json
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    convert_all_xml_to_yolo(args.input_dir, args.output_dir, class_mapping)

if __name__ == "__main__":
    main()
