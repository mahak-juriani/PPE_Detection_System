import os
import cv2
import numpy as np

def crop_person_from_image(image_path, annotations_path, output_dir, class_mapping, image_ext='jpg'):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    # Load annotations
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    annotation_file = os.path.join(annotations_path ,f"{base_filename}.txt")

    

    if not os.path.exists(annotation_file):
        print(f"No annotation file found for {image_path}")
        print(annotation_file)
        return
    
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    # Process each annotation
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Convert YOLO coordinates to pixel coordinates
        img_height, img_width, _ = image.shape
        xmin = int((x_center - width / 2) * img_width)
        xmax = int((x_center + width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        ymax = int((y_center + height / 2) * img_height)
        
        # Ensure coordinates are within image bounds
        xmin, xmax = max(0, xmin), min(img_width, xmax)
        ymin, ymax = max(0, ymin), min(img_height, ymax)

        # Crop the image
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        # Ensure output directory exists
        output_class_dir = os.path.join(output_dir)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        # Save the cropped image
        output_filename = os.path.join(output_class_dir, f"{base_filename}_{idx}.{image_ext}")
        cv2.imwrite(output_filename, cropped_image)
        print(f"Cropped image saved to {output_filename}")

def crop_all_images(input_images_dir, annotations_dir, output_dir, class_mapping, image_ext='jpg'):
    for filename in os.listdir(input_images_dir):
        if filename.lower().endswith(image_ext):
            image_path = os.path.join(input_images_dir, filename)
            crop_person_from_image(image_path, annotations_dir, output_dir, class_mapping, image_ext)

# usage
input_images_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/images'
annotations_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels'
output_dir = '/home/mahak/Desktop/Development/ppe-detection/datasets_ppe_model/images'
class_mapping = {
    'person': 0
    }

crop_all_images(input_images_dir, annotations_dir, output_dir, class_mapping)
