import os
from PIL import Image

def get_image_dimensions(image_folder):
    """
    Get the width and height of each image in the specified folder.
    """
    dimensions = {}
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith('.jpg'):
            file_path = os.path.join(image_folder, file_name)
            with Image.open(file_path) as img:
                width, height = img.size
                dimensions[file_name] = (width, height)
    return dimensions

def read_yolo_annotations(anno_file, original_width, original_height):
    """
    Read YOLO annotations and convert them to pixel coordinates.
    """
    bboxes = []
    with open(anno_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])
            x_center_pixel = x_center * original_width
            y_center_pixel = y_center * original_height
            width_pixel = width * original_width
            height_pixel = height * original_height
            bboxes.append((class_id, x_center_pixel, y_center_pixel, width_pixel, height_pixel))
    return bboxes

def update_yolo_annotations(crop_x, crop_y, crop_w, crop_h, original_width, original_height, input_anno_file, output_anno_file):
    """
    Update YOLO annotation file based on cropped image.
    """
    def clip(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    with open(input_anno_file, 'r') as infile, open(output_anno_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:])
            x_center_pixel = x_center * original_width
            y_center_pixel = y_center * original_height
            width_pixel = width * original_width
            height_pixel = height * original_height
            x_center_cropped = (x_center_pixel - crop_x) / crop_w
            y_center_cropped = (y_center_pixel - crop_y) / crop_h
            width_cropped = width_pixel / crop_w
            height_cropped = height_pixel / crop_h
            if (0 <= x_center_cropped <= 1) and (0 <= y_center_cropped <= 1) and (width_cropped > 0) and (height_cropped > 0):
                x_center_cropped = clip(x_center_cropped, 0, 1)
                y_center_cropped = clip(y_center_cropped, 0, 1)
                width_cropped = clip(width_cropped, 0, 1)
                height_cropped = clip(height_cropped, 0, 1)
                # Saving classes other than person
                if (class_id != '0'):
                    outfile.write(f"{class_id} {x_center_cropped} {y_center_cropped} {width_cropped} {height_cropped}\n")

def process_annotations_for_crops(image_folder, anno_folder, output_folder):
    """
    Process all YOLO annotation files and create cropped annotations for each image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dimensions = get_image_dimensions(image_folder)
    
    for file_name in os.listdir(anno_folder):
        if file_name.endswith('.txt'):
            base_name = os.path.splitext(file_name)[0]
            image_file = base_name + '.jpg'
            
            if image_file in dimensions:
                original_width, original_height = dimensions[image_file]
                input_anno_file = os.path.join(anno_folder, file_name)
                bboxes = read_yolo_annotations(input_anno_file, original_width, original_height)

                if bboxes:
                    for idx, (class_id, x_center_pixel, y_center_pixel, width_pixel, height_pixel) in enumerate(bboxes):
                        if class_id == '0':  # Only process class 0 (person)
                            crop_x = x_center_pixel - width_pixel / 2
                            crop_y = y_center_pixel - height_pixel / 2
                            crop_w = width_pixel
                            crop_h = height_pixel

                            # Ensure the crop region is within the image bounds
                            crop_x = max(0, crop_x)
                            crop_y = max(0, crop_y)
                            crop_w = min(crop_w, original_width - crop_x)
                            crop_h = min(crop_h, original_height - crop_y)

                            # Generate the new annotation file for the cropped instance
                            output_anno_file = os.path.join(output_folder, f"{base_name}_{idx}.txt")
                            update_yolo_annotations(crop_x, crop_y, crop_w, crop_h, original_width, original_height, input_anno_file, output_anno_file)
                            print(f"Processed: {output_anno_file}")

#usage:
image_folder = '/home/mahak/Desktop/Development/ppe-detection/datasets/images' # images path
anno_folder = '/home/mahak/Desktop/Development/ppe-detection/datasets/labels/yolo' # Folder with YOLO annotations
output_folder = '/home/mahak/Desktop/Development/ppe-detection/datasets_ppe_model/labels'  # Folder to save cropped annotations

process_annotations_for_crops(image_folder, anno_folder, output_folder)
