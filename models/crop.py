# crop.py (do not change or remove this comment)
from datetime import datetime
import os
import cv2
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG)

# Define a dictionary to map class IDs to class names
class_names = {
    0: "Chicken",
    1: "Pig",
}


def crop_objects(image_path):
    # Load the YOLOv8 model
    model = YOLO(r"models/object_detection_model.pt")

    # Load the original image
    image = image_path

    # Perform inference on the image
    results = model(image)

    img = cv2.imread(image)

    # Check if there are no results
    if len(results[0].boxes.cls) == 0:
        logging.debug("No objects detected in the image.")
    else:
        # Extract bounding boxes and class IDs
        boxes = results[0].boxes.xyxy.tolist()
        class_ids = results[0].boxes.cls.tolist()

        # Get the current datetime
        current_datetime = datetime.now().strftime("%B %d, %Y, %H:%M:%S")

        # Iterate through the bounding boxes
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            x1, y1, x2, y2 = box
            # Add padding to the bounding box
            x1 -= 32
            y1 -= 32
            x2 += 32
            y2 += 32
            # Ensure the coordinates are within the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            # Crop the object using the padded bounding box coordinates
            cropped_object = img[int(y1) : int(y2), int(x1) : int(x2)]
            # Get the class name from the class ID
            class_name = class_names.get(int(class_id), "Unknown")
            # Generate the filename with class name and formatted date
            filename = f"{class_name} - {current_datetime} {i}.jpg"
            # Define the path to the cropped folder
            cropped_folder = "cropped"
            # Ensure the cropped folder exists
            os.makedirs(cropped_folder, exist_ok=True)
            # Save the cropped object as an image in the cropped folder
            cv2.imwrite(os.path.join(cropped_folder, filename), cropped_object)
