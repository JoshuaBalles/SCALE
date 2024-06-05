# models/crop.py (do not change or remove this comment)

from datetime import datetime
import os
import cv2
import logging
from ultralytics import YOLO
from PIL import Image
import piexif

logging.basicConfig(level=logging.INFO)

# Define a dictionary to map class IDs to class names
class_names = {
    0: "Chicken",
    1: "Pig",
}


def crop_objects(image_path):
    # Function to extract the 'Captured' metadata from the input image
    def get_captured_metadata(image_path):
        try:
            image = Image.open(image_path)
            exif_data = piexif.load(image.info["exif"])
            captured_metadata = (
                exif_data["0th"]
                .get(piexif.ImageIFD.ImageDescription, b"Unknown")
                .decode("utf-8")
            )
            return captured_metadata
        except Exception as e:
            logging.error(f"Failed to get metadata from image: {e}")
            return "Unknown"

    # Extract the 'Captured' metadata from the input image
    captured_metadata = get_captured_metadata(image_path)
    logging.info(f"Extracted metadata: Captured={captured_metadata}")

    # Load the YOLOv8 model
    model_path = os.path.join("models", "object_detection_model.pt")
    logging.info(f"Loading YOLOv8 model from {model_path}")
    model = YOLO(model_path)

    # Load the original image
    logging.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)

    # Perform inference on the image
    logging.info("Performing inference on the image")
    results = model(image)

    # Check if there are no results
    if len(results[0].boxes.cls) == 0:
        logging.info("No objects detected in the image.")
    else:
        # Extract bounding boxes and class IDs
        logging.info("Extracting bounding boxes and class IDs")
        boxes = results[0].boxes.xyxy.tolist()
        class_ids = results[0].boxes.cls.tolist()

        # Iterate through the bounding boxes
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            logging.info(f"Processing object {i + 1} with class ID {class_id}")
            x1, y1, x2, y2 = box
            # Add padding to the bounding box
            x1 -= 32
            y1 -= 32
            x2 += 32
            y2 += 32
            # Ensure the coordinates are within the image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            # Crop the object using the padded bounding box coordinates
            cropped_object = image[int(y1) : int(y2), int(x1) : int(x2)]
            # Get the class name from the class ID
            class_name = class_names.get(int(class_id), "Unknown")
            # Generate the filename with class name and formatted date
            filename = f"{class_name} - {captured_metadata} {i}.jpg"
            # Define the path to the cropped folder
            cropped_folder = "cropped"
            # Ensure the cropped folder exists
            os.makedirs(cropped_folder, exist_ok=True)
            # Save the cropped object as an image in the cropped folder
            save_path = os.path.join(cropped_folder, filename)
            cv2.imwrite(save_path, cropped_object)
            logging.info(f"Cropped object saved to {save_path}")
