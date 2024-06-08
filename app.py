# app.py (do not change or remove this comment)

import logging
import os
import subprocess
import sys
from datetime import datetime

import piexif
from flask import Flask, redirect, render_template, request, url_for
from libcamera import Transform
from picamera2 import Picamera2
from PIL import Image

from models import crop

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the path to the video server script
VIDEO_SERVER_SCRIPT = os.path.join(os.getcwd(), "camera_video_server.py")

# Use the Python interpreter from the current environment
python_interpreter = sys.executable

# Global variable to hold the video server process
video_server_process = None


# Function to start the video server
def start_video_server():
    global video_server_process
    if video_server_process is None:
        logging.info("Starting video server...")
        video_server_process = subprocess.Popen(
            [python_interpreter, VIDEO_SERVER_SCRIPT]
        )
    else:
        logging.info("Video server already running.")


# Function to stop the video server
def stop_video_server():
    global video_server_process
    if video_server_process is not None:
        logging.info("Stopping video server...")
        video_server_process.terminate()  # Send termination signal to the process
        video_server_process.wait()  # Wait for the process to terminate
        logging.info("Video server stopped.")
        video_server_process = None
    else:
        logging.info("No video server process to stop.")


# Function to release the camera resources
def release_camera(picam2):
    try:
        picam2.stop()
        picam2.close()
        logging.info("Camera released.")
    except Exception as e:
        logging.error(f"Error releasing camera: {e}")


# Function to capture an image and save it with metadata,
# then perform object detection on the saved image
def capture_image():
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%B %d, %Y, %I-%M-%S %p")

    # Initialize Picamera2
    picam2 = Picamera2()

    # Configure the camera with vertical and horizontal flip
    config = picam2.create_still_configuration(
        transform=Transform(vflip=True, hflip=False)
    )
    picam2.configure(config)

    # Start the camera
    picam2.start()

    # Capture an image into memory
    logging.info("Capturing image...")
    buffer = picam2.capture_array()

    # Save the image as JPEG with EXIF metadata
    image = Image.fromarray(buffer)
    image_path = os.path.join(os.getcwd(), "capture.jpg")

    # Create EXIF metadata
    exif_dict = {
        "0th": {},
        "Exif": {},
        "1st": {},
        "thumbnail": None,
        "GPS": {},
        "Interop": {},
    }
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = timestamp

    exif_bytes = piexif.dump(exif_dict)

    # Save the image with EXIF metadata
    image.save(image_path, "jpeg", exif=exif_bytes)
    logging.info(f"Saved image at {image_path} with metadata: Captured={timestamp}")

    # Perform object detection on the saved image
    crop.crop_objects("capture.jpg")

    # Release the camera
    release_camera(picam2)


@app.before_request
def before_request():
    if request.endpoint == "capture":
        start_video_server()


@app.after_request
def after_request(response):
    if request.endpoint != "capture":
        stop_video_server()
    return response


@app.route("/")
def index():
    # Render the main home page.
    return render_template("home.html")


@app.route("/capture", methods=["GET", "POST"])
def capture():
    if request.method == "POST":
        # Stop the video server before capturing the image
        stop_video_server()

        # Capture image
        logging.info("Capturing image...")
        capture_image()

        # Restart the video server
        start_video_server()

        logging.info("Capture complete and video server restarted.")
        return redirect(url_for("capture"))

    return render_template("capture.html")


@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/track")
def track():
    return render_template("track.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
