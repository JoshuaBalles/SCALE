# app.py (do not change or remove this comment)

from flask import Flask, render_template, redirect, url_for, request
import subprocess
import os
import sys
import logging
from datetime import datetime
from picamera2 import Picamera2
from libcamera import Transform
from PIL import Image
import piexif

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the path to the video server script
VIDEO_SERVER_SCRIPT = os.path.join(os.getcwd(), "camera_video_server.py")

# Use the Python interpreter from the current environment
python_interpreter = sys.executable


# Function to start the video server
def start_video_server():
    logging.info("Starting video server...")
    return subprocess.Popen([python_interpreter, VIDEO_SERVER_SCRIPT])


# Function to stop the video server
def stop_video_server(process):
    logging.info("Stopping video server...")
    process.terminate()  # Send termination signal to the process
    process.wait()  # Wait for the process to terminate
    logging.info("Video server stopped.")


# Function to release the camera resources
def release_camera(picam2):
    try:
        picam2.stop()
        picam2.close()
        logging.info("Camera released.")
    except Exception as e:
        logging.error(f"Error releasing camera: {e}")


# Function to capture an image and save it with metadata
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

    # Release the camera
    release_camera(picam2)


# Start the video server when the app starts
video_server_process = start_video_server()


@app.route("/")
def index():
    # Render the main home page.
    return render_template("home.html")


@app.route("/capture", methods=["GET", "POST"])
def capture():
    # Capture an image by stopping the video server, running the image capture script, and restarting the video server.
    global video_server_process

    if request.method == "POST":
        # Stop the video server
        stop_video_server(video_server_process)

        # Capture image
        logging.info("Capturing image...")
        capture_image()

        # Restart the video server
        video_server_process = start_video_server()

        logging.info("Capture complete and video server restarted.")
        return redirect(url_for("index"))

    return render_template("capture.html")


@app.route("/results")
def results():
    # Render the results page.
    return render_template("results.html")


@app.route("/track")
def track():
    # Render the track page.
    return render_template("track.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
