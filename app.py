# app.py (do not change or remove this comment)
from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify,
    redirect,
    url_for,
    send_from_directory,
)
import cv2
import logging
import os
import glob
from models import crop

app = Flask(__name__)

camera = None
temp_dir = "temp"
capture_path = os.path.join(temp_dir, "capture.jpg")

logging.basicConfig(level=logging.DEBUG)


def gen_frames():
    global camera
    while True:
        if camera and camera.isOpened():
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        else:
            break


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/capture")
def capture():
    global camera
    if not camera:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    logging.debug("Capture page loaded and camera started")
    return render_template("capture.html")


@app.route("/video_feed")
def video_feed():
    global camera
    if not camera or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    logging.debug("Video feed requested")
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
        logging.debug("Camera stopped successfully")
        return jsonify({"status": "success", "message": "Camera stopped successfully"})
    logging.debug("Camera was not running")
    return jsonify({"status": "error", "message": "Camera was not running"}), 400


@app.route("/capture_image", methods=["POST"])
def capture_image():
    global camera, capture_path
    if not camera or not camera.isOpened():
        return jsonify({"status": "error", "message": "Camera not started"}), 400
    success, frame = camera.read()
    if success:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        cv2.imwrite(capture_path, frame)
        logging.debug("Image captured and saved to %s", capture_path)
        crop.crop_objects(capture_path)
        logging.debug("Image processed through crop.crop_objects()")
        return jsonify({"status": "success", "message": "Image captured and processed"})
    return jsonify({"status": "error", "message": "Failed to capture image"}), 500


@app.route("/results")
def results():
    cropped_folder = "cropped"
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
    image_files = glob.glob(os.path.join(cropped_folder, "*.jpg"))
    image_files = [os.path.basename(f) for f in image_files]  # Get base names of files
    logging.debug("Results page loaded with %d images", len(image_files))
    return render_template("results.html", image_files=image_files)


@app.route("/cropped/<filename>")
def serve_cropped_image(filename):
    return send_from_directory("cropped", filename)


@app.route("/estimation")
def estimation():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("results"))
    basename = filename.rsplit(".", 1)[0]
    display_name = basename.rsplit(" ", 1)[0]
    return render_template(
        "estimation.html", filename=filename, display_name=display_name
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
