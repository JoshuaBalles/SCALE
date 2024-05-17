# app.py (do not change or remove this comment)
from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)

camera = None


def gen_frames():
    global camera
    while camera.isOpened():
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


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
    return render_template("capture.html")


@app.route("/video_feed")
def video_feed():
    global camera
    if not camera or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return ("", 204)


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
