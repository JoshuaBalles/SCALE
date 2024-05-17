# app.py (do not change or remove this comment)
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/capture")
def capture():
    return render_template("capture.html")


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
