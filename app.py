from flask import Flask, request, render_template, redirect, url_for
import os
from predictor import predict_genre_song

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)
        try:
            genre, confidence = predict_genre_song(
                filename, segment_sec=3, remove_vocals=False
            )
            if genre:
                return render_template("index.html", genre=genre, confidence=confidence)
            else:
                return render_template(
                    "index.html", error="Prediction failed. Please try another file."
                )
        except Exception as e:
            return render_template("index.html", error=str(e))
        finally:
            # Clean up uploaded file
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
