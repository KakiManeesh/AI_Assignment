from flask import Flask, request, render_template, redirect, url_for
import os
from predictor import predict_genre_song
import predictor  # Import to access progress updates

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/progress")
def get_progress():
    """AJAX endpoint to check current progress."""
    import predictor

    progress_list = getattr(predictor, "progress_updates", [])
    if not progress_list:
        return {
            "progress": [],
            "progress_percentage": 0,
            "current_step": "",
            "has_result": False,
        }

    # Get the latest progress update
    latest = progress_list[-1]

    # Handle both old string format and new dict format
    if isinstance(latest, dict):
        return {
            "progress": progress_list,
            "progress_percentage": latest.get("progress", 0),
            "current_step": latest.get("message", ""),
            "current_segment": latest.get("current_segment", 0),
            "total_segments": latest.get("total_segments", 0),
            "has_result": latest.get("message") == "✨ Processing complete!",
        }
    else:
        # Fallback for old string format
        return {
            "progress": progress_list,
            "progress_percentage": 100 if "Processing complete!" in str(latest) else 50,
            "current_step": str(latest),
            "current_segment": 0,
            "total_segments": 0,
            "has_result": "Processing complete!" in str(latest),
        }


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
                return render_template(
                    "index.html",
                    genre=genre,
                    confidence=confidence,
                    progress_updates=predictor.progress_updates,
                )
            else:
                return render_template(
                    "index.html",
                    error="Prediction failed. Please try another file.",
                    progress_updates=predictor.progress_updates,
                )
        except Exception as e:
            return render_template(
                "index.html", error=str(e), progress_updates=predictor.progress_updates
            )
        finally:
            # Clean up uploaded file
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
