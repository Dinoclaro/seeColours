from cs50 import SQL
from flask import session, url_for, render_template, redirect, current_app
from functools import wraps
import os

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
SCORES = {"protan": (0, 100, 100), "deutan": (100, 0, 100), "tritan": (100, 100, 0)}  # RGB

# Initialize database
db = SQL("sqlite:///seeColours.db")

def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/latest/patterns/viewdecorators/
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)

    return decorated_function

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_data(user_id):
    """Retrieve user's prognosis and cone scores from the database."""
    user_data = db.execute("SELECT prognosis, red_score, green_score, blue_score FROM users WHERE id = ?", user_id)
    if user_data:
        return {
            "prognosis": user_data[0]["prognosis"],
            "red_score": user_data[0]["red_score"],
            "green_score": user_data[0]["green_score"],
            "blue_score": user_data[0]["blue_score"],
        }
    return {"prognosis": "deutan", "red_score": 0, "green_score": 100, "blue_score": 100}


def get_uploaded_file(session):
    """Retrieve the uploaded file from the session."""
    uploaded_file = session.get("uploaded_file")
    if not uploaded_file:
        return None
    return current_app.url_for("static", filename=f"uploads/{uploaded_file}")


def process_image(prognosis, filename, processor_function):
    """Generic image processing logic for simulate and daltonize."""
    print("Entered process image")

    # Build the relative input path from your app's root directory
    uploads_folder = 'static/uploads'
    input_path = os.path.join(uploads_folder , filename)
    print(f"Input Path: {input_path}")

    # Check if the file exists in the uploads directory
    # if not os.path.exists(input_path):
    #     raise FileNotFoundError(f"File {filename} not found at {input_path}")

    output_filename = f"processed_{prognosis}_{filename}"
    output_path = os.path.join(uploads_folder, output_filename)  # Save processed file in the same folder
    print(f"Output Path: {output_path}")

    if prognosis == "user_prognosis":
        user = db.execute(
            "SELECT prognosis, red_score, green_score, blue_score FROM users WHERE id = ?",
            session["user_id"],
        )[0]
        prognosis = user["prognosis"]
        scores = (user["red_score"], user["green_score"], user["blue_score"])
    elif prognosis in SCORES:
        print("in scores")
        scores = SCORES[prognosis]
    else:
        raise ValueError("Invalid prognosis type")

    print("sent for processing")
    processed_image = processor_function(input_path, prognosis, *scores)
    processed_image.save(output_path)
    return output_filename
