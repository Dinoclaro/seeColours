import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from helpers import login_required, allowed_file, get_uploaded_file, get_user_data, process_image
from image_processing import process_image_simulate as pi_sim
from image_processing import process_image_daltonize as pi_dal
from image_processing import SCORES

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Libraray to use SQLite database
db = SQL("sqlite:///seeColours.db")

# Configure upload fodler
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB limit

@app.after_request
def after_request(response):
    """Ensure response aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
@login_required
def index():
    user = db.execute("SELECT prognosis, blue_score, green_score, red_score FROM users WHERE id = ?", session["user_id"])

    if user[0]['prognosis']:
        user = user[0]
        prognosis = user['prognosis']
        blue_score = user['blue_score']
        green_score = user['green_score']
        red_score = user['red_score']

        return render_template("index.html", prognosis=prognosis, blue_score=blue_score, green_score=green_score, red_score=red_score)
    else:
        return render_template("index.html", no_results=True)

@app.route("/about")
@login_required
def about():

    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register a new user"""
     # Forget any user
    session.clear()

    if request.method == "POST":
        # Retrieve form inputs
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Validate input fields
        if not username or not password or not confirmation:
            flash("All fields are required!", "danger")
            return redirect("/register")

        # Check if passwords match
        if password != confirmation:
            flash("Passwords do not match!", "danger")
            return redirect("/register")

        # Check for existing username
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)
        if len(rows) > 0:
            flash("Username already exists!", "danger")
            return redirect("/register")

        # Hash the password
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256", salt_length=8)

        # Insert the new user into the database
        db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, hashed_password)

        flash("Registration successful! Please log in.", "success")
        return redirect("/login")

    # If user reached the route via GET
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            flash("Must provide username :(", "danger")
            return redirect("/login")

        # Ensure password was submitted
        elif not request.form.get("password"):
            flash("No password :(", "danger")
            return redirect("/login")

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            flash("Invalid username and/or password :(", "danger")
            return redirect("/login")

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        flash("Logged In!", "success")
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")

@app.route("/submit_test", methods=["GET", "POST"])
@login_required
def submit_test():
    """Save EnChroma test results to the database"""
    if request.method == "POST":
        prognosis = request.form.get("prognosis")
        red_score = request.form.get("red_score")
        green_score = request.form.get("green_score")
        blue_score = request.form.get("blue_score")

        # Validate entries
        if not prognosis or not red_score or not green_score or not blue_score:
            flash("Please provide all test results", "error")
            return redirect("/upload")

        # Update scores for the logged-in user
        db.execute("UPDATE users SET prognosis = ?, red_score = ?, green_score = ?, blue_score = ? WHERE id = ?",
           prognosis, int(red_score), int(green_score), int(blue_score), session["user_id"])

        flash("Test results saved successfully!", "success")
        return redirect("/")
    else:
        return render_template("/submit_test.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """Allow users to upload an image and navigate to the simulator."""
    if request.method == "POST":
        # Check if the action is valid
        action = request.form.get("action")
        if action not in ["simulate", "daltonize"]:
            flash("Invalid action selected. Please try again.", "error")
            return redirect(request.url)

        # Check if a file is provided
        if "file" not in request.files:
            flash("No file part. Please upload a file.", "error")
            return redirect(request.url)

        file = request.files["file"]

        # Validate the file
        if file.filename == "":
            flash("No file selected. Please choose a file.", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            session["uploaded_file"] = filename

            # Get user's prognosis
            user_data = get_user_data(session["user_id"])
            if not user_data["prognosis"]:
                flash("Prognosis not found. Using Protan as default.", "warning")

            flash("File uploaded successfully!", "success")

            # Redirect to the simulator page
            return redirect(url_for("render_page", mode=action))

        else:
            flash("Invalid file type. Please upload a valid image file.", "error")
            return redirect(request.url)

    return render_template("upload.html")


@app.route("/process_image/<mode>/<prognosis>/<filename>")
@login_required
def process_image_route(mode, prognosis, filename):
    """Unified image processing route."""
    print(f"Mode: {mode}, Prognosis: {prognosis}, Filename: {filename}")
    if not prognosis or not filename:
        return jsonify({"error": "Missing prognosis or filename"}), 400

    processor_function = pi_sim if mode == "simulate" else pi_dal
    print(f"Processor Function: {mode}")
    try:
        print("Before processing")
        output_filename = process_image(prognosis, filename, processor_function)
        print("Sent for processing")
        return jsonify({"image_url": url_for("static", filename=f"uploads/{output_filename}")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/base_simulator/<mode>")
@login_required
def render_page(mode):
    """Render the base simulator for simulate or daltonize modes."""
    if mode not in ["simulate", "daltonize"]:
        flash("Invalid mode selected.", "error")
        return redirect("/upload")

    uploaded_file = get_uploaded_file(session)
    if not uploaded_file:
        flash("No file uploaded. Please upload a file.", "error")
        return redirect("/upload")

    user_data = get_user_data(session["user_id"])

    return render_template(
        "base_simulator.html",
        mode=mode,
        image_path=uploaded_file,
        prognosis=user_data.get("prognosis", "Protan"),
        red_score=user_data.get("red_score", 0),
        green_score=user_data.get("green_score", 0),
        blue_score=user_data.get("blue_score", 0),
    )

if __name__ == "__main__":
    app.run(debug=True)
