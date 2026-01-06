import os
import json
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, send_from_directory, flash
)
import tensorflow as tf
from gradcam import get_deforestation_outline
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best_deforestation_model.h5"
UPLOAD_FOLDER = "uploads"
PRED_DIR = "predictions"
LOG_FILE = os.path.join(PRED_DIR, "predictions_log.txt")

ALLOWED_EXT = {"png", "jpg", "jpeg", "tiff", "tif"}
IMG_SIZE = (150, 150)
CLASS_NAMES = ["Deforested", "Non-Deforested"]
DB_PATH = "users.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ best_deforestation_model.h5 NOT FOUND")

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------
# FLASK APP
# -------------------------
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "secret123"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# Ensure users table (username + password only)
# -------------------------
def ensure_users_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

ensure_users_table()

# -------------------------
# LOGIN SYSTEM
# -------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id, username, password):
        # Flask-Login expects the id attribute to be a string
        self.id = str(id)
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    # user_id arrives as a string; defensively convert to int
    try:
        uid = int(user_id)
    except Exception:
        return None

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (uid,))
    user = c.fetchone()
    conn.close()

    if user:
        # row: (id, username, password)
        return User(user[0], user[1], user[2])
    return None


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image_file(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return arr, img


def predict_from_array(arr):
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]

    # handle binary output as single probability
    if len(preds) == 1:
        probs = [1 - float(preds[0]), float(preds[0])]
    else:
        probs = [float(x) for x in preds]

    total = sum(probs)
    probs = [p / total for p in probs]

    return probs, int(np.argmax(probs))


def log_prediction(fname, label, conf, probs):
    timestamp = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}\t{fname}\t{label}\t{conf:.3f}\t{json.dumps(probs)}\n")


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return redirect(url_for("login"))


# -------- LOGIN PAGE ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            login_user(User(user[0], user[1], user[2]))
            return redirect("/index")
        else:
            flash("Invalid username or password", "error")

    return render_template("login.html")


# -------- REGISTER PAGE ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        confirm = request.form["confirm"]

        # Password match
        if password != confirm:
            flash("Passwords do not match!", "error")
            return redirect("/register")

        # Minimum password length
        if len(password) < 6:
            flash("Password must be at least 6 characters!", "error")
            return redirect("/register")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Check if username exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        existing = c.fetchone()

        if existing:
            flash("Username already exists!", "error")
            conn.close()
            return redirect("/register")

        # Insert new user (username + hashed password)
        hashed_password = generate_password_hash(password)
        c.execute("INSERT INTO users(username, password) VALUES(?,?)",
                  (username, hashed_password))
        conn.commit()
        conn.close()

        flash("Account created successfully! Please login.", "success")
        return redirect("/login")

    return render_template("register.html")


# -------- LOGOUT ----------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# -------- MAIN UPLOAD PAGE ----------
@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":

        if "file" not in request.files:
            flash("No file uploaded", "error")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Invalid file format", "error")
            return redirect(url_for("index"))

        # sanitize filename
        safe_name = secure_filename(file.filename)
        fname = datetime.utcnow().strftime("%Y%m%d%H%M%S_") + safe_name
        path = os.path.join(UPLOAD_FOLDER, fname)
        file.save(path)

        # preprocess + predict
        arr, pil_img = preprocess_image_file(path)
        probs, idx = predict_from_array(arr)

        label = CLASS_NAMES[idx]
        conf = round(probs[idx] * 100, 2)

        # save predicted image
        pred_name = "pred_" + fname
        pred_path = os.path.join(PRED_DIR, pred_name)
        pil_img.save(pred_path)

        # create and save Grad-CAM outline (uses your gradcam.get_deforestation_outline)
        outline_name = "outline_" + fname
        outline_path = os.path.join(PRED_DIR, outline_name)
        outline_img = get_deforestation_outline(model, arr, idx)
        outline_img.save(outline_path)

        # log
        log_prediction(pred_name, label, conf, probs)

        top2 = sorted(
            [{"label": CLASS_NAMES[i], "prob": float(p)} for i, p in enumerate(probs)],
            key=lambda x: x["prob"],
            reverse=True
        )[:2]

        return render_template(
            "result.html",
            filename=pred_name,
            outline_filename=outline_name,
            label=label,
            confidence=conf,
            top2=top2
        )

    return render_template("index.html")


# -------- OUTLINE PAGE ----------
@app.route("/deforestation_map/<filename>/<outline_filename>/<label>/<confidence>")
@login_required
def deforestation_map(filename, outline_filename, label, confidence):
    return render_template(
        "deforestation_map.html",
        filename=filename,
        outline_filename=outline_filename,
        label=label,
        confidence=confidence
    )


# -------- STATIC IMAGES ----------
@app.route("/predictions/<path:filename>")
def prediction_file(filename):
    return send_from_directory(PRED_DIR, filename)


# -------- VIEW LOG ----------
@app.route("/log")
@login_required
def view_log():
    if not os.path.exists(LOG_FILE):
        return "No predictions logged yet."
    with open(LOG_FILE, "r") as f:
        content = f.read()
    return "<pre>" + content + "</pre>"

def calculate_deforested_area(heatmap1, heatmap2):
    import cv2
    import numpy as np

    # Convert to grayscale
    gray1 = cv2.cvtColor(np.array(heatmap1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(heatmap2), cv2.COLOR_RGB2GRAY)

    # Threshold to get deforestation regions
    _, bin1 = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY)

    # Detect new deforestation
    diff = cv2.subtract(bin2, bin1)

    deforested_pixels = np.count_nonzero(diff)
    total_pixels = diff.size

    area_percentage = (deforested_pixels / total_pixels) * 100
    return round(area_percentage, 2)
# -------- COMPARE PAGE ----------
@app.route("/compare", methods=["GET"])
@login_required
def compare_page():
    return render_template("compare.html")


# -------- COMPARE RESULT ----------
@app.route("/compare_result", methods=["POST"])
@login_required
def compare_result():
    img1 = request.files.get("image1")
    img2 = request.files.get("image2")

    if not img1 or not img2:
        flash("Please upload both images", "error")
        return redirect(url_for("compare_page"))

    os.makedirs("static/uploads", exist_ok=True)

    filename1 = secure_filename(img1.filename)
    filename2 = secure_filename(img2.filename)

    p1 = os.path.join("static/uploads", filename1)
    p2 = os.path.join("static/uploads", filename2)

    img1.save(p1)
    img2.save(p2)

    arr1, _ = preprocess_image_file(p1)
    arr2, _ = preprocess_image_file(p2)

    heat1 = get_deforestation_outline(model, arr1, 0)
    heat2 = get_deforestation_outline(model, arr2, 0)

    # % of NEW deforestation from Image-1 → Image-2
    area = calculate_deforested_area(heat1, heat2)

    # ---------- SCIENTIFIC INTERPRETATION ----------
    THRESHOLD = 5.0  # ignore noise / seasonal variation

    if area > THRESHOLD:
        change_status = "DEFORESTATION_INCREASED"
        message = (
            "Based on comparison of Image 1 with Image 2, "
            "new deforestation activity is detected in the region."
        )

    elif area < -THRESHOLD:
        change_status = "FOREST_RECOVERY"
        message = (
            "Forest recovery is observed. Vegetation cover has "
            "improved when compared with the earlier image."
        )

    else:
        change_status = "NO_SIGNIFICANT_CHANGE"
        message = (
            "No significant deforestation change is observed "
            "between the two time periods."
        )

    return render_template(
        "compare_result.html",
        change_status=change_status,
        message=message,
        image1_name=filename1,
        img2_name=filename2
    )
# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)