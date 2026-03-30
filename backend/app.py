from flask import Flask, request, jsonify, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import numpy as np
import joblib
import os
import csv
import bcrypt
import uuid
import bcrypt
import uuid

# =====================================
# APP SETUP
# =====================================
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'

db = SQLAlchemy(app)
CORS(app, supports_credentials=True)


# =====================================
# DATABASE MODELS
# =====================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100))
    grade = db.Column(db.Integer)


class Attempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)

    # ML FEATURES
    lit_acc = db.Column(db.Float)
    inf_acc = db.Column(db.Float)
    voc_acc = db.Column(db.Float)
    mid_acc = db.Column(db.Float)
    overall = db.Column(db.Float)
    time_f = db.Column(db.Float)
    diff = db.Column(db.Integer)

    # OUTPUTS
    lexile = db.Column(db.Integer)
    band = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================
# LOAD ML MODELS
# =====================================
MODEL_RF_PATH = "rf_model.pkl"
MODEL_SVM_PATH = "svm_model.pkl"

rf_model = None
svm_model = None


def load_models():
    global rf_model, svm_model

    if os.path.exists(MODEL_RF_PATH) and os.path.exists(MODEL_SVM_PATH):
        print("✅ Loading ML models...")
        rf_model = joblib.load(MODEL_RF_PATH)
        svm_model = joblib.load(MODEL_SVM_PATH)
    else:
        print("⚠️ Models not found. Run train_model.py first.")


# =====================================
# INIT DB + LOAD MODELS
# =====================================
with app.app_context():
    # If existing sqlite schema is outdated, recreate tables.
    try:
        inspector = inspect(db.engine)
        if 'user' in inspector.get_table_names():
            user_cols = [c['name'] for c in inspector.get_columns('user')]
            required = {'username', 'email', 'password_hash', 'current_lexile', 'total_xp'}
            if not required.issubset(set(user_cols)):
                db.drop_all()
                db.create_all()
        else:
            db.create_all()
    except Exception:
        db.drop_all()
        db.create_all()

    load_models()


# =====================================
# HEALTH CHECK + SERVE FRONTEND
# =====================================
@app.route("/")
def home():
    return send_file('../frontend/dashboard.html')

@app.route("/library")
def library():
    return send_file('../frontend/library.html')

@app.route("/progress")
def progress():
    return send_file('../frontend/progress.html')

@app.route("/login")
def login_page():
    return send_file('../frontend/login.html')


# =====================================
# REGISTER USER
# =====================================

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    data = request.json
    required_fields = ['username', 'email', 'password', 'name', 'grade']

    for field in required_fields:
        if field not in data or not data[field]:
            return jsonify({"error": f"Missing {field}"}), 400

    username = data['username'].strip()
    email = data['email'].strip().lower()
    password = data['password']
    name = data['name'].strip()
    grade = int(data['grade'])

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 409

    user = User(
        username=username,
        email=email,
        name=name,
        grade=grade,
        current_lexile=500 + (grade * 50)
    )
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "User registered successfully",
        "user_id": user.id,
        "username": user.username
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login_user():
    data = request.json

    if not data.get('username') or not data.get('password'):
        return jsonify({"error": "Missing username or password"}), 400

    username = data['username'].strip()
    password = data['password']

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    session_token = str(uuid.uuid4())
    session['user_id'] = user.id
    session['username'] = user.username
    session['session_token'] = session_token

    return jsonify({
        "message": "Login successful",
        "user_id": user.id,
        "username": user.username,
        "name": user.name,
        "grade": user.grade,
        "current_lexile": user.current_lexile,
        "current_band": user.current_band,
        "total_xp": user.total_xp,
        "session_token": session_token
    }), 200


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logout successful"}), 200


@app.route('/api/auth/user', methods=['GET'])
def get_current_user():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "user_id": user.id,
        "username": user.username,
        "name": user.name,
        "grade": user.grade,
        "current_lexile": user.current_lexile,
        "current_band": user.current_band,
        "total_xp": user.total_xp
    }), 200


# =====================================
# GET PASSAGE DIFFICULTY (SIMPLE LOGIC)
# =====================================
@app.route('/api/passage/<int:grade>', methods=['GET'])
def get_passage(grade):

    if grade <= 5:
        difficulty = 0
    elif grade <= 8:
        difficulty = 1
    else:
        difficulty = 2

    return jsonify({
        "difficulty": difficulty
    })


# =====================================
# ML PREDICTION (RF + SVM ENSEMBLE)
# =====================================
@app.route('/api/predict', methods=['POST'])
def predict():
    global rf_model, svm_model

    if rf_model is None or svm_model is None:
        return jsonify({"error": "Models not loaded"}), 500

    data = request.json

    try:
        features = np.array([[
            data['lit_acc'],
            data['inf_acc'],
            data['voc_acc'],
            data['mid_acc'],
            data['overall'],
            data['time_f'],
            data['diff']
        ]])
    except KeyError as e:
        return jsonify({"error": f"Missing field {str(e)}"}), 400

    # Model predictions
    rf_prob = rf_model.predict_proba(features)[0]
    svm_prob = svm_model.predict_proba(features)[0]

    # Ensemble (average)
    combined = (rf_prob + svm_prob) / 2
    band = int(np.argmax(combined))

    # Convert to Lexile
    lexile_base = [300, 500, 700, 860, 1100, 1300]
    offset = int((data['overall'] - 0.5) * 100)
    lexile = max(100, min(1400, lexile_base[band] + offset))

    return jsonify({
        "band": band,
        "lexile": lexile,
        "confidence": float(np.max(combined))
    })


# =====================================
# SAVE ATTEMPT (DATA COLLECTION CORE)
# =====================================
@app.route('/api/submit', methods=['POST'])
def submit():
    data = request.json

    required_fields = [
        'user_id','lit_acc','inf_acc','voc_acc',
        'mid_acc','overall','time_f','diff','lexile','band'
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing {field}"}), 400

    attempt = Attempt(
        user_id=data['user_id'],
        lit_acc=data['lit_acc'],
        inf_acc=data['inf_acc'],
        voc_acc=data['voc_acc'],
        mid_acc=data['mid_acc'],
        overall=data['overall'],
        time_f=data['time_f'],
        diff=data['diff'],
        lexile=data['lexile'],
        band=data['band']
    )

    db.session.add(attempt)
    db.session.commit()
    
    # Update user progress
    user = User.query.get(data['user_id'])
    if user:
        # Calculate XP based on accuracy and difficulty
        xp_gain = int((data['overall'] * 100) + (data['diff'] * 50))
        user.total_xp += xp_gain
        user.current_lexile = data['lexile']
        user.current_band = data['band']
        db.session.commit()
        
        return jsonify({
            "message": "saved",
            "xp_gain": xp_gain,
            "total_xp": user.total_xp,
            "current_lexile": user.current_lexile,
            "current_band": user.current_band
        })
    
    return jsonify({"message": "saved"})


# =====================================
# EXPORT DATASET TO CSV
# =====================================
@app.route('/api/export', methods=['GET'])
def export_csv():
    attempts = Attempt.query.all()

    with open("dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "lit_acc","inf_acc","voc_acc","mid_acc",
            "overall","time_f","diff","label"
        ])

        for a in attempts:
            writer.writerow([
                a.lit_acc,
                a.inf_acc,
                a.voc_acc,
                a.mid_acc,
                a.overall,
                a.time_f,
                a.diff,
                a.band
            ])

    return jsonify({
        "message": "dataset.csv generated",
        "rows": len(attempts)
    })


# =====================================
# GET USER HISTORY
# =====================================
@app.route('/api/history/<int:user_id>', methods=['GET'])
def history(user_id):
    attempts = Attempt.query.filter_by(user_id=user_id).all()

    return jsonify([
        {
            "lexile": a.lexile,
            "band": a.band,
            "date": a.created_at.strftime("%Y-%m-%d %H:%M")
        } for a in attempts
    ])


# =====================================
# RUN APP
# =====================================
if __name__ == "__main__":
    app.run(debug=True)