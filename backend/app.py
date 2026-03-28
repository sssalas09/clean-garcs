from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import numpy as np
import joblib
import os
import csv

# =====================================
# APP SETUP
# =====================================
app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
CORS(app)


# =====================================
# DATABASE MODELS
# =====================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
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
    db.create_all()
    load_models()


# =====================================
# HEALTH CHECK + SERVE FRONTEND
# =====================================
@app.route("/")
def home():
    return send_file('../frontend/index.html')



# =====================================
# REGISTER USER
# =====================================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json

    if not data.get("name") or not data.get("grade"):
        return jsonify({"error": "Missing name or grade"}), 400

    user = User(
        name=data['name'],
        grade=data['grade']
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"user_id": user.id})


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