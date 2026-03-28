import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------
# GENERATE DUMMY DATA
# -----------------------------
np.random.seed(42)

X = []
y = []

for _ in range(1000):
    lit = np.random.rand()
    inf = np.random.rand()
    voc = np.random.rand()
    mid = np.random.rand()
    overall = (lit + inf + voc + mid) / 4
    time_f = np.random.rand()
    diff = np.random.randint(0, 3)

    features = [lit, inf, voc, mid, overall, time_f, diff]

    # Simulated label logic (bands 0–5)
    score = overall + (diff * 0.1)

    if score < 0.3:
        label = 0
    elif score < 0.45:
        label = 1
    elif score < 0.6:
        label = 2
    elif score < 0.75:
        label = 3
    elif score < 0.9:
        label = 4
    else:
        label = 5

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# -----------------------------
# MODELS
# -----------------------------
rf = RandomForestClassifier(n_estimators=100)
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])

rf.fit(X, y)
svm.fit(X, y)

# Save models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(svm, "svm_model.pkl")

print("Models trained and saved!")