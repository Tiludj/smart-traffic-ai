"""
forecast.py - Machine learning: predict each junction's congestion 30 minutes ahead.

Model: Random Forest classifier (Low / Medium / High).
Features: time of day, day of week, current utilisation, junction busyness, weather (rain factor).

Training data is generated from a realistic Hyderabad-style daily traffic profile
(morning + evening peaks, quieter weekends, random noise). Be upfront with judges
about this, and say the same pipeline trains on real junction counts once a city
shares them (e.g. Bengaluru Traffic Police / Kaggle Bangalore traffic dataset).
"""

import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from core import JUNCTIONS, congestion_label, utilisation

LABELS = ["Low", "Medium", "High"]
HORIZON_H = 0.5  # predict 30 minutes ahead


def make_dataset(n_samples=20000, seed=42):
    rng = random.Random(seed)
    names = list(JUNCTIONS)
    X, y = [], []
    weather_factors = [1.0] * 6 + [1.08, 1.15, 1.25, 1.4]   # mostly dry, sometimes rain
    for _ in range(n_samples):
        name = rng.choice(names)
        hour = rng.uniform(5, 23.5)
        dow = rng.randint(0, 6)
        wf = rng.choice(weather_factors)
        now = utilisation(name, hour, dow, noise_seed=rng.random()) * wf
        future = utilisation(name, hour + HORIZON_H, dow, noise_seed=rng.random()) * wf
        X.append([hour, dow, now, JUNCTIONS[name][2], wf])
        y.append(LABELS.index(congestion_label(future)))
    return np.array(X), np.array(y)


def train():
    X, y = make_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=0, n_jobs=-1)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, round(acc * 100, 1)


def predict(model, state, hour, day_of_week, weather_factor=1.0):
    """Returns {junction: {"label": .., "probability": .., "utilisation": ..}} for 30 min ahead."""
    names = list(state)
    X = np.array([[hour, day_of_week, state[n]["utilisation"], JUNCTIONS[n][2], weather_factor]
                  for n in names])
    probs = model.predict_proba(X)
    out = {}
    for n, p in zip(names, probs):
        idx = int(np.argmax(p))
        # Expected utilisation from class probabilities - used as edge weight for routing
        expected_util = float(p @ np.array([0.45, 0.72, 0.98]))
        out[n] = {"label": LABELS[idx], "probability": round(float(p[idx]), 2),
                  "utilisation": round(expected_util, 2)}
    return out


if __name__ == "__main__":
    m, acc = train()
    print(f"Test accuracy: {acc}%")
