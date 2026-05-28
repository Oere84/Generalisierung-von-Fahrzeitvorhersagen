"""
Random Forest Training auf D1, D2, D3, D4, D5
================================================
Random Forest ist ein Ensemble aus unabhaengigen Entscheidungsbaeumen.
Im Gegensatz zu XGBoost (sequenziell) werden alle Baeume parallel und
unabhaengig trainiert (Bagging). Vorhersage = Mittelwert aller Baeume.

Kein Early Stopping moeglich (kein iterativer Prozess).
CPU-only (sklearn hat keine GPU-Unterstuetzung fuer Random Forests).

Aufruf:  python skripte/02_train_random_forest.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

SCRIPT_START = time.time()

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "ergebnisse"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "travel_time"
VARIANTEN = ["D1_single", "D2_multi4", "D3_mittel6", "D4_gross10", "D5_fremd", "D6_coord"]


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — ignoriert Nullwerte im Nenner."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


results = []

for variant in VARIANTEN:
    print(f"\n{'='*60}")
    print(f"  Random Forest | {variant}")
    print(f"{'='*60}")

    t0 = time.time()
    d = DATA_DIR / variant
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    features = [c for c in train.columns if c != TARGET]

    X_train, y_train = train[features].values, train[TARGET].values
    X_test, y_test = test[features].values, test[TARGET].values

    print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(features)}")

    # -----------------------------------------------------------------------
    # Random-Forest-Hyperparameter:
    # - n_estimators=300: 300 unabhaengige Baeume (mehr = stabiler, aber langsamer)
    # - max_depth=20: Maximale Baumtiefe (tiefer als XGBoost, da keine Boosting-Kaskade)
    # - min_samples_leaf=10: Mindestens 10 Samples pro Blatt (Regularisierung)
    # - max_features=0.8: Jeder Baum sieht 80% der Features (Feature-Bagging)
    # - n_jobs=-1: Alle CPU-Kerne nutzen (parallelisierbar, da Baeume unabhaengig)
    # - verbose=1: Fortschrittsanzeige waehrend des Trainings
    # -----------------------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)

    # Evaluation nur auf Test-Set (kein Val-Set noetig, da kein Early Stopping)
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    mape_test = mape(y_test, y_pred_test)

    elapsed = time.time() - t0
    print(f"\n  Test MAE: {mae_test:.2f}s  RMSE: {rmse_test:.2f}s  MAPE: {mape_test:.1f}%")
    print(f"  Dauer: {elapsed:.1f}s")

    # Modell speichern (Achtung: RF-Modelle koennen sehr gross werden, z.B. >1 GB fuer D4)
    model_path = MODEL_DIR / f"rf_{variant}.joblib"
    joblib.dump(model, model_path)
    print(f"  Gespeichert: {model_path.name}")

    results.append({
        "experiment": f"RF_{variant}",
        "modell": "RandomForest",
        "daten": variant,
        "train_n": len(train),
        "mae_test": round(mae_test, 2),
        "rmse_test": round(rmse_test, 2),
        "mape_test": round(mape_test, 1),
        "zeit_s": round(elapsed, 1),
    })

# ---------------------------------------------------------------------------
# Ergebnisse als CSV speichern + Zusammenfassung ausgeben
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "rf_results.csv", index=False)
print(f"\n{'='*60}")
print("ZUSAMMENFASSUNG Random Forest")
print(f"{'='*60}")
print(results_df[["experiment", "mae_test", "rmse_test", "mape_test", "zeit_s"]].to_string(index=False))

# Laufzeit in zentrale Timing-Datei schreiben (upsert-Logik)
elapsed_total = time.time() - SCRIPT_START
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "02_train_random_forest", "laufzeit_s": round(elapsed_total, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "02_train_random_forest"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)

print(f"\nGesamtlaufzeit: {elapsed_total:.1f}s")
