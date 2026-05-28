"""
Ridge Regression Training auf D1, D2, D3, D4, D5
===================================================
Ridge Regression ist eine lineare Regression mit L2-Regularisierung.
Die Regularisierung bestraft grosse Koeffizienten und verhindert Overfitting.

Ridge ist das einfachste Modell im Vergleich, zeigt aber die beste
Cross-Gemeinde-Generalisierung, weil es keine gemeinde-spezifischen
Muster in den Label-Encodings auswendig lernen kann.

StandardScaler wird vor dem Training angewendet, da Ridge-Koeffizienten
von der Skalierung der Features abhaengen.

Aufruf:  python skripte/04_train_ridge.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    print(f"  Ridge | {variant}")
    print(f"{'='*60}")

    t0 = time.time()
    d = DATA_DIR / variant
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    features = [c for c in train.columns if c != TARGET]

    # StandardScaler: Features auf Mittelwert=0, Std=1 normieren.
    # Fuer Ridge zwingend noetig, da die Regularisierung sonst
    # Features mit groesserer Skala staerker bestraft.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features].values)
    X_test = scaler.transform(test[features].values)
    y_train = train[TARGET].values
    y_test = test[TARGET].values

    print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(features)}")

    # Ridge mit alpha=1.0 (Standard-Regularisierungsstaerke)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_test = root_mean_squared_error(y_test, y_pred)
    mape_test = mape(y_test, y_pred)

    elapsed = time.time() - t0
    print(f"  Test MAE: {mae_test:.2f}s  RMSE: {rmse_test:.2f}s  MAPE: {mape_test:.1f}%")
    print(f"  Dauer: {elapsed:.1f}s")

    # Modell + Scaler speichern (beide werden fuer die Inferenz benoetigt)
    joblib.dump(model, MODEL_DIR / f"ridge_{variant}.joblib")
    joblib.dump(scaler, MODEL_DIR / f"ridge_scaler_{variant}.joblib")
    print(f"  Gespeichert: ridge_{variant}.joblib + scaler")

    results.append({
        "experiment": f"Ridge_{variant}",
        "modell": "Ridge",
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
results_df.to_csv(OUTPUT_DIR / "ridge_results.csv", index=False)
print(f"\n{'='*60}")
print("ZUSAMMENFASSUNG Ridge")
print(f"{'='*60}")
print(results_df[["experiment", "mae_test", "rmse_test", "mape_test", "zeit_s"]].to_string(index=False))

# Laufzeit in zentrale Timing-Datei schreiben (upsert-Logik)
elapsed_total = time.time() - SCRIPT_START
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "04_train_ridge", "laufzeit_s": round(elapsed_total, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "04_train_ridge"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)

print(f"\nGesamtlaufzeit: {elapsed_total:.1f}s")
