"""
XGBoost Training auf D1, D2, D3, D4, D5
==========================================
Speichert Modelle + Metriken. Nutzt Early Stopping auf Val-Set.

XGBoost ist ein Gradient-Boosting-Verfahren auf Entscheidungsbaeumen.
Es baut sequenziell schwache Lerner (Baeume) auf, wobei jeder neue Baum
die Fehler der bisherigen Ensemble-Vorhersage korrigiert.

Aufruf:  python skripte/01_train_xgboost.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

SCRIPT_START = time.time()

# ---------------------------------------------------------------------------
# GPU-Erkennung: Wenn CUDA (via PyTorch) verfuegbar ist, wird XGBoost
# auf der GPU trainiert (deutlich schneller bei grossen Datensaetzen).
# ---------------------------------------------------------------------------
try:
    import torch
    XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    XGB_DEVICE = "cpu"

# -- Pfade ----------------------------------------------------------------
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


def load_data(variant):
    """Laedt Train/Val/Test-CSVs fuer eine Datenvariante."""
    d = DATA_DIR / variant
    train = pd.read_csv(d / "train.csv")
    val = pd.read_csv(d / "val.csv")
    test = pd.read_csv(d / "test.csv")
    features = [c for c in train.columns if c != TARGET]
    return train, val, test, features


results = []

for variant in VARIANTEN:
    print(f"\n{'='*60}")
    print(f"  XGBoost | {variant}")
    print(f"{'='*60}")

    t0 = time.time()
    train, val, test, features = load_data(variant)

    X_train, y_train = train[features].values, train[TARGET].values
    X_val, y_val = val[features].values, val[TARGET].values
    X_test, y_test = test[features].values, test[TARGET].values

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"  Features: {len(features)}")

    # -----------------------------------------------------------------------
    # XGBoost-Hyperparameter:
    # - n_estimators=500: Maximal 500 Boosting-Runden (Early Stopping begrenzt)
    # - max_depth=8: Mittlere Baumtiefe, balanciert Kapazitaet vs. Overfitting
    # - learning_rate=0.1: Schrittweite pro Baum
    # - subsample/colsample_bytree=0.8: Stochastisches Gradient Boosting
    #   (80% Zeilen/Spalten pro Baum -> Regularisierung)
    # - min_child_weight=5: Minimale Summe der Instanzgewichte in einem Blatt
    # - reg_alpha/reg_lambda: L1/L2-Regularisierung der Blattgewichte
    # - tree_method="hist": Histogramm-basierter Algorithmus (GPU-kompatibel)
    # - early_stopping_rounds=20: Stopp wenn Val-MAE sich 20 Runden nicht bessert
    # -----------------------------------------------------------------------
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device=XGB_DEVICE,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="mae",
    )

    # Training mit Validierungs-Monitoring fuer Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Vorhersage auf Val- und Test-Set
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metriken berechnen
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    mape_test = mape(y_test, y_pred_test)

    elapsed = time.time() - t0
    print(f"\n  Val  MAE: {mae_val:.2f}s")
    print(f"  Test MAE: {mae_test:.2f}s  RMSE: {rmse_test:.2f}s  MAPE: {mape_test:.1f}%")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Dauer: {elapsed:.1f}s")

    # Modell als joblib speichern (inkl. aller Baeume + Hyperparameter)
    model_path = MODEL_DIR / f"xgboost_{variant}.joblib"
    joblib.dump(model, model_path)
    print(f"  Gespeichert: {model_path.name}")

    results.append({
        "experiment": f"XGB_{variant}",
        "modell": "XGBoost",
        "daten": variant,
        "train_n": len(train),
        "mae_val": round(mae_val, 2),
        "mae_test": round(mae_test, 2),
        "rmse_test": round(rmse_test, 2),
        "mape_test": round(mape_test, 1),
        "best_iter": model.best_iteration,
        "zeit_s": round(elapsed, 1),
    })

# ---------------------------------------------------------------------------
# Ergebnisse als CSV speichern + Zusammenfassung ausgeben
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "xgboost_results.csv", index=False)
print(f"\n{'='*60}")
print("ZUSAMMENFASSUNG XGBoost")
print(f"{'='*60}")
print(results_df[["experiment", "mae_test", "rmse_test", "mape_test", "zeit_s"]].to_string(index=False))

# Laufzeit in zentrale Timing-Datei schreiben (upsert-Logik)
elapsed_total = time.time() - SCRIPT_START
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "01_train_xgboost", "laufzeit_s": round(elapsed_total, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "01_train_xgboost"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)

print(f"\nGesamtlaufzeit: {elapsed_total:.1f}s")
