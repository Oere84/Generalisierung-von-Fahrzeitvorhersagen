"""
MLPv2 (Entity Embeddings) Training auf D1, D2, D3, D4, D5
============================================================
GPU-beschleunigt (CUDA).

Das MLPv2-Modell verwendet Entity Embeddings fuer kategorische Variablen
(from_stop, to_stop, route, line). Statt roher Label-Encoding-Werte lernt
das Netzwerk dichte Vektoren (Embeddings) fuer jede Kategorie, die
semantische Aehnlichkeiten abbilden koennen.

Architektur: Embeddings + 3 Hidden Layers (128 -> 64 -> 32) mit
BatchNorm, ReLU und Dropout. Training mit Adam-Optimizer und
ReduceLROnPlateau-Scheduler.

WICHTIG: Kein DataLoader verwendet — Daten liegen als GPU-Tensoren vor.
DataLoader mit num_workers>0 crasht unter Windows (Fork-Problem).

Aufruf:  python skripte/03_train_mlp_v2.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
# DataLoader nicht verwendet - Daten direkt als GPU-Tensors fuer Speed auf Windows

SCRIPT_START = time.time()

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "ergebnisse"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# GPU/CPU-Erkennung und Batch-Groesse anpassen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8192 if torch.cuda.is_available() else 1024
TARGET = "travel_time"
VARIANTEN = ["D1_single", "D2_multi4", "D3_mittel6", "D4_gross10", "D5_fremd", "D6_coord"]

# Kategorische Spalten, die als Entity Embeddings verarbeitet werden
CAT_COLS = ["from_stop_enc", "to_stop_enc", "route_enc", "line_enc"]

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # TensorCore-Optimierung fuer schnellere Matmul-Operationen
    torch.set_float32_matmul_precision("high")


def emb_dim(n_cat: int) -> int:
    """
    Berechnet die Embedding-Dimension basierend auf der Kategorienanzahl.
    Heuristik: sqrt(n_cat), begrenzt auf [4, 16].
    Beispiel: 100 Haltestellen -> Embedding-Dim = 10.
    """
    return min(16, max(4, int(n_cat ** 0.5)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — ignoriert Nullwerte im Nenner."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class EmbeddingMLP(nn.Module):
    """
    Multi-Layer Perceptron mit Entity Embeddings.

    Aufbau:
    1. Embedding-Schichten: Jede kategorische Variable bekommt eine eigene
       Embedding-Tabelle (Lookup von Integer -> dichter Vektor)
    2. Konkatenation: Kontinuierliche Features + alle Embedding-Vektoren
    3. Feed-Forward: 128 -> 64 -> 32 -> 1 mit BatchNorm und Dropout

    Input:
    - x_cont: Tensor der kontinuierlichen Features [batch, n_cont]
    - x_cat: Tensor der kategorischen Indices [batch, n_cat]

    Output:
    - Vorhersage der Fahrzeit in Sekunden [batch]
    """
    def __init__(self, n_cont: int, vocab_sizes: list[int], emb_dims: list[int]):
        """
        Initialisiert eine Embedding-Tabelle pro kategorische Variable
        und das Feed-Forward-Netz mit fixer Tiefe (128-64-32-1).

        Args:
            n_cont: Anzahl kontinuierlicher Features.
            vocab_sizes: Vokabulargroesse pro kategorische Spalte.
            emb_dims: Embedding-Dimension pro kategorische Spalte.
        """
        super().__init__()
        # Eine Embedding-Schicht pro kategorische Variable
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, dim) for vs, dim in zip(vocab_sizes, emb_dims)
        ])
        # Gesamte Eingabe-Dimension: kontinuierlich + alle Embeddings
        n_input = n_cont + sum(emb_dims)
        self.net = nn.Sequential(
            nn.Linear(n_input, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cont, x_cat):
        """
        Forward-Pass des Netzes.

        Schlaegt fuer jede kategorische Spalte ihren Embedding-Vektor nach,
        konkateniert diese mit den kontinuierlichen Features und schickt
        das Ergebnis durch das Feed-Forward-Netz. Output ist die skalare
        Fahrzeit-Vorhersage in Sekunden pro Sample.
        """
        # Jede kategorische Spalte durch ihre Embedding-Schicht leiten
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Kontinuierliche Features und Embeddings zusammenfuegen
        x = torch.cat([x_cont] + embs, dim=1)
        return self.net(x).squeeze(-1)


def train_one_variant(variant: str):
    """
    Trainiert ein MLPv2-Modell auf einer Datenvariante.

    Ablauf:
    1. Daten laden und in kontinuierlich/kategorisch aufteilen
    2. StandardScaler auf kontinuierliche Features fitten
    3. Vocab-Sizes bestimmen (max. Index + 1 pro kategorische Variable)
    4. Alle Daten als GPU-Tensoren anlegen (kein DataLoader!)
    5. Training mit manuellem Mini-Batch-Loop
    6. Early Stopping auf Validation-MAE (Patience=10 Epochen)
    7. Bestes Modell evaluieren und speichern
    """
    print(f"\n{'='*60}")
    print(f"  MLPv2 | {variant}")
    print(f"{'='*60}")

    t0 = time.time()
    d = DATA_DIR / variant
    train_df = pd.read_csv(d / "train.csv")
    val_df = pd.read_csv(d / "val.csv")
    test_df = pd.read_csv(d / "test.csv")

    # Features aufteilen: kontinuierlich vs. kategorisch
    all_cols = [c for c in train_df.columns if c != TARGET]
    cont_cols = [c for c in all_cols if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in all_cols]

    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"  Cont: {len(cont_cols)}  Cat: {len(cat_cols)}")

    # StandardScaler: Kontinuierliche Features auf Mittelwert=0, Std=1 normieren
    # Wichtig fuer neuronale Netze (Gradient-basiertes Lernen)
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(train_df[cont_cols].values)
    X_val_cont = scaler.transform(val_df[cont_cols].values)
    X_test_cont = scaler.transform(test_df[cont_cols].values)

    # Kategorische Features als Integer-Indices
    X_train_cat = train_df[cat_cols].values.astype(np.int64)
    X_val_cat = val_df[cat_cols].values.astype(np.int64)
    X_test_cat = test_df[cat_cols].values.astype(np.int64)

    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    y_test = test_df[TARGET].values

    # Vocab-Sizes: Maximaler Index + 1 (ueber alle Splits, damit kein OOV)
    vocab_sizes = [int(np.max(np.concatenate([X_train_cat[:, i], X_val_cat[:, i], X_test_cat[:, i]]))) + 1
                   for i in range(len(cat_cols))]
    dims = [emb_dim(vs) for vs in vocab_sizes]
    print(f"  Vocab: {dict(zip(cat_cols, vocab_sizes))}")
    print(f"  Emb dims: {dict(zip(cat_cols, dims))}")

    # -----------------------------------------------------------------------
    # Daten als GPU-Tensoren: Statt DataLoader werden alle Daten direkt auf
    # die GPU geladen. Dies vermeidet den DataLoader-Overhead und das
    # num_workers-Problem unter Windows. Mini-Batches werden per Index-Slicing
    # auf dem GPU-Tensor erzeugt.
    # -----------------------------------------------------------------------
    t_train_cont = torch.tensor(X_train_cont, dtype=torch.float32, device=DEVICE)
    t_train_cat = torch.tensor(X_train_cat, dtype=torch.long, device=DEVICE)
    t_train_y = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    t_val_cont = torch.tensor(X_val_cont, dtype=torch.float32, device=DEVICE)
    t_val_cat = torch.tensor(X_val_cat, dtype=torch.long, device=DEVICE)
    t_val_y = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
    t_test_cont = torch.tensor(X_test_cont, dtype=torch.float32, device=DEVICE)
    t_test_cat = torch.tensor(X_test_cat, dtype=torch.long, device=DEVICE)

    n_train = len(y_train)
    if torch.cuda.is_available():
        print(f"  GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")

    # Modell erstellen und auf Device verschieben
    model = EmbeddingMLP(len(cont_cols), vocab_sizes, dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Lernrate halbieren wenn Val-MAE 5 Epochen nicht sinkt
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()  # MSE als Verlustfunktion (Einheit: s^2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n_params:,}")

    # -----------------------------------------------------------------------
    # Training: Manueller Mini-Batch-Loop mit GPU-residenten Tensoren.
    # Pro Epoche werden die Trainings-Indices zufaellig permutiert.
    # -----------------------------------------------------------------------
    best_val_mae = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(60):
        model.train()
        # Zufaellige Permutation der Trainingsindizes (auf GPU)
        perm = torch.randperm(n_train, device=DEVICE)
        train_loss = 0.0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred = model(t_train_cont[idx], t_train_cat[idx])
            loss = criterion(pred, t_train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(idx)

        train_loss /= n_train  # Mittlerer MSE-Loss in s^2

        # Validation: Vorhersage in Batches (ohne Gradient)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(0, len(t_val_y), BATCH_SIZE):
                pred = model(t_val_cont[i:i+BATCH_SIZE], t_val_cat[i:i+BATCH_SIZE])
                val_preds.append(pred.cpu().numpy())

        val_pred = np.concatenate(val_preds)
        val_mae = mean_absolute_error(y_val, val_pred)
        scheduler.step(val_mae)

        # Fortschritt alle 5 Epochen ausgeben
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:>3d}  loss={train_loss:.1f}  val_mae={val_mae:.2f}s  lr={lr:.1e}")

        # Early Stopping: Bestes Modell speichern, bei 10 Epochen ohne Verbesserung stoppen
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping bei Epoch {epoch+1}")
                break

    # Bestes Modell laden und auf Test-Set evaluieren
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    test_preds = []
    with torch.no_grad():
        for i in range(0, len(t_test_cont), BATCH_SIZE):
            pred = model(t_test_cont[i:i+BATCH_SIZE], t_test_cat[i:i+BATCH_SIZE])
            test_preds.append(pred.cpu().numpy())

    y_pred_test = np.concatenate(test_preds)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    mape_test = mape(y_test, y_pred_test)

    elapsed = time.time() - t0
    print(f"\n  Best Val MAE: {best_val_mae:.2f}s")
    print(f"  Test MAE: {mae_test:.2f}s  RMSE: {rmse_test:.2f}s  MAPE: {mape_test:.1f}%")
    print(f"  Dauer: {elapsed:.1f}s")

    # Drei Dateien pro Modell: Gewichte (.pt), Scaler, Konfiguration
    torch.save(best_state, MODEL_DIR / f"mlpv2_{variant}.pt")
    joblib.dump(scaler, MODEL_DIR / f"mlpv2_scaler_{variant}.joblib")
    joblib.dump({
        "n_cont": len(cont_cols), "vocab_sizes": vocab_sizes, "emb_dims": dims,
        "cont_cols": cont_cols, "cat_cols": cat_cols,
    }, MODEL_DIR / f"mlpv2_config_{variant}.joblib")
    print(f"  Gespeichert: mlpv2_{variant}.pt + scaler + config")

    return {
        "experiment": f"MLPv2_{variant}",
        "modell": "MLPv2",
        "daten": variant,
        "train_n": len(train_df),
        "mae_val": round(best_val_mae, 2),
        "mae_test": round(mae_test, 2),
        "rmse_test": round(rmse_test, 2),
        "mape_test": round(mape_test, 1),
        "params": n_params,
        "zeit_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Alle 5 Datenvarianten nacheinander trainieren
# ---------------------------------------------------------------------------
results = []
for v in VARIANTEN:
    results.append(train_one_variant(v))

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "mlpv2_results.csv", index=False)

print(f"\n{'='*60}")
print("ZUSAMMENFASSUNG MLPv2")
print(f"{'='*60}")
print(results_df[["experiment", "mae_test", "rmse_test", "mape_test", "params", "zeit_s"]].to_string(index=False))

# Laufzeit in zentrale Timing-Datei schreiben (upsert-Logik)
elapsed_total = time.time() - SCRIPT_START
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "03_train_mlp_v2", "laufzeit_s": round(elapsed_total, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "03_train_mlp_v2"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)

print(f"\nGesamtlaufzeit: {elapsed_total:.1f}s")
