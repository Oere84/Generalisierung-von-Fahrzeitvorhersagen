"""
Cross-Gemeinde-Evaluation aller gespeicherten Modelle
=======================================================
Laedt alle 20 Modelle aus models/ und evaluiert jedes einzelne auf
allen 18 Gemeinden des Datensatzes. Ergebnis: 360 Zeilen (20 x 18).

Dies ist der zentrale Evaluationsschritt: Er zeigt, wie gut ein Modell
auf Gemeinden generalisiert, die es im Training nie gesehen hat.

Fuer jedes Modell wird unterschieden:
- "Bekannte Gemeinden": Im Trainingsset enthalten
- "Unbekannte Gemeinden": Nicht im Training → Generalisierungstest

Die Differenz (unbekannt - bekannt) ist die "Generalisierungsluecke".

Aufruf:  python skripte/05_evaluate_all.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_START = time.time()

# -- Pfade ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # -> Projekt-Root
DATEN_DIR = BASE_DIR / "Daten"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "ergebnisse"
DATA_DIR = SCRIPT_DIR / "data"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096 if torch.cuda.is_available() else 1024

# Alle 18 Gemeinden dynamisch aus dem Daten-Ordner ermitteln
GEMEINDEN = sorted([f.stem for f in (DATEN_DIR / "travel_times").glob("GM*.csv")])
TARGET = "travel_time"

# Feature-Definition: Identisch zu 00_prepare_data.py.
# Beide Feature-Sets werden berechnet; pro Modell wird dann die passende
# Untermenge verwendet (Label-Set fuer D1-D5, Coord-Set fuer D6).
FEATURE_COLS_BASE = (
    ["hour_sin", "hour_cos"]
    + [f"weekday_{d}" for d in range(1, 7)]
    + ["is_weekend", "is_rush_hour"]
    + ["month_sin", "month_cos"]
    + ["segment_dist_m"]
    + ["dwell_time", "seg_position"]
    + ["travel_time_prev"]
)
FEATURE_COLS_LABEL = FEATURE_COLS_BASE + ["from_stop_enc", "to_stop_enc", "route_enc", "line_enc"]
FEATURE_COLS_COORD = FEATURE_COLS_BASE + ["from_lat", "from_lon", "to_lat", "to_lon"]
CAT_COLS = ["from_stop_enc", "to_stop_enc", "route_enc", "line_enc"]


def features_for(variant: str) -> list[str]:
    """Waehlt das Feature-Set passend zur Variante (coord bei D6, sonst label)."""
    return FEATURE_COLS_COORD if "D6" in variant else FEATURE_COLS_LABEL


# -- Modell-Definition (identisch zu 03_train_mlp_v2.py) -------------------
def emb_dim(n_cat: int) -> int:
    """Embedding-Dimension: sqrt(n_cat), begrenzt auf [4, 16]."""
    return min(16, max(4, int(n_cat ** 0.5)))


class EmbeddingMLP(nn.Module):
    """MLPv2 mit Entity Embeddings — Architektur muss identisch zum Training sein."""
    def __init__(self, n_cont: int, vocab_sizes: list[int], emb_dims: list[int]):
        """
        Baut Embedding-Tabellen + Feed-Forward-Netz auf. Architektur muss
        bit-genau zu der im Trainingsskript passen, sonst schlaegt das
        Laden der gespeicherten Gewichte fehl.
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, dim) for vs, dim in zip(vocab_sizes, emb_dims)
        ])
        n_input = n_cont + sum(emb_dims)
        self.net = nn.Sequential(
            nn.Linear(n_input, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cont, x_cat):
        """
        Forward-Pass: kategorische Indices in Embedding-Vektoren umwandeln,
        mit kontinuierlichen Features konkatenieren und durch das
        Feed-Forward-Netz schicken. Liefert die Fahrzeit-Vorhersage in
        Sekunden pro Sample (skalar).
        """
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_cont] + embs, dim=1)
        return self.net(x).squeeze(-1)


# -- Hilfsfunktionen (identisch zu 00_prepare_data.py) --------------------
def parse_points_vectorized(series):
    """Extrahiert Lon/Lat aus WKT-POINT-Strings."""
    extracted = series.str.extract(r"POINT\(([^ ]+) ([^ ]+)\)")
    return extracted[0].astype(float).values, extracted[1].astype(float).values


def haversine_vec(lon1, lat1, lon2, lat2):
    """Grosskreisdistanz in Metern (vektorisiert)."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — ignoriert Nullwerte im Nenner."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# Maximale Zeilenanzahl pro Gemeinde (Kompromiss Speed vs. Genauigkeit)
MAX_ROWS = 500_000


def prepare_gemeinde(name: str) -> pd.DataFrame | None:
    """
    Bereitet eine Gemeinde mit der v2-Feature-Pipeline auf.

    Wendet dieselbe Aufbereitung wie in 00_prepare_data.py an:
    1. Roh-CSVs laden (travel_times + dwell_times)
    2. Merge, Ausreisser-Filterung, Haversine-Distanz
    3. Feature Engineering (zyklisch, one-hot, label-encoding, lag)

    WICHTIG: Label-Encoding erfolgt hier pro Gemeinde. Die Codes sind
    daher NICHT konsistent mit denen im Training! Das ist beabsichtigt:
    Die Cross-Evaluation testet, ob ein Modell auf neue Encodings generalisiert.
    """
    tt_path = DATEN_DIR / "travel_times" / f"{name}.csv"
    dt_path = DATEN_DIR / "dwell_times" / f"{name}.csv"
    if not tt_path.exists() or not dt_path.exists():
        return None

    df_tt = pd.read_csv(tt_path, nrows=MAX_ROWS)
    df_dt = pd.read_csv(dt_path, nrows=MAX_ROWS)

    # Zeitstempel parsen und Dauern berechnen
    df_tt["from_time"] = pd.to_datetime(df_tt["from_time"])
    df_tt["to_time"] = pd.to_datetime(df_tt["to_time"])
    df_tt["travel_time"] = (df_tt["to_time"] - df_tt["from_time"]).dt.total_seconds()

    df_dt["from_time"] = pd.to_datetime(df_dt["from_time"])
    df_dt["to_time"] = pd.to_datetime(df_dt["to_time"])
    df_dt["dwell_time"] = (df_dt["to_time"] - df_dt["from_time"]).dt.total_seconds()

    df_tt["route"] = df_tt["route"].astype("int64")
    df_dt["route"] = df_dt["route"].astype("int64")

    # Left-Join: dwell_time an travel_times
    df = pd.merge(
        df_tt, df_dt[["date", "trip", "route", "stop", "dwell_time"]],
        how="left",
        left_on=["date", "trip", "route", "from_stop"],
        right_on=["date", "trip", "route", "stop"],
    ).drop(columns=["stop"])

    # Koordinaten extrahieren (werden fuer Distanz + D6-Features verwendet)
    lon_from, lat_from = parse_points_vectorized(df["from_geometry"])
    lon_to, lat_to = parse_points_vectorized(df["to_geometry"])
    df["from_lon"] = lon_from
    df["from_lat"] = lat_from
    df["to_lon"] = lon_to
    df["to_lat"] = lat_to
    df["segment_dist_m"] = haversine_vec(lon_from, lat_from, lon_to, lat_to)

    # Strict Cleaning (identisch zu 00_prepare_data.py):
    # Duplikate, travel_time, dist, speed, dwell_time, IQR
    df = df.drop_duplicates(
        subset=["date", "trip", "route", "from_stop", "to_stop", "from_time"]
    )
    df = df[(df["travel_time"] > 0) & (df["travel_time"] <= 600)].copy()
    df = df[df["segment_dist_m"] >= 10].copy()
    speed = df["segment_dist_m"] / df["travel_time"] * 3.6
    df = df[(speed >= 1) & (speed <= 100)].copy()
    df.loc[(df["dwell_time"] < 0) | (df["dwell_time"] > 300), "dwell_time"] = np.nan

    # IQR-Filter pro (route, from_stop, to_stop), asymmetrisch 1.5/3.0 IQR, Gruppen >= 30
    group_cols = ["route", "from_stop", "to_stop"]
    q = df.groupby(group_cols)["travel_time"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    q["cnt"] = df.groupby(group_cols).size()
    q["lower"] = q["q1"] - 1.5 * (q["q3"] - q["q1"])
    q["upper"] = q["q3"] + 3.0 * (q["q3"] - q["q1"])
    q = q.reset_index()
    df = df.merge(q, on=group_cols, how="left")
    keep = (df["cnt"] < 30) | ((df["travel_time"] >= df["lower"]) & (df["travel_time"] <= df["upper"]))
    df = df[keep].drop(columns=["q1", "q3", "cnt", "lower", "upper"]).copy()

    if len(df) < 100:
        return None

    # Feature Engineering
    df["date_dt"] = pd.to_datetime(df["date"])
    hour = df["from_time"].dt.hour
    df["hour_sin"] = np.sin(hour * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(hour * 2 * np.pi / 24)

    weekday = df["date_dt"].dt.dayofweek
    for d in range(1, 7):
        df[f"weekday_{d}"] = (weekday == d).astype(int)
    df["is_weekend"] = (weekday >= 5).astype(int)
    df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 15) & (hour <= 18))).astype(int)

    month = df["date_dt"].dt.month
    df["month_sin"] = np.sin(month * 2 * np.pi / 12)
    df["month_cos"] = np.cos(month * 2 * np.pi / 12)

    # Label-Encoding: Pro Gemeinde neu (nicht aus dem Trainingsset!)
    for col, new_col in [("from_stop", "from_stop_enc"), ("to_stop", "to_stop_enc"),
                          ("route", "route_enc"), ("line", "line_enc")]:
        df[new_col] = df[col].astype("category").cat.codes

    df["dwell_time"] = df["dwell_time"].fillna(0.0)
    df = df.sort_values(["date", "trip", "from_time"])
    df["seg_position"] = df.groupby(["date", "trip"]).cumcount()
    df["travel_time_prev"] = df.groupby(["date", "trip"])["travel_time"].shift(1)
    df["travel_time_prev"] = df["travel_time_prev"].fillna(0.0)

    # Liefere ALLE moeglichen Features zurueck (label + coord); Aufrufer waehlt.
    all_feats = list(dict.fromkeys(FEATURE_COLS_LABEL + FEATURE_COLS_COORD))
    return df[all_feats + [TARGET]].copy()


def evaluate_metrics(y_true, y_pred):
    """Berechnet MAE, RMSE und MAPE fuer ein Vorhersage-Array."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


@torch.no_grad()
def predict_mlpv2(model, X_cont, X_cat, y):
    """
    Vorhersage mit einem MLPv2-Modell.
    Verwendet DataLoader mit num_workers=0 (Windows-kompatibel).
    """
    model.eval()
    ds = TensorDataset(
        torch.tensor(X_cont, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds = []
    for X_c_b, X_cat_b, _ in loader:
        preds.append(model(X_c_b.to(DEVICE), X_cat_b.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Modelle laden: Alle 20 Modelle aus dem models/-Ordner
# Vier Modelltypen: XGBoost, Random Forest, Ridge, MLPv2
# Fuenf Datenvarianten: D1-D5
# ---------------------------------------------------------------------------
print("=" * 70)
print("Cross-Gemeinde-Evaluation")
print(f"Device: {DEVICE}   Gemeinden: {len(GEMEINDEN)}")
print("=" * 70)

models = {}

ALLE_VARIANTEN = ["D1_single", "D2_multi4", "D3_mittel6", "D4_gross10", "D5_fremd", "D6_coord"]

# XGBoost-Modelle laden (sklearn-kompatibel, keine Skalierung noetig)
for variant in ALLE_VARIANTEN:
    path = MODEL_DIR / f"xgboost_{variant}.joblib"
    if path.exists():
        models[f"XGB_{variant}"] = {"type": "sklearn", "model": joblib.load(path), "variant": variant}
        print(f"  Geladen: XGB_{variant}")

# Random-Forest-Modelle laden (sklearn, keine Skalierung noetig)
for variant in ALLE_VARIANTEN:
    path = MODEL_DIR / f"rf_{variant}.joblib"
    if path.exists():
        models[f"RF_{variant}"] = {"type": "sklearn", "model": joblib.load(path), "variant": variant}
        print(f"  Geladen: RF_{variant}")

# Ridge-Modelle laden (benoetigen den StandardScaler vom Training)
for variant in ALLE_VARIANTEN:
    path = MODEL_DIR / f"ridge_{variant}.joblib"
    if path.exists():
        scaler = joblib.load(MODEL_DIR / f"ridge_scaler_{variant}.joblib")
        models[f"Ridge_{variant}"] = {
            "type": "sklearn_scaled", "model": joblib.load(path),
            "scaler": scaler, "variant": variant,
        }
        print(f"  Geladen: Ridge_{variant}")

# MLPv2-Modelle laden (benoetigen Scaler + Config fuer Architektur-Rekonstruktion)
for variant in ALLE_VARIANTEN:
    pt_path = MODEL_DIR / f"mlpv2_{variant}.pt"
    cfg_path = MODEL_DIR / f"mlpv2_config_{variant}.joblib"
    scl_path = MODEL_DIR / f"mlpv2_scaler_{variant}.joblib"
    if pt_path.exists() and cfg_path.exists() and scl_path.exists():
        config = joblib.load(cfg_path)
        scaler = joblib.load(scl_path)
        # Architektur aus Config rekonstruieren und Gewichte laden
        model = EmbeddingMLP(config["n_cont"], config["vocab_sizes"], config["emb_dims"]).to(DEVICE)
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
        models[f"MLPv2_{variant}"] = {
            "type": "mlpv2", "model": model, "scaler": scaler,
            "config": config, "variant": variant,
        }
        print(f"  Geladen: MLPv2_{variant}")

print(f"\n{len(models)} Modelle geladen.\n")

# ---------------------------------------------------------------------------
# Evaluation: Jedes Modell auf jeder Gemeinde evaluieren.
# Pro Gemeinde werden die Roh-Daten geladen und aufbereitet,
# dann alle 20 Modelle darauf evaluiert.
# ---------------------------------------------------------------------------
all_results = []

for gm in GEMEINDEN:
    print(f"[{gm}] ", end="", flush=True)
    df = prepare_gemeinde(gm)
    if df is None:
        print("uebersprungen")
        continue

    y = df[TARGET].values
    n_seg = len(df)
    print(f"{n_seg:>9,} seg  ", end="", flush=True)

    for model_name, m in models.items():
        # Pro Modell: passendes Feature-Set auswaehlen (label fuer D1-D5, coord fuer D6)
        feat_cols = features_for(m["variant"])
        X = df[feat_cols]
        cont_cols = [c for c in feat_cols if c not in CAT_COLS]
        cat_cols = [c for c in feat_cols if c in CAT_COLS]

        # Vorhersage je nach Modelltyp
        if m["type"] == "sklearn":
            # XGBoost und RF: Direkte Vorhersage ohne Skalierung
            y_pred = m["model"].predict(X)
        elif m["type"] == "sklearn_scaled":
            # Ridge: Erst skalieren, dann vorhersagen
            X_scaled = m["scaler"].transform(X)
            y_pred = m["model"].predict(X_scaled)
        elif m["type"] == "mlpv2":
            # MLPv2: Kontinuierliche Features skalieren, kategorische clippen
            X_cont = m["scaler"].transform(X[cont_cols])
            X_cat = X[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(X), 0), dtype=np.int64)
            # Kategorische Indices clippen: Neue Gemeinden haben evtl.
            # mehr Kategorien als im Training gesehen -> auf max_idx clippen
            for i, col in enumerate(cat_cols):
                max_idx = m["config"]["vocab_sizes"][i] - 1
                X_cat[:, i] = np.clip(X_cat[:, i], 0, max_idx)
            y_pred = predict_mlpv2(m["model"], X_cont, X_cat, y)

        metrics = evaluate_metrics(y, y_pred)
        all_results.append({"gemeinde": gm, "modell": model_name, "n_segmente": n_seg, **metrics})

    # Bestes Modell fuer diese Gemeinde anzeigen
    gm_results = [r for r in all_results if r["gemeinde"] == gm]
    best = min(gm_results, key=lambda r: r["MAE"])
    print(f"best: {best['modell']} MAE={best['MAE']:.1f}s")

t_elapsed = time.time() - SCRIPT_START
print(f"\nGesamtlaufzeit: {t_elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Ergebnisse speichern: 360 Zeilen (20 Modelle x 18 Gemeinden)
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)
results_df_export = results_df.copy()
results_df_export["MAE"] = results_df_export["MAE"].round(2)
results_df_export["RMSE"] = results_df_export["RMSE"].round(2)
results_df_export["MAPE"] = results_df_export["MAPE"].round(1)
results_df_export.to_csv(OUTPUT_DIR / "cross_evaluation.csv", index=False)

# Zusammenfassung: Mittlere MAE ueber alle 18 Gemeinden (Ranking)
print(f"\n{'='*70}")
print("ZUSAMMENFASSUNG: Mittlere MAE ueber alle 18 Gemeinden")
print(f"{'='*70}")
means = results_df.groupby("modell")["MAE"].mean().sort_values()
for model_name, mae_val in means.items():
    print(f"  {model_name:<25s}  MAE={mae_val:.2f}s")

# ---------------------------------------------------------------------------
# Bekannt vs. Unbekannt: Zeigt die Generalisierungsluecke pro Modell.
# "Bekannt" = Gemeinden, die im Trainingsset der jeweiligen Datenvariante waren.
# "Unbekannt" = Alle anderen der 18 Gemeinden.
# ---------------------------------------------------------------------------
known_sets = {
    "D1_single": {"GM0047"},
    "D2_multi4": {"GM0047", "GM0059", "GM0281", "GM0590"},
    "D3_mittel6": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681"},
    "D4_gross10": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681", "GM0281", "GM1950", "GM1969", "GM1930"},
    "D5_fremd": {"GM0312", "GM0590", "GM0546", "GM0629", "GM1681"},
    # D6 = gleiche Gemeinden wie D4, nur andere Features (Koordinaten statt Label-Enc)
    "D6_coord": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681", "GM0281", "GM1950", "GM1969", "GM1930"},
}

print(f"\n{'='*70}")
print("BEKANNT vs UNBEKANNT (MAE)")
print(f"{'='*70}")
print(f"  {'Modell':<25s}  {'bekannt':>10s}  {'unbekannt':>10s}  {'Diff':>8s}")
print(f"  {'-'*55}")

for model_name in means.index:
    # Datenvariante aus dem Modellnamen extrahieren
    dv = None
    for k in known_sets:
        if k in model_name:
            dv = k
            break
    if dv is None:
        continue

    known = known_sets[dv]
    m_results = results_df[results_df["modell"] == model_name]
    k_mae = m_results[m_results["gemeinde"].isin(known)]["MAE"].mean()
    u_mae = m_results[~m_results["gemeinde"].isin(known)]["MAE"].mean()
    diff = u_mae - k_mae
    print(f"  {model_name:<25s}  {k_mae:>9.2f}s  {u_mae:>9.2f}s  {diff:>+7.2f}s")

print(f"\nErgebnisse gespeichert: {OUTPUT_DIR / 'cross_evaluation.csv'}")

# -- Timing ----------------------------------------------------------------
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "05_evaluate_all", "laufzeit_s": round(t_elapsed, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "05_evaluate_all"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)
