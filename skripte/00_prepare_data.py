"""
Datenaufbereitung — 6 Datenvarianten
======================================
Erzeugt fuer jede Variante train/val/test.csv + label_mappings.pkl
in einem eigenen Unterordner unter data/.

Varianten:
  D1: GM0047 (Single)
  D2: GM0047 + GM0059 + GM0281 + GM0590 (Multi-4)
  D3: GM0047 + GM0312 + GM0590 + GM0546 + GM0629 + GM1681 (Mittel-6, divers)
  D4: D3 + GM0281 + GM1950 + GM1969 + GM1930 (Gross-10)
  D5: D3 ohne GM0047 (Fremd — haertester Generalisierungstest)
  D6: wie D4, aber mit Koordinaten-Features statt Label-Encoding
      (Ablationstest: scheitern Baum-Modelle an den Labels oder am Modell?)

Feature-Sets:
  "label": 20 Features — zyklisch, Weekday-OH, Haversine, from/to_stop_enc,
          route_enc, line_enc, dwell, seg_position, lag
  "coord": 20 Features — wie "label", aber Stop-IDs/route/line ersetzt durch
          from_lat, from_lon, to_lat, to_lon

Bereinigung (strict, ab V02b):
  - travel_time in (0, 600] s
  - dwell_time in [0, 300] s (Werte ausserhalb -> NaN)
  - segment_dist_m >= 10 m (GPS-Jitter raus)
  - speed in [1, 100] km/h (Bus muss sich bewegen, keine Rennwagen)
  - Duplikat-Entfernung: (date, trip, route, from_stop, to_stop, from_time)
  - IQR-Filter pro (gemeinde, route, from_stop, to_stop): travel_time
    ausserhalb [Q1 - 1.5*IQR, Q3 + 3*IQR] entfernt, nur bei Gruppen >= 30.
    Asymmetrisch: untere Flanke = Sensorfehler, obere = legitime Staus.

Aufruf:  python skripte/00_prepare_data.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_START = time.time()

# ---------------------------------------------------------------------------
# Pfade: BASE_DIR zeigt auf das Projekt-Wurzelverzeichnis,
# DATEN_DIR auf die Roh-CSV-Dateien, DATA_DIR auf den Ausgabe-Ordner
# fuer die aufbereiteten Datenvarianten.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # -> Projekt-Root
DATEN_DIR = BASE_DIR / "Daten"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "ergebnisse"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Datenvarianten: gemeinden = Liste von GM-IDs, features = "label" oder "coord".
# D1-D5 verwenden Label-Encoding (klassisch).
# D6 verwendet Koordinaten statt Label-Encoding als Ablationstest.
# ---------------------------------------------------------------------------
D3_GEMEINDEN = ["GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681"]
D4_GEMEINDEN = D3_GEMEINDEN + ["GM0281", "GM1950", "GM1969", "GM1930"]

VARIANTEN = {
    "D1_single":  {"gemeinden": ["GM0047"], "features": "label"},
    "D2_multi4":  {"gemeinden": ["GM0047", "GM0059", "GM0281", "GM0590"], "features": "label"},
    "D3_mittel6": {"gemeinden": D3_GEMEINDEN, "features": "label"},
    "D4_gross10": {"gemeinden": D4_GEMEINDEN, "features": "label"},
    "D5_fremd":   {"gemeinden": ["GM0312", "GM0590", "GM0546", "GM0629", "GM1681"], "features": "label"},
    "D6_coord":   {"gemeinden": D4_GEMEINDEN, "features": "coord"},
}

# ---------------------------------------------------------------------------
# Feature-Spalten (20 Features in beiden Sets, fuer Vergleichbarkeit):
#
# Gemeinsame Basis (16):
#   - Zyklische Zeit: hour_sin/cos, month_sin/cos
#   - Wochentag: One-Hot Di-So (Mo = Referenz)
#   - Binaere Flags: is_weekend, is_rush_hour
#   - segment_dist_m, dwell_time, seg_position, travel_time_prev
#
# Label-Set (4): from_stop_enc, to_stop_enc, route_enc, line_enc
# Coord-Set (4): from_lat, from_lon, to_lat, to_lon
# ---------------------------------------------------------------------------
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

TARGET = "travel_time"


# ---------------------------------------------------------------------------
# Hilfsfunktion: Extrahiert Lon/Lat aus WKT-POINT-Strings.
# ---------------------------------------------------------------------------
def parse_points_vectorized(series):
    """
    Extrahiert Laengen- und Breitengrad aus einer Pandas-Serie von
    WKT-POINT-Strings (Format: ``POINT(lon lat)``).

    Vektorisiert ueber alle Zeilen via Regex; deutlich schneller als
    eine Python-Schleife.

    Returns:
        tuple[np.ndarray, np.ndarray]: (lon_array, lat_array) als floats.
    """
    extracted = series.str.extract(r"POINT\(([^ ]+) ([^ ]+)\)")
    return extracted[0].astype(float).values, extracted[1].astype(float).values


def haversine_vec(lon1, lat1, lon2, lat2):
    """
    Berechnet die Grosskreisdistanz zwischen zwei Punkten in Metern.

    Vektorisierte Numpy-Implementierung der Haversine-Formel; arbeitet
    elementweise auf gleich langen Arrays von Koordinaten. Erdradius
    wird konstant mit 6 371 000 m angenommen (Mittelwert).
    """
    R = 6_371_000  # Erdradius in Metern
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Cleaning-Statistik: Wird pro Gemeinde und pro Schritt befuellt und am Ende
# als ergebnisse/cleaning_stats.csv exportiert. Im Paper als Tabelle zitierbar.
# ---------------------------------------------------------------------------
cleaning_stats: list[dict] = []


def _log(scope: str, step: str, before: int, after: int):
    """Fuegt einen Eintrag in die globale cleaning_stats-Liste ein."""
    cleaning_stats.append({
        "scope": scope,
        "step": step,
        "rows_before": before,
        "rows_after": after,
        "removed": before - after,
        "removed_pct": round((before - after) / max(before, 1) * 100, 3),
    })


def load_and_clean(name: str) -> pd.DataFrame:
    """
    Laedt eine Gemeinde, merged dwell_times, bereinigt Ausreisser.

    Reihenfolge der Filter:
    1. Load + Join + Zeitstempel-Parsing
    2. Koordinaten extrahieren + Haversine-Distanz
    3. Duplikate entfernen
    4. travel_time in (0, 600] s
    5. segment_dist_m >= 10 m (GPS-Jitter)
    6. speed in [1, 100] km/h
    7. dwell_time-Grenzen (< 0 oder > 300 -> NaN)
    8. IQR-Filter pro (route, from_stop, to_stop) bei Gruppen >= 30
    """
    print(f"    [{name}] Laden...", end=" ", flush=True)
    df_tt = pd.read_csv(DATEN_DIR / "travel_times" / f"{name}.csv")
    df_dt = pd.read_csv(DATEN_DIR / "dwell_times" / f"{name}.csv")

    df_tt["from_time"] = pd.to_datetime(df_tt["from_time"])
    df_tt["to_time"] = pd.to_datetime(df_tt["to_time"])
    df_tt["travel_time"] = (df_tt["to_time"] - df_tt["from_time"]).dt.total_seconds()

    df_dt["from_time"] = pd.to_datetime(df_dt["from_time"])
    df_dt["to_time"] = pd.to_datetime(df_dt["to_time"])
    df_dt["dwell_time"] = (df_dt["to_time"] - df_dt["from_time"]).dt.total_seconds()

    df_tt["route"] = df_tt["route"].astype("int64")
    df_dt["route"] = df_dt["route"].astype("int64")

    df = pd.merge(
        df_tt,
        df_dt[["date", "trip", "route", "stop", "dwell_time"]],
        how="left",
        left_on=["date", "trip", "route", "from_stop"],
        right_on=["date", "trip", "route", "stop"],
    ).drop(columns=["stop"])

    n_load = len(df)
    _log(name, "01_load_and_join", n_load, n_load)

    # Koordinaten direkt aus Geometrie extrahieren und als Spalten halten.
    # Werden fuer Distanz UND als D6-Features verwendet.
    lon_from, lat_from = parse_points_vectorized(df["from_geometry"])
    lon_to, lat_to = parse_points_vectorized(df["to_geometry"])
    df["from_lon"] = lon_from
    df["from_lat"] = lat_from
    df["to_lon"] = lon_to
    df["to_lat"] = lat_to
    df["segment_dist_m"] = haversine_vec(lon_from, lat_from, lon_to, lat_to)

    # 3. Duplikate: identische Trip-Segmente in Rohdaten oder durch Merge
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["date", "trip", "route", "from_stop", "to_stop", "from_time"]
    )
    _log(name, "02_duplicates", n_before, len(df))

    # 4. travel_time in (0, 600]
    n_before = len(df)
    df = df[(df["travel_time"] > 0) & (df["travel_time"] <= 600)].copy()
    _log(name, "03_travel_time_0_600", n_before, len(df))

    # 5. segment_dist_m >= 10 m
    n_before = len(df)
    df = df[df["segment_dist_m"] >= 10].copy()
    _log(name, "04_segment_dist_min10m", n_before, len(df))

    # 6. Geschwindigkeit in [1, 100] km/h
    n_before = len(df)
    speed = df["segment_dist_m"] / df["travel_time"] * 3.6
    df = df[(speed >= 1) & (speed <= 100)].copy()
    _log(name, "05_speed_1_100_kmh", n_before, len(df))

    # 7. dwell_time-Grenzen: Werte < 0 oder > 300 -> NaN (keine Zeilen entfernt)
    mask_dwell = (df["dwell_time"] < 0) | (df["dwell_time"] > 300)
    n_nan_set = int(mask_dwell.sum())
    df.loc[mask_dwell, "dwell_time"] = np.nan
    cleaning_stats.append({
        "scope": name, "step": "06_dwell_time_nan_set",
        "rows_before": len(df), "rows_after": len(df),
        "removed": 0, "removed_pct": 0.0,
        "note": f"{n_nan_set} dwell_time-Werte auf NaN gesetzt",
    })

    # 8. IQR-Filter pro (route, from_stop, to_stop). Asymmetrisch:
    #    [Q1 - 1.5*IQR, Q3 + 3*IQR], nur bei Gruppen >= 30.
    df = _apply_iqr_filter(df, name)

    df["gemeinde"] = name

    print(f"{n_load:>9,} -> {len(df):>9,} ({n_load - len(df):,} entfernt)")
    return df


def _apply_iqr_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """
    Asymmetrischer IQR-Filter auf travel_time pro (route, from_stop, to_stop).

    Schwellen: [Q1 - 1.5*IQR, Q3 + 3*IQR]
      - Untere Flanke 1.5x IQR: zu kurze Fahrzeiten sind fast immer Sensorfehler,
        da kann man strikt filtern.
      - Obere Flanke 3x IQR: lange Fahrzeiten haben legitime Ursachen (Stau,
        Baustelle, Schulferien-Verkehr), die will man NICHT verlieren.

    Wirkt nur auf Gruppen mit >= 30 Beobachtungen (sonst zu instabil).
    """
    group_cols = ["route", "from_stop", "to_stop"]

    q = df.groupby(group_cols)["travel_time"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    q["cnt"] = df.groupby(group_cols).size()
    q["iqr"] = q["q3"] - q["q1"]
    q["lower"] = q["q1"] - 1.5 * q["iqr"]
    q["upper"] = q["q3"] + 3.0 * q["iqr"]
    q = q.reset_index()

    n_before = len(df)
    df = df.merge(q, on=group_cols, how="left")
    keep = (df["cnt"] < 30) | ((df["travel_time"] >= df["lower"]) & (df["travel_time"] <= df["upper"]))
    df = df[keep].drop(columns=["q1", "q3", "cnt", "iqr", "lower", "upper"]).copy()
    _log(scope, "07_iqr_per_segment", n_before, len(df))
    return df


def engineer_features(df: pd.DataFrame, feature_kind: str) -> pd.DataFrame:
    """
    Wendet die Feature-Pipeline an.

    Gemeinsame Features werden immer berechnet; Label-Encoding nur bei
    feature_kind == "label" (sonst weggelassen). Koordinaten sind bereits
    als Spalten aus load_and_clean vorhanden.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    hour = df["from_time"].dt.hour

    # Zyklische Stunden-Kodierung: vermeidet kuenstlichen Sprung bei 23->0 Uhr
    df["hour_sin"] = np.sin(hour * 2 * np.pi / 24)
    df["hour_cos"] = np.cos(hour * 2 * np.pi / 24)

    # Wochentag (Mo=0 ist Referenzkategorie, wird nicht kodiert)
    weekday = df["date_dt"].dt.dayofweek
    for d in range(1, 7):
        df[f"weekday_{d}"] = (weekday == d).astype(int)
    df["is_weekend"] = (weekday >= 5).astype(int)
    df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 15) & (hour <= 18))).astype(int)

    month = df["date_dt"].dt.month
    df["month_sin"] = np.sin(month * 2 * np.pi / 12)
    df["month_cos"] = np.cos(month * 2 * np.pi / 12)

    # Label-Encoding nur bei "label"-Varianten.
    if feature_kind == "label":
        for col, new_col in [("from_stop", "from_stop_enc"), ("to_stop", "to_stop_enc"),
                              ("route", "route_enc"), ("line", "line_enc")]:
            df[new_col] = df[col].astype("category").cat.codes

    df["dwell_time"] = df["dwell_time"].fillna(0.0)
    df = df.sort_values(["date", "trip", "from_time"])
    df["seg_position"] = df.groupby(["date", "trip", "gemeinde"]).cumcount()

    df["travel_time_prev"] = df.groupby(["date", "trip", "gemeinde"])["travel_time"].shift(1)
    df["travel_time_prev"] = df["travel_time_prev"].fillna(0.0)

    return df


def split_and_save(df: pd.DataFrame, variant_name: str, feature_kind: str):
    """Zeitbasierter 70/15/15 Split + Export als CSV."""
    out_dir = DATA_DIR / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = FEATURE_COLS_LABEL if feature_kind == "label" else FEATURE_COLS_COORD
    df_model = df[feature_cols + [TARGET, "date_dt", "from_time"]].copy()
    df_model = df_model.sort_values(["date_dt", "from_time"]).reset_index(drop=True)

    n = len(df_model)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    df_train = df_model.iloc[:n_train]
    df_val = df_model.iloc[n_train:n_train + n_val]
    df_test = df_model.iloc[n_train + n_val:]

    drop_cols = ["date_dt", "from_time"]
    df_train.drop(columns=drop_cols).to_csv(out_dir / "train.csv", index=False)
    df_val.drop(columns=drop_cols).to_csv(out_dir / "val.csv", index=False)
    df_test.drop(columns=drop_cols).to_csv(out_dir / "test.csv", index=False)

    # Label-Mappings nur bei Label-Varianten speichern.
    if feature_kind == "label":
        label_maps = {}
        for col in ["from_stop", "to_stop", "route", "line"]:
            if col in df.columns:
                cat = df[col].astype("category")
                mapping = dict(enumerate(cat.cat.categories))
                label_maps[col] = mapping
        pd.to_pickle(label_maps, out_dir / "label_mappings.pkl")

    print(f"    Train: {len(df_train):>10,}  Val: {len(df_val):>8,}  Test: {len(df_test):>8,}")
    for f in ["train.csv", "val.csv", "test.csv"]:
        size = (out_dir / f).stat().st_size / 1024 / 1024
        print(f"    {f}: {size:.1f} MB")

    return len(df_train), len(df_val), len(df_test)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
print("=" * 70)
print("Datenaufbereitung (strict cleaning) — 6 Varianten")
print("=" * 70)

alle_gemeinden = sorted(set(gm for cfg in VARIANTEN.values() for gm in cfg["gemeinden"]))
print(f"\nBenoetigte Gemeinden: {len(alle_gemeinden)}")

cache = {}
for gm in alle_gemeinden:
    cache[gm] = load_and_clean(gm)

print(f"\nAlle Gemeinden geladen. Cache: {len(cache)} DataFrames")

summary = []

for variant_name, cfg in VARIANTEN.items():
    t0 = time.time()
    gemeinden = cfg["gemeinden"]
    feature_kind = cfg["features"]
    print(f"\n{'='*70}")
    print(f"  {variant_name} ({feature_kind}): {gemeinden}")
    print(f"{'='*70}")

    dfs = [cache[gm] for gm in gemeinden]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Segmente (bereinigt): {len(df):,}")

    df = engineer_features(df, feature_kind)

    # Statistiken ausgeben
    if feature_kind == "label":
        n_from_stops = df["from_stop_enc"].nunique()
        n_to_stops = df["to_stop_enc"].nunique()
        n_routes = df["route_enc"].nunique()
        n_lines = df["line_enc"].nunique()
        print(f"  Features: {len(FEATURE_COLS_LABEL)} (label) | Stops: {n_from_stops}+{n_to_stops} | "
              f"Routen: {n_routes} | Linien: {n_lines}")
    else:
        n_from_stops = df["from_stop"].nunique() if "from_stop" in df.columns else 0
        n_to_stops = df["to_stop"].nunique() if "to_stop" in df.columns else 0
        n_routes = df["route"].nunique() if "route" in df.columns else 0
        n_lines = df["line"].nunique() if "line" in df.columns else 0
        print(f"  Features: {len(FEATURE_COLS_COORD)} (coord) | Lat-Range: "
              f"[{df['from_lat'].min():.3f}, {df['from_lat'].max():.3f}] | "
              f"Lon-Range: [{df['from_lon'].min():.3f}, {df['from_lon'].max():.3f}]")

    n_tr, n_va, n_te = split_and_save(df, variant_name, feature_kind)

    elapsed = time.time() - t0
    summary.append({
        "variante": variant_name,
        "features": feature_kind,
        "gemeinden": len(gemeinden),
        "segmente": len(df),
        "train": n_tr, "val": n_va, "test": n_te,
        "from_stops": n_from_stops,
        "to_stops": n_to_stops,
        "routen": n_routes,
        "linien": n_lines,
        "zeit_s": round(elapsed, 1),
    })
    print(f"  Dauer: {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# Zusammenfassung + Cleaning-Stats exportieren
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("ZUSAMMENFASSUNG")
print(f"{'='*70}")
sum_df = pd.DataFrame(summary)
print(sum_df.to_string(index=False))
sum_df.to_csv(OUTPUT_DIR / "datenvarianten_summary.csv", index=False)

stats_df = pd.DataFrame(cleaning_stats)
stats_df.to_csv(OUTPUT_DIR / "cleaning_stats.csv", index=False)
print(f"\nCleaning-Statistik: {OUTPUT_DIR / 'cleaning_stats.csv'}")

# Kurzuebersicht: Gesamtverlust pro Gemeinde (initial -> final)
print(f"\n{'='*70}")
print("Bereinigungs-Uebersicht pro Gemeinde")
print(f"{'='*70}")
for gm in alle_gemeinden:
    gm_stats = [s for s in cleaning_stats if s["scope"] == gm]
    if not gm_stats:
        continue
    initial = gm_stats[0]["rows_before"]
    final = gm_stats[-1]["rows_after"]
    pct = (initial - final) / max(initial, 1) * 100
    print(f"  {gm}: {initial:>9,} -> {final:>9,}  (-{pct:.1f} %)")

# Laufzeit in zentrale Timing-Datei schreiben (upsert-Logik)
elapsed_total = time.time() - SCRIPT_START
timing_file = OUTPUT_DIR / "training_times.csv"
row = pd.DataFrame([{"skript": "00_prepare_data", "laufzeit_s": round(elapsed_total, 1)}])
if timing_file.exists():
    existing = pd.read_csv(timing_file)
    existing = existing[existing["skript"] != "00_prepare_data"]
    pd.concat([existing, row], ignore_index=True).to_csv(timing_file, index=False)
else:
    row.to_csv(timing_file, index=False)

print(f"\nGesamtlaufzeit: {elapsed_total:.1f}s")
print("Fertig.")
