"""
Visualisierungen der Cross-Gemeinde-Evaluation
=================================================
Erzeugt 5 Plots fuer die Projektarbeit (je als PNG + PDF):

1. Heatmap: MAE pro Modell x Gemeinde (20 x 18 Matrix)
2. Balkendiagramm: Mittlere MAE pro Modell (Ranking)
3. Bekannt vs. Unbekannt: Generalisierungsluecke pro Modell
4. Modelltyp x Datenvariante: Gruppierter Vergleich
5. Trainingszeiten: Laufzeit pro Skript

Aufruf:  python skripte/06_plot_results.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "ergebnisse"
FIG_DIR = SCRIPT_DIR.parent / "Unterlagen" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib-Einstellungen fuer Druckqualitaet
# Schriftgroessen bewusst gross gewaehlt, da Plots im Paper/Poster auf
# Spaltenbreite (~8 cm) skaliert werden und dabei stark verkleinert.
plt.rcParams.update({
    "figure.dpi": 150,       # Bildschirm-Aufloesung
    "savefig.dpi": 300,      # Export-Aufloesung (300 DPI fuer Druck)
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# -- Daten laden --------------------------------------------------------------
df = pd.read_csv(OUTPUT_DIR / "cross_evaluation.csv")

# Welche Gemeinden kennt jede Datenvariante? (fuer Plot 3)
known_sets = {
    "D1_single": {"GM0047"},
    "D2_multi4": {"GM0047", "GM0059", "GM0281", "GM0590"},
    "D3_mittel6": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681"},
    "D4_gross10": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681",
                   "GM0281", "GM1950", "GM1969", "GM1930"},
    "D5_fremd": {"GM0312", "GM0590", "GM0546", "GM0629", "GM1681"},
    "D6_coord": {"GM0047", "GM0312", "GM0590", "GM0546", "GM0629", "GM1681",
                 "GM0281", "GM1950", "GM1969", "GM1930"},
}

# MAE auf unbekannten Gemeinden pro Modell — primaere Forschungsmetrik
def unknown_mae(model_name: str) -> float:
    variant = model_name.split("_", 1)[1]
    known = known_sets[variant]
    sub = df[df["modell"] == model_name]
    return sub[~sub["gemeinde"].isin(known)]["MAE"].mean()

# Modelle nach MAE auf unbekannten Gemeinden sortieren (bestes zuerst)
all_models = df["modell"].unique().tolist()
model_order = sorted(all_models, key=unknown_mae)
gemeinde_order = sorted(df["gemeinde"].unique())

# ==========================================================================
# Plot 1: Heatmap MAE pro Modell x Gemeinde
# Zeigt die vollstaendige 20x18-Ergebnismatrix als Farbkarte.
# Gruen = niedrig (gut), Rot = hoch (schlecht).
# ==========================================================================
print("Plot 1: Heatmap...")
pivot = df.pivot(index="modell", columns="gemeinde", values="MAE")
pivot = pivot.reindex(index=model_order, columns=gemeinde_order)

fig, ax = plt.subplots(figsize=(14, 7.5))
sns.heatmap(
    pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
    vmin=5, vmax=100, linewidths=0.5, ax=ax,
    annot_kws={"fontsize": 10},
    cbar_kws={"label": "MAE (s)"},
)
ax.set_title("Cross-Gemeinde-Evaluation: MAE (s) pro Modell und Gemeinde\n(schwarz umrandet: im Training enthalten)")
ax.set_ylabel("Modell (sortiert nach MAE auf unbekannten Gemeinden)")
ax.set_xlabel("Gemeinde")
plt.xticks(rotation=90, ha="center")
plt.yticks(rotation=0)

# Trainings-Gemeinden pro Modell markieren: dicker schwarzer Rahmen
from matplotlib.patches import Rectangle
for row_idx, model_name in enumerate(model_order):
    variant = model_name.split("_", 1)[1]
    train_gms = known_sets[variant]
    for col_idx, gm in enumerate(gemeinde_order):
        if gm in train_gms:
            ax.add_patch(Rectangle(
                (col_idx, row_idx), 1, 1,
                fill=False, edgecolor="black", lw=1.8,
            ))

fig.savefig(FIG_DIR / "cross_eval_heatmap.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "cross_eval_heatmap.pdf", bbox_inches="tight")
plt.close()

# ==========================================================================
# Plot 2: Horizontales Balkendiagramm — mittlere MAE pro Modell
# Farbkodierung nach Modelltyp: Rot=XGBoost, Gruen=RF, Blau=MLP, Orange=Ridge
# ==========================================================================
print("Plot 2: Balkendiagramm MAE auf unbekannten Gemeinden...")
means = pd.Series({m: unknown_mae(m) for m in model_order})

# Farbzuordnung nach Modelltyp-Praefix
colors = []
for m in model_order:
    if m.startswith("XGB"):
        colors.append("#e74c3c")     # Rot
    elif m.startswith("RF"):
        colors.append("#2ecc71")     # Gruen
    elif m.startswith("MLPv2"):
        colors.append("#3498db")     # Blau
    elif m.startswith("Ridge"):
        colors.append("#f39c12")     # Orange
    else:
        colors.append("#95a5a6")     # Grau (Fallback)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(model_order)), means.values, color=colors)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order)
ax.set_xlabel("MAE auf unbekannten Gemeinden (s)")
ax.set_title("Cross-Gemeinde-Evaluation: MAE auf unbekannten Gemeinden pro Modell")
ax.invert_yaxis()  # Bestes Modell oben

# Wertbeschriftung rechts neben den Balken
for bar, val in zip(bars, means.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}s",
            va="center", ha="left", fontsize=11)

# Legende fuer Modelltyp-Farben
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="XGBoost"),
    Patch(facecolor="#2ecc71", label="Random Forest"),
    Patch(facecolor="#3498db", label="MLPv2 (Entity Emb.)"),
    Patch(facecolor="#f39c12", label="Ridge Regression"),
]
ax.legend(handles=legend_elements, loc="upper right")
ax.set_xlim(0, max(means.values) * 1.15)

fig.savefig(FIG_DIR / "cross_eval_mean_mae.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "cross_eval_mean_mae.pdf", bbox_inches="tight")
plt.close()

# ==========================================================================
# Plot 3: Bekannt vs. Unbekannt (Generalisierungsluecke)
# Gruppiertes Balkendiagramm: Gruen = bekannte Gemeinden, Rot = unbekannte.
# Der Abstand zwischen den Balken zeigt die Generalisierungsluecke.
# ==========================================================================
print("Plot 3: Bekannt vs. Unbekannt...")
rows = []
for model_name in model_order:
    dv = None
    for k in known_sets:
        if k in model_name:
            dv = k
            break
    if dv is None:
        continue
    known = known_sets[dv]
    m_df = df[df["modell"] == model_name]
    k_mae = m_df[m_df["gemeinde"].isin(known)]["MAE"].mean()
    u_mae = m_df[~m_df["gemeinde"].isin(known)]["MAE"].mean()
    rows.append({"modell": model_name, "bekannt": k_mae, "unbekannt": u_mae})

bv_df = pd.DataFrame(rows)
x = np.arange(len(bv_df))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5.5))
bars1 = ax.bar(x - width / 2, bv_df["bekannt"], width, label="Bekannte Gemeinden", color="#2ecc71", alpha=0.85)
bars2 = ax.bar(x + width / 2, bv_df["unbekannt"], width, label="Unbekannte Gemeinden", color="#e74c3c", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(bv_df["modell"], rotation=90, ha="center")
ax.set_ylabel("MAE (s)")
ax.set_title("Generalisierung: MAE auf bekannten vs. unbekannten Gemeinden")
ax.legend()

# Wertbeschriftung ueber den Balken
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=10)

fig.savefig(FIG_DIR / "cross_eval_known_vs_unknown.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "cross_eval_known_vs_unknown.pdf", bbox_inches="tight")
plt.close()

# ==========================================================================
# Plot 4: Modelltyp x Datenvariante (gruppierter Vergleich)
# Zeigt fuer jeden der 4 Modelltypen, wie sich die 5 Datenvarianten
# auf die mittlere Cross-Eval-MAE auswirken.
# ==========================================================================
print("Plot 4: Modelltyp vs. Datenvariante (MAE auf unbekannten Gemeinden)...")
df["modelltyp"] = df["modell"].str.split("_").str[0]
df["datenvariante"] = df["modell"].str.replace(r"^[^_]+_", "", regex=True)

# Pro (modelltyp, datenvariante) den MAE auf unbekannten Gemeinden
type_data_means_list = []
for m in all_models:
    typ = m.split("_")[0]
    dv = m.split("_", 1)[1]
    type_data_means_list.append({"modelltyp": typ, "datenvariante": dv, "MAE": unknown_mae(m)})
type_data_means = pd.DataFrame(type_data_means_list)

# Sortierung: Ridge (bestes) links, XGBoost (schlechtestes) rechts
type_order = ["Ridge", "MLPv2", "RF", "XGB"]
data_order = ["D1_single", "D2_multi4", "D3_mittel6", "D4_gross10", "D5_fremd", "D6_coord"]
data_colors = {"D1_single": "#3498db", "D2_multi4": "#2ecc71", "D3_mittel6": "#f39c12",
               "D4_gross10": "#e74c3c", "D5_fremd": "#9b59b6", "D6_coord": "#2c3e50"}

fig, ax = plt.subplots(figsize=(10, 5.5))
x_pos = np.arange(len(type_order))
# Breite und Zentrierung an Anzahl Datenvarianten anpassen (6 Balken pro Gruppe)
n_variants = len(data_order)
width = 0.9 / n_variants
offset = (n_variants - 1) / 2.0

for i, dv in enumerate(data_order):
    subset = type_data_means[type_data_means["datenvariante"] == dv]
    vals = []
    for t in type_order:
        row = subset[subset["modelltyp"] == t]
        vals.append(row["MAE"].values[0] if len(row) > 0 else 0)
    mask = [v > 0 for v in vals]
    positions = [x_pos[j] + (i - offset) * width for j in range(len(type_order)) if mask[j]]
    heights = [v for v in vals if v > 0]
    ax.bar(positions, heights, width, label=dv, color=data_colors[dv], alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(type_order)
ax.set_ylabel("MAE auf unbekannten Gemeinden (s)")
ax.set_title("Modelltyp x Datenvariante: MAE auf unbekannten Gemeinden")
ax.legend(title="Datenvariante")

fig.savefig(FIG_DIR / "cross_eval_type_vs_data.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "cross_eval_type_vs_data.pdf", bbox_inches="tight")
plt.close()

# ==========================================================================
# Plot 5: Trainingszeiten der gesamten Pipeline
# Horizontales Balkendiagramm der Skript-Laufzeiten.
# ==========================================================================
print("Plot 5: Trainingszeiten...")
timing = pd.read_csv(OUTPUT_DIR / "training_times.csv")
timing = timing.sort_values("laufzeit_s", ascending=True)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(timing["skript"], timing["laufzeit_s"], color="#3498db")
ax.set_xlabel("Laufzeit (s)")
ax.set_title("Skript-Laufzeiten")
# Beschriftung: Sekunden oder Minuten je nach Groesse
for bar, val in zip(bars, timing["laufzeit_s"]):
    label = f"{val:.0f}s" if val < 600 else f"{val/60:.0f}min"
    ax.text(val + 10, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=11)
ax.set_xlim(0, timing["laufzeit_s"].max() * 1.15)

fig.savefig(FIG_DIR / "training_times.png", bbox_inches="tight")
fig.savefig(FIG_DIR / "training_times.pdf", bbox_inches="tight")
plt.close()

print(f"\n5 Plots gespeichert in: {FIG_DIR}")
print("  - cross_eval_heatmap.png/pdf")
print("  - cross_eval_mean_mae.png/pdf")
print("  - cross_eval_known_vs_unknown.png/pdf")
print("  - cross_eval_type_vs_data.png/pdf")
print("  - training_times.png/pdf")
