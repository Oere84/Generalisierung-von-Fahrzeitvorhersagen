# Generalisierung von Fahrzeitvorhersagen im oeffentlichen Nahverkehr

Geographische vs. kategorische Stop-Repraesentation im Vergleich ueber 18 Gemeinden.

**Projektarbeit** im Weiterbildungszertifikat *Neuronale Netze und Datenkompetenz* an der Technischen Hochschule Ingolstadt, Fach: Angewandtes Wissenschaftliches Arbeiten.

## Fragestellung

Welche Kombination aus Trainingsdaten, Feature-Kodierung und Lernalgorithmus generalisiert am besten auf Gemeinden, die im Training nicht enthalten waren?

## Methodik

- **Datensatz:** Oeffentlicher Benchmark-Datensatz mit ~52 Mio. Segmenten aus 18 Gemeinden (2022-2024)
- **6 Datenvarianten** (D1-D6): Von einer einzelnen Gemeinde (701k Segmente) bis zu 10 Gemeinden (14,1 Mio. Segmente). D6 nutzt identische Trainingsdaten wie D4, ersetzt aber Label-IDs durch GPS-Koordinaten (Ablationskontrast)
- **4 Modelltypen:** Ridge Regression, Random Forest, XGBoost, MLP mit Entity Embeddings
- **24 Experimente** (4 Modelle x 6 Datenvarianten), jeweils evaluiert auf allen 18 Gemeinden (432 Ergebnisse)
- **20 Features:** Zyklische Kodierung (Stunde, Monat), Weekday One-Hot, Haversine-Distanz, Lag-Feature, plus Label- oder Koordinaten-Features fuer Haltestellen

## Ergebnisse

### Top 6 — MAE auf unbekannten Gemeinden (Forschungsfragen-relevante Metrik)

| Rang | Modell          | MAE Unbek. |
|------|-----------------|------------|
| 1    | Ridge_D6_coord  | 22,0s      |
| 2    | MLPv2_D6_coord  | 23,2s      |
| 3    | Ridge_D1_single | 24,0s      |
| 4    | Ridge_D5_fremd  | 25,1s      |
| 5    | Ridge_D3_mittel6| 25,5s      |
| 6    | Ridge_D4_gross10| 26,1s      |

RF_D6 folgt erst auf Rang 7 (26,9s), XGB_D6 auf Rang 15 (32,9s, praktisch gleichauf mit MLPv2_D5). Im Gesamtmittel ueber alle 18 Gemeinden fuehrt zwar RF_D6 mit 15,3s, dieser Wert profitiert aber stark von den 10 Trainings-Gemeinden, auf denen RF_D6 nur 6,0s erreicht. Fuer die eigentliche Forschungsfrage zaehlt die Out-of-Sample-Performance.

### Kernbefunde

- **Ridge dominiert die Out-of-Sample-Rangliste:** Ridge belegt die Plaetze 1, 3, 4, 5 und 6 — sowohl mit Koordinaten- als auch mit Label-Features. Mit 20 Koeffizienten und L2-Regularisierung kann Ridge sich nicht an Trainingsgemeinden ueberanpassen und generalisiert daher robust.
- **D4 vs. D6 (Ablation auf Unbek.):** Identische Trainingsdaten; ID-basierte Merkmale (Haltestelle, Route, Linie) werden durch Koordinaten-Features ersetzt. Random Forest verbessert sich um 29%, MLPv2 um 19%, Ridge um 16%, XGBoost um nur 7%. Im Gesamtmittel wirken die Effekte mit -57%/-48%/-38%/-18% deutlich groesser, weil die Trainings-Gemeinden den Mittelwert nach unten ziehen.
- **Tree-Modelle generalisieren auch mit Koordinaten schlechter:** RF_D6 erreicht 26,9s auf Unbek., XGB_D6 sogar nur 32,9s — beide schlechter als die beste reine Label-Variante (Ridge_D1: 24,0s).
- **Hypothesen-Abgleich:** H1 (XGB > MLP) widerlegt, H2 (mehr Daten) bestaetigt, H3 (Entity Embeddings fuer Unbek.) widerlegt, H4 (Ridge als robuste Baseline) **bestaetigt**, H5 (Feature-Kodierung sekundaer) teilweise — Koordinaten helfen, ersetzen aber kein gut regularisiertes Modell.

## Projektstruktur

```
skripte/
  00_prepare_data.py           # Datenaufbereitung fuer alle 6 Varianten
  01_train_xgboost.py          # XGBoost Training (GPU)
  02_train_random_forest.py    # Random Forest Training (CPU)
  03_train_mlp_v2.py           # MLP mit Entity Embeddings (GPU)
  04_train_ridge.py            # Ridge Regression (CPU)
  05_evaluate_all.py           # Cross-Evaluation: 24 Modelle x 18 Gemeinden
  06_plot_results.py           # Ergebnis-Visualisierungen
  notebooks/                   # Jupyter Notebooks mit Erklaerungen
  ergebnisse/                  # Evaluationsergebnisse als CSV
requirements.txt               # Python-Abhaengigkeiten
```

Nicht im GitHub-Repository enthalten: Rohdaten (`Daten/`, ~19 GB), aufbereitete Datensaetze (`skripte/data/`), trainierte Modelle (`skripte/models/`) — diese werden lokal durch die Pipeline erzeugt. Der Paper-LaTeX-Quelltext und der Experimentplan werden lokal gepflegt und sind nicht Teil dieses Repos.

## Reproduktion

### Voraussetzungen

- Python 3.10+
- CUDA-faehige GPU (fuer XGBoost und MLPv2)
- Rohdaten als CSV-Dateien in `Daten/travel_times/` und `Daten/dwell_times/` (nicht im Repository enthalten, ~19 GB)

### Installation

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt          # Windows
# Linux/macOS: source .venv/bin/activate && pip install -r requirements.txt
# GPU-PyTorch separat, z. B.: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Pipeline ausfuehren

```bash
python skripte/00_prepare_data.py          # Datenvarianten erzeugen (~9 Min.)
python skripte/01_train_xgboost.py         # XGBoost Training (~2 Min., GPU)
python skripte/02_train_random_forest.py   # Random Forest Training (~2,5 Std., CPU)
python skripte/03_train_mlp_v2.py          # MLP Training (~46 Min., GPU)
python skripte/04_train_ridge.py           # Ridge Training (~1 Min., CPU)
python skripte/05_evaluate_all.py          # Cross-Evaluation (~11 Min.)
python skripte/06_plot_results.py          # Visualisierungen
```

## Technologien

- **Python** (pandas, NumPy, scikit-learn, matplotlib)
- **PyTorch** (MLP mit Entity Embeddings, CUDA)
- **XGBoost** (GPU-beschleunigt)
- **LaTeX** (ieeeconf, biblatex)

## Nutzung generativer KI

Bei der Erstellung dieser Arbeit wurden Anthropic Claude (u.a. Claude Code), OpenAI ChatGPT und Google NotebookLM als Hilfsmittel eingesetzt — fuer Code-Entwicklung, Debugging, LaTeX-Formatierung, sprachliche Ueberarbeitung, deutschsprachige Paper-Uebersetzungen sowie quellengebundene Befragung hochgeladener Referenzpaper (NotebookLM). Forschungsfragen, Experimentdesign und Ergebnisinterpretation liegen beim Autor. Zahlen aus Fremdpapern wurden gegen die Originalquellen verifiziert.
