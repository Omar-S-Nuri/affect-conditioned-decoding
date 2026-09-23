## train_pal.py

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

# =====================================================================
# 1. PATH DEFINITIONS
# =====================================================================
INPUT_FILE = "nrc_final_evolutionary.txt"
MODEL_FILE = "pal_model.pkl"

print("🧬 Starte Training des erweiterten Limbischen Systems (PAL)...")

# =====================================================================
# 2. DATEN BASIS LADEN & FILTERN
# =====================================================================
if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: '{INPUT_FILE}' nicht gefunden!")
    print("Bitte warte, bis die 'expand_lexicon1400.py' die ersten Blöcke in die Datei geschrieben hat.")
    exit()

print(f"📥 Lade KI-erweitertes Lexikon: {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, sep="\t")

# WICHTIG: Wir trainieren NUR auf den Wörtern, die die KI bereits erfolgreich expandiert hat!
if "Is_Expanded" in df.columns:
    df_train = df[df["Is_Expanded"] == 1].dropna().copy()
else:
    # Fallback, falls die Spalte im Test-Zwischenstand noch nicht da sein sollte
    df_train = df[(df["Danger"] != 0.0) | (df["Resource_Value"] != 0.0)].dropna().copy()

print(f"📊 Gefundene Trainings-Datenbasis: {len(df_train)} fertig expandierte Wörter.")

if len(df_train) < 40:
    print("⚠️ Warnung: Es wurden noch zu wenige Wörter von der KI expandiert.")
    print("Lass die 'expand_lexicon1400.py' erst noch ein paar Minuten laufen, bevor du das PAL-System trainierst!")
    exit()

# =====================================================================
# 3. TEXT-VEKTORISIERUNG (Die Wort-Struktur mathematisch greifbar machen)
# =====================================================================
print("🔤 Vektorisiere Wörter via TF-IDF (Zeichen-N-Gramme)...")
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=25000)

X_train = vectorizer.fit_transform(df_train["Word"].astype(str).str.lower())

target_columns = ["V", "A", "D", "Danger", "Resource_Value", "Social_Bond", "Goal_Proximity"]
Y_train = df_train[target_columns].values

# =====================================================================
# 4. MULTI-OUTPUT TRAINING (7 Dimensionen gleichzeitig lernen)
# =====================================================================
print("🧠 Trainiere mathematische Instinkt-Matrix für 7 emotionale Dimensionen...")

base_model = Ridge(alpha=1.0)
pal_multi_model = MultiOutputRegressor(base_model)
pal_multi_model.fit(X_train, Y_train)

# =====================================================================
# 5. WERTEBEREICHE ERMITTELN (für korrektes Clipping bei der Anwendung)
# =====================================================================
# Wichtig: Nicht alle 7 Dimensionen liegen zwangsläufig in [0, 1]. Manche
# Lexika kodieren z.B. Valence (V) zentriert um 0 (negative Werte möglich).
# Pauschales np.clip(pred, 0.0, 1.0) würde solche Dimensionen verzerren
# (negative Valenz würde fälschlich auf 0 = "neutral" abgebildet).
# Deshalb: echte Min/Max je Spalte aus den Trainingsdaten ermitteln und
# zusammen mit dem Modell speichern, statt später zu raten oder pauschal
# zu clippen.
clip_ranges = {
    col: (float(df_train[col].min()), float(df_train[col].max()))
    for col in target_columns
}
print("📐 Ermittelte Wertebereiche pro Dimension (aus echten Trainingsdaten):")
for col, (lo, hi) in clip_ranges.items():
    print(f"   {col:<16} [{lo:.3f}, {hi:.3f}]")

# =====================================================================
# 6. MODELL SPEICHERN & SICHERN
# =====================================================================
print(f"💾 Speichere das 7-dimensionale PAL-Modell ab: {MODEL_FILE}...")
with open(MODEL_FILE, "wb") as f:
    # 4-Tupel: Vectorizer, Modell, Spaltennamen (Reihenfolge!), echte Wertebereiche.
    # Die Reihenfolge von target_columns wird mitgespeichert, damit nachgelagerte
    # Skripte (z.B. evaluate_pal.py) nicht erneut annehmen müssen, welche Position
    # welcher Dimension entspricht.
    pickle.dump((vectorizer, pal_multi_model, target_columns, clip_ranges), f)

print("🎉 PAL-System erfolgreich auf 7 biologische Merkmale trainiert und gesichert!")

# =====================================================================
# 7. INSTINKT-SCHNELLTEST
# =====================================================================
print("\n👀 INSTINKT-TEST (Live-Vorhersage für unbekannte Reize):")
print("-" * 90)


def test_instinct(word):
    vec = vectorizer.transform([word.lower()])
    predictions = pal_multi_model.predict(vec)[0]

    # Auch im Test spaltenspezifisch clippen, statt pauschal [0,1] –
    # konsistent mit dem, was evaluate_pal.py später tut.
    clipped = [
        max(clip_ranges[col][0], min(clip_ranges[col][1], predictions[i]))
        for i, col in enumerate(target_columns)
    ]

    print(f"Wort: '{word}'")
    print(f"  ↳ [Klassisch VAD]  Valence: {clipped[0]:.2f} | Arousal: {clipped[1]:.2f} | Dominance: {clipped[2]:.2f}")
    print(f"  ↳ [Evolutionär]   Danger:  {clipped[3]:.2f} | Resource: {clipped[4]:.2f} | Social:    {clipped[5]:.2f} | Goal: {clipped[6]:.2f}")
    print("-" * 90)


test_instinct("blood")
test_instinct("apple")
test_instinct("weapon")
