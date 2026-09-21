import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

# =====================================================================
# 1. PATH DEFINITIONS
# =====================================================================
# Wir laden die Datei, die gerade im Hintergrund von der KI befüllt wird!
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
# (Also alle Zeilen, bei denen Is_Expanded auf 1 steht)
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
# char_wb analysiert Wortbestandteile (Präfixe, Suffixe), damit auch unbekannte Wörter emotional verstanden werden!
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=25000)

X_train = vectorizer.fit_transform(df_train["Word"].astype(str).str.lower())

# NEU: Wir definieren alle 7 Zielmerkmale (3 klassische VAD + 4 evolutionäre Merkmale)
target_columns = ["V", "A", "D", "Danger", "Resource_Value", "Social_Bond", "Goal_Proximity"]
Y_train = df_train[target_columns].values

# =====================================================================
# 4. MULTI-OUTPUT TRAINING (7 Dimensionen gleichzeitig lernen)
# =====================================================================
print("🧠 Trainiere mathematische Instinkt-Matrix für 7 emotionale Dimensionen...")

# Ridge Regression ist extrem stabil gegen Overfitting und mathematisch pfeilschnell
base_model = Ridge(alpha=1.0)
pal_multi_model = MultiOutputRegressor(base_model)

# Das Modell lernt die Verknüpfung von Wortstrukturen zu den 7 Vektoren
pal_multi_model.fit(X_train, Y_train)

# =====================================================================
# 5. MODELL SPEICHERN & SICHERN
# =====================================================================
print(f"💾 Speichere das 7-dimensionale PAL-Modell ab: {MODEL_FILE}...")
with open(MODEL_FILE, "wb") as f:
    # Wir speichern den Vectorizer und das Multi-Output-Modell zusammen ab
    pickle.dump((vectorizer, pal_multi_model), f)

print("🎉 PAL-System erfolgreich auf 7 biologische Merkmale trainiert und gesichert!")

# =====================================================================
# 6. INSTINKT-SCHNELLTEST
# =====================================================================
print("\n👀 INSTINKT-TEST (Live-Vorhersage für unbekannte Reize):")
print("-" * 90)

def test_instinct(word):
    vec = vectorizer.transform([word.lower()])
    # Vorhersage liefert uns eine Liste mit den 7 Werten zurück
    predictions = pal_multi_model.predict(vec)[0]
    
    print(f"Wort: '{word}'")
    print(f"  ↳ [Klassisch VAD]  Valence: {predictions[0]:.2f} | Arousal: {predictions[1]:.2f} | Dominance: {predictions[2]:.2f}")
    print(f"  ↳ [Evolutionär]   Danger:  {predictions[3]:.2f} | Resource: {predictions[4]:.2f} | Social:    {predictions[5]:.2f} | Goal: {predictions[6]:.2f}")
    print("-" * 90)

# Teste drei völlig unterschiedliche biologische Reize
test_instinct("blood")
test_instinct("apple")
test_instinct("weapon")
