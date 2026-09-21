import pandas as pd
import os

LARGE_VAD_FILE = "nrc_vad.txt"
NEW_EMO_FILE = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUTPUT_FILE = "nrc_extended.txt"

print("📊 1. Lade deine ursprüngliche VAD-Datei (54.800 Wörter)...")
if not os.path.exists(LARGE_VAD_FILE):
    print(f"❌ Fehler: '{LARGE_VAD_FILE}' nicht gefunden!")
    exit()

# Wir lesen deine Datei ein (skiprows=1 bleibt, falls deine VAD-Datei einen manuellen Header hat)
df_vad = pd.read_csv(LARGE_VAD_FILE, sep="\t", skiprows=1, names=["Word", "V", "A", "D"]).dropna()
df_vad["Word"] = df_vad["Word"].astype(str).str.lower().str.strip()

print(f"✅ Geladen: {len(df_vad)} Wörter.")

print("\n⚙️ 2. Lade und formatiere das neue NRC-Emotion-Lexicon...")
if not os.path.exists(NEW_EMO_FILE):
    print(f"❌ Fehler: Die entpackte Datei '{NEW_EMO_FILE}' wurde nicht gefunden!")
    print("Bitte kopiere sie aus dem entpackten ZIP-Ordner hierher.")
    exit()

# KORREKTUR: on_bad_lines='skip' ignoriert eventuelle Lizenztexte am Anfang der Datei. 
# skiprows=1 wurde entfernt, da die echten Daten sofort in Zeile 1 starten.
df_emo_raw = pd.read_csv(
    NEW_EMO_FILE, 
    sep="\t", 
    names=["Word", "Emotion", "Association"], 
    on_bad_lines='skip'
)
df_emo_raw = df_emo_raw.dropna()
df_emo_raw["Word"] = df_emo_raw["Word"].astype(str).str.lower().str.strip()

# Wir filtern nur die 6 von dir gewünschten Basis-Emotionen heraus
target_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
df_emo_filtered = df_emo_raw[df_emo_raw["Emotion"].isin(target_emotions)]

print("🔄 Baue die Emotions-Struktur in Spalten um (Pivot)...")
# Verwandelt Zeilen-Emotionen in saubere Spalten pro Wort
df_emo_pivot = df_emo_filtered.pivot_table(
    index="Word", 
    columns="Emotion", 
    values="Association", 
    aggfunc="first"
).reset_index()

# Spaltennamen schön benennen
df_emo_pivot.columns = ["Word", "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]

print("\n🤝 3. Verschmelze beide Lexika (Merge)...")
# 'how="left"' sorgt dafür, dass deine 54.800 Wörter alle erhalten bleiben!
df_merged = pd.merge(df_vad, df_emo_pivot, on="Word", how="left")

# Wörter, die im neuen Lexikon nicht existierten, bekommen für die Emotionen den Standardwert 0.0
df_merged = df_merged.fillna(0.0)

print("\n💾 4. Speichere neue Gesamt-Datei ab...")
df_merged.to_csv(OUTPUT_FILE, sep="\t", index=False)

print(f"🎉 RIESIGER ERFOLG! Die Datei '{OUTPUT_FILE}' wurde erstellt.")
print(f"📊 Neue Struktur hat {len(df_merged)} Zeilen und folgende Spalten:")
print(list(df_merged.columns))

# Zeige eine kleine Vorschau von Wörtern wie "love" oder "monster"
print("\n👀 Kleine Vorschau der Daten:")
print(df_merged[df_merged["Word"].isin(["love", "monster", "tiger", "sunny"])])


##  ## python merge_lexicons.py