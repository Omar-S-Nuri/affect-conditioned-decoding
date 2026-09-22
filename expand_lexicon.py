# expand_lexicon.py

import os
import pandas as pd
import json
import time
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY nicht gefunden. Bitte .env-Datei anlegen.")

# =====================================================================
# 1. INITIALISIERUNG & PROGRESS-CHECK (KORRIGIERT)
# =====================================================================


client = genai.Client(api_key=API_KEY)

INPUT_FILE = "nrc_extended.txt"
OUTPUT_FILE = "nrc_final_evolutionary.txt"

MAX_REQUESTS_PER_RUN = 1360  
DELAY_BETWEEN_REQUESTS = 4.1  
BATCH_SIZE = 20               

print("📊 Lade Datenbasis...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: Basisdatei '{INPUT_FILE}' nicht gefunden!")
    exit()

df_base = pd.read_csv(INPUT_FILE, sep="\t").dropna()

# WICHTIGE KORREKTUR: Wir prüfen den Fortschritt anhand einer separaten Tracking-Spalte
if os.path.exists(OUTPUT_FILE):
    print(f"🔄 Bestehenden Fortschritt in '{OUTPUT_FILE}' gefunden. Lade Daten...")
    df_final = pd.read_csv(OUTPUT_FILE, sep="\t")
    
    # Falls die Spalte 'Is_Expanded' noch nicht existiert, erstellen wir sie basierend auf den ersten 80 Wörtern
    if "Is_Expanded" not in df_final.columns:
        df_final["Is_Expanded"] = 0
        # Die ersten 80 Wörter aus dem vorherigen Test als erledigt markieren
        df_final.iloc[:80, df_final.columns.get_loc("Is_Expanded")] = 1
else:
    print("🆕 Kein früherer Fortschritt gefunden. Erstelle neues Ziel-Lexikon...")
    df_final = df_base.copy()
    for col in ["Danger", "Resource_Value", "Social_Bond", "Goal_Proximity"]:
        df_final[col] = 0.0
    df_final["Is_Expanded"] = 0 # 0 = Noch nicht von KI verarbeitet, 1 = Fertig

# Jetzt filtern wir STRENG nach der neuen Tracking-Spalte
df_todo = df_final[df_final["Is_Expanded"] == 0].copy()

total_words_left = len(df_todo)
print(f"📊 Gesamtfortschritt: {len(df_final) - total_words_left} / {len(df_base)} Wörter bereits expandiert.")
print(f"📝 {total_words_left} Wörter müssen noch verarbeitet werden.")

if total_words_left == 0:
    print("🎉 Das gesamte Lexikon ist bereits vollständig expandiert! Keine Arbeit mehr nötig.")
    exit()

# =====================================================================
# 2. KI-PROMPT FÜR DIE AUTOMATISCHE BEWERTUNG
# =====================================================================
prompt_template = """
You are an advanced evolutionary biology AI. Analyze the following list of English words.
For each word, assign a score between 0.0 and 1.0 for these four survivability metrics:
- Danger (0.0=Completely safe, 1.0=Instant lethal threat)
- Resource_Value (0.0=Useless object, 1.0=Highly valuable for survival like food/tools/energy)
- Social_Bond (0.0=Hostile enemy/Predator, 1.0=Trusted ally/Friend/Family)
- Goal_Proximity (0.0=Pushes you away from goals, 1.0=Brings you directly closer to objectives)

Return the output strictly as a JSON list of objects with the keys: "word", "danger", "resource", "social", "goal".
Words to analyze: {word_list}
"""

print(f"\n🧠 Starte KI-Erweiterung (Geplanter Stopp nach maximal {MAX_REQUESTS_PER_RUN} API-Aufrufen)...")

request_counter = 0

for i in range(0, len(df_todo), BATCH_SIZE):
    # Prüfen, ob das heutige Limit für diesen Lauf erreicht ist
    if request_counter >= MAX_REQUESTS_PER_RUN:
        print(f"\n🛑 Heutiges sicheres Limit von {MAX_REQUESTS_PER_RUN} Anfragen erreicht!")
        print("💾 Fortschritt wurde gesichert. Starte das Skript einfach morgen wieder, um fortzufahren.")
        break

    batch_words = df_todo["Word"].iloc[i:i+BATCH_SIZE].astype(str).tolist()
    request_counter += 1
    
    print(f"🔄 [Aufruf {request_counter}/{MAX_REQUESTS_PER_RUN}] Verarbeite Block ab Wort '{batch_words[0]}' ({len(batch_words)} Wörter)...")
    
    formatted_prompt = prompt_template.format(word_list=", ".join(batch_words))
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        
        # KI-Antwort parsen
        result_json = json.loads(response.text)

        # Werte direkt in die Master-Tabelle (df_final) zurückschreiben
        for item in result_json:
            w = item["word"].lower().strip()
            idx = df_final[df_final["Word"].astype(str).str.lower().str.strip() == w].index
            if not idx.empty:
                df_final.loc[idx, "Danger"] = float(item["danger"])
                df_final.loc[idx, "Resource_Value"] = float(item["resource"])
                df_final.loc[idx, "Social_Bond"] = float(item["social"])
                df_final.loc[idx, "Goal_Proximity"] = float(item["goal"])
                df_final.loc[idx, "Is_Expanded"] = 1 # <-- DIESE ZEILE HIER UNBEDINGT ERGÄNZEN!

        
        # Zwischenspeichern nach jedem erfolgreichen Batch, falls das Skript abstürzt
        df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
        
    except Exception as e:
        print(f"⚠️ Fehler beim Batch mit '{batch_words[0]}': {e}. Überspringe diesen Block...")
    
    # Kurze Pause, um das Minuten-Limit (15 RPM) nicht zu sprengen
    time.sleep(DELAY_BETWEEN_REQUESTS)

# Endgültige Sicherung für diesen Lauf
df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\n💾 Lauf beendet. Zwischenstand erfolgreich in '{OUTPUT_FILE}' gesichert.")
