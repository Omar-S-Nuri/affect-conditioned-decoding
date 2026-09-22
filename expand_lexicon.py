# expand_lexicon2.py

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
# 1. INITIALISIERUNG & PROGRESS-CHECK
# =====================================================================

client = genai.Client(api_key=API_KEY)

INPUT_FILE = "nrc_extended.txt"
OUTPUT_FILE = "nrc_final_evolutionary.txt"

MAX_REQUESTS_PER_RUN = 1360  
DELAY_BETWEEN_REQUESTS = 4.1  
BATCH_SIZE = 20               
MAX_RETRIES = 5  
BASE_DELAY = 2.0 

print("📊 Lade Datenbasis...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: Basisdatei '{INPUT_FILE}' nicht gefunden!")
    exit()

df_base = pd.read_csv(INPUT_FILE, sep="\t").dropna()

# Fortschritt anhand der Tracking-Spalte 'Is_Expanded' überprüfen
if os.path.exists(OUTPUT_FILE):
    print(f"🔄 Bestehenden Fortschritt in '{OUTPUT_FILE}' gefunden. Lade Daten...")
    df_final = pd.read_csv(OUTPUT_FILE, sep="\t")
    
    if "Is_Expanded" not in df_final.columns:
        df_final["Is_Expanded"] = 0
        df_final.iloc[:80, df_final.columns.get_loc("Is_Expanded")] = 1
else:
    print("🆕 Kein früherer Fortschritt gefunden. Erstelle neues Ziel-Lexikon...")
    df_final = df_base.copy()
    for col in ["Danger", "Resource_Value", "Social_Bond", "Goal_Proximity"]:
        df_final[col] = 0.0
    df_final["Is_Expanded"] = 0 

# Filterung nach echten Unvollständigkeiten (Lücken werden hier automatisch erfasst!)
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
    if request_counter >= MAX_REQUESTS_PER_RUN:
        print(f"\n🛑 Sicheres Limit von {MAX_REQUESTS_PER_RUN} Anfragen für diesen Lauf erreicht!")
        break

    batch_words = df_todo["Word"].iloc[i:i+BATCH_SIZE].astype(str).tolist()
    request_counter += 1
    
    print(f"🔄 [Aufruf {request_counter}/{MAX_REQUESTS_PER_RUN}] Verarbeite Block ab Wort '{batch_words[0]}' ({len(batch_words)} Wörter)...")
    formatted_prompt = prompt_template.format(word_list=", ".join(batch_words))
    
    success = False
    
    for retry in range(MAX_RETRIES):
        try:
            # 🔄 UPGRADE: Dynamisches Modell-Fallback bei Quoten- oder Serverproblemen
            target_model = 'gemini-3.6-flash' if retry < 2 else 'gemini-1.5-flash'
            if retry >= 2:
                print(f"   🔄 Versuche Fallback-Modell '{target_model}'...")

            response = client.models.generate_content(
                model=target_model,
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            result_json = json.loads(response.text)

            # Werte sicher zurück in die Master-Tabelle schreiben
            for item in result_json:
                w = item["word"].lower().strip()
                idx = df_final[df_final["Word"].astype(str).str.lower().str.strip() == w].index
                if not idx.empty:
                    df_final.loc[idx, "Danger"] = float(item["danger"])
                    df_final.loc[idx, "Resource_Value"] = float(item["resource"])
                    df_final.loc[idx, "Social_Bond"] = float(item["social"])
                    df_final.loc[idx, "Goal_Proximity"] = float(item["goal"])
                    df_final.loc[idx, "Is_Expanded"] = 1  # Block erfolgreich abgeschlossen

            # Nach jedem erfolgreichen Block Zustand auf Festplatte sichern
            df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
            success = True
            break  # Raus aus der Retry-Schleife
            
        except Exception as e:
            error_str = str(e)
            
            # 🛑 ABFANGEN DES TÄGLICHEN QUOTEN-LIMITS (429)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print("\n🛑 QUOTEN-LIMIT ERREICHT (429): Das tägliche Free-Tier-Kontingent deines Google-Keys ist erschöpft.")
                print("💾 Fortschritt wird sauber gesichert. Starte das Skript einfach morgen wieder, um die verbleibenden Lücken zu füllen.")
                df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
                exit()  # Kontrollierter, sicherer Programmabbruch
                
            # ⏳ SERVERÜBERLASTUNG (503) -> Exponential Backoff
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                backoff_delay = BASE_DELAY * (2 ** retry)
                print(f"   ⏳ Server ausgelastet (503). Warte {backoff_delay}s vor Versuch {retry+1}/{MAX_RETRIES}...")
                time.sleep(backoff_delay)
            else:
                print(f"   ⚠️ Unvorhergesehener Fehler im Batch: {e}. Überspringe Block vorerst...")
                break  # Is_Expanded bleibt 0 -> Lücke wird beim nächsten Run automatisch erkannt
                
    if not success:
        print(f"❌ Block ab Wort '{batch_words[0]}' fehlgeschlagen. Bleibt als offene Lücke für den nächsten Durchlauf markiert.")

    time.sleep(DELAY_BETWEEN_REQUESTS)

# Endgültige Speicherung des aktuellen Meilensteins
df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\n💾 Lauf beendet. Daten lückenlos in '{OUTPUT_FILE}' gesichert.")
