## expand_lexicon_ollama2.py

# [OLLAMA LOCAL EDITION - FIXED]

import os
import pandas as pd
import json
import time
import ollama

INPUT_FILE = "nrc_extended.txt"
OUTPUT_FILE = "nrc_final_evolutionary.txt"

MAX_REQUESTS_PER_RUN = 1360  
DELAY_BETWEEN_REQUESTS = 0.5  
BATCH_SIZE = 20               

# 🔴 Nutze dein gewünschtes Modell (z.B. "qwen2.5:7b", "llama3", etc.)
LOCAL_MODEL = "llama3.1:latest" ##"llama3"

print(f"🧠 Initialisiere lokales 7D-Erweiterungssystem via Ollama [{LOCAL_MODEL}]...")
print("📊 Lade Datenbasis...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: Basisdatei '{INPUT_FILE}' nicht gefunden!")
    exit()

df_base = pd.read_csv(INPUT_FILE, sep="\t").dropna()

# Fortschritt überprüfen
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

# Filterung nach echten Lücken
df_todo = df_final[df_final["Is_Expanded"] == 0].copy()

total_words_left = len(df_todo)
print(f"📊 Gesamtfortschritt: {len(df_final) - total_words_left} / {len(df_base)} Wörter bereits expandiert.")
print(f"📝 {total_words_left} Wörter müssen noch verarbeitet werden.")

if total_words_left == 0:
    print("🎉 Das gesamte Lexikon ist bereits vollständig expandiert! Keine Arbeit mehr nötig.")
    exit()

# 🔴 KORREKTUR: Prompt erzwingt nun ein umschließendes "results"-Objekt für stabilere JSON-Generierung
prompt_template = """
You are an advanced evolutionary biology AI. Analyze the following list of English words.
For each word, assign a score between 0.0 and 1.0 for these four survivability metrics:
- danger (0.0=Completely safe, 1.0=Instant lethal threat)
- resource (0.0=Useless object, 1.0=Highly valuable for survival like food/tools/energy)
- social (0.0=Hostile enemy/Predator, 1.0=Trusted ally/Friend/Family)
- goal (0.0=Pushes you away from goals, 1.0=Brings you directly closer to objectives)

Return the output strictly as a JSON object containing a list named "results", where each element is an object with the keys "word", "danger", "resource", "social", and "goal". 
Example format:
{{
  "results": [
    {{"word": "example", "danger": 0.1, "resource": 0.5, "social": 0.8, "goal": 0.4}}
  ]
}}

Words to analyze: {word_list}
"""

print(f"\n🚀 Starte lokale KI-Erweiterung über Ollama...")

request_counter = 0

for i in range(0, len(df_todo), BATCH_SIZE):
    if request_counter >= MAX_REQUESTS_PER_RUN:
        print(f"\n🛑 Limit von {MAX_REQUESTS_PER_RUN} Aufrufen für diesen Durchlauf erreicht.")
        break

    batch_words = df_todo["Word"].iloc[i:i+BATCH_SIZE].astype(str).tolist()
    request_counter += 1
    
    print(f"🔄 [Aufruf {request_counter}] Verarbeite Block ab Wort '{batch_words[0]}' ({len(batch_words)} Wörter) lokal...")
    formatted_prompt = prompt_template.format(word_list=", ".join(batch_words))
    
    try:
        # Lokaler Ollama-Aufruf
        response = ollama.generate(
            model=LOCAL_MODEL,
            prompt=formatted_prompt,
            format="json"
        )
        
        # 🔴 KORREKTUR: Sicheres, mehrstufiges Extrahieren des JSON-Inhalts
        raw_response = response['response']
        parsed_data = json.loads(raw_response)
        
        # Falls das Modell die Liste direkt oder eingepackt in "results" liefert, fangen wir beides ab:
        if isinstance(parsed_data, dict) and "results" in parsed_data:
            result_list = parsed_data["results"]
        elif isinstance(parsed_data, list):
            result_list = parsed_data
        elif isinstance(parsed_data, dict):
            # Falls das Modell einen anderen Key gewürfelt hat, nehmen wir die erste gefundene Liste
            result_list = next((v for v in parsed_data.values() if isinstance(v, list)), [])
        else:
            result_list = []

        if not result_list:
            print(f"⚠️ Warnung: Keine valide JSON-Struktur extrahiert für Block '{batch_words[0]}'.")
            continue

        # Werte zurückschreiben
        for item in result_list:
            if not isinstance(item, dict) or "word" not in item:
                continue
            w = item["word"].lower().strip()
            idx = df_final[df_final["Word"].astype(str).str.lower().str.strip() == w].index
            if not idx.empty:
                df_final.loc[idx, "Danger"] = float(item.get("danger", 0.0))
                df_final.loc[idx, "Resource_Value"] = float(item.get("resource", 0.0))
                df_final.loc[idx, "Social_Bond"] = float(item.get("social", 0.0))
                df_final.loc[idx, "Goal_Proximity"] = float(item.get("goal", 0.0))
                df_final.loc[idx, "Is_Expanded"] = 1

        # Zwischenstand sichern
        df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
        
    except Exception as e:
        print(f"⚠️ Fehler beim lokalen Batch mit '{batch_words[0]}': {e}. Block bleibt offen...")
    
    time.sleep(DELAY_BETWEEN_REQUESTS)

df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\n💾 Lauf beendet. Daten sauber in '{OUTPUT_FILE}' gesichert.")
