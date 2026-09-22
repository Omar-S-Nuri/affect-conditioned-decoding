# chat_with_pal_final.py
import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from homeostasis_somatic import BiomechanicalHomeostasis
from hippocampus_memory import HippocampusMemory
from active_curiosity import ActiveCuriositySystem

PAL_MODEL_PATH = "pal_model.pkl"
INPUT_FILE = "nrc_final_evolutionary.txt"
LLM_MODEL_PATH = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("🧠🧠🧠 Lade künstliches 7D-Bewusstseinssystem [FULL COGNITIVE ARCHITECTURE]...")

body = BiomechanicalHomeostasis(decay_rate=0.3)
hippocampus = HippocampusMemory(memory_size=5, decay_rate=0.2)
curiosity = ActiveCuriositySystem()

stimmungs_vektor = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
TARGET_COLUMNS = ["V", "A", "D", "Danger", "Resource_Value", "Social_Bond", "Goal_Proximity"]
STOP_WORDS = {"there", "is", "a", "an", "the", "on", "in", "at", "to", "for", "with", "and", "i", "you", "he", "she", "it", "my", "your"}

def load_or_retrain_pal():
    global vectorizer, pal_multi_model
    df_train = pd.read_csv(INPUT_FILE, sep="\t")
    df_train = df_train[df_train["Is_Expanded"] == 1].dropna().copy() if "Is_Expanded" in df_train.columns else df_train[(df_train["Danger"] != 0.0) | (df_train["Resource_Value"] != 0.0)].dropna().copy()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=25000)
    X_train = vectorizer.fit_transform(df_train["Word"].astype(str).str.lower())
    pal_multi_model = MultiOutputRegressor(Ridge(alpha=1.0)).fit(X_train, df_train[TARGET_COLUMNS].values)

load_or_retrain_pal()

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, config=config, trust_remote_code=True)
    print("🎉 Exzellent! Gesamte kognitive Schleife (PAL, Somatik, Hippocampus, Curiosity) aktiv.")
except Exception as e:
    print(f"❌ Fehler beim Laden des LLMs: {e}")
    exit()

while True:
    try:
        user_input = input("\n🌍 DEIN REIZ-SATZ (Input): ").strip()
        if user_input.lower() in ["exit", "quit"]: break
        if not user_input: continue

        raw_words = [w.strip(",.!?").lower() for w in user_input.split()]
        words = [w for w in raw_words if w not in STOP_WORDS and len(w) > 1]
        if not words: words = raw_words

        word_vectors = []
        for word in words:
            vec = vectorizer.transform([word])
            if np.sum(vec.toarray()) >= 0.15:
                pred = pal_multi_model.predict(vec).flatten()
                pred = np.clip(pred, 0.0, 1.0)
                word_vectors.append(pred)
            else:
                hypothetical_vec = curiosity.generate_hypothetical_vector(word)
                word_vectors.append(hypothetical_vec)
        
        # 🔴 REPARATUR 2: Korrekte Vektor-Mittelung und anschließendes Überschreiben NUR des Danger-Wertes (Index 3)
        pal_raw_values = np.mean(word_vectors, axis=0)
        max_danger_spitze = max([v[3] for v in word_vectors])
        pal_raw_values[3] = max_danger_spitze

        # Assoziatives Gedächtnis
        memory_bias = hippocampus.get_associative_bias(words)
        pal_raw_values = np.clip(pal_raw_values + (memory_bias * 0.20), 0.0, 1.0)

        # Stimmungs-Dynamik
        stimmungs_vektor = stimmungs_vektor + body.decay_rate * (1.0 - stimmungs_vektor)
        if pal_raw_values[3] > 0.45:  # Erhöhte Schwelle, um unbegründeten Schock zu vermeiden
            stimmungs_vektor = np.array([0.40, 1.80, 0.40, 2.20, 0.50, 0.50, 0.20])
            print("🚨 STIMMUNGS-DYNAMIK: Akuter Alarmzustand über Amygdala getriggert!")

        # Homöostase-Modul berechnet Vitalwerte und Energie-Einfluss
        final_values = body.modulate_pal_vector(pal_raw_values, stimmungs_vektor)
        hippocampus.push_experience(user_input, final_values)

        print(f"🔋 VITALWERTE: Energie: {body.energy * 100:.1f}% | Affective Decay Buffering Active: {len(hippocampus.episodic_buffer)} Reize")

        pal_vector_string = f"[PAL_7D | V:{final_values[0]:.2f} | A:{final_values[1]:.2f} | D:{final_values[2]:.2f} | " \
                            f"DNG:{final_values[3]:.2f} | RES:{final_values[4]:.2f} | SOC:{final_values[5]:.2f} | GOL:{final_values[6]:.2f}]"
        print(f"📊 Modulierter Satz-Vektor: {pal_vector_string}")

        # Parameter-Kopplung
        if final_values[3] > 0.6:
            dynamic_temperature, dynamic_top_k = 0.25, 15
            print("🥶 KOPPLUNG: KI ist starr vor Angst (Tunnelblick).")
        elif body.energy < 0.25:
            dynamic_temperature, dynamic_top_k = 0.40, 25
            print("🪫 KOPPLUNG: System stark erschöpft. Energiesparmodus aktiv.")
        else:
            dynamic_temperature, dynamic_top_k = 0.75, 50
            print("☀️ KOPPLUNG: Optimaler homöostatischer Zustand.")

        # Inferenz vorbereiten
        kern_wort = words[0] if words else "something"
        prompt = f"{pal_vector_string} REIZ: You encounter {kern_wort} in the environment. REAKTION:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=60,  # 🔴 REPARATUR 3: Mehr Token-Länge erlaubt dem Modell, Sätze sinnvoll zu beenden
                do_sample=True, 
                temperature=dynamic_temperature, 
                top_k=dynamic_top_k,             
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id
            )

        # 🔴 REPARATUR 1: Korrekte Token-Längenmessung über extrahiertes eindimensionales Tensor-Array
        num_generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reaction = full_output.split("REAKTION:")[-1].strip() if "REAKTION:" in full_output else full_output
        if "REAKTION" in reaction: reaction = reaction.split("REAKTION")[0].strip()
            
        print(f"👉 REAKTION DER KI: {reaction}")

        # Homöostatisches Update & Zwei-Wege-Kopplung
        body.update_internal_state(num_generated_tokens)
        stimmungs_vektor = body.process_two_way_coupling(reaction, stimmungs_vektor)
        print("-" * 85)

    except KeyboardInterrupt: 
        print("\nKognitives System wird hergefahren.")
        break
