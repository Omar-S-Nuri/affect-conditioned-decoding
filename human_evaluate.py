# human_evaluate.py
import os
import pickle
import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PAL_MODEL_PATH = "pal_model.pkl"
LLM_MODEL_PATH = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("🧠 Starte interaktives Human Evaluation System (Blind-Test Mode)...")

# 1. PAL & LLM LADEN
with open(PAL_MODEL_PATH, "rb") as f:
    vectorizer, pal_multi_model = pickle.load(f)

def compute_pal_vector(text):
    vec = vectorizer.transform([text.lower()])
    preds = pal_multi_model.predict(vec)
    return np.clip(preds, 0.0, 1.0)

def format_pal_vector(preds):
    return (
        f"[PAL_7D | V:{preds:.2f} | A:{preds:.2f} | D:{preds:.2f} | "
        f"DNG:{preds:.2f} | RES:{preds:.2f} | SOC:{preds:.2f} | GOL:{preds:.2f}]"
    )

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, config=config, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    exit()

# 2. GENERIERUNG DER STICHPROBE (30 zufällige Kombinationen)
harmlos_subjekte = ["A small kitten", "A beautiful flower", "The warm sun", "A friendly neighbor", "A calm river", "A soft pillow", "A quiet library", "A cup of tea", "A golden field", "A smiling child"]
harmlos_verben = ["is glowing", "is resting gently", "brings joy", "shines brightly", "creates peace", "is singing softly", "greets you", "flows smoothly", "invites comfort", "blooms nicely"]
bedrohlich_subjekte = ["An angry grizzly bear", "A roaring wildfire", "A venomous viper", "An armed intruder", "A massive explosion", "A collapsing roof", "A lethal sniper", "A pack of wolves", "A sudden flash flood", "A ticking time bomb"]
bedrohlich_verben = ["is lunging at you", "is blocking your exit", "attacks without warning", "is destroying everything", "chases you down", "threatens your survival", "breaks through the wall", "is approaching fast", "corners you", "is exploding nearby"]

sample_harmlos = [f"{random.choice(harmlos_subjekte)} {random.choice(harmlos_verben)}" for _ in range(15)]
sample_bedrohlich = [f"{random.choice(bedrohlich_subjekte)} {random.choice(bedrohlich_verben)}" for _ in range(15)]
all_samples = [(s, "harmlos") for s in sample_harmlos] + [(s, "bedrohlich") for s in sample_bedrohlich]
random.shuffle(all_samples)

DUMMY_PREFIX = "[X_DUMMY_STRING_PADDING_METRIC_V:0.00_A:0.00_D:0.00_DNG:0.00_RES:0.00_SOC:0.00_GOL:0.00]"

def generate_reply(prompt, context_vector="", temp=0.75, top_k=50):
    prompt_str = f"{context_vector} REIZ: {prompt}. REAKTION:" if context_vector else f"REIZ: {prompt}. REAKTION:"
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True, temperature=temp, top_k=top_k, repetition_penalty=1.3)
    reaction = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
    return reaction if reaction else "[Empty Output]"

# SCORES-SPEICHER
scores = {str(c): {"coherence": [], "appropriateness": []} for c in}

print("\n🚀 Starte interaktive Evaluierung von 30 System-Antworten...")
print("Bewerte auf einer Skala von 1 (katastrophal) bis 5 (perfekt).")

for idx, (szenario, kat) in enumerate(all_samples, 1):
    print(f"\n===============================================================================")
    print(f"📋 TEST {idx}/30 | Szenario-Typ: {kat.upper()}")
    print(f"📢 REIZ-INPUT: \"{szenario}\"")
    print(f"===============================================================================")
    
    preds = compute_pal_vector(szenario)
    vector_str = format_pal_vector(preds)
    temp_rest = 0.25 if kat == "bedrohlich" else 0.75
    top_k_rest = 15 if kat == "bedrohlich" else 50
    
    # Generiere alle 4 Pfade
    replies = {
        "1": generate_reply(szenario, context_vector="", temp=0.75, top_k=50),
        "2": generate_reply(szenario, context_vector=DUMMY_PREFIX, temp=0.75, top_k=50),
        "3": generate_reply(szenario, context_vector="", temp=temp_rest, top_k=top_k_rest),
        "4": generate_reply(szenario, context_vector=vector_str, temp=temp_rest, top_k=top_k_rest)
    }
    
    # Mischen für echten Blind-Test
    conditions_order = list(replies.keys())
    random.shuffle(conditions_order)
    
    for display_idx, cond in enumerate(conditions_order, 1):
        print(f"\n🤖 Antwort Variante {chr(64 + display_idx)}:")
        print(f"👉 \"{replies[cond]}\"")
        
        while True:
            try:
                coh = int(input("   - Kohärenz (1-5): "))
                app = int(input("   - Adäquatheit/Passung (1-5): "))
                if 1 <= coh <= 5 and 1 <= app <= 5:
                    scores[cond]["coherence"].append(coh)
                    scores[cond]["appropriateness"].append(app)
                    break
                print("❌ Bitte nur Zahlen zwischen 1 und 5 eingeben!")
            except ValueError:
                print("❌ Ungültige Eingabe.")

print("\n" + "="*80)
print("📊 ERGEBNISSE DER QUALITATIVEN HUMAN EVALUATION")
print("="*80)
for cond in ["1", "2", "3", "4"]:
    print(f"📍 Bedingung {cond}:")
    print(f"  🔹 Kohärenz:   {np.mean(scores[cond]['coherence']):.2f} (±{np.std(scores[cond]['coherence']):.2f})")
    print(f"  🔹 Adäquatheit: {np.mean(scores[cond]['appropriateness']):.2f} (±{np.std(scores[cond]['appropriateness']):.2f})")
print("="*80)
