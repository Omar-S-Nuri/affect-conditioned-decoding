## fintune_llm_verhaltensvielfallt.py

import os
import pickle
import torch
import random
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

PAL_MODEL_PATH = "pal_model.pkl"
INPUT_FILE = "nrc_final_evolutionary.txt"
OUTPUT_DIR = "./pal_trained_model"

print("🧠 Lade künstliches Bewusstseinssystem für das erweiterte 7D-Training...")

# =====================================================================
# 1. ERWEITERTES PAL-MODELL LADEN
# =====================================================================
if not os.path.exists(PAL_MODEL_PATH):
    print(f"❌ Fehler: '{PAL_MODEL_PATH}' nicht gefunden. Bitte zuerst train_pal.py ausführen!")
    exit()

with open(PAL_MODEL_PATH, "rb") as f:
    vectorizer, pal_multi_model = pickle.load(f)


def get_7d_pal_vector(text):
    vec = vectorizer.transform([text.lower()])
    # .flatten() verwandelt die Matrix in eine einfache, flache Liste von 7 Werten
    preds = pal_multi_model.predict(vec).flatten()
    return (
        f"[PAL_7D | V:{preds[0]:.2f} | A:{preds[1]:.2f} | D:{preds[2]:.2f} | "
        f"DNG:{preds[3]:.2f} | RES:{preds[4]:.2f} | SOC:{preds[5]:.2f} | GOL:{preds[6]:.2f}]"
    )

# =====================================================================
# 2. TRAININGSDATEN GENERIEREN (PUNKT 1 & 4: Nuanciert & Satz-basiert)
# =====================================================================
print(f"📝 Lade Daten aus '{INPUT_FILE}'...")
if not os.path.exists(INPUT_FILE):
    print(f"❌ Fehler: '{INPUT_FILE}' nicht gefunden!")
    exit()

df = pd.read_csv(INPUT_FILE, sep="\t")
df_train = df[df["Is_Expanded"] == 1].dropna().copy()

# Wir begrenzen das CPU-Training auf maximal 500 Wörter für die Geschwindigkeit
df_subset = df_train.head(500)

formatted_data = []
for _, row in df_subset.iterrows():
    word = str(row["Word"])
    
    # PUNKT 4: Wir bauen aus dem Wort eine ganze Situation (Satz-Modus)
    situations = [
        f"You encounter {word} in the environment.",
        f"A sudden event involves {word}.",
        f"You are directly interacting with {word}."
    ]
    situation = random.choice(situations)
    
    vad_signal = get_7d_pal_vector(situation)
    
    # PUNKT 1: Sprachliche Vielfalt & Nuancierte Kombinationen
    dng, res, soc = row["Danger"], row["Resource_Value"], row["Social_Bond"]
    reaktion = ""
    
    # Komplexer Konflikt: Hohes Risiko, aber hoher Nutzen
    if dng > 0.6 and res > 0.6:
        reaktion = random.choice([
            "High risk, high reward! Approach with extreme caution, secure the asset, and retreat immediately.",
            "Dangerous but highly valuable. Evaluate the threat, prepare defense, and attempt to collect the resource."
        ])
    # Reine Gefahr
    elif dng > 0.6:
        reaktion = random.choice([
            "Threat detected! Survival protocol activated: Flee or defend yourself immediately.",
            "Danger imminent. Back away slowly, avoid eye contact, and seek high ground or shelter."
        ])
    # Reiner Nutzen
    elif res > 0.6:
        reaktion = random.choice([
            "Valuable resource found. Secure and store the object for energy conservation.",
            "Positive asset spotted. Collect it immediately to optimize survival capabilities."
        ])
    # Soziale Allianz
    elif soc > 0.6:
        reaktion = random.choice([
            "Trusted entity detected. Initiate cooperative protocols and establish a social alliance.",
            "Friendly presence. Extend peaceful signals and interact calmly to reinforce bonds."
        ])
    # Neutraler Zustand
    else:
        reaktion = "Maintain baseline survival state. Monitor environment, no immediate action required."
        
    full_text = f"{vad_signal} REIZ: {situation} REAKTION: {reaktion}"
    formatted_data.append({"text": full_text})

dataset = Dataset.from_list(formatted_data)

# =====================================================================
# 3. BASE-MODELL LADEN & TRAINING EINSTELLEN (Wie gehabt)
# =====================================================================
model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=128), batched=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    use_cpu=True,                
    per_device_train_batch_size=1,
    num_train_epochs=1,          
    max_steps=80,  # 40 Schritte bleiben bestehen              
    learning_rate=5e-5,
    logging_steps=5,             
    weight_decay=0.01,
    report_to="none",
    save_strategy="steps",       
    save_steps=40,                
    save_total_limit=2           
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        'input_ids': torch.stack([torch.tensor(d['input_ids']) for d in data]),
        'attention_mask': torch.stack([torch.tensor(d['attention_mask']) for d in data]),
        'labels': torch.stack([torch.tensor(d['input_ids']) for d in data])
    }
)

print("🚀 Starte CPU-Finetuning...")
# Wir überschreiben das alte Training (Force Fresh Start), damit das neue Format gelernt wird
trainer.train()

print("💾 Speichere das neue nuancierte Modell dauerhaft ab...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Gewichte im Ordner '{OUTPUT_DIR}' gesichert!")
