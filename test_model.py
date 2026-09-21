import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("⏳ Lade Ihr feinjustiertes 7D-Modell mit korrigierter Konfiguration...")

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    print("✅ 7D-Modell erfolgreich geladen!\n")
except Exception as e:
    print(f"\n❌ Fehler beim Laden des Modells: {e}\nBitte stelle sicher, dass 'finetune_llm.py' komplett fertig ist.")
    exit()

test_word = "tiger" 
# NEU: Das simulierte Gefühl sendet jetzt alle 7 biologischen Dimensionen an das LLM
simulierter_7d_vektor = "[PAL_7D | V:0.15 | A:0.85 | D:0.20 | DNG:0.90 | RES:0.10 | SOC:0.00 | GOL:0.20]"
prompt = f"{simulierter_7d_vektor} REIZ: The word is '{test_word}'. REAKTION:"

print("🧠 Generiere die evolutionäre Reaktion...")


# =====================================================================
# 3. TEXT GENERIEREN (Optimiert für sauberes Satzende)
# =====================================================================
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=40,       
        do_sample=True,          # Aktiviert echtes Sampling (beseitigt die Warnung!)
        temperature=0.7,         # Erlaubt natürliche Varianz
        top_k=50,
        eos_token_id=tokenizer.eos_token_id # Zwingt das Modell, am Satzende zu stoppen
    )

# Nur die reine Reaktion ausschneiden
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
reaktion = generated_text.split("REAKTION:")[-1].strip() if "REAKTION:" in generated_text else generated_text

print("=== ERGEBNIS DES TRAINIERTEN SYSTEMS ===")
print(f"👉 {reaktion}")
print("========================================")

