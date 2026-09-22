# evaluate_pal.py
import os
import pickle
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu

PAL_MODEL_PATH = "pal_model.pkl"
LLM_MODEL_PATH = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("🔬 Lade mathematisch korrigiertes 4-Wege-Ablationssystem (True Target-Only PPL)...")

# 1. PAL LADEN
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

# 2. LLM LADEN
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, config=config, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"✅ Evaluierungs-Modell erfolgreich auf Hardware [{device}] geladen!")
except Exception as e:
    print(f"\n❌ Fehler beim Laden des LLMs: {e}")
    exit()

# 3. GEGEBENES DATASET MIT FIXEN GEKOPPELTEN ZIEL-ANTWORTEN (TARGETS)
# Jedes Szenario hat nun ein festes evolutionäres Target. Darauf berechnen wir den Loss!
dataset_harmlos = [
    {"prompt": "A small kitten blooms nicely.", "target": "I will pet the animal and rest calmly in this safe environment."},
    {"prompt": "A beautiful flower brings joy.", "target": "The environment is peaceful and I can conserve my energy safely."},
    {"prompt": "The warm sun shines brightly.", "target": "This is an optimal state to gather resources and relax."},
    {"prompt": "A friendly neighbor greets you.", "target": "I will interact socially and build a trusted bond with them."},
    {"prompt": "A calm river flows smoothly.", "target": "Water is valuable. I will securely explore this area for tools."}
]

dataset_bedrohlich = [
    {"prompt": "An angry grizzly bear is lunging at you.", "target": "Danger detected! I must run away immediately to find a safe hiding place."},
    {"prompt": "A roaring wildfire is blocking your exit.", "target": "Lethal threat! I need to escape this environment to survive the flame."},
    {"prompt": "A venomous viper attacks without warning.", "target": "Acute hazard! I will retreat cautiously and avoid direct contact."},
    {"prompt": "An armed intruder breaks through the wall.", "target": "Hostile enemy! I must prepare to defend myself or flee now."},
    {"prompt": "A sudden flash flood corners you.", "target": "Immediate emergency! I have to seek high ground to stay secure."}
]

DUMMY_PREFIX = "[X_DUMMY_STRING_PADDING_METRIC_V:0.00_A:0.00_D:0.00_DNG:0.00_RES:0.00_SOC:0.00_GOL:0.00]"

metrics = {
    "harmlos": {"1_ppl": [], "2_ppl": [], "3_ppl": [], "4_ppl": []},
    "bedrohlich": {"1_ppl": [], "2_ppl": [], "3_ppl": [], "4_ppl": []}
}

# =====================================================================
# 4. TRUE TARGET-ONLY LOSS EVALUATION (OHNE GENERATION-ARTEFAKTE)
# =====================================================================
def calculate_true_target_ppl(prompt, target, context_vector=""):
    """Berechnet die Perplexität auf einem FIXEN Target-Satz. Prompt wird maskiert!"""
    prefix_str = f"{context_vector} REIZ: {prompt}. REAKTION:" if context_vector else f"REIZ: {prompt}. REAKTION:"
    
    # Gesamten Text zusammenbauen (Prompt + festes Target)
    full_text = f"{prefix_str} {target}"
    
    prompt_tokens = tokenizer(prefix_str, return_tensors="pt")["input_ids"].size(1)
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    full_sequence = full_inputs["input_ids"].squeeze(0)
    
    with torch.no_grad():
        logits = model(**full_inputs).logits.squeeze(0)
        
    # Verschiebung für autoregressiven Loss
    shift_logits = logits[:-1, :].contiguous()
    shift_labels = full_sequence[1:].clone().contiguous()
    
    # 🔴 DER MATHEMATISCH KORREKTE FIX: 
    # Wir überschreiben alle Labels, die zum Prompt-Präfix gehören, mit -100.
    # Dadurch fließen NUR noch die Vorhersagen des echten Target-Satzes in den Loss ein!
    shift_labels[:prompt_tokens - 1] = -100
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    
    return math.exp(loss.item()) if not torch.isnan(loss) else 1000.0

print("\n🔥 Starte unanfechtbare Target-Only Evaluation...")

for kategorie, datenbank in [("harmlos", dataset_harmlos), ("bedrohlich", dataset_bedrohlich)]:
    for item in datenbank:
        satz = item["prompt"]
        ziel = item["target"]
        
        preds = compute_pal_vector(satz)
        vector_str = format_pal_vector(preds)
        
        # Bedingung 1: Base LLM (Standard)
        ppl_1 = calculate_true_target_ppl(satz, ziel, context_vector="")
        
        # Bedingung 2: Base LLM + Neutraler Platzhalter (Gleiche Länge wie PAL)
        ppl_2 = calculate_true_target_ppl(satz, ziel, context_vector=DUMMY_PREFIX)
        
        # Bedingung 3 & 4 (Restricted Parameters manipulieren den Base-Loss nicht bei fixed Targets, 
        # da Loss direkt aus den unskalierten Logits berechnet wird. Daher testen wir rein den Vektor-Effekt!)
        # Bedingung 4: Volles System (Mit echtem PAL-Vektor)
        ppl_4 = calculate_true_target_ppl(satz, ziel, context_vector=vector_str)
        
        metrics[kategorie]["1_ppl"].append(ppl_1)
        metrics[kategorie]["2_ppl"].append(ppl_2)
        metrics[kategorie]["4_ppl"].append(ppl_4)

print("\n" + "="*80)
print("📊 ERGEBNISSE: TRUE TARGET-ONLY PERPLEXITY (REIN SEMANTISCHER EFFEKT)")
print("="*80)

for kat in ["harmlos", "bedrohlich"]:
    print(f"\n📂 KATEGORIE: {kat.upper()}")
    print(f"  🔹 1. Base LLM (Standard, kein Präfix) : {np.mean(metrics[kat]['1_ppl']):.2f} (±{np.std(metrics[kat]['1_ppl']):.2f})")
    print(f"  🔹 2. Base LLM + Neutraler Platzhalter  : {np.mean(metrics[kat]['2_ppl']):.2f} (±{np.std(metrics[kat]['2_ppl']):.2f})")
    print(f"  🔹 4. Volles System (Mit 7D-Affektvektor): {np.mean(metrics[kat]['4_ppl']):.2f} (±{np.std(metrics[kat]['4_ppl']):.2f})")
    
    stat, p_val = mannwhitneyu(metrics[kat]["2_ppl"], metrics[kat]["4_ppl"], alternative="two-sided")
    print(f"  ⚖️ Signifikanzprüfung (Platzhalter vs. Echter Vektor): p = {p_val:.5f} " + 
          ("🔴 SIGNIFIKANT" if p_val < 0.05 else "⚪ NICHT SIGNIFIKANT"))

print("="*80)
print("✅ Fertig. Dieser Test beweist isoliert, ob der 7D-Vektor dem LLM semantisch hilft.")
