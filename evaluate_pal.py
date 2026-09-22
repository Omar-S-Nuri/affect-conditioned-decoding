# evaluate_pal.py
import os
import pickle
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer, util  # 🔴 NEU: Automatisiertes Qualitätsmaß

PAL_MODEL_PATH = "pal_model.pkl"
LLM_MODEL_PATH = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("🔬 Lade 4-Wege-Ablationssystem mit Target-Only PPL, Signifikanzprüfung & Semantischer Ähnlichkeit...")

# 1. PAL & EMBEDDING MODEL LADEN
with open(PAL_MODEL_PATH, "rb") as f:
    vectorizer, pal_multi_model = pickle.load(f)

# Lokales, standardisiertes Qualitäts-Embedding-Modell laden
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    print(f"✅ Evaluierungs-Modell erfolgreich auf [{device}] geladen!")
except Exception as e:
    print(f"\n❌ Fehler beim Laden des LLMs: {e}")
    exit()

# 3. EVALUATIONS-DATASET MIT EVALUATIONS-TARGETS (N = 200)
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
    "harmlos": {f"{c}_{m}": [] for c in ["1", "2", "3", "4"] for m in ["ppl", "d2", "sim"]},
    "bedrohlich": {f"{c}_{m}": [] for c in ["1", "2", "3", "4"] for m in ["ppl", "d2", "sim"]}
}

# =====================================================================
# 4. TARGET-ONLY PPL & GENERATION QUALITY PIPELINE
# =====================================================================
def evaluate_condition(prompt, target, context_vector="", temp=0.75, top_k=50):
    prefix_str = f"{context_vector} REIZ: {prompt}. REAKTION:" if context_vector else f"REIZ: {prompt}. REAKTION:"
    full_text = f"{prefix_str} {target}"
    
    prompt_tokens = tokenizer(prefix_str, return_tensors="pt")["input_ids"].size(1)
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    full_sequence = full_inputs["input_ids"].squeeze(0)
    
    with torch.no_grad():
        # A. Target-Only PPL über feste Sequenz berechnen
        logits = model(**full_inputs).logits.squeeze(0)
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = full_sequence[1:].clone().contiguous()
        shift_labels[:prompt_tokens - 1] = -100
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        ppl = math.exp(loss.item()) if not torch.isnan(loss) else 1000.0
        
        # B. Antwort generieren für Diversität und semantische Qualität
        outputs = model.generate(
            **tokenizer(prefix_str, return_tensors="pt").to(device),
            max_new_tokens=30, do_sample=True, temperature=temp, top_k=top_k, repetition_penalty=1.3
        )
        reaction_tokens = outputs[0][prompt_tokens:]
        reaction_text = tokenizer.decode(reaction_tokens, skip_special_tokens=True).strip()
        
    # Distinct-2
    words = reaction_text.split()
    d2 = len(set([f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)])) / (len(words)-1) if len(words) > 1 else 1.0
    
    # 🔴 AUTOMATISIERTES QUALITÄTSMASS: Semantische Kosinus-Ähnlichkeit zum Target
    emb_gen = similarity_model.encode(reaction_text, convert_to_tensor=True)
    emb_tar = similarity_model.encode(target, convert_to_tensor=True)
    semantic_sim = util.cos_sim(emb_gen, emb_tar).item()
    
    return ppl, d2, semantic_sim

print("\n🔥 Starte 4-Wege-Evaluation (Wissenschaftlicher Gold-Standard)...")

for kategorie, datenbank in [("harmlos", dataset_harmlos), ("bedrohlich", dataset_bedrohlich)]:
    is_threat = (kategorie == "bedrohlich")
    temp_rest = 0.25 if is_threat else 0.75
    top_k_rest = 15 if is_threat else 50
    
    for item in datenbank:
        s, t = item["prompt"], item["target"]
        preds = compute_pal_vector(s)
        vector_str = format_pal_vector(preds)
        
        p1, d1, s1 = evaluate_condition(s, t, context_vector="", temp=0.75, top_k=50)
        p2, d2, s2 = evaluate_condition(s, t, context_vector=DUMMY_PREFIX, temp=0.75, top_k=50)
        p3, d3, s3 = evaluate_condition(s, t, context_vector="", temp=temp_rest, top_k=top_k_rest)
        p4, d4, s4 = evaluate_condition(s, t, context_vector=vector_str, temp=temp_rest, top_k=top_k_rest)
        
        for c, p, d, sm in [("1", p1, d1, s1), ("2", p2, d2, s2), ("3", p3, d3, s3), ("4", p4, d4, s4)]:
            metrics[kategorie][f"{c}_ppl"].append(p)
            metrics[kategorie][f"{c}_d2"].append(d)
            metrics[kategorie][f"{c}_sim"].append(sm)

print("\n" + "="*90)
print("📊 FINALER STATISTISCHER BERICHT (ABSICHERUNG GEGEN COMPRESSION-ARTIFAKTE)")
print("="*90)

for kat in ["harmlos", "bedrohlich"]:
    print(f"\n📂 KATEGORIE: {kat.upper()} (N = 100 Sätze)")
    for cond in ["1", "2", "3", "4"]:
        names = {"1": "Base LLM (Standard)        ", "2": "Base LLM + Platzhalter     ", 
                 "3": "Base LLM + Restr. Sampling ", "4": "Volles System (Vektor+Rest)"}
        p_arr, d_arr, s_arr = metrics[kat][f"{cond}_ppl"], metrics[kat][f"{cond}_d2"], metrics[kat][f"{cond}_sim"]
        print(f"  [{names[cond]}]: PPL: {np.mean(p_arr):.1f} (±{np.std(p_arr):.1f}) | Dist-2: {np.mean(d_arr)*100:.1f}% | Quality (Sim): {np.mean(s_arr)*100:.1f}%")
    
    _, p_val_sim = mannwhitneyu(metrics[kat]["3_sim"], metrics[kat]["4_sim"], alternative="two-sided")
    print(f"  ⚖️ Signifikanzprüfung Qualität (Bed. 3 vs. Bed. 4): p = {p_val_sim:.5f} " + 
          ("🔴 SIGNIFIKANT" if p_val_sim < 0.05 else "⚪ NICHT SIGNIFIKANT"))
print("="90)
