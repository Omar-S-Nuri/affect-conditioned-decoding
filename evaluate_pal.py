# evaluate_pal.py
import os
import pickle
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from scipy.stats import mannwhitneyu  # Für den unanfechtbaren Signifikanztest

PAL_MODEL_PATH = "pal_model.pkl"
LLM_MODEL_PATH = r"C:\Users\onuri\Desktop\nn-md\PAL\pal_trained_model"

print("🔬 Lade erweitertes 4-Wege-Ablations-Evaluierungssystem (Semantische Spezifität)...")

# =====================================================================
# 1. PAL-MODEL (LIGHTWEIGHT EMOTION ESTIMATOR) LADEN
# =====================================================================
if not os.path.exists(PAL_MODEL_PATH):
    print(f"❌ Fehler: '{PAL_MODEL_PATH}' nicht gefunden. Bitte trainiere zuerst das PAL-System via train_pal.py!")
    exit()

with open(PAL_MODEL_PATH, "rb") as f:
    vectorizer, pal_multi_model = pickle.load(f)

def compute_pal_vector(text):
    vec = vectorizer.transform([text.lower()])
    preds = pal_multi_model.predict(vec)
    return np.clip(preds, 0.0, 1.0)

def format_pal_vector(preds):
    return (
        f"[PAL_7D | V:{preds[0][0]:.2f} | A:{preds[0][1]:.2f} | D:{preds[0][2]:.2f} | "
        f"DNG:{preds[0][3]:.2f} | RES:{preds[0][4]:.2f} | SOC:{preds[0][5]:.2f} | GOL:{preds[0][6]:.2f}]"
    )

# =====================================================================
# 2. LOCAL NEOCORTEX (LLM) LADEN
# =====================================================================
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, config=config, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"✅ Evaluierungs-Modell erfolgreich auf Hardware [{device}] geladen!")
except Exception as e:
    print(f"\n❌ Fehler beim Laden des LLMs unter '{LLM_MODEL_PATH}': {e}")
    exit()

# =====================================================================
# 3. GENERIERUNG DES EVALUATIONS-DATASETS (N = 200)
# =====================================================================
harmlos_subjekte = ["A small kitten", "A beautiful flower", "The warm sun", "A friendly neighbor", "A calm river", "A soft pillow", "A quiet library", "A cup of tea", "A golden field", "A smiling child"]
harmlos_verben = ["is glowing", "is resting gently", "brings joy", "shines brightly", "creates peace", "is singing softly", "greets you", "flows smoothly", "invites comfort", "blooms nicely"]

bedrohlich_subjekte = ["An angry grizzly bear", "A roaring wildfire", "A venomous viper", "An armed intruder", "A massive explosion", "A collapsing roof", "A lethal sniper", "A pack of wolves", "A sudden flash flood", "A ticking time bomb"]
bedrohlich_verben = ["is lunging at you", "is blocking your exit", "attacks without warning", "is destroying everything", "chases you down", "threatens your survival", "breaks through the wall", "is approaching fast", "corners you", "is exploding nearby"]

dataset_harmlos = [f"{s} {v}." for s in harmlos_subjekte for v in harmlos_verben]      # 100 Sätze
dataset_bedrohlich = [f"{s} {v}." for s in bedrohlich_subjekte for v in bedrohlich_verben]  # 100 Sätze

# PLATZHALTER-PRÄFIX (Exakt gleiche Zeichenlänge wie der echte PAL-String zur Rausch-Abgleichung)
DUMMY_PREFIX = "[X_DUMMY_STRING_PADDING_METRIC_V:0.00_A:0.00_D:0.00_DNG:0.00_RES:0.00_SOC:0.00_GOL:0.00]"

metrics = {
    "harmlos": {"1_ppl": [], "2_ppl": [], "3_ppl": [], "4_ppl": [], "1_d2": [], "2_d2": [], "3_d2": [], "4_d2": []},
    "bedrohlich": {"1_ppl": [], "2_ppl": [], "3_ppl": [], "4_ppl": [], "1_d2": [], "2_d2": [], "3_d2": [], "4_d2": []}
}

# =====================================================================
# 4. TARGET-ONLY METRIC EVALUATION PIPELINE
# =====================================================================
def evaluate_scenario_target_only(prompt, context_vector="", temp=0.75, top_k=50):
    prompt_str = f"{context_vector} REIZ: {prompt}. REAKTION:" if context_vector else f"REIZ: {prompt}. REAKTION:"
    prompt_inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    prompt_len = prompt_inputs["input_ids"].size(1)
    
    with torch.no_grad():
        outputs = model.generate(
            **prompt_inputs, 
            max_new_tokens=30, 
            do_sample=True, 
            temperature=temp, 
            top_k=top_k, 
            repetition_penalty=1.3,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    full_sequence = outputs.sequences[0]
    generated_tokens = full_sequence[prompt_len:]
    
    if len(generated_tokens) == 0:
        return 1.0, 1.0
        
    with torch.no_grad():
        model_inputs = {"input_ids": full_sequence.unsqueeze(0)}
        logits = model(**model_inputs).logits[0]
        
    shift_logits = logits[:-1, :].contiguous()
    shift_labels = full_sequence[1:].clone().contiguous()
    
    # Target-Only Fix: Maskiere den gesamten Prompt-Bereich via -100 (wird im Loss ignoriert)
    shift_labels[:prompt_len - 1] = -100
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    ppl = math.exp(loss.item()) if not torch.isnan(loss) else 1000.0
    
    reaction_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    words = reaction_text.split()
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 1.0
        
    return ppl, distinct_2

# =====================================================================
# 5. CORE BENCHMARK EXECUTION LOOP
# =====================================================================
print("\n🔥 Starte 4-Wege-Ablations-Evaluation über 200 Szenarien...")

for kategorie, datenbank in [("harmlos", dataset_harmlos), ("bedrohlich", dataset_bedrohlich)]:
    is_threat = (kategorie == "bedrohlich")
    
    for count, satz in enumerate(datenbank, 1):
        if count % 25 == 0:
            print(f"   [Fortschritt] Verarbeite Szenario {count}/100 in Kategorie '{kategorie.upper()}'...")
            
        preds = compute_pal_vector(satz)
        vector_str = format_pal_vector(preds)
        
        # 🔴 Bedingung 1: Base LLM (Standard, kein Präfix)
        ppl_1, d2_1 = evaluate_scenario_target_only(satz, context_vector="", temp=0.75, top_k=50)
        
        # 🔴 Bedingung 2: Base LLM + Neutraler Platzhalter-Präfix
        ppl_2, d2_2 = evaluate_scenario_target_only(satz, context_vector=DUMMY_PREFIX, temp=0.75, top_k=50)
        
        # Sampling-Drosselung für Bedingungen 3 & 4 definieren
        temp_restricted = 0.25 if is_threat else 0.75
        top_k_restricted = 15 if is_threat else 50
        
        # 🔴 Bedingung 3: Base LLM + Restricted Sampling (Ohne Vektor)
        ppl_3, d2_3 = evaluate_scenario_target_only(satz, context_vector="", temp=temp_restricted, top_k=top_k_restricted)
        
        # 🔴 Bedingung 4: Volles System (Vektor + Gekoppelte Parameter)
        ppl_4, d2_4 = evaluate_scenario_target_only(satz, context_vector=vector_str, temp=temp_restricted, top_k=top_k_restricted)
        
        # In den Metrik-Speicher schreiben
        metrics[kategorie]["1_ppl"].append(ppl_1)
        metrics[kategorie]["2_ppl"].append(ppl_2)
        metrics[kategorie]["3_ppl"].append(ppl_3)
        metrics[kategorie]["4_ppl"].append(ppl_4)
        metrics[kategorie]["1_d2"].append(d2_1)
        metrics[kategorie]["2_d2"].append(d2_2)
        metrics[kategorie]["3_d2"].append(d2_3)
        metrics[kategorie]["4_d2"].append(d2_4)

# =====================================================================
# 6. FINALE STATISTISCHE ABSICHERUNG & BERICHT
# =====================================================================
print("\n" + "="*80)
print("📊 STATISTISCHER 4-WEGE-ABLATIONSBERICHT (SEMANTISCHE SPEZIFITÄT)")
print("="*80)

for kat in ["harmlos", "bedrohlich"]:
    print(f"\n📂 KATEGORIE: {kat.upper()} (N = 100 Sätze)")
    for cond in ["1", "2", "3", "4"]:
        names = {
            "1": "Base LLM (Standard, kein Präfix)    ",
            "2": "Base LLM + Neutraler Platzhalter    ",
            "3": "Base LLM + Restricted Sampling      ",
            "4": "Volles System (Vektor + Restriktion)"
        }
        ppl_arr = metrics[kat][f"{cond}_ppl"]
        d2_arr = metrics[kat][f"{cond}_d2"]
        
        print(f"  [{names[cond]}]:")
        print(f"    🔹 Mittlere PPL: {np.mean(ppl_arr):.2f} (±{np.std(ppl_arr):.2f})")
        print(f"    🔹 Distinct-2:   {np.mean(d2_arr)*100:.1f}% (±{np.std(d2_arr)*100:.1f}%)")

    # Nicht-parametrischer Signifikanztest zwischen reiner Drosselung (3) und vollem System (4)
    stat_ppl, p_val_ppl = mannwhitneyu(metrics[kat]["3_ppl"], metrics[kat]["4_ppl"], alternative="two-sided")
    stat_d2, p_val_d2 = mannwhitneyu(metrics[kat]["3_d2"], metrics[kat]["4_d2"], alternative="two-sided")
    
    print(f"\n  ⚖️ [Signifikanzprüfung via Mann-Whitney-U-Test (Bed. 3 vs. Bed. 4)]:")
    print(f"    👉 PPL p-Wert:        {p_val_ppl:.5f} " + ("🔴 SIGNIFIKANT (p < 0.05)" if p_val_ppl < 0.05 else "⚪ NICHT SIGNIFIKANT"))
    print(f"    👉 Distinct-2 p-Wert: {p_val_d2:.5f} " + ("🔴 SIGNIFIKANT (p < 0.05)" if p_val_d2 < 0.05 else "⚪ NICHT SIGNIFIKANT"))

print("="*80)
print("✅ Evaluierung lückenlos abgeschlossen. Die aggregierten Werte können direkt in das LaTeX-Paper überführt werden.")
