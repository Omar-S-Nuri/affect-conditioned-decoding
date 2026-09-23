"""
evaluate_pal.py
================
4-Way Semantic Specificity Ablation Matrix für Affect-Conditioned Decoding.

Isoliert den kausalen Effekt des 7D-Affekt-Vektors von zwei Konfundierungen,
die in der Vorgängerversion nicht kontrolliert wurden:

  (a) Reines Sampling-Parameter-Tuning (enges τ/top_k) senkt PPL unabhängig
      vom Vektorinhalt, weil es Repetition/Degeneration begünstigt.
  (b) Ein längerer Präfix-String senkt tendenziell die Perplexität der
      Fortsetzung, unabhängig davon, ob er semantisch sinnvoll ist.

Design-Entscheidung, die (b) löst:
  Die GENERIERUNG (welcher Text entsteht) und die MESSUNG (wie überrascht
  ist das Modell davon) laufen über unterschiedliche Kontexte. Der Präfix
  (Vektor oder Noise-Padding) darf die Generierung beeinflussen, aber die
  Perplexität wird immer unter demselben neutralen Scoring-Kontext
  ("REIZ: {prompt}. REAKTION:", OHNE Präfix) berechnet, mit maskierten
  Prompt-Tokens (ignore_index=-100). Damit ist die PPL über alle vier
  Bedingungen hinweg fair vergleichbar.

Die vier Bedingungen:
  1. Absolute Control      – Standardparameter, kein Präfix
  2. Noise Padding Control – Standardparameter, bedeutungsloser Platzhalter
                              exakt gleicher Zeichenlänge wie der PAL-Vektor
  3. Hyperparameter Baseline – enge τ/top_k, kein Präfix
  4. Full Framework        – PAL-Vektor-Präfix + enge τ/top_k

Metriken pro Bedingung: Target-Only-PPL, Distinct-2, Embedding-Similarity
zur kategorie-spezifischen Referenzbedeutung, Länge in Wörtern.
Statistik: Mann-Whitney-U zwischen Bedingung 4 und den Kontrollbedingungen.
"""

import json
import math
import random

import statistics
from dataclasses import dataclass, field
from pathlib import Path


import os
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer, util as st_util

# -----------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------

# Stelle sicher, dass pickle im Skript-Kopf importiert ist!
import pickle

# Diese globalen Variablen oben in der Nähe von MODEL_PATH definieren:
PAL_MODEL_PATH = "pal_model.pkl"
_PAL_VECTORIZER = None
_PAL_REGRESSOR = None


MODEL_PATH = "./finetuned_qwen_pal"          # Pfad zum fine-getunten Neocortex-Modell
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

STIMULI_FILE = "eval_stimuli.json"           # siehe Format weiter unten
MAX_NEW_TOKENS = 30
N_SAMPLES_PER_CONDITION = 1                  # >1, um Sampling-Varianz auszumitteln

TEMP_BASE, TOPK_BASE = 0.75, 50              # "kreativer" Standard
TEMP_RESTRICTED, TOPK_RESTRICTED = 0.25, 15  # "eingeschnürter" Zustand

# Kategorie-spezifische Referenzsätze für die Embedding-Similarity.
# Ersetzen/erweitern mit echten, vorab definierten Ziel-Verhaltensbeschreibungen.
REFERENCE_BEHAVIOR = {
    "threatening": "The subject retreats cautiously, avoids the danger and stays alert.",
    "benign": "The subject remains calm, continues normal behavior without alarm.",
}

# -----------------------------------------------------------------------
# Stimuli-Format (eval_stimuli.json):
# [
#   {"prompt": "the lion is in front of you", "category": "threatening"},
#   {"prompt": "love is in the air",          "category": "benign"},
#   ...
# ]
# Für N=200 wird eine entsprechend große, category-balancierte Datei erwartet.
# -----------------------------------------------------------------------


@dataclass
class ConditionResult:
    condition: str
    prompt: str
    category: str
    continuation: str
    perplexity: float
    word_count: int
    distinct_2: float
    embedding_similarity: float


@dataclass
class AblationReport:
    results: list = field(default_factory=list)

    def by_condition(self, condition: str):
        return [r for r in self.results if r.condition == condition]

    def ppl_values(self, condition: str):
        return [r.perplexity for r in self.by_condition(condition) if math.isfinite(r.perplexity)]


# -----------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------

def load_stimuli(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Bitte eine Stimuli-Datei im beschriebenen "
            f"JSON-Format bereitstellen (siehe Kommentar am Dateianfang)."
        )
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"{path} enthält keine Stimuli.")
    return data

def get_pal_vector_string(prompt: str) -> str:
    """
    Lädt die reale, trainierte Multi-Output-Ridge-Regression (Estimator) 
    und berechnet den 7D-Affektvektor für den gegebenen Prompt.
    """
    global _PAL_VECTORIZER, _PAL_REGRESSOR
    
    # Lazy Loading des Modells beim ersten Aufruf
    if _PAL_VECTORIZER is None or _PAL_REGRESSOR is None:
        if not os.path.exists(PAL_MODEL_PATH):
            raise FileNotFoundError(
                f"'{PAL_MODEL_PATH}' nicht gefunden. Bitte führe zuerst 'train_pal.py' aus, "
                f"um die Ridge-Regressionsmatrizen zu kalibrieren!"
            )
        with open(PAL_MODEL_PATH, "rb") as f:
            _PAL_VECTORIZER, _PAL_REGRESSOR = pickle.load(f)
            
    # Extrahiere Wörter aus dem Satz (Stopwörter-Filterung analog zum Hauptskript)
    raw_words = [w.strip(",.!?").lower() for w in prompt.split()]
    if not raw_words:
        raw_words = ["something"]
        
    # Vektoren für die Wörter berechnen
    word_vectors = []
    for word in raw_words:
        vec = _PAL_VECTORIZER.transform([word])
        # Wenn das Wort im TF-IDF-Raum existiert, prädizieren
        if np.sum(vec.toarray()) >= 0.15:
            pred = _PAL_REGRESSOR.predict(vec).flatten()
            word_vectors.append(np.clip(pred, 0.0, 1.0))
            
    # Falls keine bekannten Wörter gefunden wurden, Curiosity-Fallback simulieren
    if not word_vectors:
        pal_raw_values = np.array([0.5, 0.4, 0.5, 0.15, 0.3, 0.5, 0.4])
    else:
        # Aggregation über Mittelwert
        pal_raw_values = np.mean(word_vectors, axis=0)
        
    # Textuelle Repräsentation für den Prompt- frontier formatieren
    return (
        f"[PAL_7D | V:{pal_raw_values[0]:.2f} | A:{pal_raw_values[1]:.2f} | D:{pal_raw_values[2]:.2f} | "
        f"DNG:{pal_raw_values[3]:.2f} | RES:{pal_raw_values[4]:.2f} | SOC:{pal_raw_values[5]:.2f} | GOL:{pal_raw_values[6]:.2f}]"
    )



def make_noise_padding(reference_string: str, seed: int) -> str:
    """
    Erzeugt einen bedeutungsfreien Platzhalter-String exakt gleicher
    Zeichenlänge wie reference_string. Nutzt neutrale, hochfrequente
    Füllwörter statt reiner Zufallszeichen, damit die Tokenisierungs-
    Statistik (Token-Anzahl, Subword-Verteilung) der eines echten
    Vektor-Strings ähnlicher ist als bei Buchstabensalat.
    """
    filler_pool = [
        "context", "marker", "value", "slot", "buffer", "index",
        "field", "entry", "token", "state", "level", "unit",
    ]
    rng = random.Random(seed)
    target_len = len(reference_string)
    out = []
    cur_len = 0
    while cur_len < target_len:
        word = rng.choice(filler_pool)
        out.append(word)
        cur_len += len(word) + 1
    padded = " ".join(out)
    return padded[:target_len]


def target_only_perplexity(prompt: str, continuation: str, tokenizer, model) -> float:
    """
    Berechnet die Perplexität NUR der generierten Fortsetzung, unter einem
    fixen, präfix-freien Scoring-Kontext. Das entkoppelt die Messung vom
    Präfix, der nur die Generierung (nicht die Bewertung) beeinflusst hat.
    """
    base_prompt = f"REIZ: {prompt}. REAKTION:"
    full_text = f"{base_prompt} {continuation}"

    enc_prompt = tokenizer(base_prompt, return_tensors="pt")
    enc_full = tokenizer(full_text, return_tensors="pt").to(DEVICE)

    prompt_len = enc_prompt["input_ids"].shape[1]
    labels = enc_full["input_ids"].clone()
    labels[:, :prompt_len] = -100  # Prompt-Tokens aus der Loss ausmaskieren

    with torch.no_grad():
        out = model(**enc_full, labels=labels)
        loss = out.loss

    if loss is None or not math.isfinite(loss.item()):
        return float("inf")
    return math.exp(loss.item())


def distinct_2(text: str) -> float:
    words = text.split()
    if len(words) <= 1:
        return 1.0
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return len(set(bigrams)) / len(bigrams)


def embedding_similarity(text: str, category: str, embedder) -> float:
    reference = REFERENCE_BEHAVIOR.get(category)
    if not reference or not text.strip():
        return float("nan")
    emb_text = embedder.encode(text, convert_to_tensor=True)
    emb_ref = embedder.encode(reference, convert_to_tensor=True)
    return float(st_util.cos_sim(emb_text, emb_ref).item())


def generate_continuation(prompt: str, prefix: str, temp: float, top_k: int,
                           tokenizer, model, seed: int) -> str:
    gen_prompt = f"{prefix} REIZ: {prompt}. REAKTION:" if prefix else f"REIZ: {prompt}. REAKTION:"
    inputs = tokenizer(gen_prompt, return_tensors="pt").to(DEVICE)
    torch.manual_seed(seed)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temp,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# -----------------------------------------------------------------------
# Ablations-Kern
# -----------------------------------------------------------------------

def run_condition(condition: str, prompt: str, category: str, prefix: str,
                   temp: float, top_k: int, tokenizer, model, embedder,
                   seed: int) -> ConditionResult:
    continuation = generate_continuation(prompt, prefix, temp, top_k, tokenizer, model, seed)
    ppl = target_only_perplexity(prompt, continuation, tokenizer, model)
    d2 = distinct_2(continuation)
    sim = embedding_similarity(continuation, category, embedder)
    return ConditionResult(
        condition=condition,
        prompt=prompt,
        category=category,
        continuation=continuation,
        perplexity=ppl,
        word_count=len(continuation.split()),
        distinct_2=d2,
        embedding_similarity=sim,
    )


def run_ablation(stimuli: list, tokenizer, model, embedder) -> AblationReport:
    report = AblationReport()

    for i, item in enumerate(stimuli):
        prompt, category = item["prompt"], item["category"]

        pal_vector_str = get_pal_vector_string(prompt)

        noise_str = make_noise_padding(pal_vector_str, seed=i)

        for rep in range(N_SAMPLES_PER_CONDITION):
            seed = i * 1000 + rep

            report.results.append(run_condition(
                "1_absolute_control", prompt, category, prefix="",
                temp=TEMP_BASE, top_k=TOPK_BASE,
                tokenizer=tokenizer, model=model, embedder=embedder, seed=seed))

            report.results.append(run_condition(
                "2_noise_padding_control", prompt, category, prefix=noise_str,
                temp=TEMP_BASE, top_k=TOPK_BASE,
                tokenizer=tokenizer, model=model, embedder=embedder, seed=seed))

            report.results.append(run_condition(
                "3_hyperparameter_baseline", prompt, category, prefix="",
                temp=TEMP_RESTRICTED, top_k=TOPK_RESTRICTED,
                tokenizer=tokenizer, model=model, embedder=embedder, seed=seed))

            report.results.append(run_condition(
                "4_full_framework", prompt, category, prefix=pal_vector_str,
                temp=TEMP_RESTRICTED, top_k=TOPK_RESTRICTED,
                tokenizer=tokenizer, model=model, embedder=embedder, seed=seed))

    return report


# -----------------------------------------------------------------------
# Auswertung / Statistik
# -----------------------------------------------------------------------

def summarize(report: AblationReport) -> dict:
    conditions = [
        "1_absolute_control",
        "2_noise_padding_control",
        "3_hyperparameter_baseline",
        "4_full_framework",
    ]
    summary = {}
    for cond in conditions:
        rows = report.by_condition(cond)
        ppls = [r.perplexity for r in rows if math.isfinite(r.perplexity)]
        sims = [r.embedding_similarity for r in rows if math.isfinite(r.embedding_similarity)]
        d2s = [r.distinct_2 for r in rows]
        summary[cond] = {
            "n": len(rows),
            "ppl_mean": statistics.mean(ppls) if ppls else float("nan"),
            "ppl_stdev": statistics.stdev(ppls) if len(ppls) > 1 else float("nan"),
            "distinct_2_mean": statistics.mean(d2s) if d2s else float("nan"),
            "embedding_similarity_mean": statistics.mean(sims) if sims else float("nan"),
        }
    return summary


def significance_tests(report: AblationReport) -> dict:
    """
    Mann-Whitney-U zwischen Bedingung 4 (volles System) und den beiden
    Kontrollbedingungen 2 und 3. Signifikanz hier bedeutet: der PPL-Abfall
    lässt sich NICHT allein durch Präfix-Länge (2) oder Parameter-Restriktion
    (3) erklären.
    """
    ppl_4 = report.ppl_values("4_full_framework")
    ppl_2 = report.ppl_values("2_noise_padding_control")
    ppl_3 = report.ppl_values("3_hyperparameter_baseline")

    results = {}
    if ppl_4 and ppl_2:
        stat, p = mannwhitneyu(ppl_4, ppl_2, alternative="less")
        results["4_vs_2_noise_padding"] = {"U": stat, "p_value": p, "significant": p < 0.05}
    if ppl_4 and ppl_3:
        stat, p = mannwhitneyu(ppl_4, ppl_3, alternative="less")
        results["4_vs_3_hyperparameter"] = {"U": stat, "p_value": p, "significant": p < 0.05}
    return results


def print_report(summary: dict, sig: dict) -> None:
    print("\n=== 4-Way Ablation: Zusammenfassung ===")
    header = f"{'Bedingung':<28}{'N':>5}{'PPL (Ø)':>12}{'PPL (σ)':>12}{'Distinct-2':>12}{'Sim':>8}"
    print(header)
    print("-" * len(header))
    for cond, s in summary.items():
        print(f"{cond:<28}{s['n']:>5}{s['ppl_mean']:>12.2f}{s['ppl_stdev']:>12.2f}"
              f"{s['distinct_2_mean']:>12.3f}{s['embedding_similarity_mean']:>8.3f}")

    print("\n=== Signifikanztests (Mann-Whitney-U, einseitig 'less') ===")
    for name, r in sig.items():
        flag = "✅ signifikant" if r["significant"] else "❌ nicht signifikant"
        print(f"{name}: U={r['U']:.1f}, p={r['p_value']:.4f} → {flag}")

    print(
        "\nHinweis: Nur wenn 4_full_framework signifikant NIEDRIGERE PPL zeigt als\n"
        "sowohl 2_noise_padding_control als auch 3_hyperparameter_baseline, ist der\n"
        "PPL-Effekt dem semantischen Inhalt des 7D-Vektors zuzuschreiben und nicht\n"
        "bloß der Präfixlänge oder der Sampling-Restriktion."
    )


# -----------------------------------------------------------------------
# Einstiegspunkt
# -----------------------------------------------------------------------

def main():
    print(f"Lade Modell von {MODEL_PATH} auf {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print(f"Lade Embedding-Modell {EMBEDDING_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    stimuli = load_stimuli(STIMULI_FILE)
    print(f"{len(stimuli)} Stimuli geladen. Starte 4-Way-Ablation ...")

    report = run_ablation(stimuli, tokenizer, model, embedder)
    summary = summarize(report)
    sig = significance_tests(report)

    print_report(summary, sig)

    out_path = Path("ablation_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "significance": sig,
            "raw": [r.__dict__ for r in report.results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetaillierte Rohdaten gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()
