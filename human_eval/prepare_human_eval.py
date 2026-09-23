"""
prepare_human_eval.py
======================
Erzeugt aus den Rohdaten von evaluate_pal.py (ablation_results.json) eine
verblindete, randomisierte Stichprobe für die Human Evaluation:
Bedingung "3_hyperparameter_baseline" vs. "4_full_framework".

Ablauf:
  1. Lädt ablation_results.json (Output von evaluate_pal.py).
  2. Gruppiert die generierten Texte nach Prompt, filtert auf die beiden
     zu vergleichenden Bedingungen.
  3. Zieht eine balancierte Zufallsstichprobe von N Prompts
     (Standard: 30, je zur Hälfte "threatening"/"benign", soweit vorhanden).
  4. Vergibt jedem der 2*N Texte einen zufälligen, bedeutungslosen Code
     und mischt die Reihenfolge.
  5. Schreibt zwei Dateien:
       - rating_sheet.csv   → für die Bewerter (OHNE Bedingungs-Info!)
       - mapping_key.csv    → geheim, NICHT an Bewerter weitergeben.
                               Wird erst bei der Auswertung gebraucht.

Aufruf:
  python prepare_human_eval.py --input ../ablation_results.json --n 30 --seed 42
"""

import argparse
import csv
import json
import random
import string
from pathlib import Path

CONDITIONS_TO_COMPARE = ("3_hyperparameter_baseline", "4_full_framework")


def load_raw_results(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("raw")
    if not raw:
        raise ValueError(
            f"'{path}' enthält keinen 'raw'-Schlüssel mit Einzelergebnissen. "
            f"Bitte sicherstellen, dass evaluate_pal.py vollständig durchgelaufen ist."
        )
    return raw


def group_by_prompt(raw: list) -> dict:
    """
    { prompt: {"category": ..., "3_hyperparameter_baseline": continuation,
                              "4_full_framework": continuation} }
    Nimmt bei mehreren Samples pro Bedingung (N_SAMPLES_PER_CONDITION > 1)
    einfach das erste.
    """
    grouped = {}
    for row in raw:
        prompt = row["prompt"]
        cond = row["condition"]
        if cond not in CONDITIONS_TO_COMPARE:
            continue
        entry = grouped.setdefault(prompt, {"category": row["category"]})
        if cond not in entry:
            entry[cond] = row["continuation"]
    # nur Prompts behalten, bei denen BEIDE Bedingungen vorhanden sind
    complete = {
        p: v for p, v in grouped.items()
        if all(c in v for c in CONDITIONS_TO_COMPARE)
    }
    return complete


def stratified_sample(grouped: dict, n: int, rng: random.Random) -> list:
    by_category = {}
    for prompt, entry in grouped.items():
        by_category.setdefault(entry["category"], []).append(prompt)

    categories = list(by_category.keys())
    per_category = n // max(len(categories), 1)

    chosen = []
    for cat in categories:
        pool = by_category[cat][:]
        rng.shuffle(pool)
        chosen.extend(pool[:per_category])

    # Falls durch Rundung noch Plätze übrig sind, aus dem Restpool auffüllen
    remaining = n - len(chosen)
    if remaining > 0:
        leftover = [p for cat in categories for p in by_category[cat] if p not in chosen]
        rng.shuffle(leftover)
        chosen.extend(leftover[:remaining])

    return chosen[:n]


def random_code(rng: random.Random, length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "ITEM_" + "".join(rng.choice(alphabet) for _ in range(length))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../ablation_results.json",
                         help="Pfad zu ablation_results.json")
    parser.add_argument("--n", type=int, default=30,
                         help="Anzahl der Stimuli (nicht Texte!) für die Human Eval")
    parser.add_argument("--seed", type=int, default=42,
                         help="Zufalls-Seed, für Reproduzierbarkeit der Blindung")
    parser.add_argument("--outdir", type=str, default=".",
                         help="Zielordner für rating_sheet.csv / mapping_key.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    raw = load_raw_results(Path(args.input))
    grouped = group_by_prompt(raw)

    if len(grouped) < args.n:
        print(f"⚠️  Nur {len(grouped)} Prompts mit vollständigen Bedingungen 3+4 "
              f"gefunden, weniger als angefordert (n={args.n}). Nutze alle verfügbaren.")
        args.n = len(grouped)

    sampled_prompts = stratified_sample(grouped, args.n, rng)

    # Für jeden Prompt zwei Items erzeugen (eins pro Bedingung), verblindet.
    items = []
    for prompt in sampled_prompts:
        entry = grouped[prompt]
        for cond in CONDITIONS_TO_COMPARE:
            items.append({
                "prompt": prompt,
                "category": entry["category"],
                "condition": cond,
                "text": entry[cond],
            })

    rng.shuffle(items)  # Reihenfolge mischen, damit kein Muster erkennbar ist

    # Eindeutige Codes vergeben (Kollisionen vermeiden)
    used_codes = set()
    for item in items:
        code = random_code(rng)
        while code in used_codes:
            code = random_code(rng)
        used_codes.add(code)
        item["code"] = code

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- rating_sheet.csv: das bekommen die Bewerter. KEINE Bedingungs-Info! ---
    rating_path = outdir / "rating_sheet.csv"
    with rating_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_code", "stimulus_prompt", "generated_response",
            "coherence_1to5", "appropriateness_1to5",
        ])
        for item in items:
            writer.writerow([item["code"], item["prompt"], item["text"], "", ""])

    # --- mapping_key.csv: GEHEIM, erst für die Auswertung, nicht an Bewerter geben ---
    mapping_path = outdir / "mapping_key.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_code", "prompt", "category", "condition"])
        for item in items:
            writer.writerow([item["code"], item["prompt"], item["category"], item["condition"]])

    print(f"✅ {len(items)} verblindete Items aus {args.n} Stimuli erzeugt.")
    print(f"   Bewertungsbogen (an Rater geben):  {rating_path}")
    print(f"   Mapping-Key (GEHEIM, nicht teilen): {mapping_path}")
    print("\nNächste Schritte:")
    print("  1. rating_sheet.csv kopieren, einmal pro Bewerter (z.B. rating_sheet_rater1.csv, ...).")
    print("     Bewerter füllen NUR die Spalten coherence_1to5 / appropriateness_1to5 aus.")
    print("  2. Bewerter arbeiten unabhängig voneinander, ohne Austausch.")
    print("  3. Ausgefüllte Bögen später mit analyze_human_eval.py auswerten.")


if __name__ == "__main__":
    main()
