"""
analyze_human_eval.py
======================
Wertet die ausgefüllten Bewertungsbögen aus prepare_human_eval.py aus.

Erwartet:
  - mapping_key.csv (aus der Vorbereitung, GEHEIM gehalten bis hierher)
  - einen oder mehrere ausgefüllte rating_sheet_raterX.csv
    (gleiche Spalten wie rating_sheet.csv, aber coherence_1to5 /
    appropriateness_1to5 ausgefüllt)

Berechnet:
  - Mittelwerte pro Bedingung (3_hyperparameter_baseline vs. 4_full_framework)
  - Wilcoxon signed-rank Test (GEPAART, da dieselben Stimuli in beiden
    Bedingungen auftauchen – NICHT Mann-Whitney-U, das wäre für unabhängige
    Stichproben gedacht)
  - Inter-Rater-Übereinstimmung (Spearman-Korrelation zwischen Ratern,
    paarweise, als einfaches Reliabilitätsmaß)

Aufruf:
  python analyze_human_eval.py --mapping mapping_key.csv \
      --ratings rating_sheet_rater1.csv rating_sheet_rater2.csv rating_sheet_rater3.csv
"""
# human_eval/analyze_human_eval.py
import argparse
import csv
import os
import itertools
import statistics
from pathlib import Path
from scipy.stats import wilcoxon, spearmanr

CONDITIONS = ("3_hyperparameter_baseline", "4_full_framework")

def safe_path(path_str: str) -> Path:
    """🔴 FIX: Sucht intelligent im aktuellen Ordner ODER im human_eval-Unterordner"""
    p = Path(path_str)
    if p.exists():
        return p
    # Fallback 1: Falls aus dem Hauptordner aufgerufen und Datei liegt im Unterordner
    fallback_1 = Path("human_eval") / path_str
    if fallback_1.exists():
        return fallback_1
    # Fallback 2: Falls aus dem Unterordner aufgerufen und Datei liegt im Hauptordner
    fallback_2 = Path("..") / path_str
    if fallback_2.exists():
        return fallback_2
    return p

def load_mapping(path: Path) -> dict:
    mapping = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["item_code"]] = row
    return mapping

def load_ratings(path: Path) -> dict:
    ratings = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["item_code"]
            coh = row.get("coherence_1to5", "").strip()
            app = row.get("appropriateness_1to5", "").strip()
            if not coh or not app:
                continue
            ratings[code] = {
                "coherence": float(coh),
                "appropriateness": float(app),
            }
    return ratings

def paired_scores(mapping: dict, ratings: dict, dimension: str) -> tuple:
    by_prompt = {}
    for code, r in ratings.items():
        meta = mapping.get(code)
        if meta is None:
            continue
        by_prompt.setdefault(meta["prompt"], {})[meta["condition"]] = r[dimension]

    cond3, cond4 = [], []
    for prompt, vals in by_prompt.items():
        if all(c in vals for c in CONDITIONS):
            cond3.append(vals[CONDITIONS[0]])
            cond4.append(vals[CONDITIONS[1]])
    return cond3, cond4

def summarize_dimension(mapping: dict, ratings: dict, dimension: str, label: str):
    cond3, cond4 = paired_scores(mapping, ratings, dimension)
    if len(cond3) < 2:
        print(f"  {label}: zu wenige gepaarte Datenpunkte ({len(cond3)}) für einen Test.")
        return

    mean3, mean4 = statistics.mean(cond3), statistics.mean(cond4)
    print(f"  {label}:")
    print(f"    3_hyperparameter_baseline  Ø={mean3:.2f}  (n={len(cond3)})")
    print(f"    4_full_framework           Ø={mean4:.2f}  (n={len(cond4)})")

    try:
        stat, p = wilcoxon(cond4, cond3, alternative="greater")
        flag = "✅ signifikant" if p < 0.05 else "❌ nicht signifikant"
        print(f"    Wilcoxon (4 > 3): W={stat:.1f}, p={p:.4f} → {flag}")
    except ValueError as e:
        print(f"    Wilcoxon-Test nicht durchführbar: {e}")

def inter_rater_reliability(rating_files: list, dimension: str):
    all_ratings = [load_ratings(p) for p in rating_files]
    common_codes = set.intersection(*(set(r.keys()) for r in all_ratings))
    if len(common_codes) < 3:
        print(f"  Zu wenige gemeinsam bewertete Items ({len(common_codes)}) für Inter-Rater-Reliabilität.")
        return

    print(f"  Paarweise Spearman-Korrelation ({dimension}, n={len(common_codes)} gemeinsame Items):")
    for i, j in itertools.combinations(range(len(rating_files)), 2):
        vals_i = [all_ratings[i][c][dimension] for c in common_codes]
        vals_j = [all_ratings[j][c][dimension] for c in common_codes]
        rho, p = spearmanr(vals_i, vals_j)
        print(f"    Rater {i+1} vs. Rater {j+1}: rho={rho:.3f} (p={p:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=str, required=True, help="Pfad zu mapping_key.csv")
    parser.add_argument("--ratings", type=str, nargs="+", required=True, help="Ein/mehrere ausgefüllte rating_sheet_raterX.csv")
    args = parser.parse_args()

    # 🔴 FIX: Nutze die safe_path Funktion für intelligentes Finden der Dateien
    mapping_p = safe_path(args.mapping)
    rating_paths = [safe_path(p) for p in args.ratings]

    if not mapping_p.exists():
        print(f"❌ Fehler: Mapping-Datei '{args.mapping}' konnte nirgendwo gefunden werden!")
        exit()
        
    for rp in rating_paths:
        if not rp.exists():
            print(f"❌ Fehler: Rating-Datei '{rp}' konnte nirgendwo gefunden werden!")
            exit()

    mapping = load_mapping(mapping_p)

    print(f"=== Human Evaluation: {len(rating_paths)} Rater, {len(mapping)} Items im Mapping ===\n")

    if len(rating_paths) < 2:
        print("⚠️  Nur ein Rater übergeben. Inter-Rater-Reliabilität kann nicht berechnet werden.\n")

    merged = {}
    for path in rating_paths:
        r = load_ratings(path)
        for code, vals in r.items():
            merged.setdefault(code, []).append(vals)

    averaged = {
        code: {
            "coherence": statistics.mean(v["coherence"] for v in vals),
            "appropriateness": statistics.mean(v["appropriateness"] for v in vals),
        }
        for code, vals in merged.items()
    }

    print("--- Gepaarte Signifikanztests (gemittelt über alle Rater) ---")
    summarize_dimension(mapping, averaged, "coherence", "Syntactic Coherence")
    summarize_dimension(mapping, averaged, "appropriateness", "Contextual Appropriateness")

    if len(rating_paths) >= 2:
        print("\n--- Inter-Rater-Reliabilität ---")
        inter_rater_reliability(rating_paths, "coherence")
        inter_rater_reliability(mapping_paths, "appropriateness")

if __name__ == "__main__":
    main()
