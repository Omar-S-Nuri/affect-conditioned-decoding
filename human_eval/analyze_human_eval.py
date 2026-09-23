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

import argparse
import csv
import itertools
import statistics
from pathlib import Path

from scipy.stats import wilcoxon, spearmanr

CONDITIONS = ("3_hyperparameter_baseline", "4_full_framework")


def load_mapping(path: Path) -> dict:
    mapping = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["item_code"]] = row
    return mapping


def load_ratings(path: Path) -> dict:
    """ { item_code: {"coherence": float, "appropriateness": float} } """
    ratings = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["item_code"]
            coh = row.get("coherence_1to5", "").strip()
            app = row.get("appropriateness_1to5", "").strip()
            if not coh or not app:
                continue  # unvollständig ausgefüllte Zeile überspringen
            ratings[code] = {
                "coherence": float(coh),
                "appropriateness": float(app),
            }
    return ratings


def paired_scores(mapping: dict, ratings: dict, dimension: str) -> tuple:
    """
    Liefert zwei gleich lange Listen (cond3_scores, cond4_scores),
    gepaart über denselben Prompt.
    """
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

    # Gepaarter, nicht-parametrischer Test: Wilcoxon signed-rank.
    # alternative='greater': prüft, ob 4_full_framework > 3_hyperparameter_baseline
    try:
        stat, p = wilcoxon(cond4, cond3, alternative="greater")
        flag = "✅ signifikant" if p < 0.05 else "❌ nicht signifikant"
        print(f"    Wilcoxon (4 > 3): W={stat:.1f}, p={p:.4f} → {flag}")
    except ValueError as e:
        print(f"    Wilcoxon-Test nicht durchführbar: {e}")


def inter_rater_reliability(rating_files: list, dimension: str):
    """
    Paarweise Spearman-Korrelation zwischen allen Ratern auf derselben
    Dimension, nur über Items, die alle Rater bewertet haben.
    """
    all_ratings = [load_ratings(p) for p in rating_files]
    common_codes = set.intersection(*(set(r.keys()) for r in all_ratings))
    if len(common_codes) < 3:
        print(f"  Zu wenige gemeinsam bewertete Items ({len(common_codes)}) "
              f"für Inter-Rater-Reliabilität.")
        return

    print(f"  Paarweise Spearman-Korrelation ({dimension}, n={len(common_codes)} gemeinsame Items):")
    for i, j in itertools.combinations(range(len(rating_files)), 2):
        vals_i = [all_ratings[i][c][dimension] for c in common_codes]
        vals_j = [all_ratings[j][c][dimension] for c in common_codes]
        rho, p = spearmanr(vals_i, vals_j)
        print(f"    Rater {i+1} vs. Rater {j+1}: rho={rho:.3f} (p={p:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=str, required=True,
                         help="Pfad zu mapping_key.csv")
    parser.add_argument("--ratings", type=str, nargs="+", required=True,
                         help="Ein oder mehrere ausgefüllte rating_sheet_raterX.csv")
    args = parser.parse_args()

    mapping = load_mapping(Path(args.mapping))
    rating_paths = [Path(p) for p in args.ratings]

    print(f"=== Human Evaluation: {len(rating_paths)} Rater, "
          f"{len(mapping)} Items im Mapping ===\n")

    if len(rating_paths) < 2:
        print("⚠️  Nur ein Rater übergeben. Inter-Rater-Reliabilität kann nicht "
              "berechnet werden – für eine belastbare 'double-blind human evaluation' "
              "werden mindestens 2, besser 3 unabhängige Rater benötigt.\n")

    # Alle Ratings zusammenführen für die Signifikanztests (jede Rater-Bewertung
    # zählt als eigener Datenpunkt, gepaart über item->prompt->condition)
    merged = {}
    for path in rating_paths:
        r = load_ratings(path)
        for code, vals in r.items():
            merged.setdefault(code, []).append(vals)

    # Für die Signifikanztests: Mittelwert über Rater pro Item verwenden
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
        inter_rater_reliability(rating_paths, "appropriateness")


if __name__ == "__main__":
    main()
