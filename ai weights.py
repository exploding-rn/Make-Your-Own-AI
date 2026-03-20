"""
CLINT Industries — Dataset Weighting Script
Adds weights to model_training_data.jsonl based on source model
Output: model_training_data_weighted.jsonl
"""

import json

INPUT_FILE  = "model_training_data.jsonl"
OUTPUT_FILE = "model_training_data_weighted.jsonl"

# weights by source
WEIGHTS = {
    # Claude — highest quality
    "claude-sonnet-4-6":     5.0,
    "claude":                5.0,

    # Kimi K2 1T cloud — second highest
    "kimi-k2:1t-cloud":      4.0,

    # local bulk models
    "gemma3:4b":             1.5,
    "gemma3:12b":            1.5,
    "ministral-3:14b":       1.5,
    "granite-code:8b":       1.5,
    "granite3.3:8b":         1.5,
    "llama3.2:3b":           1.5,
}

DEFAULT_WEIGHT = 1.0  # fallback for anything unrecognized

def get_weight(source: str) -> float:
    # exact match first
    if source in WEIGHTS:
        return WEIGHTS[source]
    # partial match fallback
    for key, weight in WEIGHTS.items():
        if key in source or source in key:
            return weight
    return DEFAULT_WEIGHT

def main():
    print("🚀 CLINT Industries — Dataset Weighting")
    print(f"   Input  : {INPUT_FILE}")
    print(f"   Output : {OUTPUT_FILE}")

    counts  = {}
    total   = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj    = json.loads(line)
                source = obj.get("source", "unknown")
                weight = get_weight(source)

                obj["weight"] = weight

                f_out.write(json.dumps(obj) + "\n")
                f_out.flush()

                counts[source] = counts.get(source, 0) + 1
                total += 1

            except Exception as e:
                skipped += 1
                continue

    print(f"\n✅ Done — {total} examples weighted, {skipped} skipped")
    print(f"\n{'─'*60}")
    print(f"  {'SOURCE':<30} {'COUNT':>6}   {'WEIGHT':>6}   {'EFFECTIVE':>9}")
    print(f"{'─'*60}")
    for source, count in sorted(counts.items(), key=lambda x: -x[1]):
        weight    = get_weight(source)
        effective = count * weight
        print(f"  {source:<30} {count:>6}   {weight:>6.1f}   {effective:>9.0f}")
    print(f"{'─'*60}")
    total_effective = sum(counts[s] * get_weight(s) for s in counts)
    print(f"  {'TOTAL':<30} {total:>6}   {'':>6}   {total_effective:>9.0f}")
    print(f"{'─'*60}")
    print(f"\n  Effective total = how much each model contributes to training")
    print(f"  Claude 5x and Kimi 4x means they punch way above their count")
    print(f"\n  Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()