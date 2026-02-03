# Evaluation Runner (A/B prompts + failure set + metrics)
import json
from pathlib import Path
from datetime import datetime, timezone

from agents.retriever import retrieve
from agents.judge import judge

ROOT = Path(__file__).resolve().parents[1]
FAILURE_SET = ROOT / "eval" / "failure_set.jsonl"
OUT_DIR = ROOT / "eval" / "results"

def load_failure_set(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def citation_hit_rate(judge_obj, evidence):
    # check whether judge citations are actually inside retrieved evidence list
    cited = set(judge_obj.get("citations", []))
    if not cited:
        return 0.0
    got = set([e["chunk_id"] for e in evidence])
    hits = sum([1 for c in cited if c in got])
    return hits / max(1, len(cited))

def eval_variant(variant: str, tests):
    results = []
    correct = 0
    total = 0
    hit_rates = []
    flag_count = 0

    for t in tests:
        claim = t["claim"]
        expected = t["expected"]

        # retrieval first
        evidence = retrieve(claim, k=5)

        # judge with prompt variant
        judge_obj, flags = judge(claim, evidence, variant=variant)

        pred = judge_obj.get("verdict", "Unknown")

        is_correct = (pred == expected)
        correct += 1 if is_correct else 0
        total += 1

        hr = citation_hit_rate(judge_obj, evidence)
        hit_rates.append(hr)

        if flags:
            flag_count += 1

        results.append({
            "id": t["id"],
            "claim": claim,
            "expected": expected,
            "pred": pred,
            "is_correct": is_correct,
            "citation_hit_rate": hr,
            "flags": flags,
            "judge_obj": judge_obj,
            "top_evidence": [
                {"chunk_id": e["chunk_id"], "doc_id": e["doc_id"], "score": e["score"]}
                for e in evidence
            ]
        })

    accuracy = correct / max(1, total)
    avg_hit_rate = sum(hit_rates) / max(1, len(hit_rates))
    flag_rate = flag_count / max(1, total)

    summary = {
        "variant": variant,
        "n_tests": total,
        "accuracy": accuracy,
        "avg_citation_hit_rate": avg_hit_rate,
        "flag_rate": flag_rate,
    }

    return summary, results

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tests = load_failure_set(FAILURE_SET)
    print(f"Loaded failure set: {len(tests)} tests")

    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "").replace(".", "")
    # run A and B
    for variant in ["A", "B"]:
        summary, results = eval_variant(variant, tests)

        out_path = OUT_DIR / f"eval_{variant}_{timestamp}.json"
        payload = {
            "summary": summary,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nVariant {variant}")
        print(json.dumps(summary, indent=2))
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
