"""ES-MemEval evaluator via Kimi K2.6: F1 (token overlap) + LLM-judge (0-2 strict)."""
from __future__ import annotations
import argparse, asyncio, json, re, string, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import kimi_client, kimi_chat

TOKEN_RE = re.compile(r"\w+")


def normalize(t):
    t = t.lower()
    t = "".join(c for c in t if c not in string.punctuation)
    return TOKEN_RE.findall(t)


def f1(gold, pred):
    g = normalize(gold)
    p = normalize(pred)
    if not p or not g: return 0.0
    common = Counter(g) & Counter(p)
    n = sum(common.values())
    if n == 0: return 0.0
    prec = n / len(p)
    rec = n / len(g)
    return 2 * prec * rec / (prec + rec)


JUDGE_SYSTEM = "You are a strict evaluator."
JUDGE_TEMPLATE = """You are an impartial evaluator.
Your task is to score a model's answer to a given question against a gold (reference) answer.

Scoring criteria:
- 0: Completely wrong or irrelevant
- 1: Partially correct but incomplete, vague, or missing key information
- 2: Completely correct and contextually accurate

Question: {question}
Gold Answer: {gold}
Model Answer: {prediction}

Output only one line in the exact format: "Score: X" where X is 0, 1, or 2."""


def extract_score(text: str) -> int | None:
    """Handle reasoning traces by looking for 'Score: X' pattern first, then last digit."""
    m = re.search(r"[Ss]core\s*:\s*([0-2])", text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\b([0-2])\b", text)
    return int(nums[-1]) if nums else None


async def judge_one(client, entry, sem, model):
    async with sem:
        prompt = JUDGE_TEMPLATE.format(question=entry["question"],
                                       gold=entry.get("answer") or "",
                                       prediction=entry["hypothesis"])
        txt = await kimi_chat(client, model, JUDGE_SYSTEM, prompt, max_tokens=2000)
        return extract_score(txt)


async def run(hyp_file, out_file, model, concurrency):
    client = kimi_client()
    entries = [json.loads(l) for l in hyp_file.open() if l.strip()]
    done = {}
    if out_file.exists():
        for l in out_file.open():
            if l.strip():
                r = json.loads(l)
                done[r["question_uid"]] = r
    todo = [e for e in entries if e["question_uid"] not in done]
    print(f"{len(entries)} total, {len(done)} cached, {len(todo)} new", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)

    async def work(e):
        score = await judge_one(client, e, sem, model)
        return {"question_uid": e["question_uid"], "capability": e["capability"],
                "llm_judge": score, "f1": f1(e.get("answer") or "", e["hypothesis"])}

    tasks = [work(e) for e in todo]
    with out_file.open("a") as f:
        for i in range(0, len(tasks), 50):
            batch = tasks[i:i + 50]
            res = await asyncio.gather(*batch)
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {min(i+50, len(tasks))}/{len(tasks)}", file=sys.stderr)

    all_scored = {}
    for l in out_file.open():
        if l.strip():
            r = json.loads(l)
            all_scored[r["question_uid"]] = r
    per_cap = defaultdict(lambda: {"f1": [], "judge": []})
    no_label = 0
    for r in all_scored.values():
        c = r["capability"]
        per_cap[c]["f1"].append(r["f1"])
        if r["llm_judge"] is not None:
            per_cap[c]["judge"].append(r["llm_judge"])
        else:
            no_label += 1
    print(f"\nJudge: {model}; no_label={no_label}")
    print(f"{'Capability':<25} {'F1':>8} {'LLM-judge':>12} {'n':>6}")
    print("-" * 55)
    all_f1, all_j = [], []
    for c, v in sorted(per_cap.items()):
        f1_avg = 100 * sum(v["f1"]) / len(v["f1"]) if v["f1"] else 0
        j_avg = sum(v["judge"]) / len(v["judge"]) if v["judge"] else 0
        print(f"{c:<25} {f1_avg:>7.2f} {j_avg:>11.3f} {len(v['f1']):>6}")
        all_f1.extend(v["f1"])
        all_j.extend(v["judge"])
    print("-" * 55)
    if all_f1:
        print(f"{'Overall':<25} {100*sum(all_f1)/len(all_f1):>7.2f} {sum(all_j)/len(all_j):>11.3f} {len(all_f1):>6}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="kimi-k2.6")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.hyp, args.out, args.model, args.concurrency))


if __name__ == "__main__":
    main()
