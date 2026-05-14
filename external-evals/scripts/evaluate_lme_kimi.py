"""LongMemEval evaluator using Kimi K2.6 judge. Follows official anscheck prompts."""
from __future__ import annotations
import argparse, asyncio, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import kimi_client, kimi_chat


def get_prompt(task, question, answer, response, abstention=False):
    if abstention:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. "
                "The model could say the information is incomplete, or give other information but not what was asked.\n\n"
                f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
                "Does the model correctly identify the question as unanswerable? Answer yes or no only at the end.")
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return ("I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer. Otherwise answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps, "
                "you should answer yes. If the response contains only a subset of the required information, answer no.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only at the end.")
    if task == "temporal-reasoning":
        return ("I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer. Otherwise answer no. "
                "If the response is equivalent to the correct answer or contains all intermediate steps, answer yes. "
                "Do not penalize off-by-one errors for number of days.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only at the end.")
    if task == "knowledge-update":
        return ("I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer. Otherwise answer no. "
                "If the response contains some previous information along with an updated answer, "
                "it is correct as long as the updated answer is the required answer.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only at the end.")
    if task == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Answer yes if the response satisfies the desired response. Otherwise answer no.\n\n"
                f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only at the end.")
    raise ValueError(task)


def extract_yesno(text: str) -> bool | None:
    """Kimi K2.6 may emit reasoning; we look for the LAST yes/no token."""
    t = text.lower().strip()
    # find all yes/no occurrences
    matches = re.findall(r"\b(yes|no)\b", t)
    if not matches: return None
    return matches[-1] == "yes"


async def judge_one(client, hyp, ref, sem, model):
    async with sem:
        qtype = ref["question_type"]
        is_abs = "_abs" in hyp["question_id"]
        prompt = get_prompt(qtype, ref["question"], ref["answer"], hyp["hypothesis"], abstention=is_abs)
        txt = await kimi_chat(client, model, "You are a strict evaluator.", prompt, max_tokens=2000)
        return extract_yesno(txt)


async def run(hyp_file, ref_file, out_file, model, concurrency):
    client = kimi_client()
    hyps = [json.loads(l) for l in hyp_file.open() if l.strip()]
    refs = {e["question_id"]: e for e in json.load(ref_file.open())}
    done = {}
    if out_file.exists():
        for l in out_file.open():
            if l.strip():
                r = json.loads(l)
                done[r["question_id"]] = r
    todo = [h for h in hyps if h["question_id"] not in done]
    print(f"{len(hyps)} total, {len(done)} cached, {len(todo)} new", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)

    async def work(h):
        r = refs[h["question_id"]]
        label = await judge_one(client, h, r, sem, model)
        return {"question_id": h["question_id"], "question_type": r["question_type"], "label": label}

    tasks = [work(h) for h in todo]
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
            all_scored[r["question_id"]] = r
    per_type = defaultdict(list)
    no_label = 0
    for r in all_scored.values():
        if r["label"] is None:
            no_label += 1
            continue
        per_type[r["question_type"]].append(1 if r["label"] else 0)
    print(f"\nJudge: {model}; no_label={no_label}")
    print(f"{'Type':<30} {'Acc':>8} {'n':>6}")
    print("-" * 46)
    all_vals = []
    for t, vals in sorted(per_type.items()):
        print(f"{t:<30} {100*sum(vals)/len(vals):>7.2f} {len(vals):>6}")
        all_vals.extend(vals)
    print("-" * 46)
    if all_vals:
        print(f"{'Overall':<30} {100*sum(all_vals)/len(all_vals):>7.2f} {len(all_vals):>6}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="kimi-k2.6")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.hyp, args.ref, args.out, args.model, args.concurrency))


if __name__ == "__main__":
    main()
