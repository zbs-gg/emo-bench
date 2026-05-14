"""Mini bench: compare readers on same retrieved contexts.

For N Qs from LongMemEval_S:
1. Run Pulse cosine retrieval once (share across readers)
2. Feed same top-K to each reader
3. Grade with kimi-k2.6 judge (official anscheck prompts)
4. Report per-reader accuracy + median latency
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import embed_cohere, secret, tokenize
from evaluate_lme_kimi import get_prompt, extract_yesno


QA_SYSTEM = ("You answer using only provided chat history sessions. "
             "Be concise (1-2 sentences). If info missing, say so. "
             "For temporal questions, use timestamps. Do not hallucinate.")


def concat_session(session, max_chars=8000):
    return "\n".join(f"[{t.get('role','?')}] {t.get('content','')}" for t in session)[:max_chars]


READERS = {
    # name: (base_url, api_key_file, model, max_tokens)
    "kimi-k2.6":       ("https://api.moonshot.ai/v1",        "kimi-api-key.txt", "kimi-k2.6",             3000),
    "kimi-k2-turbo":   ("https://api.moonshot.ai/v1",        "kimi-api-key.txt", "kimi-k2-turbo-preview", 1000),
    "glm-5":           ("https://api.z.ai/api/paas/v4",      "zai-api-key.txt",  "glm-5",                 3000),
    "glm-4.6":         ("https://api.z.ai/api/paas/v4",      "zai-api-key.txt",  "glm-4.6",               3000),
}


async def reader_answer(client, model, max_tokens, question, qdate, sessions, dates):
    history = "\n\n".join(f"=== Session at {d} ===\n{concat_session(s, 6000)}"
                          for s, d in zip(sessions, dates))
    prompt = f"[Question asked on {qdate}]\n\nHistory:\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=1.0, max_tokens=max_tokens,
        )
        content = (r.choices[0].message.content or "").strip()
        return content
    except Exception as ex:
        return f"[ERROR: {str(ex)[:150]}]"


async def judge(client, hyp, ref, model="kimi-k2.6"):
    qtype = ref["question_type"]
    is_abs = "_abs" in ref["question_id"]
    prompt = get_prompt(qtype, ref["question"], ref["answer"], hyp, abstention=is_abs)
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a strict evaluator."},
                      {"role": "user", "content": prompt}],
            temperature=1.0, max_tokens=2000,
        )
        return extract_yesno(r.choices[0].message.content or "")
    except Exception:
        return None


def get_client(base_url, key_file):
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=secret(key_file), base_url=base_url)


async def run(data_file, n, top_k, seed, out_file):
    data = json.load(open(data_file))
    random.seed(seed)
    sample = random.sample(data, n)
    print(f"Mini-bench: {n} Qs, top_k={top_k}", file=sys.stderr)

    # --- Retrieval phase (once) ---
    retrievals = []
    for i, entry in enumerate(sample, 1):
        sessions = entry["haystack_sessions"]
        dates = entry.get("haystack_dates", [""] * len(sessions))
        texts = [concat_session(s) for s in sessions]
        vecs = embed_cohere(texts, "search_document")
        q_vec = embed_cohere([entry["question"]], "search_query")[0]
        order = list(np.argsort(-(vecs @ q_vec)))[:top_k]
        retrievals.append({
            "entry": entry,
            "retrieved_sessions": [sessions[int(j)] for j in order],
            "retrieved_dates": [dates[int(j)] for j in order],
        })
        if i % 5 == 0:
            print(f"  retrieval {i}/{n}", file=sys.stderr)

    # --- Reader phase ---
    reader_clients = {name: get_client(cfg[0], cfg[1]) for name, cfg in READERS.items()}
    judge_client = get_client(*READERS["kimi-k2.6"][:2])

    results = {name: {"hyps": [], "times": [], "labels": []} for name in READERS}

    for i, r in enumerate(retrievals, 1):
        e = r["entry"]
        for name, (_, _, model, max_t) in READERS.items():
            t0 = time.time()
            hyp = await reader_answer(reader_clients[name], model, max_t,
                                       e["question"], e.get("question_date", ""),
                                       r["retrieved_sessions"], r["retrieved_dates"])
            elapsed = time.time() - t0
            label = await judge(judge_client, hyp, e)
            results[name]["hyps"].append({"qid": e["question_id"], "hyp": hyp, "label": label, "t": elapsed})
            results[name]["times"].append(elapsed)
            results[name]["labels"].append(label)
        print(f"  Q {i}/{n} done", file=sys.stderr)

    # --- Report ---
    print()
    print(f"{'Reader':<20} {'Acc':>8} {'Median sec':>12} {'Avg sec':>10} {'n':>4}")
    print("-" * 56)
    for name, data in results.items():
        valid = [l for l in data["labels"] if l is not None]
        acc = 100 * sum(1 for l in valid if l) / max(len(valid), 1)
        med = float(np.median(data["times"]))
        avg = float(np.mean(data["times"]))
        print(f"{name:<20} {acc:>7.2f} {med:>11.1f} {avg:>9.1f} {len(valid):>4}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved to {out_file}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.data, args.n, args.top_k, args.seed, args.out))


if __name__ == "__main__":
    main()
