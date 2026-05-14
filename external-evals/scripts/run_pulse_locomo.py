"""Pulse on LoCoMo benchmark (Maharana et al., ACL 2024).

LoCoMo = 10 very-long-term conversations (up to 35 sessions each, ~300 turns,
~9K tokens avg). Each has ~200 QA pairs with dialogue-id evidence.

Mirrors run_pulse_lme.py:
  cosine          — pure cosine retrieval (Pulse v2_pure baseline)
  bm25            — BM25 lexical baseline
  hybrid          — RRF(cosine, BM25)
  hybrid_rerank   — RRF → top-N → LLM rerank → top-K

Output: jsonl compatible with LoCoMo official evaluator
(task_eval/evaluate_qa.py).
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time, hashlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import embed_cohere, llm_client, kimi_chat, rrf_merge, tokenize, parse_last_int, PROVIDERS


# QA prompt matching LoCoMo evaluator paper Section 4.2 (direct QA on retrieved sessions)
QA_SYSTEM = (
    "You answer questions about a past conversation between two people. "
    "Use ONLY the provided retrieved sessions as context. Be concise (1-2 sentences). "
    "If the information isn't in the retrieved sessions, say 'I don't know'. "
    "For date questions, give the exact date. For names, give the exact name. "
    "Do not hallucinate."
)

RERANK_SYSTEM = "You rate how relevant a past conversation session is to a question. End with just a number 0-10."
RERANK_TEMPLATE = ("Question: {q}\n\nSession:\n{s}\n\n"
                   "Rate relevance 0-10 (0=not relevant, 10=contains answer). "
                   "End with the number only.")


def concat_session(session_turns: list[dict], session_date: str, max_chars: int = 8000) -> str:
    """Turn a LoCoMo session (list of turns) into a single text blob."""
    parts = [f"[session date: {session_date}]"] if session_date else []
    for turn in session_turns:
        speaker = turn.get("speaker", "?")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", "")
        parts.append(f"[{dia_id}] {speaker}: {text}")
    return "\n".join(parts)[:max_chars]


def get_sessions(sample: dict) -> list[tuple[str, str, str]]:
    """Return (session_id, session_text, session_date) for each session in sample."""
    conv = sample["conversation"]
    sessions = []
    # Find all session_N keys (sorted numerically)
    keys = sorted([k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
                  key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0)
    for k in keys:
        turns = conv[k]
        if not isinstance(turns, list):
            continue
        date_key = f"{k}_date_time"
        date = conv.get(date_key, "")
        text = concat_session(turns, date)
        sessions.append((k, text, date))
    return sessions


async def rerank_one(client, question, text, sem, model):
    async with sem:
        txt = await kimi_chat(client, model, RERANK_SYSTEM,
                              RERANK_TEMPLATE.format(q=question, s=text[:3000]),
                              max_tokens=2000)
        return parse_last_int(txt)


async def answer_one(client, question, retrieved_sessions, sem, model):
    async with sem:
        history = "\n\n".join(
            f"=== {sid} ({sdate}) ===\n{stext}"
            for sid, stext, sdate in retrieved_sessions)
        prompt = f"Retrieved conversation sessions:\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_sample(client, sample, top_k, top_n, mode, qa_model, rerank_model, sem):
    from rank_bm25 import BM25Okapi
    sessions = get_sessions(sample)
    if not sessions:
        return []
    texts = [s[1] for s in sessions]
    session_ids = [s[0] for s in sessions]
    session_dates = [s[2] for s in sessions]

    if mode != "bm25":
        session_vecs = embed_cohere(texts, "search_document")

    qas = sample.get("qa", [])
    results = []

    # Pre-embed all questions for cosine/hybrid modes
    if mode in ("cosine", "hybrid", "hybrid_rerank"):
        q_vecs = embed_cohere([qa["question"] for qa in qas], "search_query")

    bm25 = BM25Okapi([tokenize(t) for t in texts]) if mode != "cosine" else None

    tasks = []
    for qi, qa in enumerate(qas):
        question = qa["question"]
        if mode == "bm25":
            order = list(np.argsort(-bm25.get_scores(tokenize(question))))
            candidates = order[:top_k]
        else:
            sims = session_vecs @ q_vecs[qi]
            cosine_order = list(np.argsort(-sims))
            if mode == "cosine":
                candidates = cosine_order[:top_k]
            else:
                bm25_order = list(np.argsort(-bm25.get_scores(tokenize(question))))
                merged = rrf_merge(cosine_order, bm25_order, 60)
                candidates = merged[:top_n] if mode == "hybrid_rerank" else merged[:top_k]

        async def pack(qa=qa, candidates=candidates):
            if mode == "hybrid_rerank":
                rerank_tasks = [rerank_one(client, qa["question"], texts[int(i)], sem, rerank_model)
                                for i in candidates]
                scores = await asyncio.gather(*rerank_tasks)
                scored = sorted(zip(scores, candidates), key=lambda x: -x[0])
                final = [i for _, i in scored[:top_k]]
            else:
                final = candidates[:top_k]
            retrieved = [(session_ids[int(i)], texts[int(i)], session_dates[int(i)]) for i in final]
            rids = [session_ids[int(i)] for i in final]
            hyp = await answer_one(client, qa["question"], retrieved, sem, qa_model)
            return qa, rids, hyp

        tasks.append(pack())

    packed_results = await asyncio.gather(*tasks)
    for qa, rids, hyp in packed_results:
        # Category 5 QAs are adversarial — they have adversarial_answer instead of answer
        # and the model is expected to decline ("I don't know").
        results.append({
            "sample_id": sample["sample_id"],
            "question": qa["question"],
            "answer": qa.get("answer", ""),
            "adversarial_answer": qa.get("adversarial_answer", ""),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "hypothesis": hyp,
            "retrieved_session_ids": rids,
        })
    return results


async def run(data_file: Path, out_file: Path, top_k: int, top_n: int, mode: str,
              qa_model: str, rerank_model: str, provider: str, concurrency: int, limit: int,
              qa_limit_per_sample: int = 0):
    client = llm_client(provider)
    data = json.load(open(data_file))
    if limit: data = data[:limit]

    done = set()
    if out_file.exists():
        for line in out_file.open():
            if line.strip():
                row = json.loads(line)
                done.add((row["sample_id"], row["question"]))

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    print(f"LoCoMo: {len(data)} samples; cached={len(done)}; mode={mode} qa={qa_model}",
          file=sys.stderr)

    with out_file.open("a") as f:
        for i, sample in enumerate(data, 1):
            print(f"[{i}/{len(data)}] sample={sample['sample_id']} qa_count={len(sample.get('qa', []))}",
                  file=sys.stderr)
            # Filter QAs not yet done for this sample
            qas = sample.get("qa", [])
            if qa_limit_per_sample:
                qas = qas[:qa_limit_per_sample]
            qas_to_do = [qa for qa in qas if (sample["sample_id"], qa["question"]) not in done]
            if not qas_to_do:
                print(f"  all cached, skip", file=sys.stderr)
                continue
            filtered_sample = dict(sample)
            filtered_sample["qa"] = qas_to_do
            try:
                results = await process_sample(client, filtered_sample, top_k, top_n, mode,
                                                qa_model, rerank_model, sem)
            except Exception as ex:
                print(f"  ERR {sample['sample_id']}: {str(ex)[:200]}", file=sys.stderr)
                continue
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  sample {sample['sample_id']}: +{len(results)} QAs. elapsed={el:.0f}s "
                  f"eta={(len(data)-i)/max(i,1)*el:.0f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["cosine", "bm25", "hybrid", "hybrid_rerank"], default="cosine")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--provider", default="qwen", choices=list(PROVIDERS.keys()))
    p.add_argument("--qa-model", default=None)
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="Limit samples (default 0=all 10)")
    p.add_argument("--qa-limit-per-sample", type=int, default=0,
                   help="Limit QAs per sample (default 0=all; for smoke tests use 10-20)")
    args = p.parse_args()
    qa = args.qa_model or PROVIDERS[args.provider]["default_model"]
    rk = args.rerank_model or PROVIDERS[args.provider]["default_model"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.data, args.out, args.top_k, args.top_n, args.mode,
                    qa, rk, args.provider, args.concurrency, args.limit,
                    args.qa_limit_per_sample))


if __name__ == "__main__":
    main()
