"""Pulse on LongMemEval: Cohere v4 embed + Kimi K2.6 QA/rerank.

Modes:
  cosine          — pure cosine retrieval (baseline Pulse v2)
  bm25            — BM25 only (lexical baseline)
  hybrid          — RRF(cosine, BM25), top-K directly
  hybrid_rerank   — RRF → top-N → LLM rerank → top-K

Produces jsonl hyps compatible with LongMemEval evaluator.
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import embed_cohere, llm_client, kimi_chat, rrf_merge, tokenize, parse_last_int, PROVIDERS


QA_SYSTEM = ("You answer using only provided chat history sessions. "
             "Be concise (1-2 sentences). If info missing, say so. "
             "For temporal questions, use timestamps. Do not hallucinate.")
RERANK_SYSTEM = "You rate how relevant a chat session is to a user question. End your output with just a number 0-10."
RERANK_TEMPLATE = ("Question: {q}\n\nSession:\n{s}\n\n"
                   "Rate relevance 0-10 (0=not relevant, 10=contains answer). "
                   "End with the number only.")


def concat_session(session, max_chars=8000):
    return "\n".join(f"[{t.get('role','?')}] {t.get('content','')}" for t in session)[:max_chars]


async def rerank_one(client, question, session_text, sem, model):
    async with sem:
        txt = await kimi_chat(client, model, RERANK_SYSTEM,
                              RERANK_TEMPLATE.format(q=question, s=session_text[:3000]),
                              max_tokens=2000)
        return parse_last_int(txt)


async def answer_one(client, question, qdate, retrieved, dates, sem, model):
    async with sem:
        history = "\n\n".join(f"=== Session at {d} ===\n{concat_session(s, 6000)}"
                              for s, d in zip(retrieved, dates))
        prompt = (f"[Question asked on {qdate}]\n\nHistory:\n\n{history}\n\n"
                  f"Question: {question}\n\nAnswer:")
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def process_one(client, entry, top_k, top_n, mode, qa_model, rerank_model, sem):
    from rank_bm25 import BM25Okapi
    sessions = entry["haystack_sessions"]
    dates = entry.get("haystack_dates", [""] * len(sessions))
    texts = [concat_session(s) for s in sessions]

    if mode == "bm25":
        bm25 = BM25Okapi([tokenize(t) for t in texts])
        order = list(np.argsort(-bm25.get_scores(tokenize(entry["question"]))))
        candidates = order[:top_k]
    else:
        vecs = embed_cohere(texts, "search_document")
        q_vec = embed_cohere([entry["question"]], "search_query")[0]
        cosine_order = list(np.argsort(-(vecs @ q_vec)))
        if mode == "cosine":
            candidates = cosine_order[:top_k]
        else:
            bm25 = BM25Okapi([tokenize(t) for t in texts])
            bm25_order = list(np.argsort(-bm25.get_scores(tokenize(entry["question"]))))
            merged = rrf_merge(cosine_order, bm25_order, 60)
            candidates = merged[:top_n] if mode == "hybrid_rerank" else merged[:top_k]

    if mode == "hybrid_rerank":
        tasks = [rerank_one(client, entry["question"], texts[int(i)], sem, rerank_model) for i in candidates]
        scores = await asyncio.gather(*tasks)
        scored = sorted(zip(scores, candidates), key=lambda x: -x[0])
        final_idx = [i for _, i in scored[:top_k]]
    else:
        final_idx = candidates[:top_k]

    retrieved = [sessions[int(i)] for i in final_idx]
    retrieved_dates = [dates[int(i)] for i in final_idx]
    hyp = await answer_one(client, entry["question"], entry.get("question_date", ""),
                           retrieved, retrieved_dates, sem, qa_model)
    return {
        "question_id": entry["question_id"],
        "hypothesis": hyp,
        "retrieved_session_ids": [entry["haystack_session_ids"][int(i)] for i in final_idx],
        "evidence_session_ids": entry.get("answer_session_ids", []),
    }


async def run(data_file, out_file, top_k, top_n, mode, qa_model, rerank_model, provider, concurrency, limit):
    client = llm_client(provider)
    data = json.load(open(data_file))
    if limit: data = data[:limit]
    done = set()
    if out_file.exists():
        for l in out_file.open():
            if l.strip(): done.add(json.loads(l)["question_id"])
    print(f"LongMemEval: {len(data)} Qs; cached={len(done)}; mode={mode} qa={qa_model}", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    with out_file.open("a") as f:
        for i, entry in enumerate(data, 1):
            if entry["question_id"] in done: continue
            try:
                r = await process_one(client, entry, top_k, top_n, mode, qa_model, rerank_model, sem)
                f.write(json.dumps(r) + "\n")
                f.flush()
            except Exception as ex:
                print(f"  ERR {entry['question_id']}: {str(ex)[:200]}", file=sys.stderr)
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(data)} elapsed={el:.0f}s eta={(len(data)-i)/max(i,1)*el:.0f}s", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["cosine", "bm25", "hybrid", "hybrid_rerank"], default="cosine")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--provider", default="glm", choices=list(PROVIDERS.keys()))
    p.add_argument("--qa-model", default=None)
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    qa = args.qa_model or PROVIDERS[args.provider]["default_model"]
    rk = args.rerank_model or PROVIDERS[args.provider]["default_model"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.data, args.out, args.top_k, args.top_n, args.mode,
                    qa, rk, args.provider, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
