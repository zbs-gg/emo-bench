"""Pulse on ES-MemEval (EvoEmo): per-seeker session retrieval.

Same modes as LME adapter: cosine / bm25 / hybrid / hybrid_rerank.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import embed_cohere, llm_client, kimi_chat, rrf_merge, tokenize, parse_last_int, PROVIDERS


QA_SYSTEM = ("You are an empathic long-term memory assistant. "
             "Answer using ONLY the provided past sessions. Do not invent facts. "
             "If the question asks about something that did NOT happen (abstention), "
             "state the information is not available. "
             "Be concise: 1-3 sentences unless synthesis requires more.")
RERANK_SYSTEM = "You rate how relevant a chat session is to a user question. End your output with just a number 0-10."
RERANK_TEMPLATE = ("Question: {q}\n\nSession:\n{s}\n\n"
                   "Rate relevance 0-10 (0=not relevant, 10=contains answer). "
                   "End with the number only.")


def concat_session(session, max_chars=8000):
    parts = [
        f"[date: {session.get('timestamp','?')}]",
        f"[topic: {session.get('topic','?')}, emotion: {session.get('emotion','?')}]",
        f"Summary: {session.get('summary','')}",
    ]
    for turn in session.get("dialogue", []):
        parts.append(f"[{turn.get('role','?')}] {turn.get('content','')}")
    return "\n".join(parts)[:max_chars]


async def rerank_one(client, question, text, sem, model):
    async with sem:
        txt = await kimi_chat(client, model, RERANK_SYSTEM,
                              RERANK_TEMPLATE.format(q=question, s=text[:3000]),
                              max_tokens=2000)
        return parse_last_int(txt)


async def answer_one(client, question, retrieved, sem, model):
    async with sem:
        history = "\n\n".join(
            f"=== Session {s.get('id','?')} ({s.get('timestamp','?')}) ===\n{concat_session(s, 5000)}"
            for s in retrieved)
        prompt = f"Past sessions (retrieved):\n\n{history}\n\nQuestion: {question}\n\nAnswer:"
        return await kimi_chat(client, model, QA_SYSTEM, prompt, max_tokens=3000)


async def run(data_file, out_file, top_k, top_n, mode, qa_model, rerank_model, provider, concurrency, limit):
    from rank_bm25 import BM25Okapi
    client = llm_client(provider)
    data = json.load(open(data_file))
    done = set()
    if out_file.exists():
        for l in out_file.open():
            if l.strip(): done.add(json.loads(l)["question_uid"])
    print(f"ES-MemEval: {len(data)} seekers; cached={len(done)}; mode={mode} qa={qa_model}", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    processed = 0

    with out_file.open("a") as f:
        for seeker in data:
            sid = seeker["id"]
            sessions = seeker["dialog_history"]
            texts = [concat_session(s) for s in sessions]

            if mode != "bm25":
                session_vecs = embed_cohere(texts, "search_document")

            qs = [(qg.get("id", ""), q) for qg in seeker.get("questions", [])
                  for q in qg.get("questions", [])]
            if not qs: continue

            # queries need embed
            if mode in ("cosine", "hybrid", "hybrid_rerank"):
                q_vecs = embed_cohere([q["question"] for _, q in qs], "search_query")

            bm25 = BM25Okapi([tokenize(t) for t in texts]) if mode != "cosine" else None

            tasks = []
            meta = []
            for idx, (qgid, q) in enumerate(qs):
                quid = f"{sid}::{qgid}::{q.get('idx','?')}::{hashlib.md5(q['question'].encode()).hexdigest()[:6]}"
                if quid in done: continue
                if limit and processed >= limit: break

                if mode == "bm25":
                    order = list(np.argsort(-bm25.get_scores(tokenize(q["question"]))))
                    candidates = order[:top_k]
                else:
                    sims = session_vecs @ q_vecs[idx]
                    cosine_order = list(np.argsort(-sims))
                    if mode == "cosine":
                        candidates = cosine_order[:top_k]
                    else:
                        bm25_order = list(np.argsort(-bm25.get_scores(tokenize(q["question"]))))
                        merged = rrf_merge(cosine_order, bm25_order, 60)
                        candidates = merged[:top_n] if mode == "hybrid_rerank" else merged[:top_k]

                async def pack(q=q, candidates=candidates, sessions=sessions, texts=texts,
                               quid=quid, qgid=qgid):
                    if mode == "hybrid_rerank":
                        rerank_tasks = [rerank_one(client, q["question"], texts[int(i)], sem, rerank_model) for i in candidates]
                        scores = await asyncio.gather(*rerank_tasks)
                        scored = sorted(zip(scores, candidates), key=lambda x: -x[0])
                        final = [i for _, i in scored[:top_k]]
                    else:
                        final = candidates[:top_k]
                    retrieved = [sessions[int(i)] for i in final]
                    rids = [sessions[int(i)].get("id", "?") for i in final]
                    hyp = await answer_one(client, q["question"], retrieved, sem, qa_model)
                    return quid, qgid, q, rids, hyp

                tasks.append(pack())
                meta.append(quid)
                processed += 1

            if not tasks: continue
            results = await asyncio.gather(*tasks)
            for (quid, qgid, q, rids, hyp) in results:
                f.write(json.dumps({
                    "question_uid": quid, "seeker_id": sid, "question_group_id": qgid,
                    "question_idx": q.get("idx"), "capability": q.get("capability"),
                    "question": q["question"], "answer": q.get("answer"),
                    "evidence": q.get("evidence", []),
                    "hypothesis": hyp,
                    "retrieved_session_ids": rids,
                }, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  seeker {sid}: +{len(tasks)} processed. total={processed} elapsed={el:.0f}s", file=sys.stderr)


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
