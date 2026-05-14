# Empathic Memory Benchmark v3 — Unified Recall@3 Leaderboard

All systems evaluated on identical n=35 corpus, single protocol:
R@3 = |retrieved_top_3 ∩ ideal_top_3| / |ideal_top_3|. Chain tests (10/35)
have no `ideal_top_3_event_ids` in the corpus (judge-evaluated, not deterministic),
so chain R@3 = 0 contributes to overall for every system — consistent across rows.

## Leaderboard (n=35, chain included as 0.000)

| System | Overall R@3 | core | stateful | multi_signal | chain | n |
|---|---|---|---|---|---|---|
| **Pulse v3** | **0.210** | 0.267 | 0.300 | 0.300 | 0.000 | 35 |
| cosine | 0.181 | 0.400 | 0.200 | 0.233 | 0.000 | 35 |
| Mem0 | 0.171 | 0.333 | 0.200 | 0.233 | 0.000 | 35 |
| LangMem | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 | 35 |
| LlamaIndex | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 | 35 |
| OpenAI Memory | 0.152 | 0.267 | 0.200 | 0.200 | 0.000 | 35 |
| hybrid | 0.152 | 0.400 | 0.133 | 0.200 | 0.000 | 35 |
| bm25 | 0.067 | 0.200 | 0.067 | 0.067 | 0.000 | 35 |
| Graphiti (Zep) | 0.048 | 0.200 | 0.033 | 0.033 | 0.000 | 35 |

## Delta vs Pulse v3

| System | Δ R@3 | Relative |
|---|---|---|
| cosine | +0.029 | +16% |
| Mem0 | +0.038 | +22% |
| LangMem | +0.048 | +29% |
| LlamaIndex | +0.048 | +29% |
| OpenAI Memory | +0.057 | +38% |
| hybrid | +0.057 | +38% |
| bm25 | +0.143 | +214% |
| Graphiti (Zep) | +0.162 | +340% |

## Method note

- All 9 systems evaluated on the same 35-test corpus `datasets/empathic-memory-corpus-v3.json`.
- `R@3 = |top_3 ∩ ideal_top_3| / |ideal_top_3|`.
- Chain tests (n=10) lack `ideal_top_3_event_ids` in the corpus — they are judge-evaluated on the chain axis (see Section 5 of paper). For this deterministic R@3 metric, chain tests contribute 0 to overall for all systems. This is *consistent across rows*: it is not a Pulse-favoring artifact, and Pulse v3 also scores 0 on chain R@3 in this protocol (its chain advantage is judge-rated, Table 2 of paper).
- Pulse v3 retrievals from bench-v3-20260429-2324.json (pre-existing 11-judge run, retrieval lists are deterministic regardless of judge pool).
- Mem0 retrievals from path_c_mem0_v3_retrievals.json (Path C run).
- Graphiti / LangMem / LlamaIndex / OpenAI Memory: fresh adapter runs 2026-05-11, OpenAI gpt-4o-mini + text-embedding-3-{small,large}.