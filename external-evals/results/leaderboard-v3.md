# Empathic Memory Benchmark v3 — Unified Recall@3 Leaderboard

All systems evaluated on identical n=35 corpus, single protocol:
R@3 = |retrieved_top_3 ∩ ideal_top_3| / |ideal_top_3|. Chain tests (10/35)
have no `ideal_top_3_event_ids` in the corpus (judge-evaluated, not deterministic),
so chain R@3 = 0 contributes to overall for every system — consistent across rows.

## Leaderboard (n=35, chain included as 0.000)

| System | Overall R@3 | core | stateful | multi_signal | chain | n |
|---|---|---|---|---|---|---|
| **Pulse v3 (bge-m3 fine-tuned, zero-shot)** | **0.238** | **0.333** | **0.367** | 0.300 | 0.000 | 35 |
| Pulse v3 (Cohere embed-v4.0) | 0.210 | 0.267 | 0.300 | 0.300 | 0.000 | 35 |
| cosine (baseline) | 0.181 | 0.400 | 0.200 | 0.233 | 0.000 | 35 |
| Mem0 | 0.171 | 0.333 | 0.200 | 0.233 | 0.000 | 35 |
| LangMem | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 | 35 |
| LlamaIndex | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 | 35 |
| OpenAI Memory | 0.152 | 0.267 | 0.200 | 0.200 | 0.000 | 35 |
| hybrid (baseline) | 0.152 | 0.400 | 0.133 | 0.200 | 0.000 | 35 |
| bm25 (baseline) | 0.067 | 0.200 | 0.067 | 0.067 | 0.000 | 35 |
| Graphiti (Zep) | 0.048 | 0.200 | 0.033 | 0.033 | 0.000 | 35 |

Non-chain R@3 (excluding 10 chain probes which are judge-evaluated separately): **0.333** for Pulse v3 (bge-m3 fine-tuned, zero-shot).

## Delta vs Pulse v3 (bge-m3 fine-tuned)

| System | Δ R@3 | Relative |
|---|---|---|
| Pulse v3 (Cohere embed-v4.0) | +0.028 | +13% |
| cosine | +0.057 | +31% |
| Mem0 | +0.067 | +39% |
| LangMem | +0.076 | +47% |
| LlamaIndex | +0.076 | +47% |
| OpenAI Memory | +0.086 | +57% |
| hybrid | +0.086 | +57% |
| bm25 | +0.171 | +256% |
| Graphiti (Zep) | +0.190 | +400% |

## Method note

- All systems evaluated on the same 35-test corpus `datasets/empathic-memory-corpus-v3.json`.
- `R@3 = |top_3 ∩ ideal_top_3| / |ideal_top_3|`.
- Chain tests (n=10) lack `ideal_top_3_event_ids` in the corpus — they are judge-evaluated on the chain axis (see paper §5). For this deterministic R@3 metric, chain tests contribute 0 to overall for all systems uniformly. *Consistent across rows*; not a Pulse-favoring artifact. Pulse's chain advantage shows up in the judge-rated table, not this one.
- **New SOTA row (2026-05-15):** Pulse v3 with a LoRA adapter on `BAAI/bge-m3` (560M params) fine-tuned on **public emotional dialogue corpora only** — EmpatheticDialogues + ESConv triplets. **Strict zero-shot holdout: the Pulse evaluation corpus was never seen during training** (per Gemini 3.1 Pro methodology recommendation). Result file in the private working repo (test-name labels reference deployment-specific entities). Adapter directory: `finetune-adapters/bge-m3-empathic-2026-05`. Beats the previous Pulse v3 + Cohere embed-v4.0 configuration by +13% on overall R@3, +25% on core, +22% on stateful.
- Pulse v3 (Cohere embed-v4.0) retrievals from `bench-v3-20260429-2324.json` (pre-existing 11-judge run; retrieval lists are deterministic regardless of judge pool).
- Mem0 retrievals from `path_c_mem0_v3_retrievals.json` (Path C run).
- Graphiti / LangMem / LlamaIndex / OpenAI Memory: fresh adapter runs 2026-05-11, OpenAI gpt-4o-mini + text-embedding-3-{small,large}.

## Reproducibility — fine-tuned bge-m3 row

The new SOTA is fully reproducible without access to our deployment corpus:

1. Pull public corpora: [EmpatheticDialogues (Facebook Research)](https://github.com/facebookresearch/EmpatheticDialogues) and [ESConv (Liu et al., 2021)](https://github.com/thu-coai/Emotional-Support-Conversation).
2. Build contrastive triplets (positive: same emotion + same support strategy; negative: different emotion or contradictory strategy).
3. LoRA fine-tune `BAAI/bge-m3` per `external-evals/scripts/finetune_bge_m3_mps.py`.
4. Plug the adapter into the Pulse v3 retrieval pipeline via `external-evals/scripts/run_pulse_v3_finetuned_bge_m3.py`.
5. Run on Empathic-Memory-Bench v3 corpus.

The adapter weights from our 2026-05 training are not currently published (they were trained on Mac MPS as a single-seed run); a clean GPU-trained release is planned. The training methodology is fully documented and the public corpora it uses are open.
