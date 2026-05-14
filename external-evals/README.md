# Pulse External Evaluations

Vendor-independent evaluation pipeline for Pulse on third-party benchmarks. Complements the primary [empathic-memory bench v3](../README.md) with large-scale external validation.

---

## Benchmarks covered

| benchmark | source | size | Pulse score | link |
|---|---|---|---|---|
| **LongMemEval_S** | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval), ICLR 2025 | 500 Qs, 40 sessions | **68.89%** | results in private working repo |
| **ES-MemEval** | [slptongji/ESD-Memory-Eval](https://github.com/slptongji), Feb 2026 | 1427 Qs, 18 seekers | **1.519/2.0 ≈ 76%** (LLM-judge) | results in private working repo |
| **LoCoMo** | [snap-research/locomo](https://github.com/snap-research/locomo), ACL 2024 | 1986 Qs × 10 convs | **32.51% F1**, 62.78% adv refusal | run snapshot in private working repo |

**LongMemEval_S:** 5 capability categories (single-session-user/assistant/preference, multi-session, knowledge-update, temporal-reasoning). Pulse strongest on `single-session-assistant` (98%), weakest on `preference` (33%).

**ES-MemEval:** emotional-support conversation memory. Comparable to published gpt-4o+RAG baselines (~75%).

**LoCoMo:** very-long conversations (up to 35 sessions, ~300 turns, ~9K tokens avg). Pulse v2_pure cosine scores per category:
- Category 1 (multi-hop): 20.22%
- Category 2 (temporal): 20.56%
- Category 3 (open-domain): 9.36% (weakest)
- Category 4 (single-hop): 27.78%
- Category 5 (adversarial refusal): 62.78% (280/446 correctly refused)

---

## Cross-benchmark summary

| metric | bench v3 (8-judge) | LongMemEval_S | ES-MemEval | LoCoMo |
|---|---|---|---|---|
| size | 35 tests × 4 sys | 500 Qs | 1427 Qs | 1986 Qs |
| Pulse config | v3 (full) | v2_pure cosine | v2 | v2_pure cosine |
| score | 6.38/10 overall | 68.89% | 76% | 32.51% F1 |
| vs baseline | +32% (×38 on stateful) | +3pts vs BM25 | comparable to gpt-4o+RAG | first-run external number |

External benchmarks evaluate **v2_pure cosine** because they don't provide `user_state` API (no emotion_tags, biometric_snapshot, user_flag in the data). The v3 conditional boosts collapse to v2_pure when state signals are absent — so v3 == v2_pure on external data. The bench v3 SOTA claim specifically covers the axes external benchmarks don't touch (stateful, chain, multi-signal).

---

## Stack

- **Embeddings:** Cohere `embed-v4.0` (multilingual SOTA, session-level chunks)
- **QA / rerank / judge LLMs:** Kimi K2.6, Qwen3-Max, GLM-5, DeepSeek V3.2, OpenAI GPT-5.4 (see `scripts/common.py` for the 7-provider OpenAI-compat stack)
- **Lexical baseline:** BM25 (rank_bm25)
- **Fusion:** Reciprocal Rank Fusion (RRF, k=60)

## Pulse configurations evaluated (external)

- `pulse_v2_pure` — cosine + recency decay. Current published external results come from this config.
- `bm25` — keyword-only baseline
- `pulse_hybrid` — RRF(cosine, BM25)
- `pulse_hybrid_rerank` — RRF → top-20 → LLM rerank → top-K

## Scripts

| script | benchmark | usage |
|---|---|---|
| [`run_pulse_lme.py`](./scripts/run_pulse_lme.py) | LongMemEval_S | `python run_pulse_lme.py --data <data> --out <out> --mode cosine --provider kimi` |
| [`run_pulse_esm.py`](./scripts/run_pulse_esm.py) | ES-MemEval | `python run_pulse_esm.py --data <data> --out <out>` |
| [`run_pulse_locomo.py`](./scripts/run_pulse_locomo.py) | LoCoMo | `python run_pulse_locomo.py --data <data> --out <out> --mode cosine --provider qwen` |
| [`evaluate_locomo.py`](./scripts/evaluate_locomo.py) | LoCoMo scorer | per-category F1 + adversarial refusal accuracy |
| [`evaluate_lme_kimi.py`](./scripts/evaluate_lme_kimi.py) | LME scorer | answer-recall scoring via Kimi |
| [`compute_judge_agreement.py`](./scripts/compute_judge_agreement.py) | — | Krippendorff α on bench v3 judges |

Top-level Makefile targets:
```bash
make lme-s         # LongMemEval_S
make locomo        # LoCoMo
make judge-agreement  # Krippendorff α on 8-judge snapshot
```

---

## Reproducing published numbers

### LongMemEval_S (68.89%)

```bash
# One-time: clone LongMemEval dataset
git clone https://github.com/xiaowu0162/LongMemEval.git ~/dev/ai/longmemeval_data

# Run
LME_DATA=~/dev/ai/longmemeval_data/longmemeval_s.json make lme-s
```

### LoCoMo (32.51%)

```bash
# One-time: clone LoCoMo dataset
git clone https://github.com/snap-research/locomo.git ~/dev/ai/locomo-data

# Run
LOCOMO_DATA=~/dev/ai/locomo-data/locomo/data/locomo10.json make locomo
```

### Bench v3 (8-judge SOTA)

See parent [`../README.md`](../README.md).

---

## Mission

Best fast LLM-agnostic engine for emotional memory in AI companions. Open source. No compromises.

See `memory/project_pulse_mission_no_compromise.md` for the mission anchor.
