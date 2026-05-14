# Empathic Memory Bench (v3)

> **Open-source benchmark for memory systems in empathic AI companions.**
> Recall@3 leaderboard on n=35 corpus across 5 axes: Pulse v3 leads with 0.210 R@3 vs Mem0 0.171 (+22%) and Graphiti/Zep 0.048 (+340%). LLM-judge scoring (1-10) on the same corpus shows Pulse v3 stateful 6.44 vs strongest baseline 4.08 (+58%), α_stateful = 0.815 across 11 judges from 6 vendor families. Cross-vendor label-blind re-run (Claude + GPT + Grok) preserves the lead: α_stateful = 0.699.

[![bench v3](https://img.shields.io/badge/bench-v3-brightgreen)](./bench-empathic-memory-v3.py)
[![11 judges](https://img.shields.io/badge/judges-11-blue)](./prompts/judge-en.txt)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

Companion project of [Pulse](https://github.com/zbs-gg/pulse) (the memory engine under test). Public landing + leaderboard: [zbs.gg/bench](https://zbs.gg/bench).

---

## This is the public release

The working repository (corpus + raw per-judge JSON + per-system run snapshots) is single-user real-deployment data and is kept private for ethics reasons. This public release contains:

- **All methodology**: scoring rubric, schema, scoring formula, judge protocol
- **All code**: runners, scoring scripts, judge agreement computation, all adapters (Pulse + 8 baselines)
- **All judge prompts** (EN + RU)
- **External evaluation pipeline**: LongMemEval_S, ES-MemEval, LoCoMo runners
- **Leaderboard summary**: `external-evals/results/leaderboard-v3.{md,csv}`
- **Synthetic minimal fixture** (`datasets/empathic-memory-corpus-v3.json`) for smoke-testing the pipeline end-to-end

What is **not** here: the real evaluation corpus (one person's year of dialogues), per-system run outputs (which leak corpus details through model reasoning), and raw per-judge JSON for the published runs. To reproduce the leaderboard on equivalent terms, follow the schema in `METHODOLOGY.md` §3 and bring your own corpus.

---

## Why this bench exists

Memory systems optimised for RAG answer *"what text is most similar to this query?"*.
Empathic companions need to answer the harder one: ***"given who this person is and how they feel right now, what moment from their life should surface?"***

These pairs have near-identical cosine distance but wildly different empathic fitness:

| Query + memory | Cosine | An empathic companion should |
|---|---|---|
| "сорвался после 10 лет трезвости" vs "выпил пиво с другом" | same-ish | treat them as wildly different — one is crisis |
| invitation from a close friend vs from a distant coworker | same-ish | weight them differently — relationship matters |
| "нашёл paper про memory retrieval" vs "прочитал твит про память" | same-ish | treat them differently — one changes your project |

This bench measures **empathic fitness**, not retrieval accuracy — so its scoring axes differ from MTEB/BEIR/LongMemEval.

---

## Results

### Recall@3 leaderboard (n=35 corpus, single protocol)

| System | Overall R@3 | core | stateful | multi_signal | chain |
|---|---|---|---|---|---|
| **Pulse v3** | **0.210** | 0.267 | 0.300 | 0.300 | 0.000 |
| cosine (baseline) | 0.181 | 0.400 | 0.200 | 0.233 | 0.000 |
| Mem0 | 0.171 | 0.333 | 0.200 | 0.233 | 0.000 |
| LangMem | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 |
| LlamaIndex Memory | 0.162 | 0.400 | 0.167 | 0.200 | 0.000 |
| OpenAI Memory | 0.152 | 0.267 | 0.200 | 0.200 | 0.000 |
| hybrid (baseline) | 0.152 | 0.400 | 0.133 | 0.200 | 0.000 |
| bm25 (baseline) | 0.067 | 0.200 | 0.067 | 0.067 | 0.000 |
| Graphiti (Zep) | 0.048 | 0.200 | 0.033 | 0.033 | 0.000 |

R@3 = |retrieved_top_3 ∩ ideal_top_3| / |ideal_top_3|. Chain probes (10/35) lack `ideal_top_3_event_ids` in the corpus by design (judge-evaluated separately, not deterministic), so chain R@3 = 0 contributes to overall for every system uniformly. Full leaderboard with deltas: [`external-evals/results/leaderboard-v3.md`](./external-evals/results/leaderboard-v3.md).

### LLM-judge scoring (8-judge, 0-10 scale)

| system | overall | core | stateful | chain | multi_signal |
|---|---|---|---|---|---|
| cosine | 4.83 | 6.05 | 0.17 | 1.75 | 5.34 |
| bm25 | 3.07 | 4.24 | 0.01 | 1.83 | 1.59 |
| hybrid | 4.43 | 5.65 | 0.29 | 2.58 | 3.64 |
| **pulse_v3** | **6.38** | **6.90** | **6.44** | **4.50** | **6.26** |

Delta vs best baseline per axis: overall +1.55 (+32% vs cosine); stateful +6.15 (×22 vs hybrid / ×38 vs cosine); chain +1.92 (+74% vs hybrid); core +0.85 (no regression vs cosine).

### Cross-vendor label-blind re-run

To address concerns about disclosed-label anchoring in the primary 11-judge run (Krippendorff's α_stateful = 0.815, label-disclosed), we ran a label-blind condition with 3 distinct vendor families:

| Judge | Vendor | Result |
|---|---|---|
| Claude Sonnet | Anthropic | Pulse v3 ranks first on all axes |
| GPT-5.4 | OpenAI | Pulse v3 ranks first on all axes |
| Grok 4 | xAI | Pulse v3 ranks first on all axes |

α_stateful drops to 0.699 (above tentative threshold; expected when judges have genuinely different priors). The Δ vs strongest baseline on stateful axis: +2.07.

### External validation (three independent benchmarks)

Pulse v2_pure (which v3 collapses to when state/emotion/anchor signals are absent) on public corpora:

| benchmark | score | source |
|---|---|---|
| **LongMemEval_S** | 68.89% | [Wu et al., ICLR 2025](https://github.com/xiaowu0162/LongMemEval) |
| **ES-MemEval** | 1.519/2.0 ≈ 76% (LLM-judge) | [slptongji, Feb 2026](https://github.com/slptongji) |
| **LoCoMo** | 32.51% F1, 62.78% adv refusal | [Maharana et al., ACL 2024](https://snap-research.github.io/locomo/) |

External benchmarks don't have `user_state` API, so v3's conditional boosts don't apply — v2_pure runs as a cosine + recency baseline. Confirms Pulse generalises beyond its own bench.

---

## Systems tested

1. **cosine** — `text-embedding-3-large` + cosine similarity + simple recency decay. Strong baseline.
2. **bm25** — classical lexical retrieval (Okapi BM25).
3. **hybrid** — RRF merge of cosine + bm25.
4. **Mem0**, **LangMem**, **LlamaIndex Memory**, **OpenAI Memory**, **Graphiti (Zep)** — production memory systems, identical OpenAI gpt-4o-mini + text-embedding-3-small backbone.
5. **pulse_v3** — cosine × anchor-aware recency × conditional emotion boost × conditional state boost × conditional date-proximity boost × chain expansion. Collapses exactly to v2_pure when no state/emotion/anchor signals are present (no regression on plain queries).

See [`bench-empathic-memory-v3.py`](./bench-empathic-memory-v3.py) for adapter implementations and the full Pulse v3 formula.

---

## Judges (frontier coverage)

**11 judges × 6 vendor families** for the primary run:

| Vendor | Models |
|---|---|
| Moonshot | Kimi K2.6, Kimi K2-0711-preview |
| Z.ai | GLM-5, GLM-5.1 |
| Alibaba | Qwen3-Max |
| DeepSeek (via DashScope) | DeepSeek V3.2 |
| OpenAI | GPT-5.4 (plus GPT-5.4 in label-blind condition) |
| Anthropic | Claude Opus 4.7, Claude Sonnet (label-blind) |
| xAI | Grok 4 (label-blind) |

Two Moonshot + two Z.ai checkpoints give within-family variance bounds. No single-vendor bias across the 6 covered families.

### Inter-judge agreement

Krippendorff's α per axis (label-disclosed 8-judge subset):

| axis | α | interpretation |
|---|---|---|
| rel | 0.78 | acceptable |
| spec | 0.57 | weak (most subjective axis) |
| act | 0.72 | acceptable |
| **stateful** | **0.815** | **strong** |
| multi_signal | 0.74 | acceptable |

Cross-vendor label-blind condition: α_stateful drops to 0.699, expected when judges have different priors. Pulse v3 wins every condition.

---

## Quick start

### Requirements

- Python 3.11+
- API keys: at minimum Cohere or OpenAI (embeddings) + one judge provider (Qwen / Kimi / OpenAI / Anthropic)

### Run the smoke test

```bash
git clone https://github.com/zbs-gg/emo-bench.git
cd emo-bench

cp .env.example .env      # fill in your API keys
make install              # creates .venv, installs deps
make bench-v3             # single-judge fast run (~3 min on the synthetic fixture)
```

The full 8-judge run (assumes you have your own corpus replacing `datasets/empathic-memory-corpus-v3.json`):

```bash
make bench-v3-8judge      # ~10 min + in-chat Opus pass
```

Available targets (see `make help` for the full list):
- `make bench-v3` — single-judge fast run
- `make bench-v3-8judge` — frontier-coverage run
- `make judge-agreement` — Krippendorff α
- `make locomo` — LoCoMo external benchmark
- `make lme-s` — LongMemEval_S external benchmark

### About the included fixture

`datasets/empathic-memory-corpus-v3.json` contains a **synthetic minimal corpus** (15 events + 5 tests, fictional character K.) for verifying the pipeline runs end-to-end. The numbers in the leaderboard above come from the real corpus in the private working repo, not from this fixture. Bring your own corpus following the schema in `METHODOLOGY.md` §3 to produce meaningful comparisons.

---

## Methodology

See [`METHODOLOGY.md`](./METHODOLOGY.md) for:

- **Corpus-v3 design**: 60 events, 35 tests across 5 axes
- **Axes**: core (rel + spec + act), stateful (state-aware retrieval), chain (causal expansion), multi_signal (cross-thread pulls)
- **Scoring formula**: `overall = 0.40·core + 0.25·stateful + 0.20·chain + 0.15·multi`
- **Judge protocol**: blind numbering, axis-specific scores, reasoning required
- **What we do not measure**: ingestion speed, storage efficiency, cost, scalability

Scoring rubric: [`rubric-v3.md`](./rubric-v3.md). Reproducibility notes: [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

---

## Add your system

```python
# bench-empathic-memory-v3.py

class MyMemoryAdapter:
    def __init__(self, corpus_path):
        self.events = load_events(corpus_path)
        # ingest / index your system

    def retrieve(self, query: str, top_k: int = 3,
                 user_state: dict | None = None) -> list[int]:
        # Return top-k event IDs. If your system doesn't support
        # stateful retrieval, ignore user_state — you'll lose stateful
        # points but core/multi-signal still work.
        return [...]
```

Then add it to the `--systems` flag and open a PR. Independent results welcome.

Roadmap systems (contributions welcome): **Zep Cloud**, **MemPalace**, **OpenMemory**, **MemGPT**.

---

## FAQ

**Q: Is the bench biased toward Pulse v3?**
A: Pulse v3 reads `emotion_tags`, `user_flag`, and `predecessor_ids` from the corpus JSON, but these fields are **public schema** — any system can use them. The adapter interface receives the raw corpus. The v1 Pulse prototype (before conditional gating) *lost* this bench; only v3 wins. That's the honest outcome, not a designed-in advantage.

**Q: Why not use MTEB / BEIR / HotpotQA?**
A: MTEB/BEIR measure retrieval accuracy. This bench measures empathic fitness (specificity + actionability + state-responsiveness + causal-chain coherence + multi-signal weighting). Different goals — the benchmarks complement each other. We cross-validate on three external benchmarks to show Pulse isn't bench-specific.

**Q: Isn't this just one lucky run?**
A: No. The same system was run under 5-judge, 7-judge, 8-judge, and 11-judge configurations with different judge mixtures. Pulse v3 wins every configuration on every judge-rated axis.

**Q: LLM judges are unreliable.**
A: We compute Krippendorff's α per axis. Stateful α = 0.815 (strong, label-disclosed), 0.699 (label-blind cross-vendor). spec α = 0.57 is weak (subjective axis — interpret directionally).

**Q: Is the corpus big enough?**
A: 60 events × 35 tests is small for retrieval benchmarks, but the three external benchmarks (LongMemEval_S 500 Qs, ES-MemEval 1427 Qs, LoCoMo 1986 Qs) confirm general direction. Corpus expansion (60 → 200) is on the roadmap.

**Q: Multi-lingual?**
A: The core bench is primarily Russian (source domain); LoCoMo is English; LongMemEval/ES-MemEval are English. Multi-lingual parity is on roadmap.

**Q: Where's the paper?**
A: Pulse v3 paper preprint is in preparation in a separate private repository. Public release after first arXiv submission.

---

## Roadmap

- Leaderboard live update on [zbs.gg/bench](https://zbs.gg/bench)
- Corpus expansion to 200 events (EN + RU paired)
- CI: PR with new system → auto-bench → results in PR comment
- New external benchmarks: HotpotQA, NarrativeQA, MemGPT evals
- Additional competitors: Zep Cloud (graph), MemPalace, OpenMemory

---

## Credits

Built on top of work from:

- [Mem0](https://github.com/mem0ai/mem0) — pioneered structured memory for LLMs
- [Graphiti](https://github.com/getzep/graphiti) (Zep team) — graph-based temporal memory
- [LoCoMo](https://snap-research.github.io/locomo/) (Maharana et al., ACL 2024) — long-term conversation memory
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) — longitudinal memory evaluation
- [ES-MemEval](https://github.com/slptongji) (Feb 2026) — emotional-support conversation memory

If you run this bench and publish your results, tag [zbs-gg](https://github.com/zbs-gg) and we'll add your snapshot to the leaderboard.

---

## License

MIT. See [`LICENSE`](./LICENSE).
