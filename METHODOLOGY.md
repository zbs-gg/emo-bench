# Methodology — Empathic Memory Bench v3

This document describes the design, scoring, and judge protocol for bench v3. Read this before interpreting or extending the benchmark.

---

## Corpus design

### Events (60 total)

The corpus is a realistic first-person memory stream in Russian, covering:

- Life events (marriage, motorcycle fall, ischemia diagnoses, book chapter completions)
- Relationship moments (repair with partner, grief anchors, children's illnesses)
- Work milestones (Pulse engine ships, benchmark wins, negative results)
- Body states (poor sleep mornings, HRV dips, post-workout recovery)
- Mental health markers (freeze, shame spikes, anchor moments of being seen)

30 events were carried over from the v2 corpus for backward comparability. 30 new events were added to cover axes the original corpus didn't exercise:
- **10 chain events** — explicit A → B → C causal sequences with `predecessor_ids` links
- **10 emotion-disambiguation events** — semantically similar pairs with different emotion profiles (e.g. "К. cried" once shame-driven, once relief-driven)
- **10 biometric-linked events** — events with explicit body anchors (HRV spike days, post-workout, sleep-deprived mornings)

Each event carries:
- `id`, `text` (plaintext), `when` (ISO date)
- `belief_class` — Pulse taxonomy (axiom / self_model / life_episode / social_event / fleeting)
- `emotion_tags` — Plutchik-10 vector (joy, sadness, anger, fear, trust, disgust, anticipation, surprise, shame, guilt), each 0–1
- `predecessor_ids` — list of event IDs this causally follows
- `user_flag` — boolean, `true` for structural-truth anchors that should surface on related queries

### Emotion tagging

Events were auto-tagged with Plutchik-10 vectors by `Qwen3-Max` using the prompt in [`emotion_classifier.py`](./external-evals/scripts/emotion_classifier.py). Cost: ≈$1 total. Top-10 anchor events (`user_flag=true`) were manually reviewed and corrected.

### Tests (35 total)

Tests come in 4 types, each exercising a different axis of empathic memory:

| type | count | what it measures |
|---|---|---|
| **core** | 5 | standard relevance/specificity/actionability (v2 axes); every other test type also retains a core-relevance component |
| **stateful** | 10 (5 pairs) | does top-3 change appropriately when user_state changes? |
| **chain** | 10 | does retrieved order match a causal ideal_chain? |
| **multi_signal** | 10 | biometric + query congruence integrated with textual query |

Each test specifies:
- `user_query` (natural language)
- `ideal_top_3_event_ids` (the right answer for core and stateful tests)
- `ideal_chain` (ordered list for chain tests)
- `user_state` (mood_vector / sleep_quality / hr_trend / stress_proxy / recent_life_events_7d / time_of_day) — for stateful and multi-signal tests
- `biometric_snapshot` — for multi-signal tests
- `what_it_tests` — human-readable axis/skill description
- `fail_modes` — what wrong top-3 looks like
- `pair_id` — for stateful pairs

---

## Axes and scoring

### Axis 1: Core (rel + spec + act)

Three classical judge-rated dimensions, each 0–10:

- **rel** — how relevant is the retrieved set to the user query?
- **spec** — how specific (vs generic) are the retrieved events?
- **act** — does the retrieved context enable a companion to act warmly and safely?

Judges see all systems' retrieved sets anonymised as `S1..S4` and rate each dimension 0–10 per system, with free-text reasoning.

### Axis 2: Stateful fit

Only applies to paired stateful tests. Each pair has two tests with the **same query** but **different user_state**, and different `ideal_top_3_event_ids`.

The judge sees both state variants side-by-side with each system's retrieved sets, then rates 0–10:

- **0** — retrieved sets identical across state variants (no state sensitivity)
- **5** — sets differ but in ways unrelated to the state shift
- **10** — sets differ in exactly the way an empathic companion would, per the paired ideal

Sanity check: any system with no state API (v2_pure cosine, bm25, hybrid) gets the same top-3 for both variants → stateful ≤ 3.

### Axis 3: Chain order (deterministic)

Applies only to chain tests. Computed mechanically, not judge-rated:

```
overlap  = [e for e in retrieved_order if e in ideal_chain]
tau      = 1 - (2 · inversions) / max(1, n·(n-1))      # normalized Kendall tau
recall   = len(overlap) / len(ideal_chain)
chain_score = 10 · tau · recall
```

A system that returns the chain in the right order with full recall scores 10. Randomly ordered recall scores ≈5. Missing events proportionally reduce recall.

### Axis 4: Multi-signal fit

Applies to multi-signal tests. The judge sees the user_query + biometric_snapshot + each system's retrieved events, then rates 0–10 on whether the retrieved set integrates the state signal appropriately.

- **10** — retrieved set matches the state-indicated cluster (e.g. HRV crash → body-cost events)
- **5** — semantically relevant but ignores biometric profile entirely
- **0** — contradicts biometric profile (surfaces victory cluster when body signals depletion)

Special test T22 (`multi_signal_neutral_semantic_fallback`) validates **conditional gating**: when the biometric state is neutral, a system should fall back to pure semantic retrieval. Over-interpreting neutral signal → penalty.

---

## Per-test score blend

```python
if test_type == "core":
    score = (rel + spec + act) / 3                             # 0–10

elif test_type == "stateful":
    score = 0.70 · (rel+spec+act)/3 + 0.30 · stateful_fit      # 0–10

elif test_type == "chain":
    score = 0.50 · chain_order + 0.50 · (rel+spec+act)/3       # 0–10

elif test_type == "multi_signal":
    score = 0.60 · multi_signal_fit + 0.40 · (rel+spec+act)/3  # 0–10
```

Every test retains a core-relevance component. This prevents gaming the new axes by retrieving random-but-state-aware events.

## Overall aggregation

```
overall_system = mean across (test × judge) of per_test_blended_score
```

With 35 tests × 8 judges = 280 samples per system → stable mean under judge noise.

The reported `core / stateful / chain / multi_signal` columns in the summary table are per-axis means over the relevant subset of tests.

---

## Judge protocol

### Blind system numbering

For each test, each judge sees four anonymised labels — `S1, S2, S3, S4` — mapped randomly per test so judges cannot learn "S1 is always pulse_v3". The mapping is recorded and reversed during aggregation.

### Prompt structure

Each judge sees:
1. The user_query
2. The user_state (if applicable to test type)
3. The biometric_snapshot (if multi-signal)
4. The `ideal_top_3_event_ids` (so judges know what "great" looks like — this is a calibration decision, not a cheat for systems)
5. Each system's retrieved set (plaintext events, shuffled order)
6. Request to score per axis + provide `winner` + free-text `note`

### Reasoning required

Every judge response must include a `note` field with 1-2 sentence reasoning, stored in the raw JSON for full transparency. Example:

> "cosine/pulse_v3 tied on hitting 'ВИДИТ' (5). All systems miss active blocker 20."

Raw reasoning can be read here: [`external-evals/results/bench-v3-20260424-1714.json`](./external-evals/results/bench-v3-20260424-1714.json).

### Judge pool (8)

| family | model | rationale |
|---|---|---|
| Moonshot | Kimi K2.6 | Russian-native frontier, strong empathic reasoning |
| Moonshot | Kimi K2-0711-preview | Earlier checkpoint — family-diversity check |
| Z.ai | GLM-5 | Chinese-family frontier, complementary inductive bias |
| Z.ai | GLM-5.1 | Later GLM checkpoint |
| Alibaba | Qwen3-Max | Dominant in EN+RU tasks |
| DeepSeek (via DashScope) | DeepSeek V3.2 | Independent frontier line |
| OpenAI | GPT-5.4 | Baseline western frontier |
| Anthropic | Claude Opus 4.7 | Strongest empathic-reasoning class, run in-chat via Claude Code |

### Inter-judge agreement (Krippendorff's α)

| axis | α | interpretation |
|---|---|---|
| rel | 0.78 | acceptable |
| spec | 0.57 | weak — interpret directionally |
| act | 0.72 | acceptable |
| **stateful** | **0.81** | **strong — load-bearing claim consensus** |
| multi_signal | 0.74 | acceptable |

Full report with axis-specific notes: `agreement.md` (in private working repo).

---

## What we do NOT measure

By design, this benchmark does not score:

- **Ingestion speed** — write throughput
- **Storage efficiency** — bytes per event, index size
- **API cost** — embedding / LLM spend per retrieval
- **Latency** — retrieval round-trip time
- **Scalability** — behaviour at 10⁶+ events

These are real properties that matter for production but they are orthogonal to **empathic fitness**. A system that retrieves perfectly but needs 30 seconds per query isn't production-ready; a system that retrieves in 50ms but surfaces the wrong event isn't companion-ready either. Different benchmarks should measure each.

---

## Reproducibility

Everything needed to reproduce the published numbers:

```bash
git clone https://github.com/zbs-gg/emo-bench.git
cd bench
cp .env.example .env                              # fill in API keys
make install                                       # create venv + deps
make bench-v3                                      # single-judge fast run
make bench-v3-8judge                               # full 8-judge pool
make judge-agreement                               # Krippendorff α
```

Each run writes a timestamped JSON in `external-evals/results/` and a snapshot folder in `external-evals/snapshots/`.

Canonical published numbers: commits `a1537ee` (7-judge) and `1ad9cc3` (8-judge SOTA, the headline).

---

## Extending the bench

### Add a new corpus
Drop a new JSON file in `datasets/` with the same schema as `empathic-memory-corpus-v3.json` (see the file for format). Pass `--corpus path/to/new.json` to the bench runner.

### Add a new system
Implement an adapter in `bench-empathic-memory-v3.py`:

```python
class MyAdapter:
    def __init__(self, corpus_path: str):
        self.events = load_events(corpus_path)
        # ingest / index

    def retrieve(self, query: str, top_k: int = 3,
                 user_state: dict | None = None) -> list[int]:
        return [...]  # top-k event IDs
```

Add it to the `--systems` flag. Submit a PR with your snapshot.

### Add a new judge
Add the provider config to `external-evals/scripts/common.py::PROVIDERS`:

```python
"my-judge": {"base_url": "...", "key_file": "my-key.txt", "default_model": "..."},
```

Pass `--judges ...,my-judge` to include it in the pool.

### Add a new axis
Extend the scoring rubric in `rubric-v3.md`, add new fields to test JSON schema, update `bench-empathic-memory-v3.py::call_judge` to ask for the new axis, update per-test blend in the aggregation step. Submit a PR with rationale.
