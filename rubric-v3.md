# Empathic Memory Bench v3 — Scoring Rubric

Extends `rubric.md` (v2 core axes) with three new axes for product-vision alignment: stateful retrieval, temporal chain, and multi-signal biometric context.

## Axes overview

| Axis | Test types that measure it | Weight | Range |
|---|---|---|---|
| **Core** (rel + spec + act) | core, stateful, multi_signal | 0.40 | 0–30 (summed) |
| **Stateful fit** | stateful pairs | 0.25 | 0–10 |
| **Chain order** | chain | 0.20 | 0–10 |
| **Multi-signal fit** | multi_signal | 0.15 | 0–10 |

**Weighted overall** = 0.40·(core/30·10) + 0.25·stateful + 0.20·chain + 0.15·multi_signal  →  0–10 scale.

Tests missing an axis (e.g. `core` tests have no state/chain/bio) → weights redistribute proportionally across axes the test *does* measure. This keeps scores comparable across test types.

---

## Axis 1: Core (rel + spec + act)

**Unchanged from v2.** See `rubric.md`.

- `rel`: How relevant is the retrieved set to the user query? (0–10)
- `spec`: How specific vs generic are the retrieved events? (0–10)
- `act`: Does the retrieved context enable a companion to act warmly and safely? (0–10)

Each system gets `total = rel + spec + act` per test, averaged across judges.

---

## Axis 2: Stateful fit

**Only applies to paired stateful tests** (test_type=`stateful`, sharing a `pair_id`).

Each stateful pair has two tests with **identical query** but **different user_state**. The ideal_top_3 is different for each state variant — by design.

### What the judge scores

The judge sees BOTH state variants side-by-side with both retrieved sets and scores on a single 0–10 scale:

> How well does the system's retrieved set CHANGE appropriately between the two state variants?
> - 0 = retrieved sets identical across state variants (no state sensitivity)
> - 5 = sets differ but the differences don't reflect the state shift in a meaningful way
> - 10 = sets differ in exactly the way an empathic companion would approve given the state difference

### Concrete scoring rubric

Anchors for the judge:

- **0/10** — retrieved set is identical for state_A and state_B
- **2/10** — 1 event differs but in a way unrelated to the state shift (noise)
- **5/10** — 2 events differ with partial state relevance (shift is present but dull)
- **7/10** — retrieved sets differ and the differences ARE state-reflective but miss one critical ideal event per variant
- **9/10** — retrieved sets match ideal_top_3 for each state variant with 1 substitution that stays in-spirit
- **10/10** — retrieved sets exactly match ideal_top_3_event_ids for each state

### Failure modes to penalize

- Same top-3 for both variants → 0
- Different top-3 but both variants surface WRONG cluster (e.g. both pull origin-wound when one should be repair) → max 3
- State shift detected but in opposite direction (e.g. shame-state pulls victory events, joy-state pulls wound events) → 1

### Sanity check

A v2_pure system (no state API) will always score ≤ 3 on stateful fit: its retrieved top-3 is deterministic given the query, so the delta between pair variants is zero noise modulo floating-point randomness.

---

## Axis 3: Chain order

**Only applies to chain tests** (test_type=`chain`). Each chain test has an `ideal_chain`: an ordered list of event IDs representing a causal/temporal sequence.

### How the score is computed

1. Extract the intersection between retrieved events and `ideal_chain`:
   - `overlap = [e for e in retrieved_order if e in ideal_chain]`
2. Compute **normalized Kendall tau distance** between `overlap` (in retrieved order) and its correct order in `ideal_chain`:
   - `tau = 1 - (2 · inversions) / max(1, n·(n-1))` where `n = len(overlap)`, `inversions` = number of swaps to sort overlap into ideal_chain order
3. Apply recall penalty for missing chain elements:
   - `recall = len(overlap) / len(ideal_chain)`
   - `chain_score = 10 · tau · recall`

### Anchors

- **10/10** — retrieved contains all ideal_chain events in correct order
- **7/10** — retrieved contains all events but 1 pair swapped
- **5/10** — retrieved contains 2/3 of chain in correct order
- **3/10** — retrieved contains correct events but reversed order
- **1/10** — retrieved contains only 1 event from chain
- **0/10** — no overlap with ideal_chain

### What the judge does NOT score

Chain order is **computed mechanically** from `ideal_chain` and `retrieved_order`. Judges do NOT see chain tests for this axis — the tau computation is deterministic. Judges may still see chain tests under `core` axis (rel/spec/act) if the test includes those fields.

### Failure modes

- System returns `set` not `list` (no order): score per random permutation → expected tau ≈ 0.5
- System returns only one event from chain: tau undefined (n=1), score = recall · 10 / len(ideal_chain)

---

## Axis 4: Multi-signal fit

**Only applies to multi-signal tests** (test_type=`multi_signal`). Each test provides a `biometric_snapshot` + `user_query`.

### What the judge scores

The judge sees:
- The user_query
- The biometric_snapshot (HRV, sleep_quality, stress_proxy, recent_life_events_7d, etc.)
- The system's retrieved events

and scores 0–10:

> Does the retrieved set integrate the provided biometric / life-state snapshot?
> A fit answer surfaces body-linked or state-relevant events when state signal is strong;
> falls back to semantic relevance when state is neutral.

### Anchors

- **10/10** — retrieved set exactly matches the cluster indicated by biometric profile (e.g. HRV crash → body-cost events)
- **8/10** — retrieved set surfaces 2/3 of ideal_top_3 with one semantic-only pull
- **5/10** — retrieved set is semantically relevant but ignores biometric profile entirely
- **3/10** — retrieved set is semantically noise AND ignores biometric profile
- **0/10** — retrieved set contradicts biometric profile (e.g. surfaces victory cluster when body signals depletion)

### Special case: neutral biometric state

Test T22 (`multi_signal_neutral_semantic_fallback`) has a neutral biometric state (HRV 70, good sleep, low stress, no dominant emotion). For this test:

- **10/10** — system falls back to pure semantic/salience retrieval (as if no biometric was provided)
- **5/10** — system over-interprets neutral state as signal and surfaces body-linked events
- **0/10** — system ignores query entirely and returns biometric-dominated events

This test validates the **conditional gating** — the bio boost should be OFF when state is neutral.

### Failure modes

- Surfaces body events on a mood/emotional query with neutral biometrics → wrong register
- Surfaces mood events on a body query with strong biometric signal → misses the signal
- Returns biometric-cluster events but wrong biometric profile (e.g. motorcycle-activation cluster instead of depletion cluster)

---

## Weighted aggregation

Per test, per system, per judge:

```
if test_type == "core":
    score = core_avg / 3.0  # rel+spec+act → 0–10
elif test_type == "stateful":
    score = 0.70·(core_avg/3.0) + 0.30·stateful_fit
elif test_type == "chain":
    score = 0.50·chain_order + 0.50·(core_avg/3.0 if core_available else 0)
elif test_type == "multi_signal":
    score = 0.60·multi_signal_fit + 0.40·(core_avg/3.0)
```

Overall system score = mean across (test × judge) pairs.

### Why the per-test blends

Every test still has a "core relevance" dimension — a stateful or multi-signal test with zero semantic relevance is not actually empathically useful. Blending core into each test type prevents gaming the new axes by retrieving random-but-state-aware events.

Chain tests can have core=0 if the chain is about past events the system retrieved in correct order without the judge seeing the chain question semantically (e.g. a "what led to X" query where the chain is the answer but each individual event doesn't score high on semantic rel).

---

## Judge prompt — per test type

### core (unchanged from v2)

See `bench-empathic-memory.py` → `prompts/judge-en.txt`.

### stateful

> You are evaluating PAIRED tests. You will see TWO state variants of the SAME query,
> each with its own ideal_top_3_event_ids and its own retrieved set per system.
>
> For each system, score:
> - `{sid}_rel`, `{sid}_spec`, `{sid}_act` — for the UNION of both variants' retrieved sets (0–10 each)
> - `{sid}_stateful` — how appropriately the retrieved set CHANGED between variants (0–10)
>
> A system that returns identical top-3 for both variants scores {sid}_stateful = 0.
> A system that returns ideal_top_3 for each variant scores {sid}_stateful = 10.

### chain

> You are evaluating a CHAIN test. The user_query asks for a causal/temporal sequence,
> and the ideal answer is the ordered list `ideal_chain`.
>
> For each system, score:
> - `{sid}_rel`, `{sid}_spec`, `{sid}_act` — for the retrieved set (0–10 each)
> - (chain_order is computed mechanically — do not score it)

### multi_signal

> You are evaluating a MULTI-SIGNAL test. The user_query is paired with a biometric_snapshot
> that describes the user's current physical/life state.
>
> For each system, score:
> - `{sid}_rel`, `{sid}_spec`, `{sid}_act` — for the retrieved set (0–10 each)
> - `{sid}_multi_signal` — does the retrieved set integrate the biometric profile
>   appropriately? (0–10)
>
> A neutral biometric snapshot should produce a retrieved set that matches pure semantic/salience
> retrieval. Non-neutral snapshots should surface body-linked or state-relevant events where appropriate.

---

## Open questions (iterate in Phase 2.3)

1. **Inter-judge agreement** on stateful axis: target IJA > 0.7. If judges disagree wildly, the anchors need sharpening.
2. **Chain order on partial overlap**: if a system retrieves 2/4 chain events in correct order, should score be tau·recall (current) or tau·recall²? Paper should fix this.
3. **Who gets 10 on stateful**: current rubric says "exactly matches ideal_top_3 for each variant" — but that ignores in-spirit alternatives. May need to loosen to "in-spirit match" for 8/10 and reserve 10/10 for exact.
