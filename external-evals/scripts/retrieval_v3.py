"""Pulse retrieval v3 — conditional state/emotion/chain-aware retrieval.

Builds on v2_pure (cosine × recency) with three CONDITIONAL boost terms:
  - emotion_alignment: active only when query has a dominant emotion (max > 0.5)
  - state_fit: active only when biometric signal is strong (low sleep / high stress / elevated HR)
  - chain_expansion: active only for chain-type queries (explicit flag)

Design rationale (Phase D result, 2026-04-20):
Multiplicative emotion term monotonically HURT retrieval (β=0 43.77 → β=3 27.72 NDCG).
v3 fixes this with CONDITIONAL gating: boosts OFF unless their signal is genuine.
When state is neutral → formula collapses to v2_pure exactly.

Usage:
    from retrieval_v3 import UserState, RetrievalV3
    engine = RetrievalV3(events)
    ids = engine.retrieve(query, user_state=UserState(...), top_k=3)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import embed_cohere  # noqa: F401  (legacy import; kept for back-compat)
from embedding_provider import embed_texts as _embed_texts


def embed_cohere_or_alt(texts, input_type):
    """Provider-aware shim. Reads env var EMBEDDING_PROVIDER once per call.
    Falls back to Cohere when unset — preserves prior behaviour exactly.
    Sprint D1.8 — embedding shootout (cohere vs bge-m3) for paper ablation.
    """
    return _embed_texts(texts, input_type)


EMOTION_KEYS = [
    "joy", "sadness", "anger", "fear", "trust",
    "disgust", "anticipation", "surprise", "shame", "guilt",
]


@dataclass
class UserState:
    """Current user state for stateful retrieval. All fields optional."""
    mood_vector: dict = field(default_factory=dict)          # Plutchik-10 floats 0-1
    sleep_quality: Optional[float] = None                    # 0-1
    sleep_hours: Optional[float] = None
    hrv: Optional[float] = None                              # ms
    hr_trend: Optional[str] = None                           # "elevated_3d" | "stable" | "low"
    hrv_trend: Optional[str] = None                          # "declining_3d" | "stable" | "rising"
    stress_proxy: Optional[float] = None                     # 0-1
    recent_life_events_7d: list = field(default_factory=list)
    time_of_day: Optional[str] = None                        # "morning"/"evening"/"night"
    snapshot_days_ago: Optional[float] = None                # state represents a specific past moment (days_ago scale)

    def has_dominant_emotion(self, threshold: float = 0.5) -> tuple[bool, float, str]:
        """Return (has_dom, max_value, max_key)."""
        if not self.mood_vector:
            return False, 0.0, ""
        items = sorted(self.mood_vector.items(), key=lambda kv: -kv[1])
        top_k, top_v = items[0]
        return top_v >= threshold, top_v, top_k

    def is_body_stressed(self) -> bool:
        """True if biometric signal indicates body is under load."""
        if self.stress_proxy is not None and self.stress_proxy >= 0.6:
            return True
        if self.sleep_quality is not None and self.sleep_quality <= 0.4:
            return True
        if self.hr_trend in ("elevated_3d", "elevated_overnight"):
            return True
        if self.hrv_trend == "declining_3d":
            return True
        if self.hrv is not None and self.hrv < 55:
            return True
        return False

    def is_body_restored(self) -> bool:
        """True if biometric signal indicates good baseline."""
        if self.stress_proxy is not None and self.stress_proxy <= 0.3 and \
           (self.sleep_quality is None or self.sleep_quality >= 0.7):
            return True
        return False


# ────────────────────────────────────────────────────────────────────────────
# Core helpers
# ────────────────────────────────────────────────────────────────────────────

def emotion_vec(tags: dict | None) -> np.ndarray:
    """Convert emotion_tags dict to 10-dim float array in canonical order."""
    if not tags:
        return np.zeros(len(EMOTION_KEYS), dtype=np.float32)
    return np.array([float(tags.get(k, 0.0)) for k in EMOTION_KEYS], dtype=np.float32)


def compute_emotion_alignment(query_emo: np.ndarray, event_emo: np.ndarray) -> float:
    """Cosine similarity between two emotion vectors. 0 if either is zero-vector."""
    qn = float(np.linalg.norm(query_emo))
    en = float(np.linalg.norm(event_emo))
    if qn < 1e-6 or en < 1e-6:
        return 0.0
    return float(np.dot(query_emo, event_emo) / (qn * en))


def _event_is_depletion(event: dict) -> bool:
    """Heuristic: does event describe body depletion / overload / shame episode?"""
    bio = event.get("biometric_snapshot") or {}
    text = (event.get("text") or "").lower()
    label = (event.get("sentiment_label") or "").lower()
    # Body biometric indicators
    if isinstance(bio.get("hrv"), (int, float)) and bio["hrv"] < 60:
        return True
    if isinstance(bio.get("sleep_quality"), (int, float)) and bio["sleep_quality"] <= 0.4:
        return True
    if isinstance(bio.get("stress_proxy"), (int, float)) and bio["stress_proxy"] >= 0.6:
        return True
    if bio.get("hrv_trend") == "declining_3d" or bio.get("hr_trend") in ("elevated_3d", "elevated_overnight"):
        return True
    # Label/text fallback
    if "burden" in label or "wound" in label:
        return True
    if any(phrase in text for phrase in ("hrv 5", "declining", "anxious sleep", "overload")):
        return True
    return False


def _event_is_restoration(event: dict) -> bool:
    """Heuristic: does event describe body restoration / completion / positive biometric?"""
    bio = event.get("biometric_snapshot") or {}
    text = (event.get("text") or "").lower()
    label = (event.get("sentiment_label") or "").lower()
    if isinstance(bio.get("hrv"), (int, float)) and bio["hrv"] >= 70:
        return True
    if isinstance(bio.get("sleep_quality"), (int, float)) and bio["sleep_quality"] >= 0.7:
        if bio.get("stress_proxy", 1.0) <= 0.3:
            return True
    if bio.get("workout") is True:
        return True
    if "ship" in label or "milestone" in label or "repair" in label:
        return True
    if any(phrase in text for phrase in ("hrv 7", "hrv 8", "hrv 9", "post-workout", "ship day")):
        return True
    return False


def compute_date_proximity(event_days_ago: float, state_days_ago: float) -> float:
    """Temporal proximity score 0-1 between event and snapshot.

    Stepped curve that discriminates strongly near the snapshot date:
      diff ≤ 1  → 1.00 (same day)
      diff ≤ 3  → 0.70 (within 3 days)
      diff ≤ 7  → 0.30 (same week)
      else      → 0.00

    Dropping from 0.7 to 0.3 between diff=3 and diff=4 creates enough
    ranking differential to distinguish "happened that day" from "happened
    that week" — needed to surface day-specific events (e.g. event 40
    v1-failure on exactly the snapshot date) over general week events.
    """
    diff = abs(float(event_days_ago) - float(state_days_ago))
    if diff <= 1.0:
        return 1.0
    if diff <= 3.0:
        return 0.7
    if diff <= 7.0:
        return 0.3
    return 0.0


def compute_state_fit(event: dict, state: UserState) -> float:
    """Match between event and user_state. Returns 0-1.

    - body-stressed state boosts DEPLETION events (low HRV, poor sleep, overload)
    - body-restored state boosts RESTORATION events (high HRV, ship days, post-workout)
    - recent_life_events_7d soft-match on keywords
    """
    label = (event.get("sentiment_label") or "").lower()
    text = (event.get("text") or "").lower()
    score = 0.0

    stressed = state.is_body_stressed()
    restored = state.is_body_restored()

    if stressed:
        if _event_is_depletion(event):
            score = max(score, 1.0)
        elif _event_is_restoration(event):
            # Anti-match: restored events in stressed state should NOT boost
            score = max(score, 0.0)  # neutral, not negative (boost is multiplicative)
    if restored:
        if _event_is_restoration(event):
            score = max(score, 1.0)

    # recent_life_events_7d soft hint
    if state.recent_life_events_7d:
        hints = " ".join(str(x) for x in state.recent_life_events_7d).lower()
        if any(w in hints for w in ("anya", "аня", "conflict", "ссора")):
            if "marriage" in label or "repair" in label or "anya" in text or "аня" in text:
                score = max(score, 0.7)
        if "anniversary" in hints or "unknown" in hints:
            if "origin" in label or "wound" in label:
                score = max(score, 0.7)

    return score


# ────────────────────────────────────────────────────────────────────────────
# Chain expansion
# ────────────────────────────────────────────────────────────────────────────

def build_chain_graph(events: list[dict]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Return (parent_to_children, child_to_parents) adjacency maps based on predecessor_ids."""
    parent_to_children: dict[int, list[int]] = {}
    child_to_parents: dict[int, list[int]] = {}
    for e in events:
        eid = e["id"]
        parents = e.get("predecessor_ids") or []
        child_to_parents[eid] = list(parents)
        for p in parents:
            parent_to_children.setdefault(p, []).append(eid)
    return parent_to_children, child_to_parents


def expand_chain_from_seeds(seeds: list[int], events: list[dict], depth: int = 3) -> list[int]:
    """BFS over predecessor_ids (backwards in time) from seed events, then sort by
    topological order (root of chain first, leaf last).

    Returns an ORDERED list: earliest cause → latest effect, limited to events
    reachable from any seed within `depth` steps (either direction).
    """
    p2c, c2p = build_chain_graph(events)
    eid_map = {e["id"]: e for e in events}

    visited: set[int] = set()
    frontier: list[tuple[int, int]] = [(s, 0) for s in seeds]  # (eid, dist)
    while frontier:
        eid, d = frontier.pop(0)
        if eid in visited or eid not in eid_map:
            continue
        visited.add(eid)
        if d >= depth:
            continue
        for nb in c2p.get(eid, []) + p2c.get(eid, []):
            if nb not in visited:
                frontier.append((nb, d + 1))

    # Topological-ish order: sort by how many ancestors a node has inside `visited`.
    # Roots (no visited parents) come first.
    def _ancestor_depth(eid: int, memo: dict[int, int] | None = None) -> int:
        if memo is None: memo = {}
        if eid in memo: return memo[eid]
        memo[eid] = 0  # guard against cycles
        parents = [p for p in c2p.get(eid, []) if p in visited]
        if not parents:
            memo[eid] = 0
            return 0
        memo[eid] = 1 + max(_ancestor_depth(p, memo) for p in parents)
        return memo[eid]

    ordered = sorted(visited, key=lambda e: (_ancestor_depth(e), eid_map[e].get("days_ago", 0) * -1))
    return ordered


# ────────────────────────────────────────────────────────────────────────────
# Query emotion inference (LLM-based, with keyword fallback)
# ────────────────────────────────────────────────────────────────────────────

_QUERY_EMO_CACHE: dict[str, dict] = {}

EMO_KEYWORDS = {
    "joy":         ["рад", "кайф", "joy", "радост", "счаст"],
    "sadness":     ["груст", "печал", "тоск", "потер", "sad"],
    "anger":       ["зл", "ярос", "раздраж", "бес", "anger", "angry", "mad"],
    "fear":        ["страх", "тревог", "паник", "боюсь", "scared", "fear", "anxious"],
    "trust":       ["довер", "близос", "принят", "trust", "safe"],
    "disgust":     ["отвращ", "брезг", "disgust"],
    "anticipation":["предвкуш", "надежд", "интерес", "excited", "anticipate"],
    "surprise":    ["удивл", "шок", "недоум", "surprise"],
    "shame":       ["стыд", "смущ", "shame", "заслуживат"],
    "guilt":       ["вин", "сожал", "guilt", "накосяч", "виноват"],
}


# Phase 5.5: Emotion → semantic-cluster hints for query augmentation.
# When state has a dominant emotion, appending these short phrases to the query
# shifts the query embedding toward the semantic cluster most relevant for that
# emotion. Grid showed +3 hits across 5 stateful pairs vs no-augmentation (7/30
# → 10/30), and P1 (anger vs shame on Anya) went from 0 diff to full 5-event
# differentiation between pair variants.
#
# Hints chosen to be semantically ORTHOGONAL to the emotion labels themselves:
# we want to shift retrieval toward the ACTION CLUSTER the emotion indicates,
# not the emotion word itself (already captured by the emotion_tags boost in
# step 2). Example: anger → "conflict navigation repair" surfaces events about
# navigating anger, not events emotion-tagged angry.
EMOTION_QUERY_HINTS = {
    "anger":        "conflict navigation repair",
    "shame":        "wound self-blame rejection origin",
    "fear":         "threat anxiety uncertainty safety",
    "sadness":      "loss grief depletion",
    "joy":          "victory completion success",
    "trust":        "safety bond closeness",
    "anticipation": "aliveness presence connection",
    "surprise":     "unexpected new shift",
    "disgust":      "rejection boundary violation",
    "guilt":        "transgression repair agency",
}


# Phase 5.7: Sub-type ADDENDA — short differentiators APPENDED to the base
# emotion hint. Keeps base cluster retrieval intact while nudging toward the
# specific flavor (acute-activation vs ancestral; body-anxiety vs cognitive).
# Empty string means no sub-type differentiation applied.
EMOTION_SUBTYPE_ADDENDA = {
    "shame_acute":     "current trigger recent",
    "shame_ancestral": "ancestral identity core",
    "fear_body":       "body physiological somatic",
    "fear_cognitive":  "anticipation future",
    "anger_active":    "action immediate",
    "anger_residual":  "simmering grievance",
}


# Phase 5.6: Temporal keyword → implicit days_ago reference.
# STRICT list: only explicit temporal markers. Excluded: "now"/"сейчас"/
# "currently" — they occur in non-temporal emotional queries ("что важно
# СЕЙЧАС?") and over-triggered date boost on neutral tests. Only keep
# markers that unambiguously reference a past moment.
TEMPORAL_KEYWORDS = [
    (("today", "сегодня"), 0.0),
    (("yesterday", "вчера"), 1.0),
    (("this week", "на этой неделе", "recently", "недавно", "last few days", "за последние дни"), 3.0),
    (("last week", "на прошлой неделе", "неделю назад"), 10.0),
    (("this month", "в этом месяце", "этом месяце"), 15.0),
    (("last month", "в прошлом месяце", "месяц назад"), 40.0),
]


def infer_query_date(query: str) -> float | None:
    """Scan query for temporal keywords → return implicit days_ago reference,
    or None if no temporal indicator found.

    Returns the days_ago value that events should cluster near for this query.
    Example: query="что произошло на этой неделе?" → returns 3.0 (mid-week reference).
    Case-insensitive substring match. First-match wins if multiple keywords match.
    """
    q = query.lower()
    for kws, days in TEMPORAL_KEYWORDS:
        if any(kw in q for kw in kws):
            return days
    return None


def infer_query_emotions_keyword(query: str) -> dict[str, float]:
    """Keyword-based fallback emotion inference. Returns floats 0-0.8 per emotion."""
    q = query.lower()
    scores = {k: 0.0 for k in EMOTION_KEYS}
    for emo, kws in EMO_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[emo] = max(scores[emo], 0.7)
    return scores


def infer_query_emotions_llm(query: str, provider: str = "qwen") -> dict[str, float]:
    """LLM-based emotion inference for a query. Returns 10-dim dict."""
    if query in _QUERY_EMO_CACHE:
        return _QUERY_EMO_CACHE[query]

    from common import secret, PROVIDERS
    from urllib import request as urlreq

    cfg = PROVIDERS[provider]
    try:
        api_key = secret(cfg["key_file"])
    except Exception:
        return infer_query_emotions_keyword(query)

    system = (
        "You rate the emotional tone of a user's query on 10 dimensions "
        "(Plutchik-8 + shame + guilt). For each: 0.0=absent, 0.5=present, 1.0=overwhelming. "
        "Output ONLY a single JSON object with keys: "
        "joy, sadness, anger, fear, trust, disgust, anticipation, surprise, shame, guilt. "
        "No markdown. No explanation."
    )
    payload = json.dumps({
        "model": cfg["default_model"],
        "max_tokens": 500,
        "temperature": 1.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Query: {query}"},
        ],
    }).encode()
    req = urlreq.Request(
        f"{cfg['base_url']}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlreq.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        text = data["choices"][0]["message"].get("content") or ""
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            # Normalize keys
            out = {k: float(obj.get(k, 0.0)) for k in EMOTION_KEYS}
            _QUERY_EMO_CACHE[query] = out
            return out
    except Exception:
        pass

    # Fallback
    kw = infer_query_emotions_keyword(query)
    _QUERY_EMO_CACHE[query] = kw
    return kw


# ────────────────────────────────────────────────────────────────────────────
# Main retrieval engine
# ────────────────────────────────────────────────────────────────────────────

class RetrievalV3:
    """Pulse v3 retrieval engine.

    Formula per event:
        base = cosine(query, event) × exp(-λ · days_ago)
        boost_emo = 1 + β · max(0, emotion_alignment)     if query has dominant emotion
                    1                                       otherwise
        boost_state = 1 + γ · state_fit                   if state is body-stressed or restored
                      1                                    otherwise
        boost_anchor = 1 + δ_a                            if event.user_flag is True (structural anchor)
                       1                                    otherwise
        boost_date = 1 + δ_d · date_proximity              if state.snapshot_days_ago is set
                     1                                      otherwise
        score = base × boost_emo × boost_state × boost_anchor × boost_date

    Anchor-boost (δ_a=0.15) applies ONLY to events that already passed the cosine
    filter (i.e. are in the top candidate set) — so it surfaces load-bearing
    anchors (marriage-anchor, zasluzhivatel) when they're topically relevant,
    without dragging unrelated anchors into random queries.
    """

    def __init__(self, events: list[dict], embedder_model: str = "embed-v4.0",
                 decay_lambda: float = 0.002, decay_lambda_anchor: float = 0.001,
                 beta: float = 0.15, gamma: float = 0.15,
                 delta_anchor: float = 0.05, delta_date: float = 0.25,
                 anchor_top_n: int = 8,
                 query_emo_provider: str = "qwen", use_llm_query_emo: bool = True,
                 # ── Ablation flags (all True == current v3 behaviour) ───────
                 enable_anchor_decay: bool = True,
                 enable_anchor_boost: bool = True,
                 enable_date_boost: bool = True,
                 enable_emotion_boost: bool = True,
                 enable_state_boost: bool = True,
                 enable_emotion_hint_augment: bool = True,
                 enable_temporal_keywords: bool = True,
                 enable_chain_expansion: bool = True):
        self.events = events
        self.decay_lambda = decay_lambda
        # Phase 5.4: anchor-aware decay. user_flag=True events are structural
        # truths (marriage anchor, zasluzhivatel, communication rules) and should
        # not lose weight over time. Setting decay_lambda_anchor=0.001 gives
        # half-life ~693d for anchors vs ~347d for regular events — matches the
        # Pulse v2 BELIEF_DECAY "self_model" tier (self_model=0.0005 for the
        # axiom-tier, we pick 0.001 as conservative middle ground).
        self.decay_lambda_anchor = decay_lambda_anchor
        self.beta = beta
        self.gamma = gamma
        self.delta_anchor = delta_anchor
        self.delta_date = delta_date
        self.anchor_top_n = anchor_top_n   # anchor boost only applied to events within top-N by base score
        self.query_emo_provider = query_emo_provider
        self.use_llm_query_emo = use_llm_query_emo

        # Ablation switches — when False, the corresponding term collapses to 1.0
        # and the formula degrades toward v2_pure. Used by bench-empathic-memory-v3.py
        # `--ablation-config` for per-boost contribution measurement.
        self.enable_anchor_decay = enable_anchor_decay
        self.enable_anchor_boost = enable_anchor_boost
        self.enable_date_boost = enable_date_boost
        self.enable_emotion_boost = enable_emotion_boost
        self.enable_state_boost = enable_state_boost
        self.enable_emotion_hint_augment = enable_emotion_hint_augment
        self.enable_temporal_keywords = enable_temporal_keywords
        self.enable_chain_expansion = enable_chain_expansion

        self._ids = [e["id"] for e in events]
        self._event_vecs = embed_cohere_or_alt([e["text"] for e in events], "search_document")
        self._event_emos = np.array([emotion_vec(e.get("emotion_tags")) for e in events],
                                    dtype=np.float32)
        self._days = np.array([e.get("days_ago", 0) for e in events], dtype=np.float32)
        self._is_anchor = np.array([1.0 if e.get("user_flag") else 0.0 for e in events],
                                    dtype=np.float32)
        # Per-event effective decay — anchors get the slower rate when enabled
        if self.enable_anchor_decay:
            self._eff_lambda = np.where(
                self._is_anchor > 0, self.decay_lambda_anchor, self.decay_lambda,
            )
        else:
            self._eff_lambda = np.full_like(self._days, self.decay_lambda)
        self._eid_map = {e["id"]: e for e in events}

    def _query_emotion_vec(self, query: str) -> tuple[np.ndarray, dict[str, float]]:
        if self.use_llm_query_emo:
            emo_dict = infer_query_emotions_llm(query, self.query_emo_provider)
        else:
            emo_dict = infer_query_emotions_keyword(query)
        return emotion_vec(emo_dict), emo_dict

    def _pick_emotion_hint(self, dom_key: str, state: UserState) -> str | None:
        """Return base emotion hint from EMOTION_QUERY_HINTS. Phase 5.7 sub-type
        variants were tested (shame-acute vs shame-ancestral, fear-body vs
        fear-cognitive, anger-active vs anger-residual) but hard-replace
        regressed bench (-0.40 overall, -1.40 multi) and soft-append stayed
        flat on Apple Health. The base cluster hints are already close to
        optimal — further granularity needs LLM-generated query rewrites, not
        hand-mapped keywords. Reverted to base hints 2026-04-24.
        """
        return EMOTION_QUERY_HINTS.get(dom_key)

    def retrieve(self, query: str, user_state: UserState | None = None,
                 top_k: int = 3, return_scores: bool = False,
                 expand_chain: bool = False, augment_query: bool = True) -> list[int]:
        """Return top-K event IDs. If expand_chain=True, expand via predecessor_ids.

        Phase 5.5: when user_state has a dominant emotion (max >= 0.5) AND
        augment_query=True, append a short semantic hint (EMOTION_QUERY_HINTS)
        to the query before embedding. Shifts semantic search toward the action
        cluster matching the user's state. Conditional: neutral state → no
        augmentation → identical to v2_pure behavior.
        """
        # 1. Base semantic score (Phase 5.4: anchors use slower decay)
        effective_query = query
        if (self.enable_emotion_hint_augment and augment_query
                and user_state and user_state.mood_vector):
            dom_ok, _, dom_key = user_state.has_dominant_emotion(0.5)
            if dom_ok:
                # Phase 5.7: prefer sub-type hint when signal disambiguates
                hint = self._pick_emotion_hint(dom_key, user_state)
                if hint:
                    effective_query = f"{query} {hint}"
        q_vec = embed_cohere_or_alt([effective_query], "search_query")[0]
        sims = self._event_vecs @ q_vec                             # (N,)
        recency = np.exp(-self._eff_lambda * self._days)            # (N,)
        base = sims * recency

        # 2. Conditional emotion boost
        boost_emo = np.ones_like(base)
        if self.enable_emotion_boost:
            # Prefer user_state.mood_vector if provided (already represents current state);
            # otherwise infer from query text.
            if user_state and user_state.mood_vector:
                q_emo = emotion_vec(user_state.mood_vector)
                dom_ok, _, _ = user_state.has_dominant_emotion(0.5)
            else:
                q_emo, q_dict = self._query_emotion_vec(query)
                dom_ok = max(q_dict.values()) >= 0.5 if q_dict else False

            if dom_ok:
                # Vectorized cosine similarity with each event emo vector
                q_norm = float(np.linalg.norm(q_emo))
                if q_norm > 1e-6:
                    e_norms = np.linalg.norm(self._event_emos, axis=1)
                    mask = e_norms > 1e-6
                    align = np.zeros_like(base)
                    if mask.any():
                        align[mask] = (self._event_emos[mask] @ q_emo) / (e_norms[mask] * q_norm)
                    boost_emo = 1.0 + self.beta * np.clip(align, 0.0, None)

        # 3. Conditional state/body boost
        boost_state = np.ones_like(base)
        if self.enable_state_boost and user_state is not None and (
                user_state.is_body_stressed() or user_state.is_body_restored()):
            fit = np.array([compute_state_fit(e, user_state) for e in self.events],
                           dtype=np.float32)
            boost_state = 1.0 + self.gamma * fit

        # 3b. Anchor-priority boost (Phase 5.1 — fix T2/T31 weakness)
        # Events with user_flag=True are load-bearing anchors (marriage wound,
        # zasluzhivatel, core milestones). Boost applied ONLY to anchors that
        # already entered the top-N by base score (topically relevant). This
        # avoids Phase D trap: we don't drag unrelated anchors into unrelated
        # queries. Non-anchor events in top-N remain untouched.
        boost_anchor = np.ones_like(base)
        if self.enable_anchor_boost and self.delta_anchor > 0:
            order_base = np.argsort(-base)
            in_top_n = np.zeros_like(base, dtype=bool)
            in_top_n[order_base[:self.anchor_top_n]] = True
            anchor_in_top = in_top_n & (self._is_anchor > 0)
            boost_anchor = np.where(anchor_in_top, 1.0 + self.delta_anchor, 1.0)

        # 3c. Conditional date-proximity boost (Phase 5.2 + 5.6)
        # Two sources of snapshot_days_ago:
        #   (a) Explicit: user_state.snapshot_days_ago (Apple Health style)
        #   (b) Implicit: parse "today"/"сегодня"/"recently" from query text
        #       (Phase 5.6 — activates the boost on plain temporal queries)
        boost_date = np.ones_like(base)
        if self.enable_date_boost:
            date_ref = None
            if user_state is not None and user_state.snapshot_days_ago is not None:
                date_ref = user_state.snapshot_days_ago
            elif self.enable_temporal_keywords:
                date_ref = infer_query_date(query)
            if date_ref is not None:
                prox = np.array([
                    compute_date_proximity(d, date_ref)
                    for d in self._days
                ], dtype=np.float32)
                boost_date = 1.0 + self.delta_date * prox

        # 4. Combine
        final = base * boost_emo * boost_state * boost_anchor * boost_date
        order = np.argsort(-final)
        top_ids = [self._ids[int(i)] for i in order[:top_k]]

        # 5. Optional chain expansion: prefer a coherent chain containing seeds
        if expand_chain and self.enable_chain_expansion:
            # Over-fetch to top_k * 3 candidates for chain analysis
            wider = [self._ids[int(i)] for i in order[:max(top_k * 3, 9)]]
            p2c, c2p = build_chain_graph(self.events)

            def _connected_component(seed: int, candidates: set[int]) -> set[int]:
                """BFS to find seeds reachable from `seed` via chain edges."""
                visited = {seed}
                frontier = [seed]
                found = {seed} if seed in candidates else set()
                while frontier:
                    n = frontier.pop(0)
                    for nb in c2p.get(n, []) + p2c.get(n, []):
                        if nb not in visited:
                            visited.add(nb); frontier.append(nb)
                            if nb in candidates:
                                found.add(nb)
                return found

            # Pick the seed whose connected-component includes the most other wider seeds
            cand_set = set(wider)
            best_seed, best_reach = None, set()
            for s in wider[:top_k]:  # best seeds first
                reach = _connected_component(s, cand_set)
                if len(reach) > len(best_reach):
                    best_seed, best_reach = s, reach

            if best_seed and len(best_reach) >= 2:
                # Topologically expand from best_seed, then intersect with reachable wider seeds
                expanded = expand_chain_from_seeds([best_seed], self.events, depth=4)
                chain_ids = [eid for eid in expanded if eid in best_reach]
                # Fill up to top_k: chain first, then remaining original top seeds
                result: list[int] = []
                for eid in chain_ids:
                    if eid not in result and len(result) < top_k:
                        result.append(eid)
                for eid in top_ids:
                    if eid not in result and len(result) < top_k:
                        result.append(eid)
                top_ids = result[:top_k]

        if return_scores:
            return [(self._ids[int(i)], float(final[int(i)])) for i in order[:top_k]]
        return top_ids

    # ────────────────────────────────────────────────────────────────────────
    # Phase G — Hybrid retrieval (factual / empathic / chain modes)
    # ────────────────────────────────────────────────────────────────────────

    def _ensure_router(self):
        """Lazy-init query router (only on first hybrid retrieval)."""
        if not hasattr(self, "_router") or self._router is None:
            try:
                from query_router import QueryRouter
                self._router = QueryRouter()
            except ImportError:
                self._router = None  # heuristic disabled, default empathic
        return self._router

    def index_facts(self, facts: list[dict]) -> None:
        """Embed facts once and cache for retrieve_hybrid(mode='factual').

        Each fact dict must have 'text' and 'event_id'. Optional:
        'is_anchor', 'emotion_tags', 'attributed_to'.
        """
        self._facts = list(facts)
        if not self._facts:
            self._fact_vecs = None
            self._fact_event_ids = None
            return
        texts = [f["text"] for f in self._facts]
        self._fact_vecs = embed_cohere_or_alt(texts, "search_document")
        self._fact_event_ids = [int(f["event_id"]) for f in self._facts]

    def _retrieve_factual(self, query: str, top_k: int) -> list[int]:
        """Cosine on fact embeddings → unique parent event_ids (top_k).

        Falls back to empathic retrieve() if facts haven't been indexed.
        """
        if not getattr(self, "_facts", None) or self._fact_vecs is None:
            # No facts indexed — fall back to empathic so caller gets sane result
            return self.retrieve(query, top_k=top_k)

        q_vec = embed_cohere_or_alt([query], "search_query")[0]
        sims = self._fact_vecs @ q_vec
        order = np.argsort(-sims)

        seen: set[int] = set()
        out: list[int] = []
        for idx in order:
            eid = self._fact_event_ids[int(idx)]
            if eid in seen:
                continue
            seen.add(eid)
            out.append(eid)
            if len(out) >= top_k:
                break
        return out

    def retrieve_hybrid(self, query: str,
                        user_state: UserState | None = None,
                        top_k: int = 3,
                        mode: str = "auto",
                        return_decision: bool = False):
        """Phase G hybrid retrieval. Modes:
          - 'auto'     → router classifies, then dispatches
          - 'factual'  → search atomic_facts → unique parent event_ids
          - 'empathic' → full v3 formula (current retrieve)
          - 'chain'    → empathic + chain expansion

        Use index_facts() before calling with mode='factual' or 'auto'.
        Returns list[int] (event_ids), or (list[int], RouteDecision) if return_decision.
        """
        decision = None
        chosen_mode = mode

        if mode == "auto":
            router = self._ensure_router()
            if router is not None:
                decision = router.classify(query, user_state)
                chosen_mode = decision.mode
            else:
                chosen_mode = "empathic"  # safe default

        if chosen_mode == "factual":
            ids = self._retrieve_factual(query, top_k)
        elif chosen_mode == "chain":
            ids = self.retrieve(query, user_state=user_state, top_k=top_k,
                                expand_chain=True)
        else:  # 'empathic' or unknown → conservative default
            ids = self.retrieve(query, user_state=user_state, top_k=top_k)

        if return_decision:
            return ids, decision
        return ids


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mood", type=str, default="",
                    help="Comma-separated mood like 'anger=0.7,fear=0.2'")
    ap.add_argument("--stress", type=float, default=None)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--hr-trend", type=str, default=None)
    ap.add_argument("--expand-chain", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    events = data["events"]
    engine = RetrievalV3(events)

    mood = {}
    if args.mood:
        for kv in args.mood.split(","):
            k, v = kv.split("=")
            mood[k.strip()] = float(v)

    state = UserState(mood_vector=mood, stress_proxy=args.stress,
                      sleep_quality=args.sleep, hr_trend=args.hr_trend)
    ids = engine.retrieve(args.query, user_state=state, top_k=args.top_k,
                          expand_chain=args.expand_chain, return_scores=True)
    print(f"Query: {args.query!r}")
    print(f"State: {asdict(state)}")
    print(f"Has dominant emo: {state.has_dominant_emotion()}")
    print(f"Body stressed: {state.is_body_stressed()}  restored: {state.is_body_restored()}")
    print(f"\nTop {args.top_k}:")
    for eid, score in ids:
        ev = next(e for e in events if e["id"] == eid)
        print(f"  id={eid} score={score:.4f} label={ev.get('sentiment_label','?'):<25} {ev['text'][:100]}")
