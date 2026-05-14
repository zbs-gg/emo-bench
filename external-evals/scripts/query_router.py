"""Query router — classify queries into {factual, empathic, chain} for hybrid Pulse.

Phase G.1.2 of hybrid Pulse architecture. Heuristic-first (0.5ms) with LLM
fallback only on confidence < 0.6. Conservative default: when uncertain, route
to `empathic` to preserve current stateful retrieval gains.

Usage:
    from query_router import QueryRouter, QueryMode
    router = QueryRouter()
    mode, conf = router.classify("когда я пошёл на пайд-парад?", user_state)
    # → ("factual", 0.9)
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib import request as urlreq


QueryMode = str  # Literal["factual", "empathic", "chain"]


# ────────────────────────────────────────────────────────────────────────────
# Heuristic patterns
# ────────────────────────────────────────────────────────────────────────────

# Reuse / mirror retrieval_v3.py:267 EMO_KEYWORDS for empathic detection
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

# Strong chain markers — explicit causal/temporal trace requests
CHAIN_KEYWORDS = [
    # English
    "lead to", "led to", "leads to", "trace", "chain", "sequence",
    "what caused", "how did this", "what's behind",
    # Russian
    "почему", "цепочка", "из-за чего", "что привело", "как так",
    "вследствие", "до этого", "перед этим", "предшеств",
]

# Factual lookup signals — wh-questions about names/dates/lists, summary asks
FACTUAL_PATTERNS = [
    # English wh-questions with concrete nouns
    re.compile(r"\bwhen (?:did|was|is)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:is|are|was) (?:my|the|a)\b", re.IGNORECASE),
    re.compile(r"\bwhere (?:did|is|was)\b", re.IGNORECASE),
    re.compile(r"\bwho (?:is|was|are)\b", re.IGNORECASE),
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\blist (?:of|all)\b", re.IGNORECASE),
    re.compile(r"\bname of\b", re.IGNORECASE),
    # English "what happened" / "tell me about" / context-summary asks
    # NB: "bring me into context" was tested but conflicted with state-aware
    # cold-open queries on bench v3 — removed for conservatism.
    re.compile(r"\bwhat (?:has happened|happened)\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s been (?:going on|happening)\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
    re.compile(r"\bcatch me up\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:should|do) i know about\b", re.IGNORECASE),
    # Russian wh-questions
    re.compile(r"\bкогда\b", re.IGNORECASE),
    re.compile(r"\bкакой (?:у меня|у нас|в|на)\b", re.IGNORECASE),
    re.compile(r"\bкакая (?:у меня|у нас)\b", re.IGNORECASE),
    re.compile(r"\bкак зовут\b", re.IGNORECASE),
    re.compile(r"\bсколько\b", re.IGNORECASE),
    re.compile(r"\bсписок\b", re.IGNORECASE),
    re.compile(r"\bимя\b", re.IGNORECASE),
    # Russian "что произошло" / "расскажи о"
    re.compile(r"\bчто произошло\b", re.IGNORECASE),
    re.compile(r"\bчто случилось\b", re.IGNORECASE),
    re.compile(r"\bчто нового\b", re.IGNORECASE),
    re.compile(r"\bрасскажи (?:мне )?(?:о|про)\b", re.IGNORECASE),
    re.compile(r"\bвведи меня в курс\b", re.IGNORECASE),
]

# Temporal markers — used in conjunction with emotion to detect empathic-temporal
TEMPORAL_KEYWORDS = [
    "today", "yesterday", "this week", "last week", "this month", "last month",
    "recently", "just now",
    "сегодня", "вчера", "на этой неделе", "на прошлой неделе",
    "в этом месяце", "в прошлом месяце", "недавно", "только что",
]


@dataclass
class RouteDecision:
    mode: QueryMode
    confidence: float
    classifier: str  # "heuristic" | "llm" | "default"
    reasoning: str   # short trace for debugging


@dataclass
class QueryRouter:
    """Heuristic-first router with optional LLM fallback.

    Decision flow:
      1. Try heuristic rules in priority order (chain > factual > empathic)
      2. If best confidence < threshold AND llm_fallback enabled → call LLM
      3. Default to `empathic` (conservatism rule)
    """
    confidence_threshold: float = 0.6
    enable_llm_fallback: bool = True
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_provider: str = "anthropic"
    _cache: dict = field(default_factory=dict)

    def classify(self, query: str, user_state: Optional[object] = None) -> RouteDecision:
        cache_key = (query.strip().lower(), self._state_signature(user_state))
        if cache_key in self._cache:
            return self._cache[cache_key]

        decision = self._classify_heuristic(query, user_state)

        if decision.confidence < self.confidence_threshold and self.enable_llm_fallback:
            llm_decision = self._classify_llm(query)
            if llm_decision is not None:
                decision = llm_decision

        self._cache[cache_key] = decision
        return decision

    def _state_signature(self, state) -> str:
        if state is None:
            return ""
        try:
            mood = getattr(state, "mood_vector", {}) or {}
            top_emo = max(mood.items(), key=lambda kv: kv[1]) if mood else (None, 0)
            return f"emo={top_emo[0]}:{top_emo[1]:.1f};stress={getattr(state,'stress_proxy','')};"
        except Exception:
            return ""

    def _classify_heuristic(self, query: str, user_state) -> RouteDecision:
        q = query.lower().strip()

        # 1. Explicit chain markers — highest priority
        for kw in CHAIN_KEYWORDS:
            if kw in q:
                return RouteDecision(
                    mode="chain", confidence=0.95, classifier="heuristic",
                    reasoning=f"chain keyword: {kw!r}",
                )

        # 2. Factual lookup patterns (wh-questions for facts/names/dates/counts)
        for pat in FACTUAL_PATTERNS:
            if pat.search(q):
                # …but if query also has emotion words, downgrade — could be empathic-temporal
                emo_hit = self._emotion_keyword_hit(q)
                temp_hit = any(t in q for t in TEMPORAL_KEYWORDS)
                if emo_hit and temp_hit:
                    return RouteDecision(
                        mode="empathic", confidence=0.8, classifier="heuristic",
                        reasoning=f"factual pattern + emotion ({emo_hit}) + temporal — likely empathic-temporal",
                    )
                return RouteDecision(
                    mode="factual", confidence=0.9, classifier="heuristic",
                    reasoning=f"factual pattern: {pat.pattern!r}",
                )

        # 3. State-loaded query — if user_state has dominant emotion or body load
        if user_state is not None:
            try:
                mood = getattr(user_state, "mood_vector", {}) or {}
                if mood:
                    top_v = max(float(v) for v in mood.values())
                    if top_v >= 0.5:
                        return RouteDecision(
                            mode="empathic", confidence=0.85, classifier="heuristic",
                            reasoning=f"user_state.mood dominant ({top_v:.2f})",
                        )
                stress_method = getattr(user_state, "is_body_stressed", None)
                if callable(stress_method) and stress_method():
                    return RouteDecision(
                        mode="empathic", confidence=0.85, classifier="heuristic",
                        reasoning="user_state.is_body_stressed=True",
                    )
                rle = getattr(user_state, "recent_life_events_7d", []) or []
                if rle:
                    return RouteDecision(
                        mode="empathic", confidence=0.7, classifier="heuristic",
                        reasoning=f"recent_life_events_7d non-empty ({len(rle)} items)",
                    )
            except Exception:
                pass

        # 4. Emotion keyword in query alone
        emo_hit = self._emotion_keyword_hit(q)
        if emo_hit:
            return RouteDecision(
                mode="empathic", confidence=0.75, classifier="heuristic",
                reasoning=f"emotion keyword: {emo_hit}",
            )

        # 5. Default — empathic with low confidence (triggers LLM fallback)
        return RouteDecision(
            mode="empathic", confidence=0.5, classifier="default",
            reasoning="no heuristic match — default to empathic (conservatism)",
        )

    def _emotion_keyword_hit(self, q_lower: str) -> Optional[str]:
        for emo, kws in EMO_KEYWORDS.items():
            for kw in kws:
                if kw in q_lower:
                    return f"{emo}:{kw}"
        return None

    def _classify_llm(self, query: str) -> Optional[RouteDecision]:
        """Anthropic Claude Haiku 4.5 fallback. Returns None on error (caller keeps heuristic)."""
        api_key = self._read_anthropic_key()
        if not api_key:
            return None

        system = (
            "Classify the user's query into exactly one mode for a memory retrieval system. "
            "Output ONLY one word: factual, empathic, or chain. "
            "Definitions: "
            "factual = lookup of a specific name, date, number, or list (e.g., 'when did X happen', 'what is my Y'). "
            "empathic = query about feelings, experiences, mood, body state, or a moment in time tied to emotion. "
            "chain = causal or temporal sequence ('why', 'what led to', 'how did X arrive at Y')."
        )
        body = {
            "model": self.llm_model,
            "max_tokens": 10,
            "system": system,
            "messages": [{"role": "user", "content": query}],
        }
        try:
            req = urlreq.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                method="POST",
            )
            with urlreq.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = data["content"][0]["text"].strip().lower()
            for mode in ("factual", "empathic", "chain"):
                if mode in text:
                    return RouteDecision(
                        mode=mode, confidence=0.7, classifier="llm",
                        reasoning=f"haiku said: {text!r}",
                    )
        except Exception as ex:
            print(f"  [router LLM error] {ex}", file=sys.stderr)
        return None

    def _read_anthropic_key(self) -> Optional[str]:
        k = os.environ.get("ANTHROPIC_API_KEY")
        if k:
            return k
        for fname in ("anthropic-api-key.txt", "anthropic.txt"):
            p = Path.home() / ".openclaw" / "secrets" / fname
            if p.exists():
                raw = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                if "=" in raw:
                    _, _, val = raw.partition("=")
                    cand = val.strip().strip('"\'')
                    if cand:
                        return cand
                if raw and not raw.startswith("ANTHROPIC_A"):
                    return raw
        return None


# ────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test the query router.")
    ap.add_argument("--query", type=str, help="Single query to classify (interactive)")
    ap.add_argument("--no-llm", action="store_true", help="Heuristic only")
    ap.add_argument("--cases", type=Path, help="JSONL of {query, expected_mode} cases")
    args = ap.parse_args()

    router = QueryRouter(enable_llm_fallback=not args.no_llm)

    if args.query:
        d = router.classify(args.query)
        print(f"query: {args.query}")
        print(f"  → {d.mode} (confidence={d.confidence:.2f}, "
              f"classifier={d.classifier}, reasoning={d.reasoning})")
    elif args.cases:
        right = wrong = 0
        for line in args.cases.open():
            if not line.strip():
                continue
            row = json.loads(line)
            q, expected = row["query"], row["expected_mode"]
            d = router.classify(q)
            ok = "✓" if d.mode == expected else "✗"
            if d.mode == expected:
                right += 1
            else:
                wrong += 1
            print(f"{ok} [{d.mode}/{expected}] conf={d.confidence:.2f} cls={d.classifier:.<10} {q[:60]}")
        print(f"\naccuracy: {right}/{right+wrong} = {right/(right+wrong)*100:.1f}%")
    else:
        # Built-in self-test cases (covers Russian + English + state edge cases)
        cases = [
            ("когда я ходил на pride-парад?", "factual"),
            ("когда был мой первый митап?", "factual"),
            ("сколько у меня собак?", "factual"),
            ("what is my pet's name?", "factual"),
            ("when did Caroline transition?", "factual"),
            ("list of all my hobbies", "factual"),
            ("почему я тогда злился?", "chain"),
            ("what led to the conflict with Anna?", "chain"),
            ("trace the chain of events that brought me here", "chain"),
            ("из-за чего у нас расходятся?", "chain"),
            ("что я тогда чувствовал?", "empathic"),
            ("как я переживал тот разрыв", "empathic"),
            ("when I was scared and alone", "empathic"),
            ("что происходило когда мне было плохо", "empathic"),
            ("страх перед публикой", "empathic"),
            ("памяти где я был в ресурсе", "empathic"),
        ]
        right = wrong = 0
        for q, expected in cases:
            d = router.classify(q)
            ok = "✓" if d.mode == expected else "✗"
            if d.mode == expected:
                right += 1
            else:
                wrong += 1
            print(f"{ok} [{d.mode}/{expected}] conf={d.confidence:.2f} cls={d.classifier:.<10} {q}")
        print(f"\naccuracy: {right}/{right+wrong} = {right/(right+wrong)*100:.1f}%")
