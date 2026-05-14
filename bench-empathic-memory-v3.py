#!/usr/bin/env python3
"""
Empathic Memory Bench v3 — fork of v2 with new product-vision axes.

Four scoring axes (see rubric-v3.md):
  - core (rel + spec + act, 0-30) — legacy v2 axis
  - stateful_fit (0-10) — paired tests, same query/different user_state
  - chain_order (0-10) — Kendall tau on temporal/causal chain order
  - multi_signal_fit (0-10) — biometric_snapshot + retrieved integration

Corpus: datasets/empathic-memory-corpus-v3.json (60 events, 35 tests).

Usage:
  python bench-empathic-memory-v3.py --systems cosine,bm25,hybrid --lang ru
  python bench-empathic-memory-v3.py --systems cosine --tests T6,T7 --dry-run  # no judges
  python bench-empathic-memory-v3.py --systems all --judges kimi --snapshot    # commit snapshot

Fork rationale: bench-empathic-memory.py (v2) remains the canonical 15-system bench on
sonya corpus. v3 adds axes no external benchmark measures (stateful/chain/multi-signal)
and is the one that validates the Pulse product claims.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

# Reuse shared utilities from external-evals
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR / "external-evals" / "scripts"))
from common import (
    embed_cohere, tokenize, rrf_merge, secret, PROVIDERS,
    DO_VENDOR_FALLBACK, call_vendor_fallback,
)


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

CORPUS_FILE = BENCH_DIR / "datasets" / "empathic-memory-corpus-v3.json"
RESULTS_DIR = BENCH_DIR / "external-evals" / "results"
SNAPSHOT_DIR = BENCH_DIR / "external-evals" / "snapshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 3


# ────────────────────────────────────────────────────────────────────────────
# System adapters
#
# Each adapter: Callable[[query, user_state, biometric_snapshot], list[int]]
# Returns ORDERED list of event IDs (top K).
# user_state/biometric_snapshot may be None; adapters that ignore them still accept them.
# ────────────────────────────────────────────────────────────────────────────

class SystemAdapter:
    name: str

    def prepare(self, events: list[dict]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, user_state: dict | None = None,
                 biometric_snapshot: dict | None = None, top_k: int = TOP_K) -> list[int]:
        raise NotImplementedError


class CosineAdapter(SystemAdapter):
    """Pure cosine similarity via Cohere embed-v4.0. Query-only (ignores state)."""
    name = "cosine"

    def __init__(self):
        self._vecs = None
        self._ids = None

    def prepare(self, events: list[dict]) -> None:
        texts = [e["text"] for e in events]
        self._ids = [e["id"] for e in events]
        self._vecs = embed_cohere(texts, "search_document")

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None, top_k=TOP_K) -> list[int]:
        q_vec = embed_cohere([query], "search_query")[0]
        sims = self._vecs @ q_vec
        order = np.argsort(-sims)[:top_k]
        return [self._ids[int(i)] for i in order]


class BM25Adapter(SystemAdapter):
    """BM25 lexical retrieval. Query-only."""
    name = "bm25"

    def __init__(self):
        self._bm25 = None
        self._ids = None

    def prepare(self, events: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        texts = [e["text"] for e in events]
        self._ids = [e["id"] for e in events]
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None, top_k=TOP_K) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [self._ids[int(i)] for i in order]


class HybridAdapter(SystemAdapter):
    """RRF merge of cosine + BM25. Query-only."""
    name = "hybrid"

    def __init__(self):
        self._cos = CosineAdapter()
        self._bm25 = BM25Adapter()

    def prepare(self, events: list[dict]) -> None:
        self._cos.prepare(events)
        self._bm25.prepare(events)

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None, top_k=TOP_K) -> list[int]:
        q_vec = embed_cohere([query], "search_query")[0]
        cos_sims = self._cos._vecs @ q_vec
        cos_order = [self._cos._ids[int(i)] for i in np.argsort(-cos_sims)]
        bm25_scores = self._bm25._bm25.get_scores(tokenize(query))
        bm_order = [self._bm25._ids[int(i)] for i in np.argsort(-bm25_scores)]
        merged = rrf_merge(cos_order, bm_order, 60)
        return merged[:top_k]


class PulseV2Adapter(SystemAdapter):
    """Pulse v2_pure baseline: cosine × recency. Query-only (ignores state)."""
    name = "pulse_v2"

    def __init__(self):
        self._cos = CosineAdapter()
        self._events = None

    def prepare(self, events: list[dict]) -> None:
        self._events = events
        self._cos.prepare(events)

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None, top_k=TOP_K) -> list[int]:
        q_vec = embed_cohere([query], "search_query")[0]
        sims = self._cos._vecs @ q_vec
        DECAY_LAMBDA = 0.002
        scored = []
        for i, ev in enumerate(self._events):
            days_ago = ev.get("days_ago", 0)
            recency = np.exp(-DECAY_LAMBDA * days_ago)
            score = float(sims[i]) * recency
            scored.append((score, self._cos._ids[i]))
        scored.sort(key=lambda x: -x[0])
        return [eid for _, eid in scored[:top_k]]


class PulseV3Adapter(SystemAdapter):
    """Pulse v3: v2_pure base + CONDITIONAL emotion/state/chain boosts.

    Accepts user_state (dict) and biometric_snapshot (dict), merges them into UserState,
    calls RetrievalV3.retrieve(). For chain tests (inferred from test context), expands
    via predecessor_ids. See retrieval_v3.py for formula details.

    Ablation: pass `ablation_kwargs` dict with enable_* booleans (e.g.
    {"enable_anchor_boost": False, ...}). See ABLATION_CONFIGS below for presets.
    """
    name = "pulse_v3"

    def __init__(self, beta: float = 0.15, gamma: float = 0.15,
                 use_llm_query_emo: bool = True,
                 ablation_kwargs: dict | None = None):
        self.beta = beta
        self.gamma = gamma
        self.use_llm_query_emo = use_llm_query_emo
        self.ablation_kwargs = ablation_kwargs or {}
        self._engine = None

    def prepare(self, events: list[dict]) -> None:
        sys.path.insert(0, str(BENCH_DIR / "external-evals" / "scripts"))
        from retrieval_v3 import RetrievalV3
        self._engine = RetrievalV3(
            events, beta=self.beta, gamma=self.gamma,
            use_llm_query_emo=self.use_llm_query_emo,
            **self.ablation_kwargs,
        )

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None,
                 top_k=TOP_K, expand_chain: bool = False) -> list[int]:
        from retrieval_v3 import UserState
        # Merge user_state dict + biometric_snapshot dict into UserState
        merged = {}
        if user_state:
            merged.update(user_state)
        if biometric_snapshot:
            # biometric fields override state if both present
            for k, v in biometric_snapshot.items():
                if k in ("hrv", "sleep_quality", "sleep_hours", "hr_trend",
                         "hrv_trend", "stress_proxy", "time_of_day"):
                    merged[k] = v

        mood_vector = merged.pop("mood_vector", {})
        state = UserState(
            mood_vector=mood_vector,
            sleep_quality=merged.get("sleep_quality"),
            sleep_hours=merged.get("sleep_hours"),
            hrv=merged.get("hrv"),
            hr_trend=merged.get("hr_trend"),
            hrv_trend=merged.get("hrv_trend"),
            stress_proxy=merged.get("stress_proxy"),
            recent_life_events_7d=merged.get("recent_life_events_7d", []),
            time_of_day=merged.get("time_of_day"),
        ) if (mood_vector or any(merged.get(k) is not None for k in
              ("sleep_quality", "hrv", "hr_trend", "hrv_trend", "stress_proxy"))) else None

        return self._engine.retrieve(query, user_state=state, top_k=top_k,
                                     expand_chain=expand_chain)


class PulseHybridAdapter(PulseV3Adapter):
    """Phase G hybrid Pulse: facts (factual mode) + state-aware sessions (empathic) +
    predecessor chain (chain mode), routed by query classifier.

    Reuses PulseV3Adapter.prepare() for events; additionally extracts atomic facts
    via fact_extractor.py (gpt-4o-mini) and indexes them on the engine.

    Optional kwargs:
      facts_path: path to JSONL of pre-extracted facts (skips re-extraction)
      fact_model: gpt-4o-mini default; pass other model for ablation
    """
    name = "pulse_hybrid"

    def __init__(self, beta: float = 0.15, gamma: float = 0.15,
                 use_llm_query_emo: bool = True,
                 ablation_kwargs: dict | None = None,
                 facts_path: Path | None = None,
                 fact_model: str = "gpt-4o-mini"):
        super().__init__(beta=beta, gamma=gamma,
                         use_llm_query_emo=use_llm_query_emo,
                         ablation_kwargs=ablation_kwargs)
        self.facts_path = facts_path
        self.fact_model = fact_model
        self._facts: list[dict] = []

    def prepare(self, events: list[dict]) -> None:
        super().prepare(events)
        sys.path.insert(0, str(BENCH_DIR / "external-evals" / "scripts"))

        # Load or extract facts
        if self.facts_path and Path(self.facts_path).exists():
            self._facts = []
            for line in Path(self.facts_path).open():
                if line.strip():
                    self._facts.append(json.loads(line))
            print(f"[pulse_hybrid] loaded {len(self._facts)} facts from "
                  f"{self.facts_path}", file=sys.stderr)
        else:
            from fact_extractor import FactExtractor
            extractor = FactExtractor(model=self.fact_model)
            print(f"[pulse_hybrid] extracting facts via {self.fact_model} for "
                  f"{len(events)} events…", file=sys.stderr)
            ef_list = extractor.extract_batch(events, stderr_print=True)
            self._facts = [f.to_dict() for f in ef_list]
            if self.facts_path:
                Path(self.facts_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(self.facts_path).open("w") as fout:
                    for f in self._facts:
                        fout.write(json.dumps(f, ensure_ascii=False) + "\n")
                print(f"[pulse_hybrid] cached facts to {self.facts_path}",
                      file=sys.stderr)

        # Index facts on the engine
        self._engine.index_facts(self._facts)

    def retrieve(self, query: str, user_state=None, biometric_snapshot=None,
                 top_k=TOP_K, expand_chain: bool = False) -> list[int]:
        from retrieval_v3 import UserState
        # Same UserState assembly as parent
        merged = {}
        if user_state:
            merged.update(user_state)
        if biometric_snapshot:
            for k, v in biometric_snapshot.items():
                if k in ("hrv", "sleep_quality", "sleep_hours", "hr_trend",
                         "hrv_trend", "stress_proxy", "time_of_day"):
                    merged[k] = v
        mood_vector = merged.pop("mood_vector", {})
        state = UserState(
            mood_vector=mood_vector,
            sleep_quality=merged.get("sleep_quality"),
            sleep_hours=merged.get("sleep_hours"),
            hrv=merged.get("hrv"),
            hr_trend=merged.get("hr_trend"),
            hrv_trend=merged.get("hrv_trend"),
            stress_proxy=merged.get("stress_proxy"),
            recent_life_events_7d=merged.get("recent_life_events_7d", []),
            time_of_day=merged.get("time_of_day"),
        ) if (mood_vector or any(merged.get(k) is not None for k in
              ("sleep_quality", "hrv", "hr_trend", "hrv_trend", "stress_proxy"))) else None

        # If caller forced expand_chain (existing bench-v3 chain test path), respect it
        forced_mode = "chain" if expand_chain else "auto"
        return self._engine.retrieve_hybrid(query, user_state=state,
                                            top_k=top_k, mode=forced_mode)


# Sprint D2.10: state-aware baselines added to address peer-review concern that
# query-only baselines (cosine/bm25/hybrid) return identical top-K across
# stateful pair variants by construction → make their stateful axis score
# near-zero by design rather than by retrieval quality.
# See external-evals/scripts/baselines_state_aware.py for adapter docs.
sys.path.insert(0, str(BENCH_DIR / "external-evals" / "scripts"))
from baselines_state_aware import (
    CosineStateAdapter,
    HybridStateAdapter,
    StateConcatOnlyAdapter,
)


ADAPTERS: dict[str, type[SystemAdapter]] = {
    "cosine": CosineAdapter,
    "bm25": BM25Adapter,
    "hybrid": HybridAdapter,
    "cosine_state": CosineStateAdapter,
    "hybrid_state": HybridStateAdapter,
    "state_concat_only": StateConcatOnlyAdapter,
    "pulse_v2": PulseV2Adapter,
    "pulse_v3": PulseV3Adapter,
    "pulse_hybrid": PulseHybridAdapter,
}


# Ablation presets for --ablation-config. Each entry toggles specific enable_*
# booleans on RetrievalV3 to isolate the contribution of each Phase-5 boost.
# Values passed through to PulseV3Adapter(ablation_kwargs=...) → RetrievalV3(**).
#
# Sanity property: `v2_pure` and `no_boosts` should produce bit-identical
# retrievals (within ~0.001 Cohere embedding noise) — integrity check.
ABLATION_CONFIGS: dict[str, dict] = {
    # Full Pulse v3 (default, current behaviour)
    "full": {},

    # Everything off → cosine × exp(-0.002·days), identical to v2_pure
    "no_boosts": {
        "enable_anchor_decay": False,
        "enable_anchor_boost": False,
        "enable_date_boost": False,
        "enable_emotion_boost": False,
        "enable_state_boost": False,
        "enable_emotion_hint_augment": False,
        "enable_temporal_keywords": False,
        "enable_chain_expansion": False,
    },

    # Same as no_boosts (integrity alias — should produce same scores)
    "v2_pure": {
        "enable_anchor_decay": False,
        "enable_anchor_boost": False,
        "enable_date_boost": False,
        "enable_emotion_boost": False,
        "enable_state_boost": False,
        "enable_emotion_hint_augment": False,
        "enable_temporal_keywords": False,
        "enable_chain_expansion": False,
    },

    # Only anchor-related boosts (Phase 5.1 + 5.4)
    "anchor_only": {
        "enable_anchor_decay": True,
        "enable_anchor_boost": True,
        "enable_date_boost": False,
        "enable_emotion_boost": False,
        "enable_state_boost": False,
        "enable_emotion_hint_augment": False,
        "enable_temporal_keywords": False,
        "enable_chain_expansion": False,
    },

    # Only emotion-related boosts (Phase 5.5 + query-emotion alignment)
    "emotion_only": {
        "enable_anchor_decay": False,
        "enable_anchor_boost": False,
        "enable_date_boost": False,
        "enable_emotion_boost": True,
        "enable_state_boost": False,
        "enable_emotion_hint_augment": True,
        "enable_temporal_keywords": False,
        "enable_chain_expansion": False,
    },

    # Only date-proximity boosts (Phase 5.2 + 5.6)
    "date_only": {
        "enable_anchor_decay": False,
        "enable_anchor_boost": False,
        "enable_date_boost": True,
        "enable_emotion_boost": False,
        "enable_state_boost": False,
        "enable_emotion_hint_augment": False,
        "enable_temporal_keywords": True,
        "enable_chain_expansion": False,
    },

    # Only body-state fit (Phase 3 body-stressed / body-restored)
    "state_only": {
        "enable_anchor_decay": False,
        "enable_anchor_boost": False,
        "enable_date_boost": False,
        "enable_emotion_boost": False,
        "enable_state_boost": True,
        "enable_emotion_hint_augment": False,
        "enable_temporal_keywords": False,
        "enable_chain_expansion": False,
    },
}


# ────────────────────────────────────────────────────────────────────────────
# Live progress — atomic-write current.json that a web UI can poll
# ────────────────────────────────────────────────────────────────────────────

class LiveProgress:
    """Thread-safe progress writer for /live dashboard.

    Writes `<out_dir>/current.json` atomically after every verdict arrives,
    via write-to-tmp + rename. Pollers see either the pre-update or post-update
    state — never torn JSON.

    When bench completes, `finalize()` renames current.json → final.json so
    pollers can stop.
    """

    def __init__(self, out_dir: Path, run_id: str, ablation_config: str,
                 systems: list[str], judges: list[str], tests: list[dict]):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._recent: list[dict] = []

        self.state = {
            "run_id": run_id,
            "ablation_config": ablation_config,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "systems": systems,
            "judges": judges,
            "n_tests": len(tests),
            "progress": {
                "tests_done": 0,
                "tests_total": len(tests),
                "judge_calls_done": 0,
                "judge_calls_total": 0,  # set by set_judge_plan()
            },
            "tests": [
                {
                    "test_id": t["id"],
                    "test_name": t["name"],
                    "test_type": t.get("test_type", "core"),
                    "user_query": t["user_query"],
                    "user_state": t.get("user_state") or t.get("user_state_overlay"),
                    "biometric_snapshot": t.get("biometric_snapshot"),
                    "ideal": t.get("ideal_top_3_event_ids") or t.get("ideal_chain"),
                    "status": "pending",
                    "retrievals": {},
                    "verdicts": {},
                    "scores": {},
                    "pair_id": t.get("pair_id"),
                }
                for t in tests
            ],
            "recent_verdicts": [],
        }
        self._write()

    def set_judge_plan_size(self, total_calls: int) -> None:
        with self.lock:
            self.state["progress"]["judge_calls_total"] = int(total_calls)
            self._write()

    def set_retrievals(self, tid: str, retrievals_by_system: dict) -> None:
        with self.lock:
            for t in self.state["tests"]:
                if t["test_id"] == tid:
                    t["retrievals"] = retrievals_by_system
                    # keep status "pending" until first verdict arrives
                    break
            self._write()

    def add_verdict(self, tid: str, judge: str, verdict: dict,
                    scores: dict | None = None) -> None:
        with self.lock:
            for t in self.state["tests"]:
                if t["test_id"] == tid:
                    t["verdicts"][judge] = verdict
                    if t["status"] == "pending":
                        t["status"] = "running"
                    if scores:
                        t["scores"] = scores
                    break
            self.state["progress"]["judge_calls_done"] += 1

            # Stream of recent verdicts for the bottom ticker (last 10)
            brief = {
                "test_id": tid,
                "judge": judge,
                "winner": verdict.get("winner", "-"),
                "note": (verdict.get("note") or "")[:160],
                "at": datetime.now(timezone.utc).isoformat(),
            }
            self._recent.append(brief)
            if len(self._recent) > 10:
                self._recent = self._recent[-10:]
            self.state["recent_verdicts"] = list(self._recent)

            self._write()

    def mark_test_done(self, tid: str, scores: dict | None = None) -> None:
        with self.lock:
            for t in self.state["tests"]:
                if t["test_id"] == tid:
                    t["status"] = "done"
                    if scores:
                        t["scores"] = scores
                    break
            self.state["progress"]["tests_done"] = sum(
                1 for t in self.state["tests"] if t["status"] == "done"
            )
            self._write()

    def finalize(self) -> None:
        with self.lock:
            self.state["status"] = "complete"
            self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._write()
            # Rename so pollers can stop
            cur = self.out_dir / "current.json"
            fin = self.out_dir / "final.json"
            if cur.exists():
                cur.replace(fin)

    def _write(self) -> None:
        """Atomic write to current.json (tmp + rename)."""
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.out_dir / "current.json.tmp"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, default=str))
        tmp.replace(self.out_dir / "current.json")


class BalanceError(RuntimeError):
    """Raised when an API provider rejects the call due to balance / quota /
    rate-limit / auth issue. Halts the run so we don't accumulate partial data.
    See feedback_stop_on_api_balance_failure.md — we explicitly refuse to
    fallback, retry, or aggregate partial data on balance failures."""


_BALANCE_ERROR_MARKERS = (
    "insufficient balance", "insufficient_balance",
    "payment required", "quota exceeded",
    "http error 402", "http error 401",
    "exceeds the quota", "out of credit",
    "subscription has expired", "suspended due to",
    "billing issue",
)
# Note: HTTP 429 (rate-limit / concurrency cap on Z.ai) is intentionally
# NOT in this list. 429 is transient — handled by retry-with-backoff in
# call_judge. Halt only for actual balance / auth failures.


def _looks_like_balance_error(text: str) -> bool:
    """Detect balance / quota / auth failures ONLY in error-response strings.

    `call_judge` returns either:
      - successful model content (which may legitimately contain words like
        "402" or "billing" inside a judge's note — must NOT match here),
      - or a JSON-shaped error envelope `{"error": "http: HTTP Error 402 ..."}`
        when urlopen raised (these are the only responses we want to trip on).

    We only check the first ~240 chars AND only when the response starts with
    the error envelope — this avoids substring collisions with legitimate
    scoring reasoning.
    """
    if not text:
        return False
    head = text.strip()[:240].lower()
    # Only trip if the response clearly *is* an error envelope, not a
    # legit judge verdict that happens to mention a number/word.
    if not head.startswith('{"error"'):
        return False
    return any(m in head for m in _BALANCE_ERROR_MARKERS)


class RunCheckpoint:
    """Append-only JSONL checkpoint so a crashed / killed bench run can
    resume without re-issuing completed judge calls.

    Each line: {"test_id": "T1", "judge": "kimi", "kind": "test", "verdict": {...}, "at": "..."}.

    Thread-safe: one lock protects both the in-memory done-set and the file
    handle. `fsync()` on every append so checkpoint survives hard kills.

    Integration:
      1. On startup: `verdicts = ckpt.replay_verdicts()` pre-populates already-scored
         (test_id, judge) pairs.
      2. In judge_plan loop: `if ckpt.already_done(tid, jp): skip`.
      3. After each `f.result()`: `ckpt.append(tid, jp, verdict, kind)`.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._done: set[tuple[str, str]] = set()
        self._loaded = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._done.add((obj["test_id"], obj["judge"]))
                    self._loaded += 1
                except Exception:
                    continue
        if self._loaded:
            print(f"[checkpoint] resumed {self._loaded} verdicts from {self.path}",
                  file=sys.stderr)

    def replay_verdicts(self) -> dict[str, dict[str, dict]]:
        """Replay checkpoint into a {test_id → judge → verdict} dict."""
        result: dict[str, dict[str, dict]] = {}
        if not self.path.exists():
            return result
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    result.setdefault(obj["test_id"], {})[obj["judge"]] = obj["verdict"]
                except Exception:
                    continue
        return result

    def already_done(self, tid: str, judge: str) -> bool:
        return (tid, judge) in self._done

    def n_loaded(self) -> int:
        return self._loaded

    def append(self, tid: str, judge: str, verdict: dict, kind: str = "test") -> None:
        with self.lock:
            line = json.dumps({
                "test_id": tid,
                "judge": judge,
                "kind": kind,
                "verdict": verdict,
                "at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._done.add((tid, judge))


# ────────────────────────────────────────────────────────────────────────────
# Chain order — deterministic scoring
# ────────────────────────────────────────────────────────────────────────────

def kendall_tau_normalized(retrieved_order: list[int], ideal_chain: list[int]) -> tuple[float, float]:
    """Return (tau, recall) where tau = normalized Kendall distance on overlap
    and recall = len(overlap) / len(ideal_chain). Chain score = 10 · tau · recall.

    tau = 1 - 2 * inversions / max(1, n*(n-1)), in [0, 1] where 1 = fully sorted.
    """
    pos = {eid: i for i, eid in enumerate(ideal_chain)}
    overlap = [e for e in retrieved_order if e in pos]
    recall = len(overlap) / max(1, len(ideal_chain))

    n = len(overlap)
    if n < 2:
        # Cannot compute tau; return recall-only score
        return (1.0 if n == 1 else 0.0), recall

    # Count inversions in the mapped ranks of overlap
    ranks = [pos[e] for e in overlap]
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] > ranks[j]:
                inversions += 1

    max_inv = n * (n - 1) / 2
    tau = 1.0 - (inversions / max_inv) if max_inv else 1.0
    return tau, recall


def chain_score(retrieved_order: list[int], ideal_chain: list[int]) -> float:
    """Deterministic chain order score, 0-10."""
    tau, recall = kendall_tau_normalized(retrieved_order, ideal_chain)
    return 10.0 * tau * recall


# ────────────────────────────────────────────────────────────────────────────
# Judge prompts per test type
# ────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_CORE = """You are an impartial judge evaluating empathic memory retrieval systems.
You score retrieved event sets on three axes (0-10 each):
  rel  — how relevant the retrieved set is to the user query AND emotional context
  spec — how specific (vs generic) the retrieved events are
  act  — whether the retrieved context enables a companion to act warmly and safely

Output a single JSON object: {"S01_rel": 7, "S01_spec": 6, "S01_act": 8, ..., "winner": "S01", "note": "short reasoning"}
Return ONLY the JSON. No markdown fences. No commentary."""


JUDGE_SYSTEM_STATEFUL = JUDGE_SYSTEM_CORE + """

ADDITIONAL AXIS for STATEFUL TESTS:
  {sid}_stateful — how appropriately the retrieved set CHANGED between the two user_state variants (0-10).
    0  = retrieved sets identical across variants (no state sensitivity at all)
    5  = sets differ but differences don't reflect the state shift meaningfully
    10 = sets differ in exactly the way an empathic companion would approve given the state shift
  Judge each system independently; include _stateful per system."""


JUDGE_SYSTEM_MULTI = JUDGE_SYSTEM_CORE + """

ADDITIONAL AXIS for MULTI-SIGNAL TESTS:
  {sid}_multi_signal — does the retrieved set integrate the biometric_snapshot? (0-10)
    0  = ignores biometric signal entirely
    5  = semantically relevant but biometric ignored
    10 = retrieved cluster matches the biometric profile (e.g. depletion bio → body-cost events)
  When biometric_snapshot is neutral (normal HRV, good sleep, no strong emotion),
  a 10 means system correctly FELL BACK to pure semantic retrieval."""


def build_user_prompt_core(test: dict, results: dict[str, list[dict]],
                            blind_map: dict[str, str]) -> str:
    """Core-axis user prompt. Used for test_type in {core, chain}."""
    items = sorted(blind_map.items(), key=lambda kv: kv[1])
    blocks = []
    for real_code, blind in items:
        rs = results.get(real_code, [])
        lines = [f"=== {blind} ==="]
        for i, r in enumerate(rs, 1):
            lines.append(f"[{i}] id={r['id']} {r['text'][:300]}")
        blocks.append("\n".join(lines) if rs else f"=== {blind} ===\n(empty)")
    ideal = test.get("ideal_top_3_event_ids") or test.get("ideal_chain", [])
    return f"""Test: {test['name']}
Type: {test['test_type']}
User query: "{test['user_query']}"

What this test checks: {test.get('what_it_tests','')}

Ideal event IDs: {ideal}
Why: {test.get('ideal_explanation','')}

Failure modes to penalize:
{chr(10).join('- ' + fm for fm in test.get('fail_modes', []))}

{chr(10).join(blocks)}

Score EACH system on rel/spec/act (0-10). Pick a winner. Return JSON only."""


def build_user_prompt_stateful(test_A: dict, test_B: dict,
                                results_A: dict[str, list[dict]],
                                results_B: dict[str, list[dict]],
                                blind_map: dict[str, str]) -> str:
    items = sorted(blind_map.items(), key=lambda kv: kv[1])
    def _block(label: str, results: dict[str, list[dict]]):
        parts = [label]
        for real, blind in items:
            rs = results.get(real, [])
            lines = [f"  --- {blind} ---"]
            for i, r in enumerate(rs, 1):
                lines.append(f"  [{i}] id={r['id']} {r['text'][:250]}")
            parts.append("\n".join(lines) if rs else f"  --- {blind} ---\n  (empty)")
        return "\n".join(parts)

    return f"""STATEFUL PAIRED TEST — pair_id: {test_A.get('pair_id')}
Query (same for both variants): "{test_A['user_query']}"

=== VARIANT A ({test_A['name']}) ===
user_state: {json.dumps(test_A['user_state'], ensure_ascii=False)}
ideal_top_3: {test_A['ideal_top_3_event_ids']}
why: {test_A.get('ideal_explanation','')}

=== VARIANT B ({test_B['name']}) ===
user_state: {json.dumps(test_B['user_state'], ensure_ascii=False)}
ideal_top_3: {test_B['ideal_top_3_event_ids']}
why: {test_B.get('ideal_explanation','')}

What this pair tests: {test_A.get('what_it_tests','')}

{_block('### RETRIEVED FOR VARIANT A ###', results_A)}

{_block('### RETRIEVED FOR VARIANT B ###', results_B)}

For each system, score on:
  - {{sid}}_rel, {{sid}}_spec, {{sid}}_act — for the UNION of both variants' retrieved sets
  - {{sid}}_stateful — how appropriately the set CHANGED between variant A and B (0-10)

A system returning identical top-3 for both → {{sid}}_stateful = 0.
A system returning ideal_top_3 for each variant → {{sid}}_stateful = 10.
Winner = system with best combined core + stateful. Return JSON only."""


def build_user_prompt_multi(test: dict, results: dict[str, list[dict]],
                             blind_map: dict[str, str]) -> str:
    items = sorted(blind_map.items(), key=lambda kv: kv[1])
    blocks = []
    for real, blind in items:
        rs = results.get(real, [])
        lines = [f"=== {blind} ==="]
        for i, r in enumerate(rs, 1):
            lines.append(f"[{i}] id={r['id']} {r['text'][:300]}")
        blocks.append("\n".join(lines) if rs else f"=== {blind} ===\n(empty)")
    return f"""MULTI-SIGNAL TEST
Test: {test['name']}
User query: "{test['user_query']}"
Biometric snapshot: {json.dumps(test['biometric_snapshot'], ensure_ascii=False)}
{('User-state overlay: ' + json.dumps(test['user_state_overlay'], ensure_ascii=False)) if test.get('user_state_overlay') else ''}

What this test checks: {test.get('what_it_tests','')}

Ideal event IDs: {test['ideal_top_3_event_ids']}
Why: {test.get('ideal_explanation','')}

Failure modes to penalize:
{chr(10).join('- ' + fm for fm in test.get('fail_modes', []))}

{chr(10).join(blocks)}

For each system, score: rel/spec/act (0-10 each) + {{sid}}_multi_signal (0-10).
If biometric_snapshot is neutral (HRV ~70, good sleep, low stress), a 10 on multi_signal
means system correctly fell back to pure semantic/salience retrieval.
Return JSON only."""


# ────────────────────────────────────────────────────────────────────────────
# Judge callers (OpenAI-compatible — Kimi/GLM)
# ────────────────────────────────────────────────────────────────────────────

_TRANSIENT_HTTP_CODES = ("429", "500", "502", "503", "504")


def call_judge(provider: str, model: str, system: str, user: str,
               max_tokens: int = 8000, timeout: int = 240,
               max_retries: int = 4) -> str:
    """OpenAI-compatible chat completions caller.

    Kimi K2.6 reasoning model → 8000+ max_tokens. gpt-5 family uses max_completion_tokens.

    Retries with exponential backoff (10s, 20s, 40s) on transient errors:
    - HTTP 429 (rate limit / concurrency cap): Z.ai uses concurrent-request limits;
      with parallel judges these can briefly trip and need a backoff.
    - HTTP 5xx (server errors): transient.
    - Timeout: transient.

    Non-transient errors (401 auth, 402 payment, etc.) propagate immediately
    so the BalanceError detector at the call site can halt the run cleanly.
    """
    from urllib import request as urlreq
    cfg = PROVIDERS[provider]
    api_key = secret(cfg["key_file"])
    # gpt-5* models need max_completion_tokens; other OpenAI-compat APIs use max_tokens
    payload_body = {
        "model": model,
        "temperature": 1.0,  # Kimi K2.6 requires 1.0; OpenAI ignores if not supported
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith("gpt-5"):
        payload_body["max_completion_tokens"] = max_tokens
    else:
        payload_body["max_tokens"] = max_tokens
    payload = json.dumps(payload_body).encode("utf-8")
    req = urlreq.Request(
        f"{cfg['base_url']}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            with urlreq.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            # Reasoning-content fallback: Kimi puts chain-of-thought there and JSON may be inside too
            if not text.strip() and msg.get("reasoning_content"):
                text = msg["reasoning_content"]
            return text
        except Exception as ex:
            last_err = ex
            err_str = str(ex).lower()
            # D2.10b: silent vendor-direct fallback for DO judges where the
            # caller's PAT is locked behind a higher subscription tier (HTTP
            # 401). Other DO judges (sonnet-4.6, haiku-4.5) continue working,
            # so we only fall back when DO_VENDOR_FALLBACK has a mapping.
            if (
                provider in DO_VENDOR_FALLBACK
                and ("401" in err_str or "unauthorized" in err_str)
            ):
                print(f"    [vendor-fallback] {provider} got 401 on DO — "
                      f"routing via {DO_VENDOR_FALLBACK[provider]['vendor']}",
                      file=sys.stderr)
                return call_vendor_fallback(provider, system, user,
                                            max_tokens=max_tokens,
                                            timeout=timeout)
            is_transient = (
                any(code in err_str for code in _TRANSIENT_HTTP_CODES)
                or "timeout" in err_str
                or "timed out" in err_str
                or "connection reset" in err_str
            )
            if is_transient and attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s
                print(f"    [retry {attempt + 1}/{max_retries}] {provider} transient error "
                      f"({str(ex)[:80]}), backing off {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            # Non-transient OR exhausted retries — return error envelope
            break
    return f'{{"error": "http: {str(last_err)[:200]}"}}'


def extract_verdict_json(text: str) -> dict | None:
    """Find the last valid JSON object in text. Handles reasoning traces + code fences."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and any(isinstance(k, str) and k.startswith("S") for k in obj):
                return obj
        except Exception:
            continue
    return None


# ────────────────────────────────────────────────────────────────────────────
# Per-test scoring
# ────────────────────────────────────────────────────────────────────────────

def score_core_from_verdict(verdict: dict, blind_code: str) -> dict:
    """Extract rel/spec/act from a deblinded verdict for one system."""
    return {
        "rel": float(verdict.get(f"{blind_code}_rel", 0)),
        "spec": float(verdict.get(f"{blind_code}_spec", 0)),
        "act": float(verdict.get(f"{blind_code}_act", 0)),
    }


def deblind_verdict(blind_verdict: dict, blind_map: dict[str, str]) -> dict:
    """Convert S01_rel → systemname_rel using blind_map."""
    reverse = {v: k for k, v in blind_map.items()}
    out = {}
    for k, v in blind_verdict.items():
        if k in ("note", "error"):
            out[k] = v
            continue
        if k == "winner":
            out["winner"] = reverse.get(v, v)
            continue
        parts = k.split("_", 1)
        if len(parts) == 2 and parts[0] in reverse:
            out[f"{reverse[parts[0]]}_{parts[1]}"] = v
        else:
            out[k] = v
    return out


def weighted_test_score(test_type: str, core_avg: float | None, stateful: float | None,
                        chain: float | None, multi: float | None) -> float:
    """Per-test weighted score 0-10. Redistributes weights across available axes."""
    if test_type == "core":
        return (core_avg or 0) / 3.0
    if test_type == "stateful":
        core = (core_avg or 0) / 3.0
        return 0.70 * core + 0.30 * (stateful or 0)
    if test_type == "chain":
        core = (core_avg or 0) / 3.0
        return 0.50 * (chain or 0) + 0.50 * core
    if test_type == "multi_signal":
        core = (core_avg or 0) / 3.0
        return 0.60 * (multi or 0) + 0.40 * core
    return 0


# ────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ────────────────────────────────────────────────────────────────────────────

def make_blind_mapping(system_names: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    blinds = [f"S{i+1:02d}" for i in range(len(system_names))]
    rng.shuffle(blinds)
    return dict(zip(system_names, blinds))


def events_by_id(events: list[dict]) -> dict[int, dict]:
    return {e["id"]: e for e in events}


def run_retrieval(systems: dict[str, SystemAdapter], test: dict,
                  eid_map: dict[int, dict]) -> dict[str, list[dict]]:
    """Run every system's retrieve() on this test; return per-system list of event dicts."""
    out = {}
    q = test["user_query"]
    user_state = test.get("user_state") or test.get("user_state_overlay")
    bio = test.get("biometric_snapshot")
    is_chain = test.get("test_type") == "chain"
    # Phase 5.8 tested top_k = len(ideal_chain) for chain tests but it uplifted
    # baselines (cosine/hybrid chain averages) more than pulse_v3 (which was
    # already close to ceiling for cosine-reachable chain events). Net-gap
    # narrowed. Kept TOP_K=3 everywhere for fair comparison with paper bench-v2.
    for name, sys_a in systems.items():
        # Chain expansion only supported by pulse_v3 adapter; others ignore it silently
        try:
            ids = sys_a.retrieve(q, user_state=user_state, biometric_snapshot=bio,
                                 top_k=TOP_K, expand_chain=is_chain)
        except TypeError:
            ids = sys_a.retrieve(q, user_state=user_state, biometric_snapshot=bio, top_k=TOP_K)
        out[name] = [eid_map[i] for i in ids if i in eid_map]
    return out


def pair_stateful_tests(tests: list[dict]) -> list[tuple[dict, dict]]:
    """Group stateful tests by pair_id → return list of (A, B) pairs."""
    pairs: dict[str, list[dict]] = {}
    for t in tests:
        if t.get("test_type") == "stateful" and t.get("pair_id"):
            # Pair id like "P1-anger" / "P1-shame" → base "P1"
            base = t["pair_id"].split("-")[0]
            pairs.setdefault(base, []).append(t)
    paired = []
    for base, ts in pairs.items():
        if len(ts) == 2:
            paired.append((ts[0], ts[1]))
    return paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", type=str, default="cosine,bm25,hybrid",
                    help=f"Comma-separated systems from {list(ADAPTERS)}; 'all' for all")
    ap.add_argument("--tests", type=str, default="",
                    help="Comma-separated test IDs (e.g. T6,T7) — default all")
    ap.add_argument("--judges", type=str, default="kimi",
                    help=f"Comma-separated judge providers from {list(PROVIDERS)}")
    ap.add_argument("--judge-model", type=str, default="")
    ap.add_argument("--dry-run", action="store_true", help="Skip judge calls; score only chain order")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parallel-judges", type=int, default=3)
    ap.add_argument("--snapshot", type=str, default="",
                    help="If set, write snapshot dir 'snapshots/<snapshot>/'")
    ap.add_argument("--ablation-config", type=str, default="full",
                    choices=list(ABLATION_CONFIGS),
                    help="Pulse v3 ablation preset: full (default) | no_boosts | "
                         "v2_pure | anchor_only | emotion_only | date_only | state_only")
    ap.add_argument("--embedding", type=str, default="cohere",
                    choices=["cohere", "bge-m3"],
                    help="Embedding provider for Pulse v3 retrieval. "
                         "'cohere' = embed-v4.0 ($0.10/M, D=1536, default). "
                         "'bge-m3' = DO inference bge-m3 ($0.02/M, D=1024, "
                         "multilingual, MIT). Sprint D1.8 ablation.")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to append-only JSONL checkpoint. Defaults to "
                         "results/checkpoints/<snapshot-or-auto>.jsonl. "
                         "On re-run, already-scored (test_id, judge) pairs are "
                         "skipped — so a killed run can resume exactly where "
                         "it stopped.")
    args = ap.parse_args()

    # --- Embedding provider toggle (Sprint D1.8) ---
    # retrieval_v3.py reads EMBEDDING_PROVIDER env var via embedding_provider.py.
    # Setting here keeps baseline systems (cosine/bm25/hybrid/pulse_v2) untouched
    # — they import embed_cohere directly from common and ignore this var.
    os.environ["EMBEDDING_PROVIDER"] = args.embedding
    if args.embedding != "cohere":
        print(f"[embedding] Pulse v3 retrieval using provider: {args.embedding}",
              file=sys.stderr)

    # --- Load corpus ---
    data = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    events = data["events"]
    all_tests = data["tests"]
    eid_map = events_by_id(events)

    if args.tests:
        tid_set = set(args.tests.split(","))
        all_tests = [t for t in all_tests if t["id"] in tid_set]
    print(f"Loaded {len(events)} events, {len(all_tests)} tests from {CORPUS_FILE.name}", file=sys.stderr)

    # --- Init systems ---
    sys_names = list(ADAPTERS) if args.systems == "all" else args.systems.split(",")
    systems: dict[str, SystemAdapter] = {}
    ablation_kwargs = ABLATION_CONFIGS.get(args.ablation_config, {})
    if args.ablation_config != "full" and "pulse_v3" in sys_names:
        print(f"[ablation] pulse_v3 running with preset '{args.ablation_config}': "
              f"{ablation_kwargs}", file=sys.stderr)
    for n in sys_names:
        if n not in ADAPTERS:
            sys.exit(f"Unknown system: {n}. Available: {list(ADAPTERS)}")
        if n == "pulse_v3" and ablation_kwargs:
            a = ADAPTERS[n](ablation_kwargs=ablation_kwargs)
        else:
            a = ADAPTERS[n]()
        print(f"Preparing {n}...", file=sys.stderr)
        a.prepare(events)
        systems[n] = a

    # --- Init live progress tracker for /live webapp page ---
    live_dir = BENCH_DIR / "external-evals" / "results" / "bench-v3-live"
    run_id = datetime.now(timezone.utc).isoformat()
    live = LiveProgress(
        out_dir=live_dir,
        run_id=run_id,
        ablation_config=args.ablation_config,
        systems=list(systems),
        judges=args.judges.split(",") if args.judges else [],
        tests=all_tests,
    )

    # --- Run retrieval for every test ---
    retrievals: dict[str, dict[str, list[dict]]] = {}
    for i, t in enumerate(all_tests, 1):
        print(f"[retrieve] {i}/{len(all_tests)} {t['id']} ({t.get('test_type','?')}) {t['name']}",
              file=sys.stderr)
        retrievals[t["id"]] = run_retrieval(systems, t, eid_map)
        live.set_retrievals(t["id"], {sn: [r["id"] for r in retrievals[t["id"]][sn]]
                                       for sn in systems})

    # --- Compute deterministic chain scores ---
    chain_scores: dict[str, dict[str, float]] = {}  # test_id → system → score
    for t in all_tests:
        if t.get("test_type") == "chain":
            chain_scores[t["id"]] = {}
            for sn in systems:
                rids = [r["id"] for r in retrievals[t["id"]][sn]]
                chain_scores[t["id"]][sn] = chain_score(rids, t["ideal_chain"])

    # --- Judge calls (unless --dry-run) ---
    judge_providers = args.judges.split(",") if args.judges else []

    # --- Checkpoint: append-only JSONL, resume if file exists ---
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_tag = args.snapshot or f"auto-{args.ablation_config}"
        ckpt_path = RESULTS_DIR / "checkpoints" / f"{ckpt_tag}.jsonl"
    ckpt = RunCheckpoint(ckpt_path)

    # Restore previously-completed verdicts (if any)
    verdicts: dict[str, dict[str, dict]] = ckpt.replay_verdicts()

    blind_maps: dict[str, dict[str, str]] = {}
    paired = pair_stateful_tests(all_tests)
    paired_ids = {p[0]["id"]: p for p in paired} | {p[1]["id"]: p for p in paired}

    # Re-hydrate live dashboard with resumed verdicts so webapp shows
    # partial progress from a prior run (not just new ones).
    for tid, per_judge in verdicts.items():
        for jp, v in per_judge.items():
            live.add_verdict(tid, jp, v)

    judge_plan = []  # list[(test_or_pair_marker, judge_provider)]
    skipped = 0
    for t in all_tests:
        if t.get("test_type") == "stateful":
            # Only schedule judge on the first member of each pair (A)
            pair = paired_ids.get(t["id"])
            if pair and pair[0]["id"] == t["id"]:
                for jp in judge_providers:
                    if ckpt.already_done(t["id"], jp):
                        skipped += 1
                        continue
                    judge_plan.append((("pair", pair), jp))
            continue
        if t.get("test_type") == "chain":
            # Chain still gets a judge for rel/spec/act
            for jp in judge_providers:
                if ckpt.already_done(t["id"], jp):
                    skipped += 1
                    continue
                judge_plan.append((("test", t), jp))
            continue
        for jp in judge_providers:
            if ckpt.already_done(t["id"], jp):
                skipped += 1
                continue
            judge_plan.append((("test", t), jp))

    if skipped:
        print(f"[checkpoint] skipping {skipped} already-completed judge calls "
              f"(resumed from {ckpt_path})", file=sys.stderr)

    def _do_judge(marker, jp):
        kind = marker[0]
        model = args.judge_model or PROVIDERS[jp]["default_model"]
        if kind == "test":
            t = marker[1]
            if t.get("test_type") == "multi_signal":
                system_prompt = JUDGE_SYSTEM_MULTI
                user_prompt_fn = build_user_prompt_multi
            else:
                system_prompt = JUDGE_SYSTEM_CORE
                user_prompt_fn = build_user_prompt_core
            blind = make_blind_mapping(list(systems), seed=args.seed + hash(t["id"]) % 1000)
            blind_maps[t["id"]] = blind
            user = user_prompt_fn(t, retrievals[t["id"]], blind)
            text = call_judge(jp, model, system_prompt, user)
            # Halt on balance/quota errors — see feedback_stop_on_api_balance_failure.md
            if _looks_like_balance_error(text):
                raise BalanceError(f"{jp}: {text[:300]}")
            bv = extract_verdict_json(text) or {"error": "no_json", "raw": text[:300]}
            return ("test", t["id"], jp, deblind_verdict(bv, blind))
        else:
            pair = marker[1]
            tA, tB = pair
            blind = make_blind_mapping(list(systems), seed=args.seed + hash(tA.get("pair_id","")) % 1000)
            blind_maps[tA["id"]] = blind
            blind_maps[tB["id"]] = blind
            user = build_user_prompt_stateful(tA, tB,
                                              retrievals[tA["id"]], retrievals[tB["id"]],
                                              blind)
            text = call_judge(jp, model, JUDGE_SYSTEM_STATEFUL, user)
            if _looks_like_balance_error(text):
                raise BalanceError(f"{jp}: {text[:300]}")
            bv = extract_verdict_json(text) or {"error": "no_json", "raw": text[:300]}
            return ("pair", tA["id"], jp, deblind_verdict(bv, blind))

    if not args.dry_run and judge_plan:
        live.set_judge_plan_size(len(judge_plan) + skipped)
        # Reflect already-completed work in the progress bar
        for _ in range(skipped):
            # not using add_verdict because those were already added from replay above;
            # but we do need to move the counter. add_verdict already incremented it during
            # the re-hydrate loop, so skip adjustments here.
            pass
        print(f"[judge] {len(judge_plan)} judge calls to run "
              f"(parallel={args.parallel_judges}, skipped {skipped} from checkpoint)",
              file=sys.stderr)
        done = 0
        halted = False
        with ThreadPoolExecutor(max_workers=args.parallel_judges) as pool:
            futs = [pool.submit(_do_judge, m, jp) for (m, jp) in judge_plan]
            try:
                for f in as_completed(futs):
                    try:
                        kind, tid, jp, v = f.result()
                    except BalanceError as be:
                        halted = True
                        print(f"\n[HALT — balance] {be}", file=sys.stderr)
                        print("Cancelling remaining judge calls. Checkpoint is on "
                              f"disk at {ckpt_path} — re-run the same command "
                              "to resume.", file=sys.stderr)
                        for other in futs:
                            other.cancel()
                        break
                    verdicts.setdefault(tid, {})[jp] = v
                    # Persist to checkpoint BEFORE logging so a crash after the
                    # print() still leaves the verdict on disk.
                    ckpt.append(tid, jp, v, kind=kind)
                    done += 1
                    err = v.get("error", "")
                    winner = v.get("winner", "-")
                    print(f"  [{done}/{len(judge_plan)}] {kind} {tid} {jp} → winner={winner} {err[:40]}",
                          file=sys.stderr)
                    # Live update so the webapp sees the verdict arrive in real time.
                    # For stateful pairs the verdict is scheduled only on the "A" test;
                    # copy it to "B" for UI completeness.
                    live.add_verdict(tid, jp, v)
                    if kind == "pair":
                        # tid is tA["id"]; find the paired tB and mirror the verdict there
                        pair = paired_ids.get(tid)
                        if pair and pair[1]["id"] != tid:
                            live.add_verdict(pair[1]["id"], jp, v)
            finally:
                pass
        if halted:
            live.finalize()
            print("Partial data preserved in checkpoint; no aggregation / snapshot written.",
                  file=sys.stderr)
            sys.exit(2)

    # --- Aggregate ---
    out: dict = {
        "_meta": {
            "bench": "empathic-memory-v3",
            "version": 3,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus": CORPUS_FILE.name,
            "n_events": len(events),
            "n_tests": len(all_tests),
            "systems": list(systems),
            "judges": judge_providers,
            "dry_run": args.dry_run,
        },
        "tests": [],
        "chain_scores": chain_scores,
    }

    # Per-test records + per-system aggregates
    per_system_scores: dict[str, list[float]] = {sn: [] for sn in systems}
    per_system_axes: dict[str, dict[str, list[float]]] = {
        sn: {"core": [], "stateful": [], "chain": [], "multi_signal": []} for sn in systems
    }

    for t in all_tests:
        tid = t["id"]
        tt = t.get("test_type", "core")
        rec = {
            "test_id": tid,
            "test_name": t["name"],
            "test_type": tt,
            "user_query": t["user_query"],
            "ideal": t.get("ideal_top_3_event_ids") or t.get("ideal_chain"),
            "retrievals": {sn: [r["id"] for r in retrievals[tid][sn]] for sn in systems},
        }
        vlist = verdicts.get(tid, {})
        rec["verdicts"] = vlist

        # Per-system per-test scoring
        per_sys: dict[str, dict] = {}
        for sn in systems:
            # Core avg (rel+spec+act) from available judges
            rels, specs, acts, statefuls, multis = [], [], [], [], []
            for j, v in vlist.items():
                if "error" in v: continue
                if f"{sn}_rel" in v: rels.append(float(v[f"{sn}_rel"]))
                if f"{sn}_spec" in v: specs.append(float(v[f"{sn}_spec"]))
                if f"{sn}_act" in v: acts.append(float(v[f"{sn}_act"]))
                if f"{sn}_stateful" in v: statefuls.append(float(v[f"{sn}_stateful"]))
                if f"{sn}_multi_signal" in v: multis.append(float(v[f"{sn}_multi_signal"]))
            core_sum = None
            if rels and specs and acts:
                core_sum = (np.mean(rels) + np.mean(specs) + np.mean(acts))
            stateful_v = float(np.mean(statefuls)) if statefuls else None
            multi_v = float(np.mean(multis)) if multis else None
            chain_v = chain_scores.get(tid, {}).get(sn) if tt == "chain" else None

            score = weighted_test_score(tt, core_sum, stateful_v, chain_v, multi_v)

            per_sys[sn] = {
                "rel": np.mean(rels) if rels else None,
                "spec": np.mean(specs) if specs else None,
                "act": np.mean(acts) if acts else None,
                "core_sum": core_sum,
                "stateful": stateful_v,
                "chain": chain_v,
                "multi_signal": multi_v,
                "weighted": score,
            }
            per_system_scores[sn].append(score)
            if tt == "core" or tt == "multi_signal" or tt == "stateful":
                if core_sum is not None:
                    per_system_axes[sn]["core"].append(core_sum / 3.0)
            if tt == "stateful" and stateful_v is not None:
                per_system_axes[sn]["stateful"].append(stateful_v)
            if tt == "chain" and chain_v is not None:
                per_system_axes[sn]["chain"].append(chain_v)
            if tt == "multi_signal" and multi_v is not None:
                per_system_axes[sn]["multi_signal"].append(multi_v)

        rec["scores"] = per_sys
        out["tests"].append(rec)
        # Update live dashboard: test is now done with aggregated scores
        live.mark_test_done(tid, per_sys)

    # Overall aggregate
    out["aggregate"] = {}
    for sn in systems:
        scores = per_system_scores[sn]
        ax = per_system_axes[sn]
        out["aggregate"][sn] = {
            "overall_weighted_mean": float(np.mean(scores)) if scores else 0.0,
            "n_tests": len(scores),
            "core_mean": float(np.mean(ax["core"])) if ax["core"] else None,
            "stateful_mean": float(np.mean(ax["stateful"])) if ax["stateful"] else None,
            "chain_mean": float(np.mean(ax["chain"])) if ax["chain"] else None,
            "multi_signal_mean": float(np.mean(ax["multi_signal"])) if ax["multi_signal"] else None,
        }

    # --- Write output ---
    ts = time.strftime("%Y%m%d-%H%M")
    out_path = args.out or RESULTS_DIR / f"bench-v3-{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[write] {out_path}", file=sys.stderr)

    # --- Console summary ---
    print("\n=== bench v3 summary ===", file=sys.stderr)
    print(f"{'system':<12} {'overall':>8} {'core':>7} {'stateful':>9} {'chain':>7} {'multi':>7} {'n':>4}", file=sys.stderr)
    for sn in systems:
        a = out["aggregate"][sn]
        def _f(x): return f"{x:.2f}" if x is not None else "  -  "
        print(f"{sn:<12} {a['overall_weighted_mean']:>8.3f} "
              f"{_f(a['core_mean']):>7} {_f(a['stateful_mean']):>9} "
              f"{_f(a['chain_mean']):>7} {_f(a['multi_signal_mean']):>7} "
              f"{a['n_tests']:>4}",
              file=sys.stderr)

    # --- Optional snapshot ---
    if args.snapshot:
        snap = SNAPSHOT_DIR / args.snapshot
        snap.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(out_path, snap / "result.json")
        shutil.copy(CORPUS_FILE, snap / "corpus-v3.json")
        shutil.copy(BENCH_DIR / "rubric-v3.md", snap / "rubric-v3.md")
        frozen = snap / "scripts-frozen"
        frozen.mkdir(exist_ok=True)
        shutil.copy(__file__, frozen / Path(__file__).name)
        # Summary
        lines = [f"# Snapshot: {args.snapshot}", f"",
                 f"- corpus: {CORPUS_FILE.name} ({len(events)} events, {len(all_tests)} tests)",
                 f"- systems: {', '.join(systems)}",
                 f"- judges: {', '.join(judge_providers) or '(dry-run)'}",
                 f"- timestamp: {out['_meta']['timestamp']}", f"",
                 f"## Results", f""]
        lines.append(f"| system | overall | core | stateful | chain | multi_signal | n |")
        lines.append(f"|---|---|---|---|---|---|---|")
        for sn in systems:
            a = out["aggregate"][sn]
            def _ff(x): return f"{x:.2f}" if x is not None else "—"
            lines.append(f"| {sn} | {a['overall_weighted_mean']:.3f} | "
                         f"{_ff(a['core_mean'])} | {_ff(a['stateful_mean'])} | "
                         f"{_ff(a['chain_mean'])} | {_ff(a['multi_signal_mean'])} | "
                         f"{a['n_tests']} |")
        (snap / "summary.md").write_text("\n".join(lines))
        print(f"[snapshot] {snap}", file=sys.stderr)

    # Finalize live dashboard: rename current.json → final.json so pollers stop
    live.finalize()


if __name__ == "__main__":
    main()
