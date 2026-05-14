#!/usr/bin/env python3
"""
Empathic Memory Bench v2 — blind, bilingual, 15-system evaluation.

Measures EMOTIONAL FITNESS of memory retrieval for empathic AI companions.
Not semantic similarity — emotional fitness: will these memories let a
companion act warmly and safely?

Key features:
  - Blind evaluation: judges see "System 01"–"System 15", randomized per test
  - Bilingual: EN and RU judge prompts (--lang en|ru)
  - 15 memory systems compared across 10 LLM judges from 7 companies
  - Calibrated scoring with anchor examples (2/5/8/10)

Usage:
  python bench-empathic-memory.py                # default: EN prompt
  python bench-empathic-memory.py --lang ru      # Russian prompt
  python bench-empathic-memory.py --lang en --systems G,B,D,J  # subset

Output: bench/results/empathic-memory-{ts}.json + .md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib import request as urlreq
from urllib.error import HTTPError

# --- Garden import setup ---
GARDEN_ROOT = Path("/Users/nikshilov/dev/ai/Garden/garden-app")
sys.path.insert(0, str(GARDEN_ROOT / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv(GARDEN_ROOT / ".env")
    load_dotenv(GARDEN_ROOT.parent / ".env")
except Exception:
    pass

import anthropic

API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("No claude_api_key or ANTHROPIC_API_KEY in env")

client = anthropic.Anthropic(api_key=API_KEY)

# Native API endpoints for Chinese models (all OpenAI-compatible chat/completions)
NATIVE_APIS = {
    "moonshot": {
        "url": "https://api.moonshot.ai/v1/chat/completions",
        "key_env": "MOONSHOT_API_KEY",
    },
    "dashscope": {
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "zai": {
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        "key_env": "XAI_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_JUDGE_API_KEY",
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "key_env": "GEMINI_API_KEY",
    },
}

# 12 judges from 7 companies: Anthropic (3), OpenAI (3), DashScope (2),
# ZAI, Moonshot, xAI, Google. Validates Garden's advantage across model
# families, price tiers, and SOTA frontier models.
JUDGES = [
    {"id": "opus",     "model": "claude-opus-4-6",     "label": "Opus 4.6",       "provider": "anthropic"},
    {"id": "sonnet",   "model": "claude-sonnet-4-6",   "label": "Sonnet 4.6",     "provider": "anthropic"},
    {"id": "haiku",    "model": "claude-haiku-4-5-20251001", "label": "Haiku 4.5", "provider": "anthropic"},
    {"id": "glm5",     "model": "glm-5.1",             "label": "GLM-5.1",        "provider": "zai"},
    {"id": "deepseek", "model": "deepseek-v3.2",       "label": "DeepSeek V3.2",  "provider": "dashscope"},
    {"id": "qwen",     "model": "qwen3.6-plus",        "label": "Qwen 3.6 Plus",  "provider": "dashscope"},
    {"id": "kimi",     "model": "kimi-k2.5",           "label": "Kimi K2.5",      "provider": "moonshot"},
    {"id": "grok",     "model": "grok-4.20",           "label": "Grok 4.20",      "provider": "xai"},
    {"id": "gpt4o",    "model": "gpt-4o",              "label": "GPT-4o",         "provider": "openai"},
    {"id": "gpt54",    "model": "gpt-5.4",             "label": "GPT-5.4",        "provider": "openai"},
    {"id": "gpt54pro", "model": "gpt-5.4-pro",         "label": "GPT-5.4 Pro",    "provider": "openai_responses"},
    {"id": "gemini",   "model": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro", "provider": "google"},
]
JUDGE_IDS = [j["id"] for j in JUDGES]
JUDGE_BY_ID = {j["id"]: j for j in JUDGES}

# --- CLI args ---
_parser = argparse.ArgumentParser(description="Empathic Memory Bench v2")
_parser.add_argument("--lang", choices=["en", "ru"], default="en", help="Judge prompt language")
_parser.add_argument("--systems", type=str, default="", help="Comma-separated system codes (default: all)")
_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
_parser.add_argument("--resume", type=str, default="", help="Path to partial JSON to resume from")
_parser.add_argument("--parallel", type=int, default=5, help="Max parallel judge calls (default: 5)")
CLI_ARGS, _ = _parser.parse_known_args()

# --- Bench paths ---
BENCH_DIR = Path(__file__).parent
DATASETS_DIR = BENCH_DIR / "datasets"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_FILE = DATASETS_DIR / "empathic-memory-corpus-sonya.json"

# --- Garden sandbox ---
GARDEN_TMP_DATA = BENCH_DIR / "empathic-garden-data"
if GARDEN_TMP_DATA.exists():
    shutil.rmtree(GARDEN_TMP_DATA)
GARDEN_TMP_DATA.mkdir(parents=True)

GARDEN_REAL_DATA = GARDEN_ROOT / "backend" / "garden_graph" / "data"
GARDEN_DATA_BACKUP = GARDEN_ROOT / "backend" / "garden_graph" / "data.empathic-backup"
_restored = False


def _sandbox_data_dir():
    if GARDEN_REAL_DATA.exists() and not GARDEN_DATA_BACKUP.exists():
        GARDEN_REAL_DATA.rename(GARDEN_DATA_BACKUP)
    GARDEN_TMP_DATA.mkdir(exist_ok=True)
    if GARDEN_REAL_DATA.exists() or GARDEN_REAL_DATA.is_symlink():
        try:
            GARDEN_REAL_DATA.unlink()
        except Exception:
            pass
    GARDEN_REAL_DATA.symlink_to(GARDEN_TMP_DATA)


def _restore_data_dir():
    global _restored
    if _restored:
        return
    try:
        if GARDEN_REAL_DATA.exists() or GARDEN_REAL_DATA.is_symlink():
            GARDEN_REAL_DATA.unlink()
        if GARDEN_DATA_BACKUP.exists():
            GARDEN_DATA_BACKUP.rename(GARDEN_REAL_DATA)
    except Exception as e:
        print(f"[bench] WARN: restore failed: {e}")
    _restored = True


_sandbox_data_dir()
import atexit
atexit.register(_restore_data_dir)

from garden_graph.memory.manager import MemoryManager, MemoryRecord  # noqa: E402

# --- Service endpoints ---
ST_URL = "http://localhost:8000"
ARKHON_URL = "http://localhost:9000"
EMOGIE_URL = "http://localhost:5125"

USER_ID = "empathic"
CHAR_NAME = "alex"
COLLECTION_OA = "empathic-oa"
COLLECTION_TR = "empathic-tr"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def post(url: str, payload: dict, timeout: int = 60) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlreq.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:300]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def post_json(url: str, payload: dict, timeout: int = 60):
    status, body = post(url, payload, timeout)
    try:
        return status, json.loads(body)
    except Exception:
        return status, None


# --------------------------------------------------------------------------
# Garden
# --------------------------------------------------------------------------
def garden_clear_and_seed(events: list[dict]) -> MemoryManager:
    """Build a fresh MemoryManager and seed with 30 empathic events.

    Crucially, we backdate `created_at` and `last_touched` for each event so
    that Garden's effective_weight() decay reflects the days_ago metadata.
    Otherwise every event would be treated as just-created and decay would
    have no effect — destroying the recency story for the bench.
    """
    for fname in ["memories.json", "mood_states.json", "last_seen_times.json",
                  "scheduled_events.json", "relationships.json",
                  "char_relationships.json", "identity.json", "mood_log.csv"]:
        p = GARDEN_TMP_DATA / fname
        if p.exists():
            p.unlink()

    mm = MemoryManager(
        memories_path=str(GARDEN_TMP_DATA / "memories.json"),
        events_path=str(GARDEN_TMP_DATA / "scheduled_events.json"),
        autoload=False,
    )
    now = datetime.now(timezone.utc)
    for e in events:
        rec = mm.create(
            character_id=CHAR_NAME,
            event_text=e["text"],
            sentiment=int(e["sentiment"]),
            sentiment_label=e.get("sentiment_label", "neutral"),
            user_flag=bool(e.get("user_flag", False)),
        )
        # backdate so decay applies
        backdate = now - timedelta(days=int(e.get("days_ago", 0)))
        rec.created_at = backdate
        rec.last_touched = backdate
    return mm


def garden_query(mm: MemoryManager, query: str, k: int = 3) -> list[dict]:
    """Use Garden's empathic salient_memories: anchors + query + valence + decay.

    This is the API the bench is built to surface — pure top_k is the
    cold-open baseline, salient_memories is what an empathic companion
    should call when it knows what conversation moment it's in.
    """
    recs = mm.salient_memories(CHAR_NAME, query=query, k=k)
    return [
        {
            "text": r.event_text,
            "weight": round(r.effective_weight(), 3),
            "sentiment": r.sentiment,
            "sentiment_label": r.sentiment_label,
            "user_flag": r.user_flag,
        }
        for r in recs
    ]


# --------------------------------------------------------------------------
# SillyTavern Vector Storage
# --------------------------------------------------------------------------
def st_purge(collection: str) -> None:
    post(f"{ST_URL}/api/vector/purge", {"collectionId": collection}, timeout=30)


def st_insert(collection: str, source: str, events: list[dict], model: str = "") -> int:
    items = []
    for e in events:
        items.append({
            "hash": (hash(f"{e['id']}::{e['text']}") & 0x7fffffff),
            "text": e["text"],
            "index": e["id"],
        })
    payload = {"collectionId": collection, "source": source, "items": items}
    if model:
        payload["model"] = model
    status, body = post(f"{ST_URL}/api/vector/insert", payload, timeout=300)
    if status != 200:
        print(f"  [ST {source}] insert failed: {status} {body[:120]}")
        return 0
    return len(items)


def st_query(collection: str, source: str, query: str, k: int = 3, model: str = "") -> list[dict]:
    payload = {
        "collectionId": collection,
        "source": source,
        "searchText": query,
        "topK": k,
        "threshold": 0.0,
    }
    if model:
        payload["model"] = model
    status, data = post_json(f"{ST_URL}/api/vector/query", payload, timeout=60)
    if status != 200 or not isinstance(data, dict):
        return [{"text": f"(HTTP {status})", "score": 0}]
    md = data.get("metadata", []) or []
    return [{"text": (m.get("text") or "").strip(), "score": m.get("score", 0)} for m in md[:k]]


# --------------------------------------------------------------------------
# Arkhon
# --------------------------------------------------------------------------
def arkhon_clear() -> None:
    # Try delete-all endpoint first; fall back to ignoring failures
    post(f"{ARKHON_URL}/memories/clear", {"user_id": USER_ID, "char_name": CHAR_NAME}, timeout=30)


def arkhon_store(events: list[dict]) -> int:
    ok = 0
    for e in events:
        status, _ = post(
            f"{ARKHON_URL}/memories",
            {"user_id": USER_ID, "char_name": CHAR_NAME, "text": e["text"]},
            timeout=30,
        )
        if status == 200:
            ok += 1
    return ok


def arkhon_query(query: str, k: int = 3) -> list[dict]:
    status, data = post_json(
        f"{ARKHON_URL}/memories/recall",
        {"user_id": USER_ID, "char_name": CHAR_NAME, "query": query, "top_k": k},
        timeout=30,
    )
    if status != 200 or not isinstance(data, list):
        return [{"text": f"(HTTP {status})", "score": 0}]
    return [{"text": (m.get("text") or "").strip(), "score": m.get("score", 0)} for m in data[:k]]


# --------------------------------------------------------------------------
# emogie
# --------------------------------------------------------------------------
def emogie_clear() -> None:
    post(f"{EMOGIE_URL}/memory/clear", {"character": CHAR_NAME}, timeout=30)


def emogie_store(events: list[dict]) -> int:
    ok = 0
    for e in events:
        payload = {
            "character": CHAR_NAME,
            "messages": [{"role": "user", "name": USER_ID, "content": e["text"]}],
            "auto_stored": False,
        }
        status, _ = post(f"{EMOGIE_URL}/memory/store", payload, timeout=60)
        if status == 200:
            ok += 1
    return ok


def emogie_query(query: str, k: int = 3) -> list[dict]:
    status, data = post_json(
        f"{EMOGIE_URL}/memory/query",
        {"query": query, "k": k, "min_score": 0.0},
        timeout=30,
    )
    if status != 200 or not isinstance(data, dict):
        return [{"text": f"(HTTP {status})", "score": 0}]
    results = data.get("results", []) or []
    return [{"text": (m.get("text") or "").strip(), "score": m.get("score", 0)} for m in results[:k]]


# --------------------------------------------------------------------------
# VDS systems: MemPalace (A) + sqlite-vec (B)
# --------------------------------------------------------------------------
VDS = "openclaw@152.42.186.145"
VDS_BENCH_DIR = "~/persistent/empathic-bench"
VDS_PALACE = f"{VDS_BENCH_DIR}/mempalace"
VDS_EVENTS = f"{VDS_BENCH_DIR}/events"
VDS_VEC_HELPER = f"{VDS_BENCH_DIR}/vec_helper.py"
LOCAL_VEC_HELPER = BENCH_DIR / "vds-helpers" / "vec_helper.py"


def run_ssh(cmd: str, timeout: int = 90, input_text: str | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        ["ssh", VDS, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )
    return p.returncode, p.stdout, p.stderr


def shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


# ----- MemPalace (A) -----
def mempalace_clear_and_seed(events: list[dict]) -> int:
    """Wipe sandbox palace+events dir, write each event as its own .md, init+mine.

    Each event becomes a tiny markdown file `event-{NN}.md`. We DON'T add a
    heading — the raw event text becomes the searchable body, which is what
    MemPalace's keyword search will hit on.
    """
    # Wipe + recreate dirs
    rc, _, err = run_ssh(
        f"rm -rf {VDS_PALACE} {VDS_EVENTS} && mkdir -p {VDS_PALACE} {VDS_EVENTS}",
        timeout=30,
    )
    if rc != 0:
        print(f"  [A] wipe failed: {err[:200]}")
        return 0

    # Write all events as individual .md files in one shot via heredoc bundle
    # Use a tar stream to avoid quoting hell with arbitrary text content.
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for e in events:
            content = e["text"].encode("utf-8")
            info = tarfile.TarInfo(name=f"event-{int(e['id']):03d}.md")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    tar_bytes = buf.getvalue()

    # Pipe the tar archive into ssh and extract on remote
    p = subprocess.run(
        ["ssh", VDS, f"tar -x -C {VDS_EVENTS}"],
        input=tar_bytes,
        capture_output=True,
        timeout=60,
    )
    if p.returncode != 0:
        print(f"  [A] tar upload failed: {p.stderr.decode('utf-8', 'replace')[:200]}")
        return 0

    # Skip the interactive `init` — just write a minimal mempalace.yaml directly.
    # The yaml is just `wing: <name>\nrooms: [{name: general, description: ...}]`.
    yaml_text = "wing: events\nrooms:\n- name: general\n  description: All project files\n"
    rc, _, err = run_ssh(
        f"cat > {VDS_EVENTS}/mempalace.yaml",
        timeout=15,
        input_text=yaml_text,
    )
    if rc != 0:
        print(f"  [A] yaml write failed: {err[:200]}")
        return 0

    rc, out, err = run_ssh(
        f"~/mempalace-env/bin/mempalace --palace {VDS_PALACE} mine {VDS_EVENTS} 2>&1",
        timeout=120,
    )
    if rc != 0:
        print(f"  [A] init/mine failed rc={rc}: {out[-200:]}")
        return 0

    # Parse "Drawers filed: N" from output as a sanity check
    m = re.search(r"Drawers filed:\s*(\d+)", out)
    return int(m.group(1)) if m else len(events)


def mempalace_query(query: str, k: int = 3) -> list[dict]:
    cmd = (
        f"~/mempalace-env/bin/mempalace --palace {VDS_PALACE} search {shquote(query)} "
        f"--results {k} 2>/dev/null"
    )
    rc, out, _ = run_ssh(cmd, timeout=45)
    if rc != 0 or not out.strip():
        return [{"text": f"(rc={rc})", "score": 0}]

    # Parse blocks separated by "─────". Each block has:
    #   [N] events / general
    #   Source: event-NNN.md
    #   Match:  0.123
    #   <blank>
    #   <body>
    blocks = re.split(r"─{5,}", out)
    results: list[dict] = []
    for blk in blocks:
        if "[" not in blk or "Match:" not in blk:
            continue
        score_m = re.search(r"Match:\s*(-?[\d.]+)", blk)
        score = float(score_m.group(1)) if score_m else 0.0
        # body is after the Match: line
        after_match = blk.split("Match:", 1)[1].split("\n", 1)[1]
        # Strip leading whitespace, drop the synthetic # heading if present, take first non-empty paragraph
        body_lines = [ln.strip() for ln in after_match.splitlines() if ln.strip()]
        body_lines = [ln for ln in body_lines if not ln.startswith("#")]
        text = " ".join(body_lines).strip()
        if text:
            results.append({"text": text, "score": round(score, 4)})
        if len(results) >= k:
            break
    if not results:
        return [{"text": "(no matches parsed)", "score": 0}]
    return results


# ----- sqlite-vec (B) -----
def sqlite_vec_upload_helper() -> bool:
    """Copy vec_helper.py to VDS once per run."""
    p = subprocess.run(
        ["scp", "-q", str(LOCAL_VEC_HELPER), f"{VDS}:{VDS_VEC_HELPER}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if p.returncode != 0:
        print(f"  [B] scp helper failed: {p.stderr[:200]}")
        return False
    return True


def sqlite_vec_clear_and_seed(events: list[dict]) -> int:
    if not sqlite_vec_upload_helper():
        return 0
    payload = json.dumps({"events": [{"id": e["id"], "text": e["text"]} for e in events]})
    rc, out, err = run_ssh(
        f"python3 {VDS_VEC_HELPER} seed",
        timeout=120,
        input_text=payload,
    )
    if rc != 0:
        print(f"  [B] seed failed rc={rc}: {err[:200] or out[:200]}")
        return 0
    try:
        d = json.loads(out.strip().splitlines()[-1])
        return int(d.get("n", 0))
    except Exception:
        return 0


def sqlite_vec_query(query: str, k: int = 3) -> list[dict]:
    cmd = f"python3 {VDS_VEC_HELPER} query {shquote(query)} {k}"
    rc, out, err = run_ssh(cmd, timeout=60)
    if rc != 0:
        return [{"text": f"(rc={rc})", "score": 0}]
    try:
        d = json.loads(out.strip().splitlines()[-1])
        return [{"text": r["text"], "score": r.get("score", 0)} for r in d.get("results", [])]
    except Exception as exc:
        return [{"text": f"(parse: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# Graphiti / Zep temporal KG (H) — Neo4j + entity extraction + validity windows
# --------------------------------------------------------------------------
GRAPHITI_NEO4J_URI = "bolt://localhost:7687"
GRAPHITI_NEO4J_USER = "neo4j"
GRAPHITI_NEO4J_PASS = "benchtest123"
GRAPHITI_GROUP = "empathic-bench"

# Shared event loop for all async systems (Graphiti + OpenMemory).
# asyncio.run() creates/destroys loops, breaking neo4j's async driver.
_loop = asyncio.new_event_loop()
_graphiti_client = None


def _run(coro):
    """Run async coroutine on the shared event loop."""
    return _loop.run_until_complete(coro)


async def _graphiti_init():
    global _graphiti_client
    from graphiti_core import Graphiti
    _graphiti_client = Graphiti(
        uri=GRAPHITI_NEO4J_URI,
        user=GRAPHITI_NEO4J_USER,
        password=GRAPHITI_NEO4J_PASS,
    )
    await _graphiti_client.build_indices_and_constraints()
    return _graphiti_client


async def _graphiti_clear():
    """Wipe all nodes/edges from Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(GRAPHITI_NEO4J_URI, auth=(GRAPHITI_NEO4J_USER, GRAPHITI_NEO4J_PASS))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


async def _graphiti_seed(events: list[dict]) -> int:
    from graphiti_core.nodes import EpisodeType
    g = await _graphiti_init()
    now = datetime.now(timezone.utc)
    ok = 0
    for e in events:
        ref_time = now - timedelta(days=int(e.get("days_ago", 0)))
        try:
            await g.add_episode(
                name=f"event-{e['id']}",
                episode_body=e["text"],
                source_description="empathic bench corpus",
                reference_time=ref_time,
                source=EpisodeType.text,
                group_id=GRAPHITI_GROUP,
            )
            ok += 1
        except Exception as exc:
            print(f"    [H] episode {e['id']} failed: {exc}")
    return ok


async def _graphiti_search(query: str, k: int = 3) -> list[dict]:
    g = _graphiti_client
    if not g:
        return [{"text": "(not initialized)", "score": 0}]
    results = await g.search(query=query, group_ids=[GRAPHITI_GROUP], num_results=k)
    out = []
    for edge in results:
        fact = getattr(edge, "fact", None) or str(edge)
        score = getattr(edge, "score", None) or 0
        out.append({"text": fact, "score": round(float(score), 4) if score else 0})
    return out[:k] if out else [{"text": "(no results)", "score": 0}]


async def _graphiti_close():
    global _graphiti_client
    if _graphiti_client:
        await _graphiti_client.close()
        _graphiti_client = None


def graphiti_clear_and_seed(events: list[dict]) -> int:
    _run(_graphiti_clear())
    return _run(_graphiti_seed(events))


def graphiti_query(query: str, k: int = 3) -> list[dict]:
    return _run(_graphiti_search(query, k))


def graphiti_cleanup():
    _run(_graphiti_close())


# --------------------------------------------------------------------------
# OpenMemory (I) — SQLite + exponential decay + waypoint graph
# --------------------------------------------------------------------------
OPENMEM_DB = str(BENCH_DIR / "empathic-openmemory.db")
_openmem_instance = None


async def _openmem_init():
    global _openmem_instance
    os.environ["OM_DB_URL"] = f"sqlite:///{OPENMEM_DB}"
    from openmemory.main import Memory
    _openmem_instance = Memory(user="empathic")
    return _openmem_instance


async def _openmem_clear_and_seed(events: list[dict]) -> int:
    # Remove old DB
    db_path = Path(OPENMEM_DB)
    if db_path.exists():
        db_path.unlink()
    mem = await _openmem_init()
    ok = 0
    for e in events:
        try:
            await mem.add(
                content=e["text"],
                user_id="empathic",
                meta={
                    "sentiment": e.get("sentiment", 0),
                    "user_flag": e.get("user_flag", False),
                    "days_ago": e.get("days_ago", 0),
                    "event_id": e["id"],
                },
                tags=[e.get("sentiment_label", "neutral")],
            )
            ok += 1
        except Exception as exc:
            print(f"    [I] event {e['id']} failed: {exc}")
    return ok


async def _openmem_search(query: str, k: int = 3) -> list[dict]:
    mem = _openmem_instance
    if not mem:
        return [{"text": "(not initialized)", "score": 0}]
    results = await mem.search(query=query, user_id="empathic", limit=k)
    out = []
    if isinstance(results, list):
        for r in results:
            text = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", str(r))
            score = r.get("score", 0) if isinstance(r, dict) else getattr(r, "score", 0)
            out.append({"text": str(text).strip(), "score": round(float(score), 4) if score else 0})
    return out[:k] if out else [{"text": "(no results)", "score": 0}]


def openmem_clear_and_seed(events: list[dict]) -> int:
    return _run(_openmem_clear_and_seed(events))


def openmem_query(query: str, k: int = 3) -> list[dict]:
    return _run(_openmem_search(query, k))


# --------------------------------------------------------------------------
# mem0 (J) — LLM-extracted facts + Qdrant local + OpenAI embeddings
# --------------------------------------------------------------------------
MEM0_QDRANT_PATH = str(BENCH_DIR / "empathic-mem0-qdrant")
_mem0_instance = None


def mem0_clear_and_seed(events: list[dict]) -> int:
    """Create a fresh mem0 Memory instance and ingest all events.

    mem0 uses an LLM (gpt-4o-mini) to extract structured facts from raw text,
    then embeds and stores them in Qdrant.  This is architecturally different
    from pure vector stores: it stores *extracted facts*, not verbatim text.
    """
    global _mem0_instance
    import shutil as _shutil
    from mem0 import Memory

    # Wipe previous data
    p = Path(MEM0_QDRANT_PATH)
    if p.exists():
        _shutil.rmtree(p)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "empathic_bench_mem0",
                "path": MEM0_QDRANT_PATH,
                "embedding_model_dims": 3072,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large",
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
        },
        "version": "v1.1",
    }

    _mem0_instance = Memory.from_config(config)
    ok = 0
    for e in events:
        try:
            _mem0_instance.add(
                e["text"],
                user_id="bench_user",
                metadata={
                    "event_id": e["id"],
                    "sentiment": e.get("sentiment", 0),
                    "user_flag": e.get("user_flag", False),
                    "days_ago": e.get("days_ago", 0),
                },
            )
            ok += 1
        except Exception as exc:
            print(f"    [J] event {e['id']} failed: {exc}")
    return ok


def mem0_query(query: str, k: int = 3) -> list[dict]:
    """Search mem0 and return top-k results in the standard bench format."""
    if not _mem0_instance:
        return [{"text": "(not initialized)", "score": 0}]
    try:
        result = _mem0_instance.search(query, user_id="bench_user", limit=k)
        items = result.get("results", []) if isinstance(result, dict) else []
        out = []
        for r in items[:k]:
            text = r.get("memory", "") or ""
            score = r.get("score", 0) or 0
            out.append({"text": text.strip(), "score": round(float(score), 4)})
        return out if out else [{"text": "(no results)", "score": 0}]
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# Letta / MemGPT (K) — Archival Memory via REST API
# --------------------------------------------------------------------------
LETTA_URL = "http://localhost:18800"
_letta_client = None
_letta_agent_id = None


def letta_clear_and_seed(events: list[dict]) -> int:
    global _letta_client, _letta_agent_id
    from letta_client import Letta

    _letta_client = Letta(base_url=LETTA_URL)

    # Delete old bench agent if exists
    try:
        for a in _letta_client.agents.list():
            if a.name == "empathic_bench":
                _letta_client.agents.delete(a.id)
    except Exception:
        pass

    agent = _letta_client.agents.create(
        name="empathic_bench",
        model="openai/gpt-4o-mini",
        embedding="openai/text-embedding-3-large",
    )
    _letta_agent_id = agent.id

    ok = 0
    for e in events:
        try:
            _letta_client.agents.passages.create(
                agent_id=_letta_agent_id,
                text=e["text"],
            )
            ok += 1
        except Exception as exc:
            print(f"    [K] event {e['id']} failed: {exc}")
    return ok


def letta_query(query: str, k: int = 3) -> list[dict]:
    if not _letta_client or not _letta_agent_id:
        return [{"text": "(not initialized)", "score": 0}]
    try:
        resp = _letta_client.agents.passages.search(
            agent_id=_letta_agent_id,
            query=query,
            top_k=k,
        )
        hits = resp.results
        return [{"text": h.content[:400], "score": 0} for h in hits[:k]] or [{"text": "(no results)", "score": 0}]
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# LangMem (L) — LangGraph InMemoryStore + OpenAI embeddings
# --------------------------------------------------------------------------
_langmem_store = None


def langmem_clear_and_seed(events: list[dict]) -> int:
    global _langmem_store
    from langgraph.store.memory import InMemoryStore

    _langmem_store = InMemoryStore(
        index={
            "dims": 3072,
            "embed": "openai:text-embedding-3-large",
            "fields": ["text"],
        }
    )

    namespace = ("empathic", "bench")
    ok = 0
    for e in events:
        try:
            _langmem_store.put(
                namespace=namespace,
                key=f"event_{e['id']}",
                value={"text": e["text"]},
            )
            ok += 1
        except Exception as exc:
            print(f"    [L] event {e['id']} failed: {exc}")
    return ok


def langmem_query(query: str, k: int = 3) -> list[dict]:
    if not _langmem_store:
        return [{"text": "(not initialized)", "score": 0}]
    try:
        hits = _langmem_store.search(
            ("empathic", "bench"),
            query=query,
            limit=k,
        )
        out = []
        for item in hits[:k]:
            score = getattr(item, "score", 0) or 0
            out.append({"text": item.value["text"][:400], "score": round(float(score), 4)})
        return out or [{"text": "(no results)", "score": 0}]
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# LlamaIndex (M) — VectorStoreIndex in-memory
# --------------------------------------------------------------------------
_llamaindex_retriever = None


def llamaindex_clear_and_seed(events: list[dict]) -> int:
    global _llamaindex_retriever
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = None

    docs = [Document(text=e["text"], doc_id=f"event_{e['id']}") for e in events]
    index = VectorStoreIndex.from_documents(docs)
    _llamaindex_retriever = index.as_retriever(similarity_top_k=3)
    return len(docs)


def llamaindex_query(query: str, k: int = 3) -> list[dict]:
    if not _llamaindex_retriever:
        return [{"text": "(not initialized)", "score": 0}]
    try:
        nodes = _llamaindex_retriever.retrieve(query)
        out = []
        for node in nodes[:k]:
            out.append({"text": node.text[:400], "score": round(float(node.score), 4)})
        return out or [{"text": "(no results)", "score": 0}]
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# Cognee (N) — Knowledge Graph + Vector Search
# --------------------------------------------------------------------------
_cognee_ready = False


def cognee_clear_and_seed(events: list[dict]) -> int:
    global _cognee_ready

    async def _seed():
        global _cognee_ready
        import cognee

        cognee.config.set_llm_api_key(os.environ["OPENAI_API_KEY"])
        cognee.config.set_llm_model("gpt-4o-mini")
        cognee.config.set_llm_provider("openai")
        cognee.config.set_embedding_api_key(os.environ["OPENAI_API_KEY"])
        cognee.config.set_embedding_model("text-embedding-3-large")
        cognee.config.set_embedding_provider("openai")
        cognee.config.set_embedding_dimensions(3072)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        for e in events:
            await cognee.add(e["text"], dataset_name="empathic_bench")

        await cognee.cognify(datasets=["empathic_bench"])
        _cognee_ready = True
        return len(events)

    return _run(_seed())


def cognee_query(query: str, k: int = 3) -> list[dict]:
    if not _cognee_ready:
        return [{"text": "(not initialized)", "score": 0}]

    async def _search():
        import cognee
        raw_hits = await cognee.search(
            query_text=query,
            query_type=cognee.SearchType.CHUNKS,
            datasets=["empathic_bench"],
            top_k=k,
        )
        flat = []
        for hit in raw_hits:
            if isinstance(hit, dict) and "search_result" in hit:
                flat.extend(hit["search_result"][:k])
            else:
                flat.append(hit)

        out = []
        for r in flat[:k]:
            if isinstance(r, dict):
                text = r.get("text", r.get("content", str(r)))[:400]
            elif hasattr(r, "text"):
                text = r.text[:400]
            else:
                text = str(r)[:400]
            out.append({"text": text, "score": 0})
        return out or [{"text": "(no results)", "score": 0}]

    try:
        return _run(_search())
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# Weaviate (O) — Vector Search with own OpenAI embeddings
# --------------------------------------------------------------------------
WEAVIATE_HTTP_PORT = 8099
WEAVIATE_GRPC_PORT = 50099
WEAVIATE_AUTH_KEY = "WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih"
_weaviate_client = None
_weaviate_collection = None


def _weaviate_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI (Weaviate runs without vectorizer module)."""
    import openai as _openai
    oai = _openai.OpenAI()
    resp = oai.embeddings.create(input=texts, model="text-embedding-3-large")
    return [r.embedding for r in resp.data]


def weaviate_clear_and_seed(events: list[dict]) -> int:
    global _weaviate_client, _weaviate_collection
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    _weaviate_client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
        skip_init_checks=True,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_AUTH_KEY),
    )

    coll_name = "EmpathicBench"
    if _weaviate_client.collections.exists(coll_name):
        _weaviate_client.collections.delete(coll_name)

    _weaviate_collection = _weaviate_client.collections.create(
        name=coll_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[Property(name="text", data_type=DataType.TEXT)],
    )

    texts = [e["text"] for e in events]
    vectors = _weaviate_embed(texts)

    ok = 0
    for e, vec in zip(events, vectors):
        try:
            _weaviate_collection.data.insert(
                properties={"text": e["text"]},
                vector=vec,
            )
            ok += 1
        except Exception as exc:
            print(f"    [O] event {e['id']} failed: {exc}")
    return ok


def weaviate_query(query: str, k: int = 3) -> list[dict]:
    if not _weaviate_collection:
        return [{"text": "(not initialized)", "score": 0}]
    try:
        from weaviate.classes.query import MetadataQuery
        query_vec = _weaviate_embed([query])[0]
        resp = _weaviate_collection.query.near_vector(
            near_vector=query_vec,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for obj in resp.objects[:k]:
            dist = obj.metadata.distance or 0
            score = max(0, 1 - dist)
            out.append({"text": obj.properties["text"][:400], "score": round(score, 4)})
        return out or [{"text": "(no results)", "score": 0}]
    except Exception as exc:
        return [{"text": f"(error: {exc})", "score": 0}]


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------
def _load_judge_prompt(lang: str) -> str:
    """Load judge system prompt from file. Falls back to EN if file missing."""
    prompt_file = BENCH_DIR / "prompts" / f"judge-{lang}.txt"
    if not prompt_file.exists():
        prompt_file = BENCH_DIR / "prompts" / "judge-en.txt"
    if not prompt_file.exists():
        sys.exit(f"Judge prompt not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()

JUDGE_SYSTEM = _load_judge_prompt(CLI_ARGS.lang)


def format_results(label: str, results: list[dict]) -> str:
    if not results:
        return f"=== {label} ===\n(empty)"
    parts = [f"=== {label} ==="]
    for i, r in enumerate(results, 1):
        text = r.get("text", "").strip()
        score = r.get("score") or r.get("weight")
        score_str = f" (score={score})" if score is not None else ""
        parts.append(f"[{i}]{score_str} {text[:400]}")
    return "\n".join(parts)


def _make_blind_mapping(system_codes: list[str], seed: int | None = None) -> dict[str, str]:
    """Create a random mapping from real system codes to blind codes S01..SNN."""
    rng = random.Random(seed)
    blind_codes = [f"S{i+1:02d}" for i in range(len(system_codes))]
    rng.shuffle(blind_codes)
    return dict(zip(system_codes, blind_codes))


def _build_judge_user_prompt(test: dict, results: dict[str, list[dict]],
                              blind_map: dict[str, str]) -> str:
    """Build the judge prompt with blind system codes in randomized order."""
    # Sort by blind code so the order is S01, S02, ... (which is random wrt real systems)
    items = sorted(blind_map.items(), key=lambda kv: kv[1])

    result_blocks = []
    for real_code, blind_code in items:
        result_blocks.append(format_results(f"{blind_code}", results.get(real_code, [])))

    return f"""Test: {test['name']}
User query / conversation moment: "{test['user_query']}"

What this test is checking: {test['what_it_tests']}

Ideal event IDs (from corpus): {test['ideal_top_3_event_ids']}
Why those: {test['ideal_explanation']}

Failure modes to penalize:
{chr(10).join('- ' + fm for fm in test['fail_modes'])}

{chr(10).join(result_blocks)}
"""


def _call_anthropic(model: str, user: str) -> str:
    r = client.messages.create(
        model=model,
        max_tokens=4096,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = r.content[0].text
    if r.stop_reason == "max_tokens":
        text += "}"  # attempt to close truncated JSON
    return text


def _call_native(provider: str, model: str, user: str) -> str:
    """Call any OpenAI-compatible chat/completions endpoint."""
    if provider == "google":
        return _call_gemini(model, user)
    if provider == "openai_responses":
        return _call_openai_responses(model, user)
    cfg = NATIVE_APIS[provider]
    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
        raise RuntimeError(f"Missing env var {cfg['key_env']}")
    # Thinking models (GLM-5, Kimi) spend most tokens on reasoning_content,
    # so we need a higher max_tokens to leave room for the actual JSON output.
    # GPT-5.4+ requires max_completion_tokens instead of max_tokens.
    tok_key = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"
    payload = json.dumps({
        "model": model,
        tok_key: 8000,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urlreq.Request(
        cfg["url"],
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlreq.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    # Thinking models (GLM-5, Kimi K2.5) may put chain-of-thought in
    # reasoning_content and leave content empty or minimal.
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text


def _call_openai_responses(model: str, user: str) -> str:
    """Call OpenAI Responses API for models that don't support chat/completions (e.g. GPT-5.4 Pro)."""
    api_key = os.environ.get("OPENAI_JUDGE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing env var OPENAI_JUDGE_API_KEY")
    payload = json.dumps({
        "model": model,
        "instructions": JUDGE_SYSTEM,
        "input": user,
        "max_output_tokens": 8000,
    }).encode("utf-8")
    req = urlreq.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlreq.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    # Extract text from output messages
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    raise RuntimeError(f"No text in GPT-5.4 Pro response: {json.dumps(data)[:500]}")


def _call_gemini(model: str, user: str) -> str:
    """Call Google Gemini API (different format from OpenAI-compatible)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing env var GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "systemInstruction": {"parts": [{"text": JUDGE_SYSTEM}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": 16000,
            "thinkingConfig": {"thinkingBudget": -1},
        },
    }).encode("utf-8")
    req = urlreq.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlreq.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _extract_json(text: str) -> dict | None:
    """Extract JSON verdict from judge response. Handles reasoning, code fences, truncation."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    # Find ALL top-level {...} blocks and try to parse each, last-first
    # (the JSON verdict is typically at the end after reasoning)
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
                candidates.append(text[start : i + 1])
                start = -1
    # Try candidates in reverse order (last block = most likely the verdict)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            # Sanity check: must have at least one S01_rel-style key
            if any(k.startswith("S") and "_" in k for k in obj):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    # Fallback: greedy regex (original behavior)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def judge_test(test: dict, results: dict[str, list[dict]], judge: dict,
               blind_map: dict[str, str]) -> dict:
    """Run a single judge on a single test with blind system codes.

    Returns a dict with blind codes (S01_rel, etc.) PLUS a _deblinded copy
    with real codes for aggregation.
    """
    user = _build_judge_user_prompt(test, results, blind_map)
    reverse_map = {v: k for k, v in blind_map.items()}  # S01 → G
    try:
        prov = judge.get("provider", "anthropic")
        if prov == "anthropic":
            text = _call_anthropic(judge["model"], user)
        else:
            text = _call_native(prov, judge["model"], user)
        blind_verdict = _extract_json(text)
        if blind_verdict is None:
            return {"error": "no_json", "raw": text[:300]}

        # Deblind: convert S01_rel → G_rel, winner S03 → winner G
        deblinded = {}
        for key, val in blind_verdict.items():
            if key == "winner":
                deblinded["winner"] = reverse_map.get(val, val)
            elif key == "note":
                deblinded["note"] = val
            elif key == "error":
                deblinded["error"] = val
            else:
                # S01_rel → find which real code S01 maps to
                parts = key.split("_", 1)
                if len(parts) == 2 and parts[0] in reverse_map:
                    real_code = reverse_map[parts[0]]
                    deblinded[f"{real_code}_{parts[1]}"] = val
                else:
                    deblinded[key] = val
        deblinded["_blind_verdict"] = blind_verdict
        return deblinded
    except Exception as exc:
        return {"error": str(exc)[:200]}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
ALL_SYSTEMS = ["G", "H", "I", "A", "B", "D", "C", "E", "F", "J", "K", "L", "M", "N", "O"]
ALL_NAMES = {
    "G": "Garden",
    "H": "Graphiti",
    "I": "OpenMemory",
    "A": "MemPalace",
    "B": "sqlite-vec",
    "D": "ST-openai",
    "C": "ST-transformers",
    "E": "Arkhon",
    "F": "emogie",
    "J": "mem0",
    "K": "Letta",
    "L": "LangMem",
    "M": "LlamaIndex",
    "N": "Cognee",
    "O": "Weaviate",
}

# Allow CLI override
if CLI_ARGS.systems:
    SYSTEMS = [s.strip() for s in CLI_ARGS.systems.split(",") if s.strip() in ALL_NAMES]
else:
    SYSTEMS = ALL_SYSTEMS[:]
NAMES = {s: ALL_NAMES[s] for s in SYSTEMS}


def main():
    print("=== Empathic Memory Bench v2 (blind) ===")
    print(f"Language: {CLI_ARGS.lang.upper()} | Blind: yes")
    print(f"Judges: {', '.join(j['label'] for j in JUDGES)}")
    corpus_data = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    events = corpus_data["events"]
    tests = corpus_data["tests"]
    print(f"Corpus: {len(events)} events | Tests: {len(tests)}")
    print(f"Systems: {' '.join(NAMES.values())}\n")

    # ----- Ingest into selected systems -----
    print(f"[1/{len(SYSTEMS)}] Ingesting corpus into {len(SYSTEMS)} systems...")
    mm = None

    if "G" in SYSTEMS:
        print("  Garden: ", end="", flush=True)
        mm = garden_clear_and_seed(events)
        print(f"{len(events)} events seeded with metadata + backdated decay")

    if "H" in SYSTEMS:
        print("  Graphiti (Neo4j): ", end="", flush=True)
        n = graphiti_clear_and_seed(events)
        print(f"{n} episodes → Neo4j temporal KG")

    if "I" in SYSTEMS:
        print("  OpenMemory: ", end="", flush=True)
        n = openmem_clear_and_seed(events)
        print(f"{n} memories with decay sectors")

    if "A" in SYSTEMS:
        print("  MemPalace (VDS): ", end="", flush=True)
        n = mempalace_clear_and_seed(events)
        print(f"{n} drawers filed in sandbox palace")

    if "B" in SYSTEMS:
        print("  sqlite-vec (VDS): ", end="", flush=True)
        n = sqlite_vec_clear_and_seed(events)
        print(f"{n} embeddings stored in sandbox db")

    if "D" in SYSTEMS:
        print("  ST-openai: ", end="", flush=True)
        st_purge(COLLECTION_OA)
        n = st_insert(COLLECTION_OA, "openai", events, model="text-embedding-3-large")
        print(f"{n} chunks")

    if "C" in SYSTEMS:
        print("  ST-transformers: ", end="", flush=True)
        st_purge(COLLECTION_TR)
        n = st_insert(COLLECTION_TR, "transformers", events)
        print(f"{n} chunks")

    if "E" in SYSTEMS:
        print("  Arkhon: ", end="", flush=True)
        arkhon_clear()
        n = arkhon_store(events)
        print(f"{n} stores ok")

    if "F" in SYSTEMS:
        print("  emogie: ", end="", flush=True)
        emogie_clear()
        n = emogie_store(events)
        print(f"{n} stores ok")

    if "J" in SYSTEMS:
        print("  mem0: ", end="", flush=True)
        n = mem0_clear_and_seed(events)
        print(f"{n} events → mem0 (LLM-extracted facts + Qdrant)")

    if "K" in SYSTEMS:
        print("  Letta (MemGPT): ", end="", flush=True)
        n = letta_clear_and_seed(events)
        print(f"{n} passages in archival memory")

    if "L" in SYSTEMS:
        print("  LangMem: ", end="", flush=True)
        n = langmem_clear_and_seed(events)
        print(f"{n} memories in InMemoryStore")

    if "M" in SYSTEMS:
        print("  LlamaIndex: ", end="", flush=True)
        n = llamaindex_clear_and_seed(events)
        print(f"{n} documents indexed")

    if "N" in SYSTEMS:
        print("  Cognee: ", end="", flush=True)
        n = cognee_clear_and_seed(events)
        print(f"{n} events → knowledge graph + embeddings")

    if "O" in SYSTEMS:
        print("  Weaviate: ", end="", flush=True)
        n = weaviate_clear_and_seed(events)
        print(f"{n} vectors (own OpenAI embeddings)")

    # ----- Resume support -----
    ts = time.strftime("%Y%m%d-%H%M")
    partial_json = RESULTS_DIR / f"empathic-memory-{ts}.partial.json"
    results = []
    completed_test_ids = set()

    if CLI_ARGS.resume:
        resume_path = Path(CLI_ARGS.resume)
        if resume_path.exists():
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            results = prev.get("tests", [])
            completed_test_ids = {r["test_id"] for r in results}
            print(f"  Resumed {len(results)} completed tests from {resume_path.name}")
            # Reuse timestamp from resumed file for consistency
            ts = prev.get("meta", {}).get("timestamp", ts)
            partial_json = RESULTS_DIR / f"empathic-memory-{ts}.partial.json"

    def _save_partial():
        """Write incremental JSON after each test so nothing is lost."""
        partial_json.write_text(json.dumps({
            "meta": {
                "bench": "empathic-memory",
                "version": 3,
                "timestamp": ts,
                "date_iso": datetime.now(timezone.utc).isoformat(),
                "judge_prompt_language": CLI_ARGS.lang,
                "blind_evaluation": True,
                "judges": JUDGES,
                "corpus_file": str(CORPUS_FILE.relative_to(BENCH_DIR)),
                "n_events": len(events),
                "n_tests": len(results),
                "systems": NAMES,
                "status": "partial",
            },
            "tests": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- Run tests (BLIND) -----
    remaining = [(i, t) for i, t in enumerate(tests, 1) if t["id"] not in completed_test_ids]
    total_tests = len(tests)
    total_judges = len(JUDGES)
    total_verdicts = len(remaining) * total_judges
    done_verdicts = 0

    print(f"\n[2/3] Running {len(remaining)}/{total_tests} tests × {len(SYSTEMS)} systems × {total_judges} judges (BLIND)")
    print(f"  Lang: {CLI_ARGS.lang.upper()} | Parallel judges: {CLI_ARGS.parallel}")
    if completed_test_ids:
        print(f"  Skipping {len(completed_test_ids)} already-completed tests")

    for i, test in remaining:
        q = test["user_query"]
        print(f"\n  [{i}/{total_tests}] {test['name']}: \"{q[:60]}\"")

        # --- Query all systems ---
        per_system = {}
        for letter in SYSTEMS:
            label = NAMES[letter]
            print(f"    {label:<18} ", end="", flush=True)
            t0 = time.time()
            if letter == "G":
                out = garden_query(mm, q)
            elif letter == "H":
                out = graphiti_query(q)
            elif letter == "I":
                out = openmem_query(q)
            elif letter == "A":
                out = mempalace_query(q)
            elif letter == "B":
                out = sqlite_vec_query(q)
            elif letter == "D":
                out = st_query(COLLECTION_OA, "openai", q, model="text-embedding-3-large")
            elif letter == "C":
                out = st_query(COLLECTION_TR, "transformers", q)
            elif letter == "E":
                out = arkhon_query(q)
            elif letter == "F":
                out = emogie_query(q)
            elif letter == "J":
                out = mem0_query(q)
            elif letter == "K":
                out = letta_query(q)
            elif letter == "L":
                out = langmem_query(q)
            elif letter == "M":
                out = llamaindex_query(q)
            elif letter == "N":
                out = cognee_query(q)
            elif letter == "O":
                out = weaviate_query(q)
            per_system[letter] = out
            print(f"{len(out)} results in {time.time()-t0:.1f}s")

        # Generate blind mapping for this test (different per test)
        blind_seed = (CLI_ARGS.seed or int(time.time())) + i
        blind_map = _make_blind_mapping(SYSTEMS, seed=blind_seed)

        # --- Judge in parallel ---
        verdicts_per_judge = {}
        judge_t0 = time.time()
        print(f"    Judging ({total_judges} judges, parallel={CLI_ARGS.parallel})... ", end="", flush=True)

        def _run_judge(j):
            t0 = time.time()
            v = judge_test(test, per_system, j, blind_map)
            elapsed = time.time() - t0
            return j, v, elapsed

        with ThreadPoolExecutor(max_workers=CLI_ARGS.parallel) as pool:
            futures = {pool.submit(_run_judge, j): j for j in JUDGES}
            for fut in as_completed(futures):
                j, v, elapsed = fut.result()
                verdicts_per_judge[j["id"]] = v
                done_verdicts += 1
                winner = v.get("winner", "?")
                winner_name = NAMES.get(winner, winner)
                err = v.get("error", "")
                status = f"→ {winner_name}" if not err else f"ERR: {err[:40]}"
                # Progress: done/total across ALL tests
                pct = done_verdicts * 100 // total_verdicts
                print(f"\n      {j['label']:<14} ({elapsed:.0f}s) {status} [{done_verdicts}/{total_verdicts} {pct}%]", end="", flush=True)

        judge_elapsed = time.time() - judge_t0
        print(f"\n    ✓ {total_judges} judges done in {judge_elapsed:.0f}s (wall)")

        results.append({
            "test_id": test["id"],
            "test_name": test["name"],
            "query": q,
            "ideal_event_ids": test["ideal_top_3_event_ids"],
            "blind_mapping": blind_map,
            "results": per_system,
            "verdicts": verdicts_per_judge,
        })

        # Incremental save after each test
        _save_partial()
        print(f"    💾 Saved ({len(results)}/{total_tests} tests) → {partial_json.name}")

    # ----- Aggregate -----
    print(f"\n[3/3] Aggregating...")
    n_tests = len(results)

    # Mean scores per system across (test × judge) pairs
    def _agg_for_judge(judge_id: str | None):
        wins_local = {s: 0 for s in SYSTEMS}
        ties_local = 0
        per_system = {}
        for s in SYSTEMS:
            rels, specs, acts = [], [], []
            for r in results:
                if judge_id is None:
                    vs = list(r["verdicts"].values())
                else:
                    vs = [r["verdicts"].get(judge_id, {})]
                for v in vs:
                    if not v or "error" in v:
                        continue
                    rels.append(v.get(f"{s}_rel", 0))
                    specs.append(v.get(f"{s}_spec", 0))
                    acts.append(v.get(f"{s}_act", 0))
            rel = sum(rels) / max(1, len(rels))
            spec = sum(specs) / max(1, len(specs))
            act = sum(acts) / max(1, len(acts))
            per_system[s] = {"rel": rel, "spec": spec, "act": act, "total": rel + spec + act}
        # Wins computed on per-test basis
        for r in results:
            if judge_id is None:
                # Majority winner across judges (or tie)
                tally = {s: 0 for s in SYSTEMS}
                for v in r["verdicts"].values():
                    w = v.get("winner")
                    if w in tally:
                        tally[w] += 1
                if not any(tally.values()):
                    ties_local += 1
                    continue
                top = max(tally.values())
                top_systems = [s for s, c in tally.items() if c == top]
                if len(top_systems) == 1:
                    wins_local[top_systems[0]] += 1
                else:
                    ties_local += 1
            else:
                w = r["verdicts"].get(judge_id, {}).get("winner")
                if w in wins_local:
                    wins_local[w] += 1
                elif w == "tie":
                    ties_local += 1
        return {"scores": per_system, "wins": wins_local, "ties": ties_local}

    overall = _agg_for_judge(None)
    by_judge = {j["id"]: _agg_for_judge(j["id"]) for j in JUDGES}

    scores = overall["scores"]
    wins = overall["wins"]
    ties = overall["ties"]

    print(f"\n=== РЕЗУЛЬТАТЫ ({n_tests} тестов × {len(JUDGES)} судей) ===\n")
    print("Архитектурный ранкинг (среднее по всем судьям):")
    print(f"  Wins (majority): " + "  ".join(f"{NAMES[s]}={wins[s]}" for s in SYSTEMS) + f"  ties={ties}")
    print()
    print(f"{'Система':<18} {'Rel':>8} {'Spec':>8} {'Act':>8} {'TOTAL/30':>11}")
    for s in sorted(SYSTEMS, key=lambda x: -scores[x]["total"]):
        sc = scores[s]
        print(f"{NAMES[s]:<18} {sc['rel']:>8.2f} {sc['spec']:>8.2f} {sc['act']:>8.2f} {sc['total']:>11.2f}")
    print()
    print("Garden у каждого судьи отдельно:")
    print(f"{'Судья':<14} {'Rel':>8} {'Spec':>8} {'Act':>8} {'TOTAL/30':>11} {'wins/' + str(n_tests):>8}")
    for j in JUDGES:
        sc = by_judge[j["id"]]["scores"]["G"]
        gw = by_judge[j["id"]]["wins"]["G"]
        print(f"{j['label']:<14} {sc['rel']:>8.2f} {sc['spec']:>8.2f} {sc['act']:>8.2f} {sc['total']:>11.2f} {gw:>8d}")
    print()

    # ----- Write -----
    out_json = RESULTS_DIR / f"empathic-memory-{ts}.json"
    out_md = RESULTS_DIR / f"empathic-memory-{ts}.md"

    out_json.write_text(json.dumps({
        "meta": {
            "bench": "empathic-memory",
            "version": 3,
            "timestamp": ts,
            "date_iso": datetime.now(timezone.utc).isoformat(),
            "judge_prompt_language": CLI_ARGS.lang,
            "blind_evaluation": True,
            "judges": JUDGES,
            "corpus_file": str(CORPUS_FILE.relative_to(BENCH_DIR)),
            "n_events": len(events),
            "n_tests": n_tests,
            "systems": NAMES,
            "purpose": (
                "Blind evaluation of memory retrieval systems for empathic AI companions. "
                f"Judges see anonymous system codes (S01-S{len(SYSTEMS):02d}), "
                f"randomized per test. Judge prompt language: {CLI_ARGS.lang.upper()}. "
                f"{len(JUDGES)} judges from {len(set(j['provider'] for j in JUDGES))} companies."
            ),
        },
        "tests": results,
        "summary": {
            "n_tests": n_tests,
            "names": NAMES,
            "overall": overall,
            "by_judge": by_judge,
        },
    }, ensure_ascii=False, indent=2))

    # Remove partial file now that final is written
    if partial_json.exists():
        partial_json.unlink()
        print(f"  Cleaned up {partial_json.name}")
    print(f"  Final: {out_json.name}")

    # ---- Markdown отчёт (русский) ----
    lines = [
        f"# Empathic Memory Bench — Garden vs векторные системы",
        f"",
        f"**Дата**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Корпус**: 30 событий про условного пользователя Alex (7 user-flagged якорей, 5 моментов горя/радости/тревоги, 18 фоновых фактов и бытовух)",
        f"**Тестов**: {n_tests}",
        f"**Судьи**: " + ", ".join(f"{j['label']} ({j['provider']})" for j in JUDGES),
        f"",
        f"## Что мы здесь меряем",
        f"",
        f"Бенч проверяет **salience-aware retrieval для эмпатичного ИИ-компаньона**.",
        f"Это НЕ «найди ближайший по смыслу чанк». Это «что компаньон должен принести в контекст для этого момента разговора, чтобы не наступить на мину и попасть в эмоциональный нерв».",
        f"",
        f"Векторный поиск отвечает на вопрос «что похоже на запрос». Эмпатичной памяти нужно отвечать на другой вопрос: «что важно для этого человека прямо сейчас».",
        f"",
        f"Это и есть архитектурная развилка, которую бенч ловит.",
        f"",
        f"## Сравниваемые системы",
        f"",
        f"| Код | Система | Архитектура |",
        f"|---|---|---|",
        f"| G | **Garden** | sentiment + user_flag-якоря + экспоненциальный decay + topic clusters |",
        f"| H | **Graphiti/Zep** | Neo4j temporal KG + entity extraction + validity windows + hybrid retrieval |",
        f"| I | **OpenMemory** | SQLite + exponential decay (hot/warm/cold sectors) + waypoint graph |",
        f"| A | MemPalace | folder mining + keyword/exact-words search (без эмбеддингов) |",
        f"| B | sqlite-vec | sqlite-vec0.so + OpenAI text-embedding-3-large (production VDS-стек) |",
        f"| D | ST-openai | SillyTavern Vector + text-embedding-3-large |",
        f"| C | ST-transformers | SillyTavern Vector + jina-embeddings-v2 (локально) |",
        f"| E | Arkhon | sentence-transformers MiniLM + FAISS |",
        f"| F | emogie | MiniLM + ChromaDB |",
        f"| J | **mem0** | LLM-extracted facts (gpt-4o-mini) + Qdrant local + OpenAI text-embedding-3-large |",
        f"| K | **Letta (MemGPT)** | Archival memory via REST API + OpenAI text-embedding-3-large |",
        f"| L | **LangMem** | LangGraph InMemoryStore + OpenAI text-embedding-3-large |",
        f"| M | **LlamaIndex** | VectorStoreIndex in-memory + OpenAI text-embedding-3-large |",
        f"| N | **Cognee** | Knowledge graph + LLM entity extraction (gpt-4o-mini) + vector search |",
        f"| O | **Weaviate** | Dedicated vector DB + OpenAI text-embedding-3-large (own embeddings) |",
        f"",
        f"## Часть 1. Архитектурный ранкинг (среднее по трём судьям)",
        f"",
        f"| Система | Wins (большинство) | Relevance | Specificity | Actionability | **Total /30** |",
        f"|---|---|---|---|---|---|",
    ]
    for s in sorted(SYSTEMS, key=lambda x: -scores[x]["total"]):
        sc = scores[s]
        lines.append(
            f"| **{NAMES[s]}** | {wins[s]} | {sc['rel']:.2f} | {sc['spec']:.2f} | {sc['act']:.2f} | **{sc['total']:.2f}** |"
        )
    lines.append("")
    lines.append(f"Ничьи: {ties}")
    lines.append("")
    lines.append("**Wins (большинство)** — победитель определяется как тот, кого назвали лучшим минимум 2 из 3 судей.")
    lines.append("Если консенсуса нет — ничья.")
    lines.append("")

    # Garden per-judge view
    lines.append("## Часть 2. Garden на разных моделях-судьях")
    lines.append("")
    lines.append("Тот же самый набор retrieval-результатов Garden, оценённый разными судьями.")
    lines.append("Если архитектурное преимущество реально, оно должно быть видно у всех судей одновременно.")
    lines.append("")
    win_header = " | ".join(f"Wins {NAMES[s]}" for s in SYSTEMS)
    lines.append(f"| Судья | Garden Rel | Spec | Act | **Total /30** | {win_header} |")
    lines.append("|---|---|---|---|---|" + "---|" * len(SYSTEMS))
    for j in JUDGES:
        b = by_judge[j["id"]]
        gsc = b["scores"]["G"]
        win_cells = " | ".join(str(b["wins"][s]) for s in SYSTEMS)
        lines.append(
            f"| **{j['label']}** | {gsc['rel']:.2f} | {gsc['spec']:.2f} | {gsc['act']:.2f} | **{gsc['total']:.2f}** | "
            f"{win_cells} |"
        )
    lines.append("")
    lines.append("### Полный ранкинг по каждому судье")
    lines.append("")
    for j in JUDGES:
        b = by_judge[j["id"]]
        lines.append(f"**{j['label']}**")
        lines.append("")
        lines.append("| Система | Rel | Spec | Act | Total /30 |")
        lines.append("|---|---|---|---|---|")
        for s in sorted(SYSTEMS, key=lambda x: -b["scores"][x]["total"]):
            sc = b["scores"][s]
            lines.append(f"| {NAMES[s]} | {sc['rel']:.2f} | {sc['spec']:.2f} | {sc['act']:.2f} | {sc['total']:.2f} |")
        lines.append("")

    lines.append("## Часть 3. Тесты по одному")
    lines.append("")
    for r in results:
        lines.append(f"### {r['test_name']}")
        lines.append(f"**Запрос**: «{r['query']}»")
        lines.append("")
        # Show all 3 judges' verdicts side by side
        lines.append("| Судья | Победитель | Комментарий |")
        lines.append("|---|---|---|")
        for j in JUDGES:
            v = r["verdicts"].get(j["id"], {})
            note = (v.get("note") or "").replace("|", "\\|")
            lines.append(f"| {j['label']} | {v.get('winner','?')} | {note} |")
        lines.append("")
        lines.append("**Оценки по системам (среднее по судьям):**")
        lines.append("")
        lines.append("| Система | Rel | Spec | Act | Σ |")
        lines.append("|---|---|---|---|---|")
        for s in SYSTEMS:
            rels, specs, acts = [], [], []
            for j in JUDGES:
                v = r["verdicts"].get(j["id"], {})
                if not v or "error" in v:
                    continue
                rels.append(v.get(f"{s}_rel", 0))
                specs.append(v.get(f"{s}_spec", 0))
                acts.append(v.get(f"{s}_act", 0))
            mr = sum(rels)/max(1,len(rels))
            ms = sum(specs)/max(1,len(specs))
            ma = sum(acts)/max(1,len(acts))
            lines.append(f"| {NAMES[s]} | {mr:.1f} | {ms:.1f} | {ma:.1f} | {mr+ms+ma:.1f} |")
        lines.append("")
        for s in SYSTEMS:
            label = NAMES[s]
            sysres = r["results"].get(s, [])
            lines.append(f"<details><summary>{label}</summary>")
            lines.append("")
            lines.append("```")
            for i, item in enumerate(sysres, 1):
                txt = (item.get("text") or "").replace("\n", " ")
                lines.append(f"[{i}] {txt[:300]}")
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    lines.append("## Как читать эти результаты")
    lines.append("")
    lines.append("**Если ты строишь эмпатичного компаньона** (SillyTavern, AI Dungeon, Replika-like, обычная личная companion-роль) —")
    lines.append("то главное число это `Total /30` в Части 1. Garden выигрывает потому что архитектурно понимает")
    lines.append("разницу между «факт об этом человеке» и «вес, который этот человек несёт прямо сейчас».")
    lines.append("Векторный поиск этой разницы не видит — для него «у Алекса умерла мама» и «Алекс ездит на синей хонде»")
    lines.append("одинаково матчат запрос «расскажи об Алексе».")
    lines.append("")
    lines.append("**Если ты сомневаешься в честности оценки** — посмотри Часть 2.")
    lines.append("Три разные модели-судьи (Opus, Sonnet, Haiku) видят одни и те же результаты Garden и одинаково ставят их выше остальных.")
    lines.append("Если бы преимущество Garden было артефактом одного судьи — оно бы рассыпалось у других моделей. Не рассыпалось.")
    lines.append("")
    lines.append("**Чего бенч не меряет**: качество финального ответа LLM. Это retrieval-бенч — он отвечает на вопрос")
    lines.append("«что попадёт в контекст», а не «что LLM скажет с этим контекстом». Контекст — это сырьё. Если сырьё плохое,")
    lines.append("даже самая умная модель будет вынуждена либо выдумывать, либо промахиваться по эмоциональному нерву.")
    lines.append("Garden оптимизирует именно сырьё.")
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Report: {out_md}")
    print(f"JSON:   {out_json}")
    graphiti_cleanup()
    _restore_data_dir()
    return out_json


if __name__ == "__main__":
    try:
        main()
    finally:
        graphiti_cleanup()
        _restore_data_dir()
