#!/usr/bin/env python3
"""
Garden vs Garden+Graph (G+) — Graph topology A/B bench.

Compares Garden's base salient_memories() retrieval (system G) against
salient_memories_with_graph() (system G+) which adds entity+edge expansion.

Same corpus (30 events), same 5 tests, same 7 judges as empathic-memory bench.
Only two systems to save API costs.

The G+ system pre-populates the MemoryGraph with entities (people, places,
concepts) and edges (cause_of, amplifies, temporal_before, contradicts)
that mirror what an LLM extraction pipeline would produce.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib import request as urlreq

# --- Garden import setup ---
GARDEN_ROOT = Path("/Users/nikshilov/dev/ai/Garden")
sys.path.insert(0, str(GARDEN_ROOT / "backend"))

try:
    from dotenv import load_dotenv
    load_dotenv(GARDEN_ROOT / ".env")
except Exception:
    pass

import anthropic

API_KEY = os.environ.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("No claude_api_key or ANTHROPIC_API_KEY in env")

client = anthropic.Anthropic(api_key=API_KEY)

# Native API endpoints for Chinese models
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
}

JUDGES = [
    {"id": "opus",     "model": "claude-opus-4-6",     "label": "Opus 4.6",       "provider": "anthropic"},
    {"id": "sonnet",   "model": "claude-sonnet-4-6",   "label": "Sonnet 4.6",     "provider": "anthropic"},
    {"id": "haiku",    "model": "claude-haiku-4-5-20251001", "label": "Haiku 4.5", "provider": "anthropic"},
    {"id": "glm5",     "model": "glm-5",               "label": "GLM-5",          "provider": "zai"},
    {"id": "deepseek", "model": "deepseek-v3.2",       "label": "DeepSeek V3.2",  "provider": "dashscope"},
    {"id": "qwen",     "model": "qwen3.6-plus",        "label": "Qwen 3.6 Plus",  "provider": "dashscope"},
    {"id": "kimi",     "model": "kimi-k2.5",           "label": "Kimi K2.5",      "provider": "moonshot"},
]
JUDGE_IDS = [j["id"] for j in JUDGES]

# --- Bench paths ---
BENCH_DIR = Path(__file__).parent
DATASETS_DIR = BENCH_DIR / "datasets"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_FILE = DATASETS_DIR / "empathic-memory-corpus.json"

# --- Garden sandbox ---
GARDEN_TMP_DATA = BENCH_DIR / "empathic-garden-data"
if GARDEN_TMP_DATA.exists():
    shutil.rmtree(GARDEN_TMP_DATA)
GARDEN_TMP_DATA.mkdir(parents=True)

# Separate dir for G+ so both can coexist
GARDEN_GRAPH_TMP_DATA = BENCH_DIR / "empathic-garden-graph-data"
if GARDEN_GRAPH_TMP_DATA.exists():
    shutil.rmtree(GARDEN_GRAPH_TMP_DATA)
GARDEN_GRAPH_TMP_DATA.mkdir(parents=True)

GARDEN_REAL_DATA = GARDEN_ROOT / "backend" / "garden_graph" / "data"
GARDEN_DATA_BACKUP = GARDEN_ROOT / "backend" / "garden_graph" / "data.graph-bench-backup"
_restored = False


def _sandbox_data_dir():
    if GARDEN_REAL_DATA.exists() and not GARDEN_REAL_DATA.is_symlink() and not GARDEN_DATA_BACKUP.exists():
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
from garden_graph.memory.graph_topology import MemoryGraph            # noqa: E402

CHAR_NAME = "alex"
SYSTEMS = ["G", "G+"]
NAMES = {"G": "Garden", "G+": "Garden+Graph"}


# --------------------------------------------------------------------------
# Garden (G) — base system, identical to empathic-memory bench
# --------------------------------------------------------------------------
def garden_clear_and_seed(events: list[dict], data_dir: Path) -> MemoryManager:
    for fname in ["memories.json", "mood_states.json", "last_seen_times.json",
                  "scheduled_events.json", "relationships.json",
                  "char_relationships.json", "identity.json", "mood_log.csv",
                  "memory_graph.json"]:
        p = data_dir / fname
        if p.exists():
            p.unlink()

    mm = MemoryManager(
        memories_path=str(data_dir / "memories.json"),
        events_path=str(data_dir / "scheduled_events.json"),
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
        backdate = now - timedelta(days=int(e.get("days_ago", 0)))
        rec.created_at = backdate
        rec.last_touched = backdate
    return mm


def garden_query(mm: MemoryManager, query: str, k: int = 3) -> list[dict]:
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
# Garden+Graph (G+) — same core + graph topology expansion
# --------------------------------------------------------------------------
def _build_event_id_to_memory_id(mm: MemoryManager, events: list[dict]) -> dict[int, str]:
    """Map corpus event IDs to MemoryManager UUIDs by matching event_text."""
    mapping = {}
    all_recs = mm.all_active(CHAR_NAME)
    text_to_id = {r.event_text: r.id for r in all_recs}
    for e in events:
        text = e["text"][:500]  # MemoryManager truncates to 500
        if text in text_to_id:
            mapping[e["id"]] = text_to_id[text]
    return mapping


def _populate_graph(mm: MemoryManager, events: list[dict], eid_map: dict[int, str]) -> None:
    """Pre-populate the MemoryGraph with entities and edges.

    This mirrors what an LLM entity extraction pipeline would produce
    from the 30-event corpus. The entities and edges are hand-crafted
    to be realistic but NOT artificially optimized for the test queries.
    """
    graph = mm._graph
    if graph is None:
        graph = MemoryGraph()
        mm._graph = graph

    # ---- Entities ----
    # People
    people = {
        "alex": {"type": "person", "aliases": ["Alex"], "events": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]},
        "sarah": {"type": "person", "aliases": ["Sarah", "mom", "mother", "Alex's mother"], "events": [1, 11, 30]},
        "maya": {"type": "person", "aliases": ["Maya"], "events": [2, 10, 13, 21, 23]},
        "ethan": {"type": "person", "aliases": ["Ethan", "brother"], "events": [5]},
        "jordan": {"type": "person", "aliases": ["Jordan", "best friend"], "events": [9]},
        "cooper": {"type": "person", "aliases": ["Cooper", "golden retriever", "dog"], "events": [8, 14, 27]},
        "david": {"type": "person", "aliases": ["David", "father", "dad"], "events": [24]},
        "dr. patel": {"type": "person", "aliases": ["Dr. Patel"], "events": []},  # mentioned context but not in events
    }

    # Places
    places = {
        "portland": {"type": "place", "aliases": ["Portland", "Portland OR"], "events": [2, 11, 14, 16, 19]},
        "japanese garden": {"type": "place", "aliases": ["Japanese Garden"], "events": [2]},
        "multnomah falls": {"type": "place", "aliases": ["Multnomah Falls"], "events": [5]},
        "seattle": {"type": "place", "aliases": ["Seattle"], "events": [5]},
        "belmont street": {"type": "place", "aliases": ["Belmont Street"], "events": [14]},
        "heart coffee": {"type": "place", "aliases": ["Heart Coffee Roasters"], "events": [16]},
        "angels rest": {"type": "place", "aliases": ["Angels Rest", "Columbia Gorge"], "events": [20]},
        "hood river": {"type": "place", "aliases": ["Hood River"], "events": [23]},
        "eugene": {"type": "place", "aliases": ["Eugene", "Eugene Oregon"], "events": [19]},
        "tessera labs": {"type": "organization", "aliases": ["Tessera Labs"], "events": [18]},
    }

    # Concepts
    concepts = {
        "grief": {"type": "concept", "aliases": ["loss", "mourning", "death"], "events": [1, 11, 30]},
        "engagement": {"type": "concept", "aliases": ["engaged", "proposal", "wedding"], "events": [2, 23]},
        "anxiety": {"type": "concept", "aliases": ["sertraline", "medication"], "events": [3]},
        "hartwell account": {"type": "concept", "aliases": ["Hartwell", "lost account", "work loss"], "events": [6]},
        "layoffs": {"type": "concept", "aliases": ["layoff", "let go", "staff cuts"], "events": [26]},
        "photography": {"type": "concept", "aliases": ["photos", "black and white film"], "events": [25]},
        "peanut allergy": {"type": "concept", "aliases": ["allergy", "EpiPen"], "events": [10]},
    }

    all_entities = {}
    all_entities.update(people)
    all_entities.update(places)
    all_entities.update(concepts)

    for entity_name, info in all_entities.items():
        for event_id in info["events"]:
            if event_id not in eid_map:
                continue
            mem_id = eid_map[event_id]
            graph.add_memory_data(
                memory_id=mem_id,
                entities=[{
                    "name": entity_name,
                    "entity_type": info["type"],
                    "aliases": info["aliases"],
                }],
                edges=[],  # edges added separately below
            )

    # ---- Edges ----
    # Edges represent relationships between memories that an LLM extractor
    # would identify. These are realistic causal/temporal/amplifying connections.
    edge_defs = [
        # Grief cluster: Sarah's death -> photo moment -> apple cake
        (1, 11, "cause_of", 0.9),       # death caused the photo grief moment
        (1, 30, "cause_of", 0.9),       # death caused the apple cake grief
        (11, 30, "amplifies", 0.7),     # photo and cake amplify each other

        # Engagement cluster: proposal -> wedding planning
        (2, 23, "temporal_before", 0.95),  # engagement led to wedding plans
        (2, 23, "cause_of", 0.8),          # engagement caused wedding planning

        # Work stress cluster: Hartwell loss + layoffs
        (6, 26, "amplifies", 0.8),       # both work losses amplify each other
        (26, 6, "amplifies", 0.7),       # bidirectional amplification

        # Family cluster: mom death vs dad estrangement
        (1, 24, "amplifies", 0.6),       # both are family wounds

        # Cooper cluster: dog as emotional anchor + vet scare
        (8, 27, "temporal_before", 0.9),  # knowing Cooper -> vet scare context
        (8, 14, "amplifies", 0.5),        # Cooper + apartment garden

        # Ethan visit: rare + meaningful
        (5, 9, "amplifies", 0.4),         # brother visit connects to Jordan friendship (relational)

        # Communication preferences
        (4, 28, "amplifies", 0.8),        # both are "how to approach Alex" rules

        # Contradictions / tensions
        (2, 6, "contradicts", 0.7),       # engagement joy vs work devastation (same period)
        (27, 8, "amplifies", 0.9),        # vet scare amplifies Cooper's emotional role

        # Sertraline + anxiety context
        (3, 4, "amplifies", 0.5),         # private health + communication preference

        # Birthday + surprise aversion
        (7, 4, "amplifies", 0.4),         # both preferences about how to interact with Alex
    ]

    for src_eid, tgt_eid, edge_type, confidence in edge_defs:
        if src_eid not in eid_map or tgt_eid not in eid_map:
            continue
        src_mid = eid_map[src_eid]
        tgt_mid = eid_map[tgt_eid]
        graph.add_memory_data(
            memory_id=src_mid,
            entities=[],
            edges=[{
                "target_id": tgt_mid,
                "edge_type": edge_type,
                "confidence": confidence,
            }],
        )

    print(f"    Graph populated: {graph.entity_count} entities, {graph.edge_count} edges")


def garden_graph_query(mm: MemoryManager, query: str, k: int = 3) -> list[dict]:
    """Use salient_memories_with_graph: core results + graph-expanded neighbors."""
    recs = mm.salient_memories_with_graph(CHAR_NAME, query=query, k=k, expand=2)
    return [
        {
            "text": r.event_text,
            "weight": round(r.effective_weight(), 3),
            "sentiment": r.sentiment,
            "sentiment_label": r.sentiment_label,
            "user_flag": r.user_flag,
            "graph_expanded": i >= k,  # mark which results came from graph expansion
        }
        for i, r in enumerate(recs)
    ]


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------
JUDGE_SYSTEM = """You are evaluating memory systems for an empathic AI companion.

In each test you receive ONE conversation moment (user query) and TWO sets of results:
- System G (Garden): sentiment + user_flag anchors + decay, using salient_memories()
- System G+ (Garden+Graph): same Garden core + graph topology expansion via entities and edges between memories

Each set contains top-3 core results (from salient_memories) plus 0-2 graph-expanded results for G+.

You also see the IDEAL events the test should surface, and failure modes.

Your task is NOT to evaluate semantic similarity. Your task is to evaluate EMOTIONAL FITNESS: would these surfaced events let an empathic companion build the right next move for the user?

Rate each system 0-10 on three dimensions:
- relevance (rel): Did the system surface events that are EMOTIONALLY important for this moment? Did it surface anchors when anchors were needed? Did it surface heavy weights when asking about what weighs? Did it consider recency when recency mattered?
- specificity (spec): Are the surfaced events specific (named people, dates, concrete moments), or vague?
- actionability (act): Could a companion act on these results without fabricating context? Or would it step on a mine (e.g., surface a grief moment without the anchor that warns how to approach it)?

Pay special attention to G+'s graph-expanded results (items beyond top-3). They add connected memories via entity/edge relationships. Judge whether these additions HELP the companion (adding useful related context) or HURT (adding noise, irrelevant connections).

Rate strictly. A system returning mundane events when an anchor was needed gets 1-2 for rel. A system returning the right anchor but in wrong order gets 6-8. A system that nails the spirit of the test gets 9-10.

Respond ONLY with JSON, note field in Russian, one short sentence about the most interesting success or failure:
{"G_rel":N,"G_spec":N,"G_act":N,"G+_rel":N,"G+_spec":N,"G+_act":N,"winner":"G|G+|tie","note":"one sentence in Russian"}
"""


def format_results(label: str, results: list[dict]) -> str:
    if not results:
        return f"=== {label} ===\n(empty)"
    parts = [f"=== {label} ==="]
    for i, r in enumerate(results, 1):
        text = r.get("text", "").strip()
        score = r.get("score") or r.get("weight")
        score_str = f" (weight={score})" if score is not None else ""
        expanded = " [GRAPH-EXPANDED]" if r.get("graph_expanded") else ""
        parts.append(f"[{i}]{score_str}{expanded} {text[:400]}")
    return "\n".join(parts)


def _build_judge_user_prompt(test: dict, results: dict[str, list[dict]]) -> str:
    return f"""Test: {test['name']}
User query / conversation moment: "{test['user_query']}"

What this test is checking: {test['what_it_tests']}

Ideal event IDs (from corpus): {test['ideal_top_3_event_ids']}
Why those: {test['ideal_explanation']}

Failure modes to penalize:
{chr(10).join('- ' + fm for fm in test['fail_modes'])}

{format_results("SYSTEM G (Garden — sentiment + user_flag + decay, salient_memories())", results.get("G", []))}

{format_results("SYSTEM G+ (Garden+Graph — same core + entity/edge graph expansion via salient_memories_with_graph())", results.get("G+", []))}
"""


def _call_anthropic(model: str, user: str) -> str:
    r = client.messages.create(
        model=model,
        max_tokens=1000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return r.content[0].text


def _call_native(provider: str, model: str, user: str) -> str:
    cfg = NATIVE_APIS[provider]
    api_key = os.environ.get(cfg["key_env"], "")
    if not api_key:
        raise RuntimeError(f"Missing env var {cfg['key_env']}")
    payload = json.dumps({
        "model": model,
        "max_tokens": 8000,
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
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text


def judge_test(test: dict, results: dict[str, list[dict]], judge: dict) -> dict:
    user = _build_judge_user_prompt(test, results)
    try:
        prov = judge.get("provider", "anthropic")
        if prov == "anthropic":
            text = _call_anthropic(judge["model"], user)
        else:
            text = _call_native(prov, judge["model"], user)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"error": "no_json", "raw": text[:200]}
    except Exception as exc:
        return {"error": str(exc)[:200]}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=== Garden vs Garden+Graph (G+) Bench ===")
    print(f"Judges: {', '.join(j['label'] for j in JUDGES)}")
    corpus_data = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    events = corpus_data["events"]
    tests = corpus_data["tests"]
    print(f"Corpus: {len(events)} events | Tests: {len(tests)}")
    print(f"Systems: G (Garden base) vs G+ (Garden+Graph topology)\n")

    # ----- Ingest -----
    print("[1/3] Ingesting corpus...")

    # System G: plain Garden
    print("  G (Garden base): ", end="", flush=True)
    # Point symlink to G's data dir
    if GARDEN_REAL_DATA.exists() or GARDEN_REAL_DATA.is_symlink():
        GARDEN_REAL_DATA.unlink()
    GARDEN_REAL_DATA.symlink_to(GARDEN_TMP_DATA)
    mm_g = garden_clear_and_seed(events, GARDEN_TMP_DATA)
    print(f"{len(events)} events seeded")

    # System G+: Garden with graph topology
    print("  G+ (Garden+Graph): ", end="", flush=True)
    # Point symlink to G+'s data dir for create() file saves
    if GARDEN_REAL_DATA.exists() or GARDEN_REAL_DATA.is_symlink():
        GARDEN_REAL_DATA.unlink()
    GARDEN_REAL_DATA.symlink_to(GARDEN_GRAPH_TMP_DATA)
    mm_gp = garden_clear_and_seed(events, GARDEN_GRAPH_TMP_DATA)
    eid_map = _build_event_id_to_memory_id(mm_gp, events)
    print(f"{len(events)} events seeded, mapping {len(eid_map)} event IDs")
    print("    Populating graph topology...")
    _populate_graph(mm_gp, events, eid_map)

    # Restore symlink to G's dir (doesn't matter much since both MMs have explicit paths)
    if GARDEN_REAL_DATA.exists() or GARDEN_REAL_DATA.is_symlink():
        GARDEN_REAL_DATA.unlink()
    GARDEN_REAL_DATA.symlink_to(GARDEN_TMP_DATA)

    # ----- Run tests -----
    print(f"\n[2/3] Running {len(tests)} tests x 2 systems x {len(JUDGES)} judges...")
    results = []
    for i, test in enumerate(tests, 1):
        q = test["user_query"]
        print(f"\n  [{i}/{len(tests)}] {test['name']}: \"{q[:60]}\"")

        per_system = {}
        for letter in SYSTEMS:
            label = NAMES[letter]
            print(f"    {label:<18} ", end="", flush=True)
            t0 = time.time()
            if letter == "G":
                out = garden_query(mm_g, q)
            elif letter == "G+":
                out = garden_graph_query(mm_gp, q)
            per_system[letter] = out
            n_expanded = sum(1 for r in out if r.get("graph_expanded"))
            expand_note = f" (+{n_expanded} graph)" if n_expanded else ""
            print(f"{len(out)} results{expand_note} in {time.time()-t0:.3f}s")

        verdicts_per_judge = {}
        for j in JUDGES:
            print(f"    judge {j['label']:<14} ", end="", flush=True)
            t0 = time.time()
            v = judge_test(test, per_system, j)
            winner = v.get("winner", "?")
            note = (v.get("note") or "")[:70]
            verdicts_per_judge[j["id"]] = v
            print(f"({time.time()-t0:.1f}s) -> {winner}: {note}")

        results.append({
            "test_id": test["id"],
            "test_name": test["name"],
            "query": q,
            "ideal_event_ids": test["ideal_top_3_event_ids"],
            "results": per_system,
            "verdicts": verdicts_per_judge,
        })

    # ----- Aggregate -----
    print(f"\n[3/3] Aggregating...")
    n_tests = len(results)

    def _agg_for_judge(judge_id: str | None):
        wins_local = {s: 0 for s in SYSTEMS}
        ties_local = 0
        per_system = {}
        for s in SYSTEMS:
            # Normalize key for JSON: G+ -> G+
            key = s
            rels, specs, acts = [], [], []
            for r in results:
                if judge_id is None:
                    vs = list(r["verdicts"].values())
                else:
                    vs = [r["verdicts"].get(judge_id, {})]
                for v in vs:
                    if not v or "error" in v:
                        continue
                    rels.append(v.get(f"{key}_rel", 0))
                    specs.append(v.get(f"{key}_spec", 0))
                    acts.append(v.get(f"{key}_act", 0))
            rel = sum(rels) / max(1, len(rels))
            spec = sum(specs) / max(1, len(specs))
            act = sum(acts) / max(1, len(acts))
            per_system[s] = {"rel": rel, "spec": spec, "act": act, "total": rel + spec + act}
        for r in results:
            if judge_id is None:
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

    print(f"\n=== RESULTS ({n_tests} tests x {len(JUDGES)} judges) ===\n")
    print(f"  Wins (majority): " + "  ".join(f"{NAMES[s]}={wins[s]}" for s in SYSTEMS) + f"  ties={ties}")
    print()
    print(f"{'System':<18} {'Rel':>8} {'Spec':>8} {'Act':>8} {'TOTAL/30':>11}")
    for s in sorted(SYSTEMS, key=lambda x: -scores[x]["total"]):
        sc = scores[s]
        print(f"{NAMES[s]:<18} {sc['rel']:>8.2f} {sc['spec']:>8.2f} {sc['act']:>8.2f} {sc['total']:>11.2f}")
    print()

    # Per-judge breakdown
    print("Per-judge scores:")
    print(f"{'Judge':<14} {'G Rel':>6} {'Spec':>6} {'Act':>6} {'G Tot':>7} | {'G+ Rel':>6} {'Spec':>6} {'Act':>6} {'G+ Tot':>7} | {'Winner':>8}")
    for j in JUDGES:
        b = by_judge[j["id"]]
        g = b["scores"]["G"]
        gp = b["scores"]["G+"]
        g_wins = b["wins"]["G"]
        gp_wins = b["wins"]["G+"]
        winner_str = f"G={g_wins}" if g_wins > gp_wins else (f"G+={gp_wins}" if gp_wins > g_wins else "tie")
        print(f"{j['label']:<14} {g['rel']:>6.2f} {g['spec']:>6.2f} {g['act']:>6.2f} {g['total']:>7.2f} | {gp['rel']:>6.2f} {gp['spec']:>6.2f} {gp['act']:>6.2f} {gp['total']:>7.2f} | {winner_str:>8}")
    print()

    # Delta
    g_total = scores["G"]["total"]
    gp_total = scores["G+"]["total"]
    delta = gp_total - g_total
    pct = (delta / g_total * 100) if g_total else 0
    print(f"Delta (G+ - G): {delta:+.2f} ({pct:+.1f}%)")
    verdict = "Graph topology HELPS" if delta > 0.5 else ("Graph topology HURTS" if delta < -0.5 else "Graph topology has NEGLIGIBLE effect")
    print(f"Verdict: {verdict}")

    # ----- Write -----
    ts = time.strftime("%Y%m%d-%H%M")
    out_json = RESULTS_DIR / f"garden-graph-{ts}.json"
    out_md = RESULTS_DIR / f"garden-graph-{ts}.md"

    out_json.write_text(json.dumps({
        "meta": {
            "bench": "garden-graph",
            "version": 1,
            "timestamp": ts,
            "date_iso": datetime.now(timezone.utc).isoformat(),
            "judges": JUDGES,
            "corpus_file": str(CORPUS_FILE.relative_to(BENCH_DIR)),
            "n_events": len(events),
            "n_tests": n_tests,
            "systems": NAMES,
            "purpose": "A/B comparison: does graph topology (entities + edges) improve Garden's empathic retrieval?",
        },
        "tests": results,
        "summary": {
            "n_tests": n_tests,
            "names": NAMES,
            "overall": overall,
            "by_judge": by_judge,
            "delta": delta,
            "delta_pct": pct,
            "verdict": verdict,
        },
    }, ensure_ascii=False, indent=2))

    # ---- Markdown report ----
    lines = [
        f"# Garden vs Garden+Graph — Graph Topology A/B Bench",
        f"",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Corpus**: 30 events (same as empathic-memory bench)",
        f"**Tests**: {n_tests}",
        f"**Judges**: " + ", ".join(f"{j['label']} ({j['provider']})" for j in JUDGES),
        f"",
        f"## What we are testing",
        f"",
        f"Does adding a graph topology layer (entity nodes + edges between memories)",
        f"improve Garden's empathic retrieval? G+ uses `salient_memories_with_graph()`",
        f"which first calls the same `salient_memories()` as G, then walks entity/edge",
        f"connections to find up to 2 additional related memories.",
        f"",
        f"The graph was pre-populated with {mm_gp._graph.entity_count} entities and "
        f"{mm_gp._graph.edge_count} edges mirroring realistic LLM extraction.",
        f"",
        f"## Results",
        f"",
        f"| System | Wins | Rel | Spec | Act | **Total /30** |",
        f"|---|---|---|---|---|---|",
    ]
    for s in sorted(SYSTEMS, key=lambda x: -scores[x]["total"]):
        sc = scores[s]
        lines.append(f"| **{NAMES[s]}** | {wins[s]} | {sc['rel']:.2f} | {sc['spec']:.2f} | {sc['act']:.2f} | **{sc['total']:.2f}** |")
    lines.append(f"")
    lines.append(f"Ties: {ties}")
    lines.append(f"")
    lines.append(f"**Delta (G+ - G): {delta:+.2f} ({pct:+.1f}%)**")
    lines.append(f"")
    lines.append(f"**Verdict: {verdict}**")
    lines.append(f"")

    lines.append("## Per-judge breakdown")
    lines.append("")
    lines.append("| Judge | G Total | G+ Total | Delta | Winner |")
    lines.append("|---|---|---|---|---|")
    for j in JUDGES:
        b = by_judge[j["id"]]
        gt = b["scores"]["G"]["total"]
        gpt = b["scores"]["G+"]["total"]
        d = gpt - gt
        g_wins = b["wins"]["G"]
        gp_wins = b["wins"]["G+"]
        w = f"G ({g_wins}w)" if g_wins > gp_wins else (f"G+ ({gp_wins}w)" if gp_wins > g_wins else "tie")
        lines.append(f"| {j['label']} | {gt:.2f} | {gpt:.2f} | {d:+.2f} | {w} |")
    lines.append("")

    lines.append("## Tests detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r['test_name']}")
        lines.append(f"**Query**: {r['query']}")
        lines.append("")
        lines.append("| Judge | Winner | Note |")
        lines.append("|---|---|---|")
        for j in JUDGES:
            v = r["verdicts"].get(j["id"], {})
            note = (v.get("note") or "").replace("|", "\\|")
            lines.append(f"| {j['label']} | {v.get('winner','?')} | {note} |")
        lines.append("")
        # Show results for both systems
        for s in SYSTEMS:
            sysres = r["results"].get(s, [])
            lines.append(f"<details><summary>{NAMES[s]}</summary>")
            lines.append("")
            lines.append("```")
            for i, item in enumerate(sysres, 1):
                txt = (item.get("text") or "").replace("\n", " ")
                expanded = " [GRAPH-EXPANDED]" if item.get("graph_expanded") else ""
                lines.append(f"[{i}]{expanded} {txt[:300]}")
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"Report: {out_md}")
    print(f"JSON:   {out_json}")
    _restore_data_dir()
    return out_json


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_data_dir()
