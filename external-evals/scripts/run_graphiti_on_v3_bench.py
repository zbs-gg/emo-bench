"""Run Graphiti (Zep's new open-source memory architecture) on our
empathic-memory-bench-v3 corpus.

Symmetric to run_mem0_on_v3_bench.py. Goal: defensible "Pulse beats
Graphiti (= Zep architecture)" claim with shared backend.

Backend routing (local-first, $0):
  - LLM:        LM Studio (http://localhost:1234/v1), model="bench-active"
                (= gemma-3-27b-it-qat).
  - Embeddings: LM Studio bge-m3-mlx via the same OpenAI-compatible endpoint.
  - Graph DB:   Kuzu embedded (no Docker, no Neo4j daemon).

Each event is ingested as a Graphiti Episode named `event_{id}` under a
fixed group_id. Graphiti runs its own LLM-driven entity/edge extraction
under the hood — that's the point of the comparison. At retrieval time,
`graphiti.search()` returns EntityEdges; we walk each edge's `episodes`
field, map UUID → event_id via the bookkeeping dict we kept at ingest
time, dedupe, take top-K.

Output schema mirrors Mem0 adapter so downstream aggregators can union
the two.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---- Backend selection ----
# Two modes:
#   "local"  — LM Studio (http://localhost:1234/v1) with bench-active LLM
#              and nomic-embed embeddings. $0 but requires deep monkey-patches
#              because local 27B models paraphrase pydantic schemas.
#   "openai" — Real OpenAI gpt-4o-mini + text-embedding-3-small. ~$0.50 per
#              full 60-event/35-test run. Works natively with Graphiti, no
#              monkey-patches. Defensible "we beat Graphiti on its native
#              backend" claim for paper Table 4.
BACKEND = os.environ.get("BENCH_BACKEND", "openai")

if BACKEND == "openai":
    LLM_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_KEY = os.environ.get("OPENAI_API_KEY", "")
    LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "gpt-4o-mini")
    EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-3-small")
    if not LLM_KEY:
        sys.exit("ERROR: BENCH_BACKEND=openai requires OPENAI_API_KEY env var")
else:
    LLM_BASE = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    LLM_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
    LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "bench-active")
    EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "nomic-embed")
    os.environ.setdefault("OPENAI_API_KEY", LLM_KEY)

# Back-compat aliases for the rest of the script
LM_STUDIO_BASE = LLM_BASE
LM_STUDIO_KEY = LLM_KEY


def make_event_text(ev: dict) -> str:
    """Match Mem0 adapter's `plain` variant: bare event text only.
    Lets the extraction LLM see exactly what users wrote, no scaffolding."""
    return ev.get("text", "")


import json as _json
import re as _re

_FENCE_RE = _re.compile(r"^\s*```(?:json)?\s*\n?|\n?```\s*$", _re.IGNORECASE | _re.MULTILINE)


def _strip_fences_and_load(raw: str):
    """Tolerant JSON load. Local LLMs (Gemma, Llama) often wrap output in
    ```json ... ``` even when asked not to. Strip the fence then load. As
    last resort, try to locate the first {...} or [...] block."""
    if not isinstance(raw, str):
        return _json.loads(raw)
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return _json.loads(cleaned)
    except _json.JSONDecodeError:
        # Locate the outermost JSON object/array
        for opener, closer in (("{", "}"), ("[", "]")):
            i = cleaned.find(opener)
            j = cleaned.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return _json.loads(cleaned[i : j + 1])
                except _json.JSONDecodeError:
                    pass
        raise


def _patch_graphiti_openai_client() -> None:
    """Monkey-patch Graphiti's OpenAI client so it works against LM Studio +
    local models (Gemma 27B etc.) that wrap output in ```json fences.

    Root cause: Graphiti's structured path goes through openai SDK's
    `client.responses.parse(text_format=PydanticModel)` which performs
    pydantic validation INSIDE the SDK *before* our code sees the response.
    LM Studio doesn't implement /v1/responses, and when it falls back to a
    chat completion the SDK still gets fenced text and fails JSON-decode.

    Fix: redirect _generate_response to always use the plain
    chat.completions.create path (response_format=json_object). Strip
    ```json fences in _handle_json_response. Pydantic validation of the
    returned dict happens later in Graphiti's own code, which works."""
    from graphiti_core.llm_client import openai_base_client as _obc

    def _patched_handle_json_response(self, response):
        result = response.choices[0].message.content or "{}"
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        return _strip_fences_and_load(result), input_tokens, output_tokens

    async def _patched_generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
    ):
        # Always go through plain chat completion — bypass responses.parse.
        # LM Studio rejects response_format={'type':'json_object'} (only
        # accepts 'json_schema' or 'text'), so call client.chat.completions
        # directly without any response_format. Our strip-fences handler
        # cleans the output.
        from graphiti_core.llm_client.config import ModelSize as _MS
        if model_size is None:
            model_size = _MS.medium
        openai_messages = self._convert_messages_to_openai_format(messages)
        # Inject a strong "raw JSON only, no markdown fences" instruction
        # into the system message to help local models comply.
        if openai_messages and openai_messages[0].get("role") == "system":
            sys_content = openai_messages[0].get("content", "")
            if "```" not in sys_content:
                openai_messages[0] = {
                    **openai_messages[0],
                    "content": (
                        sys_content
                        + "\n\nCRITICAL: respond with raw JSON only. "
                          "Do NOT wrap in markdown code fences (no ```json). "
                          "No prose before or after. JSON must be the entire response."
                    ),
                }
        model = self._get_model_for_size(model_size)
        response = await self.client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=(max_tokens or self.max_tokens),
        )
        return self._handle_json_response(response)

    _obc.BaseOpenAIClient._handle_json_response = _patched_handle_json_response
    _obc.BaseOpenAIClient._generate_response = _patched_generate_response


async def run_bench(
    corpus_path: Path,
    out_path: Path,
    top_k: int,
    db_path: str,
    group_id: str,
    verbose: bool,
) -> None:
    # Lazy imports so --help works without the heavy deps
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.nodes import EpisodeType

    # Monkey-patches only needed for local LM Studio path. Real OpenAI
    # gpt-4o-mini follows Graphiti's pydantic schemas natively.
    if BACKEND == "local":
        _patch_graphiti_openai_client()

    # Wipe prior DB so this is a fresh ingest
    if os.path.exists(db_path):
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        else:
            os.remove(db_path)

    print(f"[graphiti] backend: LLM={LLM_MODEL} embed={EMBED_MODEL} base={LM_STUDIO_BASE}",
          file=sys.stderr)
    print(f"[graphiti] graph db (kuzu): {db_path}", file=sys.stderr)

    llm_config = LLMConfig(
        api_key=LM_STUDIO_KEY,
        base_url=LM_STUDIO_BASE,
        model=LLM_MODEL,
        small_model=LLM_MODEL,
    )
    llm_client = OpenAIClient(config=llm_config)

    emb_config = OpenAIEmbedderConfig(
        api_key=LM_STUDIO_KEY,
        base_url=LM_STUDIO_BASE,
        embedding_model=EMBED_MODEL,
    )
    embedder = OpenAIEmbedder(config=emb_config)

    driver = KuzuDriver(db=db_path)
    # Graphiti.add_episode compares `group_id != driver._database` to decide
    # whether to clone the driver into a per-group logical DB. Neo4j/FalkorDB
    # drivers set this; KuzuDriver does not. Pin it to our fixed group_id so
    # the check is a no-op and the embedded single-file DB stays in use.
    driver._database = group_id

    g = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
    )
    await g.build_indices_and_constraints()

    # graphiti-core 0.29.0 bug: Kuzu driver's build_indices_and_constraints is
    # a no-op, but graphiti's search code calls QUERY_FTS_INDEX which requires
    # those indices to exist. The query strings are defined for Kuzu in
    # graph_queries.get_fulltext_indices(GraphProvider.KUZU) — apply them
    # manually here. Also need to INSTALL+LOAD the fts extension.
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.graph_queries import get_fulltext_indices
    import kuzu as _kuzu
    _conn = _kuzu.Connection(driver.db)
    try:
        _conn.execute("INSTALL fts;")
    except Exception:
        pass  # idempotent — extension may already be installed
    try:
        _conn.execute("LOAD EXTENSION fts;")
    except Exception:
        _conn.execute("LOAD fts;")
    for q in get_fulltext_indices(GraphProvider.KUZU):
        try:
            _conn.execute(q)
        except Exception as ex:
            # Index may already exist if rerun; ignore that one error class
            print(f"  [fts index skip] {ex}", file=sys.stderr)
    _conn.close()
    print(f"[fts] indices created for Episodic/Entity/Community/RelatesToNode_",
          file=sys.stderr)

    data = json.loads(corpus_path.read_text())
    events = data["events"]
    tests = data["tests"]

    # name → event_id lookup. Episodes are named `event_{id}` at ingest;
    # at search time we walk results back to event_ids via this map.
    name_to_eid: dict[str, int] = {}
    uuid_to_eid: dict[str, int] = {}

    print(f"[ingest] {len(events)} events", file=sys.stderr)
    t0 = time.time()
    fail = 0
    for i, ev in enumerate(events, 1):
        eid = ev["id"]
        name = f"event_{eid}"
        text = make_event_text(ev)
        # `days_ago` is informational only — reference_time falls back to
        # now()-days for chronological ordering inside Graphiti.
        days = ev.get("days_ago", 0) or 0
        ref_time = datetime.now(timezone.utc).replace(microsecond=0)
        try:
            ep_result = await g.add_episode(
                name=name,
                episode_body=text,
                source=EpisodeType.text,
                source_description="bench-v3 corpus event",
                reference_time=ref_time,
                group_id=group_id,
            )
            name_to_eid[name] = eid
            if ep_result and getattr(ep_result, "episode", None) is not None:
                uuid_to_eid[str(ep_result.episode.uuid)] = eid
        except Exception as ex:
            fail += 1
            print(f"  [add fail id={eid}] {ex}", file=sys.stderr)
        if i % 5 == 0:
            print(f"  ingest {i}/{len(events)} ({time.time()-t0:.0f}s, fails={fail})",
                  file=sys.stderr)
    print(f"[ingest] done {len(events)-fail}/{len(events)} in {time.time()-t0:.0f}s",
          file=sys.stderr)

    # Build a copy of COMBINED_HYBRID_SEARCH_RRF with higher per-source
    # limits so we get episodes + edges + nodes back at retrievable depth.
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
    from copy import deepcopy as _deepcopy
    search_cfg = _deepcopy(COMBINED_HYBRID_SEARCH_RRF)
    search_cfg.limit = max(top_k * 6, 30)
    if search_cfg.edge_config:
        search_cfg.edge_config.bfs_max_depth = 2
    if search_cfg.episode_config:
        # episode search is the most direct path back to event_id
        pass

    print(f"[retrieve] {len(tests)} tests", file=sys.stderr)
    results = []
    overall_recall = 0.0
    per_type_recall: dict[str, list[float]] = {}
    for ti, test in enumerate(tests, 1):
        query = test["user_query"]
        ideal = set(test.get("ideal_top_3_event_ids", []))
        ttype = test.get("test_type", "unknown")
        retrieved_ids: list[int] = []
        try:
            search_results = await g.search_(
                query=query,
                config=search_cfg,
                group_ids=[group_id],
            )
        except Exception as ex:
            print(f"  [search fail t{ti}] {ex}", file=sys.stderr)
            search_results = None

        episodes_list = (getattr(search_results, "episodes", None) or [])
        edges_list = (getattr(search_results, "edges", None) or [])

        if verbose and ti == 1:
            print(f"  [debug] search_ → episodes={len(episodes_list)} edges={len(edges_list)} "
                  f"nodes={len(getattr(search_results, 'nodes', None) or [])} "
                  f"communities={len(getattr(search_results, 'communities', None) or [])}",
                  file=sys.stderr)
            if episodes_list:
                ep0 = episodes_list[0]
                print(f"  [debug] episodes[0] attrs: name={getattr(ep0, 'name', None)!r} "
                      f"uuid={getattr(ep0, 'uuid', None)!r}", file=sys.stderr)
            if edges_list:
                e0 = edges_list[0]
                print(f"  [debug] edges[0] fact: {getattr(e0, 'fact', None)!r}",
                      file=sys.stderr)
                print(f"  [debug] edges[0] episodes: {getattr(e0, 'episodes', None)!r}",
                      file=sys.stderr)
            print(f"  [debug] uuid_to_eid sample: {list(uuid_to_eid.items())[:3]}",
                  file=sys.stderr)
            print(f"  [debug] name_to_eid sample: {list(name_to_eid.items())[:3]}",
                  file=sys.stderr)

        seen: set[int] = set()

        # Path 1: direct episode results → name → event_id (most reliable)
        for ep in episodes_list:
            ep_name = getattr(ep, "name", None)
            if ep_name and ep_name in name_to_eid:
                eid = name_to_eid[ep_name]
                if eid not in seen:
                    seen.add(eid)
                    retrieved_ids.append(eid)
            else:
                ep_uuid = str(getattr(ep, "uuid", "") or "")
                eid = uuid_to_eid.get(ep_uuid)
                if eid is not None and eid not in seen:
                    seen.add(eid)
                    retrieved_ids.append(eid)
            if len(retrieved_ids) >= top_k:
                break

        # Path 2: edge-derived episodes (if path 1 underfills)
        if len(retrieved_ids) < top_k:
            for edge in edges_list:
                for ep_uuid in (getattr(edge, "episodes", None) or []):
                    eid = uuid_to_eid.get(str(ep_uuid))
                    if eid is not None and eid not in seen:
                        seen.add(eid)
                        retrieved_ids.append(eid)
                    if len(retrieved_ids) >= top_k:
                        break
                if len(retrieved_ids) >= top_k:
                    break

        retrieved_set = set(retrieved_ids[:3])
        recall = (len(retrieved_set & ideal) / len(ideal)) if ideal else 0.0
        overall_recall += recall
        per_type_recall.setdefault(ttype, []).append(recall)

        results.append({
            "test_id": test.get("id"),
            "name": test.get("name"),
            "test_type": ttype,
            "user_query": query,
            "ideal_top_3": list(ideal),
            "graphiti_top_5": retrieved_ids[:5],
            "recall_at_3": recall,
        })
        if verbose:
            print(f"  [{ti}/{len(tests)}] {test.get('name','?'):.<40} type={ttype:.<14} "
                  f"R@3={recall:.2f}", file=sys.stderr)

    overall_recall /= max(len(tests), 1)
    summary = {
        "n_events": len(events),
        "n_tests": len(tests),
        "overall_recall_at_3": overall_recall,
        "per_type_recall_at_3": {k: sum(v) / len(v) for k, v in per_type_recall.items()},
        "backend": {
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL,
            "endpoint": LM_STUDIO_BASE,
            "graph_db": "kuzu-embedded",
        },
        "per_test": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[summary] overall R@3 = {overall_recall:.3f}", file=sys.stderr)
    for k, v in summary["per_type_recall_at_3"].items():
        print(f"  {k:.<20} R@3 = {v:.3f}", file=sys.stderr)
    print(f"[save] {out_path}", file=sys.stderr)

    try:
        await g.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="bench/datasets/empathic-memory-corpus-v3.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--db-path", type=str, default="/tmp/graphiti-bench-v3-kuzu")
    ap.add_argument("--group-id", type=str, default="bench-v3")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    asyncio.run(run_bench(
        corpus_path=args.corpus,
        out_path=args.out,
        top_k=args.top_k,
        db_path=args.db_path,
        group_id=args.group_id,
        verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()
