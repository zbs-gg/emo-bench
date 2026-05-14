"""Embedding provider abstraction for DO bge-m3 with Cohere + local-mlx alternates.

Selection precedence:
  1. explicit `provider` arg
  2. env var `EMBEDDING_PROVIDER` (`bge-m3` | `cohere` | `local-mlx`)
  3. fallback: `bge-m3`

All providers return L2-normalised np.float32 (N, D) arrays — drop-in for
existing cosine pipelines (event @ q_vec is dot-product on unit vectors).

BGE-M3 (DO):        D=1024, $0.02/M, multilingual, MIT-licensed (FlagEmbedding)
Cohere embed-v4.0:  D=1536, $0.10/M direct, legacy/reproduction path only
local-mlx (mlx):    D=1024, $0/call, on-device M-series, bge-m3-mlx-fp16
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import secret, embed_cohere

# DigitalOcean Inference OpenAI-compatible endpoint
_DO_BASE_URL = "https://inference.do-ai.run/v1"
_DO_BGE_M3_MODEL = "bge-m3"

# bge-m3 max sequence length is 8192 tokens; we conservatively cap by chars
# to stay well under the limit on Russian text (Cyrillic = 1-2 bytes/char).
_MAX_CHARS_PER_TEXT = 8000

# Empirical default batch — DO endpoint accepts large batches.
_DEFAULT_BATCH = 64


def _embed_bge_m3(texts: list[str], batch: int = _DEFAULT_BATCH) -> np.ndarray:
    """Embed via DO bge-m3. OpenAI-compat schema: {"data":[{"embedding":[...]}, ...]}.

    Note: bge-m3 has no input_type concept (it's a single dense model, not
    asymmetric like Cohere search_query / search_document). We accept and
    ignore the input_type arg for caller-side parity.
    """
    import requests

    api_key = secret("do-token.txt")
    url = f"{_DO_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    truncated = [t[:_MAX_CHARS_PER_TEXT] for t in texts]
    vecs: list[list[float]] = []

    for i in range(0, len(truncated), batch):
        chunk = truncated[i:i + batch]
        payload = {"input": chunk, "model": _DO_BGE_M3_MODEL}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(
                f"bge-m3 embedding failed: HTTP {r.status_code} — "
                f"{r.text[:400]}"
            )
        data = r.json()
        if "data" not in data:
            raise RuntimeError(f"bge-m3: unexpected response shape: {list(data.keys())}")
        for item in data["data"]:
            vecs.append(item["embedding"])

    m = np.asarray(vecs, dtype=np.float32)
    # L2-normalise so caller can use dot-product = cosine similarity
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    m /= norms
    return m


# --- local-mlx (on-device, free) -----------------------------------------

_MLX_HELPER = "/Users/nikshilov/dev/ai/Garden/pulse/scripts/extract/mlx_embed_helper.py"
_MLX_PYTHON = "/Users/nikshilov/dev/ai/Garden/pulse/.venv-mlx/bin/python"
_MLX_MODEL  = "/Volumes/Celeste/llm/mlx-community/bge-m3-mlx-fp16"

_mlx_proc: subprocess.Popen | None = None
_mlx_lock = Lock()
_mlx_call_id = 0


def _ensure_mlx() -> subprocess.Popen:
    """Spawn the long-running mlx helper once. Subsequent calls reuse it
    so the bge-m3 model stays in memory between batches."""
    global _mlx_proc
    if _mlx_proc is not None and _mlx_proc.poll() is None:
        return _mlx_proc
    proc = subprocess.Popen(
        [_MLX_PYTHON, _MLX_HELPER, "--model-path", _MLX_MODEL],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Helper prints "loading..." then "ready" on stderr, AND emits a
    # `__startup__` JSON line on stdout once the model is loaded.
    # Wait for the stdout startup signal so we know stdin is ready
    # to accept requests; the buffered startup line gets consumed here.
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("mlx helper terminated before startup signal")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("id", "").startswith("__"):
            break
    _mlx_proc = proc
    return proc


def _embed_local_mlx(texts: list[str], batch: int = _DEFAULT_BATCH) -> np.ndarray:
    """Embed via the local on-device MLX helper. Free; ~15 ms/text on M4 Max."""
    global _mlx_call_id
    truncated = [t[:_MAX_CHARS_PER_TEXT] for t in texts]
    vecs: list[list[float]] = []
    with _mlx_lock:
        proc = _ensure_mlx()
        for i in range(0, len(truncated), batch):
            chunk = truncated[i:i + batch]
            _mlx_call_id += 1
            req_id = f"emb_{_mlx_call_id}"
            payload = {"id": req_id, "texts": chunk}
            try:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except BrokenPipeError as e:
                raise RuntimeError(f"mlx helper pipe closed: {e}")
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("mlx helper closed stdout (no response)")
            resp = json.loads(line)
            if resp.get("id") != req_id:
                raise RuntimeError(
                    f"mlx helper id mismatch: sent {req_id}, got {resp.get('id')}"
                )
            if "error" in resp:
                raise RuntimeError(f"mlx helper error: {resp['error']}")
            vecs.extend(resp["embeddings"])
    m = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    m /= norms
    return m


def embed_texts(
    texts: list[str],
    input_type: Literal["search_query", "search_document"],
    provider: str | None = None,
    batch: int = _DEFAULT_BATCH,
) -> np.ndarray:
    """Provider-aware embed wrapper.

    Args:
        texts: list of strings to embed.
        input_type: 'search_query' | 'search_document'. Honoured by Cohere
                    asymmetric legacy model; ignored by bge-m3 (symmetric).
        provider: 'bge-m3' | 'cohere'. If None, reads env EMBEDDING_PROVIDER.
        batch: batch size per HTTP call.

    Returns:
        np.ndarray (N, D) of L2-normalised float32 embeddings.
    """
    if provider is None:
        provider = os.environ.get("EMBEDDING_PROVIDER", "bge-m3").lower()

    if provider in ("bge-m3", "bge_m3", "bgem3"):
        return _embed_bge_m3(texts, batch=batch)
    if provider == "cohere":
        return embed_cohere(texts, input_type, batch=batch)
    if provider in ("local-mlx", "local_mlx", "mlx", "bge-m3-mlx"):
        return _embed_local_mlx(texts, batch=batch)
    if provider in ("openai-small", "text-embedding-3-small", "openai-3-small"):
        return _embed_openai(texts, model="text-embedding-3-small", batch=batch)
    if provider in ("openai-large", "text-embedding-3-large", "openai-3-large"):
        return _embed_openai(texts, model="text-embedding-3-large", batch=batch)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. "
        f"Allowed: 'bge-m3' | 'cohere' | 'local-mlx' | "
        f"'text-embedding-3-small' | 'text-embedding-3-large'."
    )


def _embed_openai(texts: list[str], model: str, batch: int = _DEFAULT_BATCH) -> np.ndarray:
    """Embed via OpenAI text-embedding-3-{small,large}. Used to produce
    backbone-matched Pulse v3 rows against memory-system baselines that
    also use OpenAI embeddings. Returns L2-normalised float32."""
    import urllib.request
    import json as _json
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set for openai-embedding-3 provider")
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        body = _json.dumps({"input": chunk, "model": model}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            d = _json.loads(r.read())
        for item in d["data"]:
            vecs.append(item["embedding"])
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def active_provider() -> str:
    """Resolve the provider name without performing any network calls.
    Useful for snapshot summaries / logging."""
    return os.environ.get("EMBEDDING_PROVIDER", "bge-m3").lower()


def embedding_dim(provider: str | None = None) -> int:
    """Return expected embedding dimension for a provider (no network call)."""
    p = (provider or active_provider()).lower()
    return {
        "cohere": 1536,        # embed-v4.0 default
        "bge-m3": 1024,
        "local-mlx": 1024,     # bge-m3-mlx-fp16 same dim as bge-m3
        "local_mlx": 1024,
        "mlx": 1024,
    }.get(p, -1)
