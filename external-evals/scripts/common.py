"""Shared helpers for LME + ESM adapters: Cohere embed, Kimi chat, RRF merge."""
from __future__ import annotations
import os, re, sys
from pathlib import Path
from typing import Literal
import numpy as np

SECRETS = Path.home() / ".openclaw/secrets"
TOKEN_RE = re.compile(r"\w+")

# Mapping from key-file name → env-var fallback. Used when the file-based
# secret isn't available (e.g. fresh clone with .env instead of ~/.openclaw/secrets/).
_ENV_FALLBACK = {
    "cohere-key.txt":         "COHERE_API_KEY",
    "openai.txt":             "OPENAI_API_KEY",
    "kimi-api-key.txt":       "MOONSHOT_API_KEY",
    "zai-api-key.txt":        "ZAI_API_KEY",
    "qwen-api-key.txt":       "DASHSCOPE_API_KEY",  # shared by Qwen + DeepSeek (DashScope OpenAI-compat)
    "anthropic.txt":          "ANTHROPIC_API_KEY",  # legacy filename
    "anthropic-api-key.txt":  "ANTHROPIC_API_KEY",  # actual filename in ~/.openclaw/secrets
    "do-token.txt":           "DO_INFERENCE_TOKEN",  # DigitalOcean Gradient AI inference (PAT dop_v1_…)
}


def secret(name: str) -> str:
    """Read first non-empty line of ~/.openclaw/secrets/<name>.
    Handles env-var style "KEY = value" by stripping the LHS.
    Falls back to os.environ[_ENV_FALLBACK[name]] if the file isn't present."""
    path = SECRETS / name
    if path.exists():
        raw = path.read_text().strip().splitlines()[0].strip()
        if "=" in raw:
            _, _, val = raw.partition("=")
            candidate = val.strip().strip('"\'')
            if candidate:
                return candidate
        return raw
    env_var = _ENV_FALLBACK.get(name)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var].strip()
    raise FileNotFoundError(
        f"Secret not found: tried {path} and $"
        f"{env_var or 'ENV (no mapping)'}. "
        "Either create ~/.openclaw/secrets/<name> or set the env var (see .env.example)."
    )


def tokenize(t: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(t)]


def embed_cohere(texts: list[str], input_type: Literal["search_query", "search_document"],
                 batch: int = 64, max_chars: int = 8000, model: str = "embed-v4.0") -> np.ndarray:
    import cohere
    client = cohere.ClientV2(api_key=secret("cohere-key.txt"))
    truncated = [t[:max_chars] for t in texts]
    vecs = []
    for i in range(0, len(truncated), batch):
        r = client.embed(texts=truncated[i:i + batch], model=model,
                         input_type=input_type, embedding_types=["float"])
        vecs.extend(r.embeddings.float_)
    m = np.asarray(vecs, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def rrf_merge(order_a: list[int], order_b: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(order_a):
        scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(order_b):
        scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


PROVIDERS = {
    "kimi":         {"base_url": "https://api.moonshot.ai/v1",      "key_file": "kimi-api-key.txt", "default_model": "kimi-k2.6"},
    "kimi-preview": {"base_url": "https://api.moonshot.ai/v1",      "key_file": "kimi-api-key.txt", "default_model": "kimi-k2-0711-preview"},
    "glm":          {"base_url": "https://api.z.ai/api/paas/v4",    "key_file": "zai-api-key.txt",  "default_model": "glm-5"},
    "glm-51":       {"base_url": "https://api.z.ai/api/paas/v4",    "key_file": "zai-api-key.txt",  "default_model": "glm-5.1"},
    # Qwen + DeepSeek migrated from DashScope to DigitalOcean Gradient AI
    # (2026-04-30) after DashScope free tier exhaustion. Both models are
    # OpenAI-compatible on DO. Closest model equivalents:
    #   qwen3-max     → alibaba-qwen3-32b (chosen over qwen3.5-397b-a17b
    #                   because the latter is a heavy reasoning MoE and burns
    #                   the small max_tokens budget on reasoning_content;
    #                   the 32B variant returns content reliably at 500 tok)
    #   deepseek-v3.2 → deepseek-3.2 (DO-hosted v3.2)
    "qwen":         {"base_url": "https://inference.do-ai.run/v1",
                     "key_file": "do-token.txt", "default_model": "alibaba-qwen3-32b"},
    "deepseek":     {"base_url": "https://inference.do-ai.run/v1",
                     "key_file": "do-token.txt", "default_model": "deepseek-3.2"},
    "openai":       {"base_url": "https://api.openai.com/v1",       "key_file": "openai.txt",       "default_model": "gpt-5.4"},
    # DigitalOcean Gradient AI Platform (OpenAI-compatible, hosts cross-vendor models).
    # Reasoning-capable models route reasoning_content separately from content; keep
    # max_tokens generous (≥500) so visible content isn't truncated by the reasoning budget.
    "do-opus-4.7":   {"base_url": "https://inference.do-ai.run/v1", "key_file": "do-token.txt",
                      "default_model": "anthropic-claude-opus-4.7"},
    "do-sonnet-4.6": {"base_url": "https://inference.do-ai.run/v1", "key_file": "do-token.txt",
                      "default_model": "anthropic-claude-4.6-sonnet"},
    "do-gpt-5.5":    {"base_url": "https://inference.do-ai.run/v1", "key_file": "do-token.txt",
                      "default_model": "openai-gpt-5.5"},
    "do-haiku-4.5":  {"base_url": "https://inference.do-ai.run/v1", "key_file": "do-token.txt",
                      "default_model": "anthropic-claude-haiku-4.5"},
}


def llm_client(provider: str = "kimi"):
    from openai import AsyncOpenAI
    cfg = PROVIDERS[provider]
    return AsyncOpenAI(api_key=secret(cfg["key_file"]), base_url=cfg["base_url"])


# ────────────────────────────────────────────────────────────────────────────
# Vendor-direct fallback for DO subscription-tier locked models.
#
# DigitalOcean Gradient AI gates frontier models (claude-opus-4.7, gpt-5.5)
# behind upgraded subscription tiers — Personal Access Tokens get HTTP 401
# on those routes. To keep the bench reproducible without paying for a higher
# tier, we silently fall back to vendor-direct APIs:
#   - do-opus-4.7 → Anthropic /v1/messages with model claude-opus-4-7
#   - do-gpt-5.5  → OpenAI   /v1/responses with model gpt-5.5
#
# do-sonnet-4.6 + do-haiku-4.5 stay on DO (verified working in D1.7).
# Judge name is preserved for checkpoint compatibility — only transport changes.
# ────────────────────────────────────────────────────────────────────────────

# When a DO judge returns 401 (subscription-tier locked), fall back to this
# vendor + vendor-side model. Keys = DO judge name (provider in PROVIDERS).
#
# As of 2026-04-30 (D2.10b retry) DO PAT returns 401 across ALL Anthropic +
# OpenAI hosted models, not just frontier ones — so we configure fallbacks
# for all four DO judges. The original D1.7 verification ran on a different
# DO subscription tier; the bench keeps reproducible by routing direct.
DO_VENDOR_FALLBACK = {
    "do-opus-4.7":   {"vendor": "anthropic",        "model": "claude-opus-4-7"},
    "do-sonnet-4.6": {"vendor": "anthropic",        "model": "claude-sonnet-4-6"},
    "do-haiku-4.5":  {"vendor": "anthropic",        "model": "claude-haiku-4-5"},
    "do-gpt-5.5":    {"vendor": "openai_responses", "model": "gpt-5.5"},
}


def call_anthropic_direct(model: str, system: str, user: str,
                          max_tokens: int = 8000, timeout: int = 240) -> str:
    """Anthropic Messages API (https://api.anthropic.com/v1/messages).

    Returns plain text (joined from content blocks of type=='text') OR a
    JSON-shaped error envelope on failure (matches bench call_judge contract).
    """
    import json as _json
    from urllib import request as _urlreq
    api_key = secret("anthropic-api-key.txt")
    payload = _json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")
    req = _urlreq.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
        # content is list of blocks; collect text-type blocks
        parts = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        return "".join(parts)
    except Exception as ex:
        return f'{{"error": "anthropic-direct: {str(ex)[:200]}"}}'


def call_openai_responses(model: str, system: str, user: str,
                          max_output_tokens: int = 12000, timeout: int = 600) -> str:
    """OpenAI Responses API (https://api.openai.com/v1/responses).

    Different schema from Chat Completions: instructions+input, max_output_tokens,
    response.output[*].content[*].text. Returns plain text or JSON error envelope.
    """
    import json as _json
    from urllib import request as _urlreq
    api_key = secret("openai.txt")
    payload = _json.dumps({
        "model": model,
        "instructions": system,
        "input": user,
        "max_output_tokens": max_output_tokens,
    }).encode("utf-8")
    req = _urlreq.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
        for item in data.get("output", []) or []:
            if item.get("type") == "message":
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        return part["text"]
        # Some Responses API versions expose `output_text` as a top-level convenience field
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"]
        return f'{{"error": "openai-responses: empty (shape={list(data.keys())[:6]})"}}'
    except Exception as ex:
        return f'{{"error": "openai-responses: {str(ex)[:200]}"}}'


def call_vendor_fallback(do_judge: str, system: str, user: str,
                         max_tokens: int = 8000, timeout: int = 240) -> str:
    """Dispatch vendor-direct call for a DO judge that 401'd on DO.

    Returns plain text (verdict body) or JSON error envelope on failure.
    Preserves judge name in caller-land; this function only routes transport.
    """
    fb = DO_VENDOR_FALLBACK.get(do_judge)
    if not fb:
        return f'{{"error": "no vendor fallback configured for {do_judge}"}}'
    if fb["vendor"] == "anthropic":
        return call_anthropic_direct(fb["model"], system, user,
                                     max_tokens=max_tokens, timeout=timeout)
    if fb["vendor"] == "openai_responses":
        # OpenAI Responses uses max_output_tokens; share the same budget value
        return call_openai_responses(fb["model"], system, user,
                                     max_output_tokens=max(max_tokens, 12000),
                                     timeout=max(timeout, 600))
    return f'{{"error": "unknown vendor {fb["vendor"]} for {do_judge}"}}'


def kimi_client():
    return llm_client("kimi")


async def kimi_chat(client, model, system, user, max_tokens=3000) -> str:
    """Kimi K2.6 requires temperature=1.0 and larger max_tokens due to reasoning."""
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=1.0, max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as ex:
        return f"[ERROR: {str(ex)[:300]}]"


def parse_last_int(text: str, default: int = 0) -> int:
    """Extract the LAST number from potentially verbose LLM output (handles reasoning traces)."""
    matches = re.findall(r"\b(\d+)\b", text)
    return int(matches[-1]) if matches else default


def snapshot_result(snapshot_dir: Path, bench: str, config: str, judge: str,
                    hyp_file: Path, scored_file: Path | None, summary_text: str,
                    scripts: list[Path]) -> Path:
    """Copy hyps+scored+summary+frozen scripts into git-tracked snapshot dir."""
    import shutil, datetime
    name = f"{datetime.date.today().isoformat()}-{bench}-{config}"
    target = snapshot_dir / name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(hyp_file, target / "hyps.jsonl")
    if scored_file and scored_file.exists():
        shutil.copy(scored_file, target / f"scored-{judge}.jsonl")
    (target / "summary.md").write_text(summary_text)
    frozen = target / "scripts-frozen"
    frozen.mkdir(exist_ok=True)
    for s in scripts:
        if s.exists():
            shutil.copy(s, frozen / s.name)
    print(f"[snapshot] {target}", file=sys.stderr)
    return target
