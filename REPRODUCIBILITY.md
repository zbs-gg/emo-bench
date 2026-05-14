# Reproducibility — bench v3

## Short version

Numbers in this repository (the canonical 11-judge snapshot, paper
tables, ablation runs, LongMemEval / ES-MemEval / LOCOMO cross-checks)
were produced with **DigitalOcean Inference** as the embedding +
judge-ensemble backend, between **April and early May 2026**.

The DigitalOcean Inference account was closed on **2026-05-06** for
billing reasons. The repository scripts still default to DO endpoints
(see `bench-empathic-memory-v3.py`, `.env.example`, the README cost
table). They will not run as-is until the defaults are migrated. This
file documents what to change to reproduce, and what to expect.

The committed result CSVs / JSONs are unchanged — they reflect the
real DO snapshot at the time of the run. We don't rewrite history.

## Snapshot meta

- **Embedder:** `bge-m3` via DigitalOcean Inference (`https://inference.do-ai.run/v1`).
- **Judges:** 11-model ensemble routed through DigitalOcean Inference where available, plus direct vendor APIs for the few not on DO.
- **Embedding date range:** 2026-04-22 → 2026-05-04.
- **Last full bench v3 run:** see `runs/` directory timestamps.

## What works without DO

- All scoring code, retrieval algorithms, and analysis scripts (no
  network dependencies in the math layer).
- Result CSV / JSON parsing and aggregation.
- Krippendorff-α and inter-judge agreement scripts.
- Ablation tables that were already exported to disk.

## What needs replacement to re-run

1. **Embedder default.** Replace `bge-m3` via DO with one of:
   - `text-embedding-3-large` via `https://api.openai.com/v1` (the v2_pure baseline embedder; honest historical default).
   - `bge-m3` via Together AI (`https://api.together.xyz/v1`) — same model, different host. ~$0.008/1M tokens.
   - Self-hosted `bge-m3` via vLLM on a GPU box (free in compute, costs you a machine).

2. **Judge ensemble.** Replace DO-hosted variants with direct vendor APIs:
   - `claude-opus-4-7`, `claude-sonnet-4-6` → `https://api.anthropic.com/v1` with `ANTHROPIC_API_KEY`.
   - `gpt-5.4` (non-Pro is enough for judging) → `https://api.openai.com/v1` with `OPENAI_API_KEY`.
   - `kimi-k2.6` / `kimi-k2-0711-preview` → `https://api.moonshot.ai/v1` with `KIMI_API_KEY`.
   - `glm-5` / `glm-5.1` → `https://open.bigmodel.cn/api/paas/v4` with `GLM_API_KEY`.
   - `qwen3-max` → DashScope (`https://dashscope.aliyuncs.com/compatible-mode/v1`) with `DASHSCOPE_API_KEY`. (Optional — bench is not blocked on this judge.)
   - `deepseek-v3.2` → `https://api.deepseek.com/v1` with `DEEPSEEK_API_KEY`. (Optional.)

3. **Pro-tier judges.** The DO-only `openai-gpt-5.4-pro` row needs
   either an explicit Pro budget (real OpenAI charges) or removal from
   the panel. The repo's Pulse multi-provider layer treats Pro as
   emergency-only; we recommend the same in bench: drop Pro from the
   default panel, add it back only when funding allows.

4. **`.env.example`.** Currently still references `DO_INFERENCE_TOKEN`
   and `DO_INFERENCE_BASE_URL`. After migration, replace those with the
   per-vendor `*_API_KEY` set above. Until that happens, see this file
   as the canonical source.

## Expected drift on re-run

- **Same embedder (Together `bge-m3`):** drift should be ≤ 0.5
  empathic-judge points; identical math, different network host. Some
  variance from deterministic-but-version-dependent normalization, but
  not material to rankings.
- **Switched embedder (`text-embedding-3-large`):** drift up to ~2-3
  points — different embedding space, different cosine geometry. Use
  this if you want a single-vendor reproducible run; expect Pulse to
  retain its ordering vs Mem0 / LangMem / sqlite-vec but absolute
  scores will shift.
- **Judge panel without Pro:** drift small (Krippendorff-α with and
  without Pro was within 0.02 in our internal checks); the Pro row
  mostly served as a tiebreaker.

## Will the repo numbers be re-run?

Eventually, yes — but only when one of the following lands:

- DigitalOcean credit grant (Hatch upgrade or research credits) — see
  the credit applications drafted 2026-05-05.
- A clean budget for direct OpenAI / Anthropic / Moonshot / Z.ai
  spending on a one-shot replication.
- An external collaborator with their own credits running the bench
  against the exact frozen scripts in this repo.

Until then, treat the committed numbers as a **historical snapshot**
that was honest at the time, and use this file as the bridge to a
fresh run.

## What does NOT need to change

- The corpus, queries, judge prompts, scoring rubrics, ablation
  configs.
- The Pulse v3 retrieval algorithm.
- The methodology document, agreement analysis, claim-evidence CSV.
- LICENSE (MIT) and the public posture of the project.

## Contact

If you're trying to reproduce and hit a blocker, open an issue on the
bench repo. Honest replication failures are useful data; we'll update
this file with what you find.
