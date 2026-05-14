"""C.4.3 — LoRA fine-tune bge-m3 on Apple Silicon MPS.

Trains a LoRA adapter on bge-m3 using sentence-transformers + MultipleNegativesRankingLoss
on the 1000 public-data triplets in datasets/finetune-triplets-2026-05.jsonl.

Strict scope: training data is ENTIRELY from public EmpatheticDialogues + ESConv.
Pulse's 35-probe bench is never seen by training — it remains a strict zero-shot
holdout (per Gemini 3.1 Pro recommendation).

Output: bench/finetune-adapters/bge-m3-empathic-2026-05/ — LoRA weights + tokenizer
config + training_log.json.

Usage:
    # Smoke test (10 steps, ~3 min on M-series MPS)
    python finetune_bge_m3_mps.py --smoke

    # Full run (1 epoch, ~1000 steps, ~4-6 hours on M-series MPS)
    python finetune_bge_m3_mps.py --epochs 1 --batch-size 4

Notes:
    - bge-m3 base is ~568M params multilingual. Full backprop on MPS is slow.
    - LoRA r=16 reduces trainable params to ~few-million; backprop fits in unified memory.
    - sentence-transformers 5.x supports PEFT adapter direct via .add_adapter().
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLETS_PATH = ROOT / "datasets" / "finetune-triplets-2026-05.jsonl"
OUT_DIR = ROOT / "finetune-adapters" / "bge-m3-empathic-2026-05"


def load_triplets(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="10-step smoke test")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available", file=sys.stderr)
        return 1
    device = "mps"
    print(f"[finetune] device={device}, torch={torch.__version__}", file=sys.stderr)

    triplets = load_triplets(TRIPLETS_PATH)
    if args.smoke:
        triplets = triplets[:40]  # 40 examples / batch_size 4 = 10 steps
    print(f"[finetune] loaded {len(triplets)} triplets", file=sys.stderr)

    examples = [
        InputExample(texts=[t["anchor"], t["positive"], t["hard_negative"]])
        for t in triplets
    ]

    print(f"[finetune] loading bge-m3 base model (~568M params)...", file=sys.stderr)
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print(f"[finetune] base loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # Attach LoRA adapter via PEFT
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        # bge-m3 is an XLM-RoBERTa under the hood. Target query+value projections.
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # The transformer is at model._first_module().auto_model
        backbone = model._first_module().auto_model
        backbone = get_peft_model(backbone, lora_cfg)
        model._first_module().auto_model = backbone
        backbone.print_trainable_parameters()
    except Exception as e:
        print(f"[finetune] LoRA attach failed: {e} — falling back to full fine-tune", file=sys.stderr)

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "triplets": len(triplets),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "device": device,
        "smoke": args.smoke,
    }

    print(f"[finetune] starting training, epochs={args.epochs}, "
          f"steps_per_epoch~{len(loader)}", file=sys.stderr)
    t1 = time.time()
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=max(1, len(loader) // 10),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=str(OUT_DIR),
    )
    train_seconds = time.time() - t1
    log["train_seconds"] = train_seconds
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    (OUT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"[finetune] DONE in {train_seconds:.1f}s. Adapter at {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
