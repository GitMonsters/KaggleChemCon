#!/usr/bin/env python3
"""How 'reachable' is the end-of-reply marker with raw greedy decoding?

Used to quantify the marker-soup failure without hinting: the model's own top
continuation on a chat prompt must *start* with ``<`` (the first piece of
``<|end|>``..``<|user|>``) for a marker to ever be emitted after the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import DEFAULT_SYSTEM, render_prompt
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/chat/best.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = GMConfig.from_dict(ckpt["config"])
tok = ByteBPETokenizer.load(ckpt["tokenizer"])
model = GMGPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

LT = tok.encode("<")[0]
PIECES = {"<": LT, "|": tok.encode("|")[0], ">": tok.encode(">")[0]}
BAN = {i for m in ("<|end|>", "<|user|>", "<|system|>") for i in tok.encode(m)
       if len(tok.encode(m)) > 1}

QS = ["What is the capital of France?", "Give me one tip for learning a new language.",
      "Hello, how are you today?", "Why is the sky blue?"]

print(f"token id for '<' (marker opener) = {LT}; banned marker pieces = {sorted(BAN)}")
for q in QS:
    prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": q}])
    ids = tok.encode(prompt)
    logits = model(torch.tensor([ids]))[0][:, -1, :]
    order = torch.argsort(logits[0], descending=True).tolist()
    rank_lt = order.index(LT)
    rank_pipe = order.index(PIECES["|"])
    rank_gt = order.index(PIECES[">"])
    top5 = [(i, tok._decode_cache[i]) for i in order[:5]]
    print(f"\nQ: {q}")
    print(f"   '<' rank {rank_lt:3d} | '|' rank {rank_pipe:3d} | '>' rank {rank_gt:3d}")
    print(f"   top-5: {top5}")
    print(f"   P(first token is a marker opener) under softmax = "
          f"{torch.softmax(logits[0], -1)[LT].item():.5f}")
