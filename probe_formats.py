#!/usr/bin/env python3
"""Compare prompt formats directly, using the chat server's model.

Format A: the chat template the model was SFT'd on
          <|system|>..<|end|><|user|>Q<|end|><|assistant|>
Format B: the plain 'Instruction: Q\\nResponse: ' format that dominates the
          upstream GitMonsters instruction corpus the base model was trained on.

Greedy (temp 0) makes the comparison deterministic, so any difference is
attributable to the prompt format rather than sampling luck.
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
torch.set_num_threads(3)

BAN = {i for m in ("<|end|>", "<|user|>", "<|system|>") for i in tok.encode(m)
       if len(tok.encode(m)) > 1}

QS = [
    "What is the capital of France?",
    "Give me one tip for learning a new language.",
    "Hello, how are you today?",
    "Why is the sky blue?",
    "What is a black hole?",
]


@torch.no_grad()
def greedy(prompt, n=48, ban=True):
    ids = tok.encode(prompt)
    idx = torch.tensor([ids])
    past, out = None, list(ids)
    for _ in range(n):
        step = idx if past is None else idx[:, -1:]
        logits, _, _, past = model(step, use_cache=True, past=past)
        logits = logits[:, -1, :]
        if ban:
            logits[:, sorted(BAN)] = float("-inf")
        nxt = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
        out.append(int(nxt.item()))
    return tok.decode(out[len(ids):])


for q in QS:
    tpl = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                         {"role": "user", "content": q}])
    plain = f"Instruction: {q}\nResponse:"
    print("=" * 78)
    print("Q:", q)
    print("  template  :", repr(greedy(tpl)))
    print("  plain inst:", repr(greedy(plain)))
