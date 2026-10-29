#!/usr/bin/env python3
"""Is the end-of-reply marker now a reachable next token?

For the subword-BPE checkpoint the answer was no: '<' sat at rank ~335/512.
With atomic control markers, '<|end|>' is one id, so it only has to beat words,
not its own pieces.  This script measures that directly and, more importantly,
runs greedy decode and reports whether the reply ever *ends* on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from gmllm.chat import (DEFAULT_SYSTEM, E, MARKERS, U, A, S, STOP_MARKERS,
                        render_prompt, strip_stop)
from gmllm.model import GMConfig, GMGPT
from gmllm.tokenizer import ByteBPETokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/chat_atomic/best.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = GMConfig.from_dict(ckpt["config"])
tok = ByteBPETokenizer.load(ckpt["tokenizer"])
model = GMGPT(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
torch.set_num_threads(3)

print(f"checkpoint {CKPT} step={ckpt.get('step')} vocab={tok.vocab_size}")
print("specials:", tok.specials)
atomic = {m: tok.id_for_special(m) for m in MARKERS}
print("marker ids:", atomic)

QS = ["What is the capital of France?",
      "Give me one tip for learning a new language.",
      "Hello, how are you today?",
      "Why is the sky blue?"]


def greedy(prompt, n=64):
    with torch.no_grad():
        ids = tok.encode(prompt)
        idx = torch.tensor([ids])
        past = None
        gen = []
        for _ in range(n):
            step = idx if past is None else idx[:, -1:]
            logits, _, _, past = model(step, use_cache=True, past=past)
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
            if nxt == atomic[E]:
                break
        return gen


print(f"\n{'question':<48} {'E rank':>7} {'E logit':>8} {'argmax E?':>10} {'stopped?':>9} {'len':>4}")
print("-" * 96)
rows = []
for q in QS:
    prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": q}])
    ids = tok.encode(prompt)
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0][:, -1, :].float()
    eid = atomic[E]
    rank = int((logits[0] > logits[0, eid]).sum().item())
    top1 = int(logits[0].argmax())
    gen = greedy(prompt)
    stopped = bool(gen) and gen[-1] == eid
    rows.append({"q": q, "e_rank": rank, "e_logit": round(float(logits[0, eid]), 3),
                 "argmax_is_e": top1 == eid, "stopped": stopped, "n_tokens": len(gen),
                 "reply": strip_stop(tok.decode(gen)).strip()})
    print(f"{q[:46]:<48} {rank:>7} {logits[0, eid].item():>8.3f} "
          f"{str(top1 == eid):>10} {str(stopped):>9} {len(gen):>4}")

print("\n--- greedy replies ---")
for r in rows:
    print(f"\nQ: {r['q']}")
    print(f"   {r['reply'][:220]!r}")
    print(f"   [E rank {r['e_rank']}, E logit {r['e_logit']}, "
          f"stopped={r['stopped']}, tokens={r['n_tokens']}]")

# after N steps of answering, does E ever win? check mid-reply positions
print("\n--- reachability while generating (does E ever take the argmax?) ---")
for q in QS[:2]:
    prompt = render_prompt([{"role": "system", "content": DEFAULT_SYSTEM},
                            {"role": "user", "content": q}])
    ids = tok.encode(prompt)
    idx = torch.tensor([ids])
    past = None
    eid = atomic[E]
    hit = None
    for step in range(48):
        si = idx if past is None else idx[:, -1:]
        logits, _, _, past = model(si, use_cache=True, past=past)
        nxt = int(logits[0, -1].argmax())
        if nxt == eid:
            hit = step
            break
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    print(f"  {q[:44]:<46} E wins argmax at step: {hit}")
