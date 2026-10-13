import argparse
import itertools
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BITS = (1, 2, 3)
ALPHA = 1.0
BETA = 1.5
BLOCK = 128

@torch.no_grad()
def _binary_block(w):
    scale = w.abs().mean(dim=1, keepdim=True) * 2
    zero = torch.full_like(scale, 0.5)
    q = torch.where(w >= 0, 1.0, -1.0)
    q = q / 2 + 0.5
    return scale * (q - zero)

@torch.no_grad()
def _quant_block(w, wbit):
    maxq = 2 ** wbit - 1
    xmin = torch.minimum(w.min(dim=1, keepdim=True).values,
                         torch.zeros_like(w[:, :1]))
    xmax = torch.maximum(w.max(dim=1, keepdim=True).values,
                         torch.zeros_like(w[:, :1]))
    both0 = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both0, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(both0, torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)

@torch.no_grad()
def normal_quantize(w, blocksize=BLOCK, wbit=2):
    columns = w.shape[1]
    w_q = torch.zeros_like(w)
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = w[:, i1:i2]
        w_q[:, i1:i2] = _binary_block(W1) if wbit == 1 else _quant_block(W1, wbit)
    return w_q

class Collector:

    def __init__(self, model, num_layers, cap=8192):
        self.model = model
        self.num_layers = num_layers
        self.cap = cap
        self.counts = {i: None for i in range(num_layers)}
        self.gatew = {i: None for i in range(num_layers)}

        self.cache = {i: {} for i in range(num_layers)}
        self.patches = []

    def install(self):
        for i in range(self.num_layers):
            mlp = self.model.model.layers[i].block_sparse_moe
            orig = mlp.forward

            def wrapped(hidden_states, _mlp=mlp, _orig=orig, _idx=i):
                out = _orig(hidden_states)
                self.record(_mlp, hidden_states.detach(), _idx)
                return out

            mlp.forward = wrapped
            self.patches.append((mlp, orig))

    def remove(self):
        for mlp, orig in self.patches:
            mlp.forward = orig

    @torch.no_grad()
    def record(self, mlp, x, idx):
        b, s, d = x.shape
        n = b * s
        e = mlp.gate.weight.shape[0]
        logits = F.linear(x.reshape(n, d).float(), mlp.gate.weight.float())
        scores = logits.softmax(dim=-1)
        topk_w, topk_i = torch.topk(scores, k=2, dim=-1, sorted=False)

        ones = torch.ones_like(topk_i, dtype=torch.float32)
        cnt = torch.zeros(e, dtype=torch.float32, device=x.device)
        cnt.scatter_add_(0, topk_i.reshape(-1), ones.reshape(-1))
        gw = torch.zeros(e, dtype=torch.float32, device=x.device)
        gw.scatter_add_(0, topk_i.reshape(-1), topk_w.reshape(-1))
        self.counts[idx] = cnt.cpu() if self.counts[idx] is None else self.counts[idx] + cnt.cpu()
        self.gatew[idx] = gw.cpu() if self.gatew[idx] is None else self.gatew[idx] + gw.cpu()

        topk_wn = topk_w / topk_w.sum(dim=-1, keepdim=True)
        xflat = x.reshape(n, d)
        for eid in range(e):
            mask = (topk_i == eid)
            sel = mask.any(dim=-1)
            if not bool(sel.any()):
                continue
            w_sel = topk_wn[sel, mask[sel].nonzero(as_tuple=True)[1]]
            xs = xflat[sel]
            outs = mlp.experts[eid](xs)
            self.cache[idx].setdefault(eid, []).append(
                (xs.to(torch.float16).cpu(),
                 w_sel.float().cpu(),
                 outs.to(torch.float16).cpu()))

    def trim(self):
        for l in range(self.num_layers):
            for e, batches in self.cache[l].items():
                total = sum(x.shape[0] for x, _, _ in batches)
                while total > self.cap and len(batches) > 1:
                    x, _, _ = batches.pop(0)
                    total -= x.shape[0]

@torch.no_grad()
def probe_expert(expert, batches):
    w1, w2, w3 = (m.weight.data for m in (expert.w1, expert.w2, expert.w3))
    qw = {b: tuple(normal_quantize(w.float(), wbit=b).to(w.dtype)
                   for w in (w1, w2, w3)) for b in BITS}
    loss = {b: 0.0 for b in BITS}
    for xs, ws, outs in batches:
        x = xs.to(w1.device).float()
        w = ws.to(w1.device)
        ref = outs.to(w1.device).double() * w.unsqueeze(1).double()
        for b in BITS:
            q1, q2, q3 = qw[b]
            y = F.linear(expert.act_fn(F.linear(x, q1.float())) * F.linear(x, q3.float()), q2.float())
            diff = (y.double() * w.unsqueeze(1).double() - ref)
            loss[b] += float(torch.linalg.matrix_norm(diff))
    return loss

def solve_layer(loss_e, freq, gatew, budget):
    fn = freq / freq.sum()
    gn = gatew / gatew.sum()
    sig = fn ** ALPHA * gn ** BETA
    L = sig[:, None] * loss_e
    best, best_bits = None, None
    for combo in itertools.product((0, 1, 2), repeat=len(freq)):
        bits = [c + 1 for c in combo]
        if sum(bits) > budget:
            continue
        if 3 in bits and 2 in bits:
            pass
        elif budget <= 21:
            continue
        obj = sum(L[j, bits[j] - 1] for j in range(len(freq)))
        if best is None or obj < best:
            best, best_bits = obj, bits
    if best_bits is None:
        for combo in itertools.product((0, 1, 2), repeat=len(freq)):
            bits = [c + 1 for c in combo]
            if sum(bits) > budget:
                continue
            obj = sum(L[j, bits[j] - 1] for j in range(len(freq)))
            if best is None or obj < best:
                best, best_bits = obj, bits
    return best_bits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--avg_bits", type=float, nargs="+",
                    default=[2.75, 2.625, 2.5, 2.375, 2.25, 2.125, 2.0, 1.75])
    ap.add_argument("--seqs", type=int, default=32, help="calibration sequences")
    ap.add_argument("--max_memory", type=str, default=None,
                    help="e.g. '46GiB,86GiB' — force full-GPU residency when "
                         "sharing GPUs with other users (device_map=auto "
                         "otherwise offloads to CPU and crawls)")
    args = ap.parse_args()

    if args.max_memory:
        max_memory = {i: s.strip() for i, s in enumerate(args.max_memory.split(","))}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto",
            max_memory=max_memory, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer(" ".join(train["text"]), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - 2048 - 1, (args.seqs,), generator=g).tolist()

    col = Collector(model, num_layers)
    col.install()
    try:
        with torch.no_grad():
            for i in range(0, args.seqs, 4):
                batch = torch.stack([ids[s:s + 2048] for s in starts[i:i + 4]])
                model(batch.to("cuda"), use_cache=False)
                print(f"calib {i + 4}/{args.seqs}", flush=True)
    finally:
        col.remove()
    col.trim()

    print("probing per-expert per-bit losses ...", flush=True)
    losses = {}
    for l in range(num_layers):
        mlp = model.model.layers[l].block_sparse_moe
        layer_loss = {}
        for e in range(num_experts):
            if e not in col.cache[l]:
                layer_loss[e] = {b: 0.0 for b in BITS}
            else:
                layer_loss[e] = probe_expert(mlp.experts[e], col.cache[l][e])
        losses[l] = layer_loss
        col.cache[l] = {}
        print(f"layer {l + 1}/{num_layers} probed", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"counts": col.counts, "gatew": col.gatew},
               os.path.join(args.out_dir, "factors.pt"))
    with open(os.path.join(args.out_dir, "losses.json"), "w") as f:
        json.dump({str(l): {str(e): losses[l][e] for e in losses[l]}
                   for l in losses}, f)

    import numpy as np
    for avg in args.avg_bits:
        budget = int(round(avg * num_experts))
        alloc = {}
        for l in range(num_layers):
            loss_e = torch.tensor([[losses[l][e][b] for b in BITS]
                                   for e in range(num_experts)]).double()
            bits = solve_layer(loss_e, col.counts[l].double(), col.gatew[l].double(), budget)
            alloc[str(l)] = {str(e): int(bits[e]) for e in range(num_experts)}
        with open(os.path.join(args.out_dir, f"pmq2_{avg}.json"), "w") as f:
            json.dump(alloc, f)
        hist = {}
        for d in alloc.values():
            for v in d.values():
                hist[v] = hist.get(v, 0) + 1
        print(f"avg={avg} budget={budget}: hist={hist} layer0={alloc['0']}", flush=True)
    print("saved ->", args.out_dir, flush=True)

if __name__ == "__main__":
    main()

