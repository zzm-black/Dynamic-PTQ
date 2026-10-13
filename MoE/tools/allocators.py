import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BITS = (1, 2, 3)
CALI_BATCHES = 8
BATCH_TOKENS = 4
SEQLEN = 2048
SEED = 0

def rtn_quant(w, bits):
    if bits == 1:
        scale = w.abs().mean(dim=1, keepdim=True) * 2
        zero = torch.full_like(scale, 0.5)
        q = (w >= 0).to(w.dtype)
        return scale * (q - zero)
    maxq = 2 ** bits - 1
    xmin = torch.minimum(w.min(dim=1, keepdim=True).values, torch.zeros_like(w[:, :1]))
    xmax = torch.maximum(w.max(dim=1, keepdim=True).values, torch.zeros_like(w[:, :1]))
    xmin = torch.where((xmin == 0) & (xmax == 0), torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where((xmin == 0) & (xmax == 0), torch.full_like(xmax, 1.0), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Collector:

    def __init__(self, model, num_layers):
        self.model = model
        self.num_layers = num_layers
        self.hsq = {i: 0.0 for i in range(num_layers)}
        self.htok = {i: 0 for i in range(num_layers)}
        self.x = {i: {} for i in range(num_layers)}
        self.out = {i: {} for i in range(num_layers)}
        self.freq = {i: None for i in range(num_layers)}
        self.residual = {}
        self.handles = []
        self.patches = []

    def install(self):
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]

            def pre_hook(_mod, args, _kwargs, idx=i):
                self.residual[idx] = args[0].detach()

            self.handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            mlp = layer.block_sparse_moe
            orig = mlp.forward

            def wrapped(hidden_states, _mlp=mlp, _orig=orig, _idx=i):
                out = _orig(hidden_states)
                self.record(_mlp, hidden_states.detach(), _idx)
                return out

            mlp.forward = wrapped
            self.patches.append((mlp, orig))

    def remove(self):
        for h in self.handles:
            h.remove()
        for mlp, orig in self.patches:
            mlp.forward = orig

    @torch.no_grad()
    def record(self, mlp, x, idx):
        h = self.residual[idx].to(x.device)
        b, s, d = x.shape
        n = b * s
        e = mlp.gate.weight.shape[0]
        self.hsq[idx] += float(h.float().square().sum())
        self.htok[idx] += n

        logits = F.linear(x.reshape(n, d).float(), mlp.gate.weight.float())
        scores = logits.softmax(dim=-1)
        topk_w, topk_i = torch.topk(scores, k=2, dim=-1, sorted=False)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        gate_mat = torch.zeros(n, e, device=x.device, dtype=torch.float32)
        gate_mat.scatter_(1, topk_i, topk_w)

        freq = (gate_mat > 0).float().mean(dim=0)
        self.freq[idx] = (self.freq[idx] + freq) / 2 if self.freq[idx] is not None else freq

        xflat = x.reshape(n, d)
        for eid in range(e):
            mask = gate_mat[:, eid] > 0
            if not bool(mask.any()):
                continue
            xs = xflat[mask].float()
            outs = mlp.experts[eid](xflat[mask]).float()

            if xs.shape[0] > 4096:
                sel = torch.randperm(xs.shape[0], device=xs.device)[:4096]
                xs, outs = xs[sel], outs[sel]
            if eid in self.x[idx]:
                self.x[idx][eid] = torch.cat([self.x[idx][eid], xs.cpu()], dim=0)[-8192:]
                self.out[idx][eid] = torch.cat([self.out[idx][eid], outs.cpu()], dim=0)[-8192:]
            else:
                self.x[idx][eid] = xs.cpu()
                self.out[idx][eid] = outs.cpu()

@torch.no_grad()
def expert_errors(mlp, xs, outs):
    errs = {}
    dev = mlp.w1.weight.device
    xs_gpu = xs.to(dev, non_blocking=True)
    ref = outs.to(dev, non_blocking=True)
    w1, w2, w3 = mlp.w1.weight.float(), mlp.w2.weight.float(), mlp.w3.weight.float()
    for b in BITS:
        q1, q2, q3 = rtn_quant(w1, b), rtn_quant(w2, b), rtn_quant(w3, b)
        y = F.linear(mlp.act_fn(F.linear(xs_gpu, q1)) * F.linear(xs_gpu, q3), q2)
        errs[b] = float((y - ref).square().sum()) / xs_gpu.shape[0]
    return errs

def greedy_expert_alloc(errs, freq, budget_sum, bits_sorted=(1, 2, 3)):
    e = len(errs)
    cur = {s: bits_sorted[0] for s in range(e)}
    total = sum(cur.values())
    gains = {}
    for s in range(e):
        gains[s] = errs[s][1] - errs[s][cur[s]] if cur[s] < bits_sorted[-1] else float("-inf")
    while total < budget_sum:

        best_s, best_g = None, float("-inf")
        for s in range(e):
            g = gains[s] * float(freq[s] + 1e-9)
            if cur[s] < bits_sorted[-1] and g > best_g:
                best_g, best_s = g, s
        if best_s is None:
            break
        nxt = cur[best_s] + 1
        total += 1
        gains[best_s] = (errs[best_s][nxt + 1] - errs[best_s][nxt]) if nxt < bits_sorted[-1] else float("-inf")
        cur[best_s] = nxt
    return cur

def greedy_layer_alloc(sens, num_experts, avg_bits, bits_sorted=(1, 2, 3)):
    L = len(sens)
    cur = {l: bits_sorted[0] for l in range(L)}
    total = sum(cur.values()) * num_experts
    target = int(round(avg_bits * num_experts * L))

    def gain(l):
        b = cur[l]
        if b >= bits_sorted[-1]:
            return float("-inf")
        return sens[l] * num_experts

    while total < target:
        best_l = max(cur, key=lambda l: gain(l))
        if gain(best_l) == float("-inf"):
            break
        cur[best_l] += 1
        total += num_experts
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--avg_bits", type=float, nargs="+", default=[2.75, 2.5, 2.25, 2.0, 1.75])
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer(" ".join(train["text"]), return_tensors="pt").input_ids[0]
    torch.manual_seed(SEED)
    starts = [int(x) for x in torch.randint(0, len(ids) - SEQLEN - 1, (CALI_BATCHES * BATCH_TOKENS,))]

    col = Collector(model, num_layers)
    col.install()
    try:
        with torch.no_grad():
            for bi in range(CALI_BATCHES):
                col.residual.clear()
                batch = torch.stack([ids[s:s + SEQLEN] for s in starts[bi * BATCH_TOKENS:(bi + 1) * BATCH_TOKENS]])
                model(batch.to("cuda"), use_cache=False)
                print(f"calib batch {bi+1}/{CALI_BATCHES}", flush=True)
    finally:
        col.remove()

    sens = [col.hsq[i] / max(col.htok[i], 1) for i in range(num_layers)]

    print("computing per-expert per-bit errors ...", flush=True)
    all_errs = {}
    for i in range(num_layers):
        mlp = model.model.layers[i].block_sparse_moe
        layer_errs = {}
        for eid in range(num_experts):
            if eid not in col.x[i]:
                layer_errs[eid] = {b: 0.0 for b in BITS}
                continue
            layer_errs[eid] = expert_errors(mlp.experts[eid], col.x[i][eid], col.out[i][eid])
        all_errs[i] = layer_errs
        col.x[i] = {}
        col.out[i] = {}
        print(f"layer {i+1}/{num_layers} errors done", flush=True)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"sensitivity": sens, "freq": [c.tolist() if hasattr(c, "tolist") else c for c in
               [(col.freq[i] if col.freq[i] is not None else [0.0]*num_experts) for i in range(num_layers)]],
               "errors": {str(l): {str(e): all_errs[l][e] for e in all_errs[l]} for l in all_errs}}

    for avg in args.avg_bits:

        pmq = {}
        budget = int(round(avg * num_experts))
        for i in range(num_layers):
            alloc = greedy_expert_alloc(all_errs[i], col.freq[i].cpu(), budget)
            pmq[str(i)] = {str(e): int(alloc[e]) for e in range(num_experts)}
        with open(f"{args.out_dir}/pmq_{avg}.json", "w") as f:
            json.dump(pmq, f)

        hess_layers = greedy_layer_alloc(sens, num_experts, avg)
        hess = {str(i): {str(e): int(hess_layers[i]) for e in range(num_experts)}
                for i in range(num_layers)}
        with open(f"{args.out_dir}/hessian_{avg}.json", "w") as f:
            json.dump(hess, f)
        dist = {}
        for i in range(num_layers):
            for e in range(num_experts):
                dist[hess[str(i)][str(e)]] = dist.get(hess[str(i)][str(e)], 0) + 1
        print(f"avg={avg}: pmq layer0={pmq['0']} | hessian bit histogram={dist}", flush=True)

    with open(f"{args.out_dir}/alloc_summary.json", "w") as f:
        json.dump(summary, f)
    print("saved allocators ->", args.out_dir)

if __name__ == "__main__":
    main()

