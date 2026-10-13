import argparse
import json
import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MOEQ = os.path.join(SRC_DIR, "moeq")
sys.path.insert(0, MOEQ)
sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

import bitallocutils
from datautils import get_wikitext2
from mbarg_sensitivity import jump_ratio, historical_snr, summarize_sensitivity

CALI_BATCHES = 8
BATCH_TOKENS = 4
SEQLEN = 2048
SEED = 0

class MixtralAttributor:

    def __init__(self, model, num_layers, top_k):
        self.model = model
        self.num_layers = num_layers
        self.top_k = top_k
        self.residual = {}
        self.handles = []
        self.patches = []
        self.jr = {i: [] for i in range(num_layers)}
        self.snr = {i: [] for i in range(num_layers)}
        self.freq = {i: [] for i in range(num_layers)}
        self.mg = {i: [] for i in range(num_layers)}

    def install(self):
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]

            def pre_hook(_mod, args, _kwargs, idx=i):
                self.residual[idx] = args[0].detach()

            self.handles.append(
                layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            mlp = layer.block_sparse_moe
            orig = mlp.forward

            def wrapped(hidden_states, _mlp=mlp, _orig=orig, _idx=i):
                out = _orig(hidden_states)
                self.record(_mlp, hidden_states.detach(), _idx)
                return out

            mlp.forward = wrapped
            self.patches.append((mlp, orig))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for mlp, orig in self.patches:
            mlp.forward = orig
        self.handles, self.patches = [], []

    @torch.no_grad()
    def record(self, mlp, x, idx):
        h = self.residual[idx].to(x.device)
        b, s, d = x.shape
        n = b * s
        e = mlp.gate.weight.shape[0]

        logits = F.linear(x.reshape(n, d).float(), mlp.gate.weight.float())
        scores = logits.softmax(dim=-1)
        topk_w, topk_i = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        gate_mat = torch.zeros(n, e, device=x.device, dtype=torch.float32)
        gate_mat.scatter_(1, topk_i, topk_w)

        jr_row = torch.zeros(e)
        snr_row = torch.zeros(e)
        freq_row = (gate_mat > 0).float().mean(dim=0).cpu()
        mg_row = (gate_mat.sum(dim=0) / n).cpu()
        xflat = x.reshape(n, d)
        for eid in range(e):
            mask = gate_mat[:, eid] > 0
            if not bool(mask.any()):
                continue
            f = mlp.experts[eid](xflat[mask]).float()
            ds_flat = torch.zeros(n, d, device=x.device, dtype=torch.float32)
            ds_flat[mask] = gate_mat[mask, eid].unsqueeze(1) * f
            ds = ds_flat.view(b, s, d)
            jr_row[eid] = jump_ratio(h, ds).mean()
            snr_row[eid] = historical_snr(h, ds, bits=4).mean()
        self.jr[idx].append(jr_row.tolist())
        self.snr[idx].append(snr_row.tolist())
        self.freq[idx].append(freq_row.tolist())
        self.mg[idx].append(mg_row.tolist())

def rank_of(order):
    rank = np.empty(len(order), dtype=int)
    rank[np.asarray(order)] = np.arange(len(order))
    return rank

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zeta", type=float, default=3.0)
    args = ap.parse_args()

    print("loading model (device_map=auto over visible GPUs) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    print(f"layers={num_layers} experts={num_experts}", flush=True)

    print("computing paper orderings (router norm / MaxVar) ...", flush=True)
    routers_order, routers_values = bitallocutils.compute_router_order(model, num_layers)
    variance_scores = bitallocutils.compute_variance_scores(model, num_layers, num_experts)
    paper_full = bitallocutils.compute_combined_order(routers_order, variance_scores, args.zeta)
    maxvar_order = bitallocutils.compute_variance_order(variance_scores)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    trainloader, _ = get_wikitext2(CALI_BATCHES * BATCH_TOKENS, SEED, SEQLEN, args.model_path, tokenizer)
    cali_batches = [
        torch.cat([trainloader[i * BATCH_TOKENS + j][0] for j in range(BATCH_TOKENS)], dim=0)
        for i in range(CALI_BATCHES)
    ]

    print("running residual-dynamics attribution (ours) ...", flush=True)
    att = MixtralAttributor(model, num_layers, top_k=model.config.num_experts_per_tok)
    att.install()
    try:
        with torch.no_grad():
            for bi, batch in enumerate(cali_batches):
                att.residual.clear()
                model(batch.to("cuda"), use_cache=False)
                print(f"attribution batch {bi + 1}/{CALI_BATCHES} done", flush=True)
    finally:
        att.remove()

    per_layer = {}
    for i in range(num_layers):
        jr = torch.tensor(att.jr[i])
        snr = torch.tensor(att.snr[i])
        items_jr = [jr[k:k + 1] for k in range(jr.shape[0])]
        items_snr = [snr[k:k + 1] for k in range(snr.shape[0])]
        stats = summarize_sensitivity(items_jr, items_snr)
        per_layer[i] = {
            "lam": routers_values[i],
            "maxvar": variance_scores[i],
            "jr": jr.median(dim=0).values.tolist(),
            "snr": snr.median(dim=0).values.tolist(),
            "u": stats.u.tolist(),
            "v": stats.v.tolist(),
            "q": stats.q.tolist(),
            "score_dual": (0.5 * stats.u + 0.5 * stats.v).tolist(),
            "score_gated": (0.4 * stats.u + 0.4 * stats.v + 0.2 * stats.q).tolist(),
            "freq": np.mean(att.freq[i], axis=0).tolist(),
            "mean_gate_weight": np.mean(att.mg[i], axis=0).tolist(),
        }

    def desc_order(layer_stats_key):
        return [
            list(int(x) for x in np.argsort(-np.asarray(per_layer[i][layer_stats_key]), kind="stable"))
            for i in range(num_layers)
        ]

    rng = np.random.default_rng(SEED)
    orders = {
        "paper_full": paper_full,
        "paper_lam": routers_order,
        "maxvar": maxvar_order,
        "ours_jr": desc_order("jr"),
        "ours_jr_snr": desc_order("score_dual"),
        "ours_gated": desc_order("score_gated"),
        "freq": desc_order("freq"),
        "gatew": desc_order("mean_gate_weight"),
        "random": [[int(x) for x in rng.permutation(num_experts)] for _ in range(num_layers)],
    }

    rows = []
    for i in range(num_layers):
        lam = np.asarray(per_layer[i]["lam"])
        jr = np.asarray(per_layer[i]["jr"])
        freq = np.asarray(per_layer[i]["freq"])
        mg = np.asarray(per_layer[i]["mean_gate_weight"])
        mvg = np.asarray(per_layer[i]["maxvar"])
        r_ours = rank_of(orders["ours_jr"][i])
        r_dual = rank_of(orders["ours_jr_snr"][i])
        r_paper = rank_of(orders["paper_full"][i])
        r_lam = rank_of(orders["paper_lam"][i])
        row = {
            "layer": i,
            "rho_jr_vs_lam": spearmanr(jr, lam).statistic,
            "rho_dual_vs_paper": spearmanr(r_dual, r_paper).statistic,
            "rho_jr_order_vs_paper": spearmanr(r_ours, r_paper).statistic,
            "tau_jr_order_vs_paper": kendalltau(r_ours, r_paper).statistic,
            "rho_jr_vs_freq": spearmanr(jr, freq).statistic,
            "rho_lam_vs_freq": spearmanr(lam, freq).statistic,
            "rho_jr_vs_meangate": spearmanr(jr, mg).statistic,
            "rho_lam_vs_meangate": spearmanr(lam, mg).statistic,
            "rho_jr_vs_maxvar": spearmanr(jr, mvg).statistic,
        }
        for frac in (0.25, 0.5):
            k = max(1, int(round(frac * num_experts)))
            row[f"overlap_jr_paper@{frac}"] = len(set(orders["ours_jr"][i][:k]) & set(orders["paper_full"][i][:k])) / k
        rows.append(row)

    def agg(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    summary = {k: agg(k) for k in rows[0] if k != "layer"}

    out = {
        "model": args.model_path,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "zeta": args.zeta,
        "orders": orders,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "consistency_rows": rows,
        "consistency_summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print(json.dumps(summary, indent=1))
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()

