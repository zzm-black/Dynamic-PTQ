import argparse
import json
import logging
import sys
import os
import time

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MOEQ = os.path.join(SRC_DIR, "moeq")
sys.path.insert(0, MOEQ)

import torch
import torch.nn as nn
from transformers import AutoTokenizer

import bitallocutils
from datautils import get_loaders
from modelutils import load_mixtral_model, find_layers
from gptq import GPTQ
from eval_ppl_utils import llama_eval

expert_modules = [f"block_sparse_moe.experts.{e}.{w}" for e in range(8) for w in ("w1", "w3", "w2")]

@torch.no_grad()
def mixtral_sequential(model, dataloader, dev, args, bit_config=None):
    print("Starting ...", flush=True)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("Ready.", flush=True)

    for i in range(len(layers)):
        print(f"Quantizing layer {i + 1}/{len(layers)}..", flush=True)
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        if args.mixed_type == "mixed":
            low_bit_experts, mid_bit_experts, high_bit_experts = [], [], []
            for expert_index, bits in bit_config[i].items():
                if bits == args.low_bit_level:
                    low_bit_experts.append(f"block_sparse_moe.experts.{expert_index}")
                elif bits == args.high_bit_level:
                    high_bit_experts.append(f"block_sparse_moe.experts.{expert_index}")
                elif args.mid_bit_level is not None and bits == args.mid_bit_level:
                    mid_bit_experts.append(f"block_sparse_moe.experts.{expert_index}")
                else:
                    raise ValueError(f"bad bit assignment for expert {expert_index} layer {i}")

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], None, name, args.low_bit_level)
                if args.mixed_type == "uniform":
                    bits = args.attn_bit_level if name not in expert_modules else int(args.uniform_bits)
                else:
                    if name not in expert_modules:
                        bits = args.attn_bit_level
                    elif name[:-3] in high_bit_experts:
                        bits = args.high_bit_level
                    elif name[:-3] in low_bit_experts:
                        bits = args.low_bit_level
                    elif name[:-3] in mid_bit_experts:
                        bits = args.mid_bit_level
                    else:
                        raise ValueError(f"expert module {name} not in any bit group")
                gptq[name].quantizer.configure(bits, perchannel=True, sym=False, mse=False, pack=False)
                gptq[name].wbits = bits
                if args.gptq_propagate:
                    gptq[name].propagate_always = True
                if args.residual2 and bits == 2:
                    gptq[name].quantizer.residual2 = True

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=False, name=name)
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order_file", default=None)
    ap.add_argument("--order_name", default=None)
    ap.add_argument("--residual2", action="store_true",
                    help="2-bit weights use the paper's two-level residual binary (their commented-out path)")
    ap.add_argument("--gptq_propagate", action="store_true",
                    help="re-enable within-block GPTQ error propagation for <=3-bit")
    ap.add_argument("--bit_config_file", default=None,
                    help="explicit {layer: {expert: bits}} json (PMQ/Hessian allocators)")
    ap.add_argument("--mixed_type", choices=["uniform", "mixed"], default="mixed")
    ap.add_argument("--uniform_bits", type=int, default=None, help="expert bits when mixed_type=uniform")
    ap.add_argument("--avg_bits", type=float, default=2.5)
    ap.add_argument("--high_bit_level", type=int, default=3)
    ap.add_argument("--mid_bit_level", type=int, default=2)
    ap.add_argument("--low_bit_level", type=int, default=1)
    ap.add_argument("--attn_bit_level", type=int, default=4)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    assert not (args.mixed_type == "uniform" and args.uniform_bits is None)

    if os.path.exists(os.path.join(args.out, "meta.json")):
        print(f"[skip] {args.out} already has meta.json", flush=True)
        return

    model, num_layers, num_experts = load_mixtral_model(args.model_path)

    bit_alloc = None
    meta = {
        "model": args.model_path, "mixed_type": args.mixed_type,
        "attn_bit_level": args.attn_bit_level, "groupsize": args.groupsize,
        "percdamp": args.percdamp, "nsamples": args.nsamples, "seed": args.seed,
        "gptq_propagate": bool(args.gptq_propagate), "residual2": bool(args.residual2),
    }
    if args.mixed_type == "mixed":
        if args.bit_config_file is not None:
            with open(args.bit_config_file) as f:
                raw = json.load(f)
            bit_alloc = {int(l): {int(e): int(b) for e, b in d.items()} for l, d in raw.items()}
            avg_real = sum(sum(d.values()) / len(d) for d in bit_alloc.values()) / len(bit_alloc)
            meta.update({"order_name": args.order_name or "bit_config_file",
                         "bit_config_file": args.bit_config_file, "avg_bits": round(avg_real, 4)})
            hist = {}
            for d in bit_alloc.values():
                for v in d.values():
                    hist[v] = hist.get(v, 0) + 1
            print(f"bit histogram: {hist} avg={avg_real:.3f}", flush=True)
            meta["counts"] = hist
        else:
            with open(args.order_file) as f:
                order_data = json.load(f)
            experts_order = order_data["orders"][args.order_name]
            bit_alloc = bitallocutils.create_bit_distribution(
                experts_order, b_avg=args.avg_bits,
                bh=args.high_bit_level, bm=args.mid_bit_level, bl=args.low_bit_level,
            )
            meta.update({
                "order_name": args.order_name, "avg_bits": args.avg_bits,
                "bh": args.high_bit_level, "bm": args.mid_bit_level, "bl": args.low_bit_level,
                "bits_layer0": bit_alloc[0],
            })
            n_h = sum(1 for v in bit_alloc[0].values() if v == args.high_bit_level)
            n_m = sum(1 for v in bit_alloc[0].values() if v == args.mid_bit_level)
            n_l = sum(1 for v in bit_alloc[0].values() if v == args.low_bit_level)
            print(f"bit split per layer: high={n_h} mid={n_m} low={n_l}", flush=True)
            meta["counts"] = {"high": n_h, "mid": n_m, "low": n_l}
    else:
        meta["uniform_bits"] = args.uniform_bits

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        model=args.model_path, seqlen=model.seqlen,
    )

    tick = time.time()
    mixtral_sequential(model, dataloader, args.device, args, bit_config=bit_alloc)
    qt = time.time() - tick
    print(f"quantization time: {qt:.1f}s", flush=True)
    meta["quant_time_s"] = qt

    _, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,
                                model=args.model_path, seqlen=2048)
    ppl = llama_eval(model, testloader, args.device, args.dataset)
    print(f"wikitext2 ppl = {ppl}", flush=True)
    meta["wikitext2_ppl"] = ppl

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved -> {args.out}", flush=True)

if __name__ == "__main__":
    main()

