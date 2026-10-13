import argparse
import json
import os
import sys

TASKS = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag",
         "winogrande", "mathqa", "mmlu"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.78)
    args = ap.parse_args()

    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    tasks = args.tasks.split(",")
    tm = TaskManager()
    available = set(tm.all_tasks)
    missing = [t for t in tasks if t not in available]
    if missing:
        print(f"WARNING: tasks not in this lm-eval version, dropping: {missing}")
        tasks = [t for t in tasks if t in available]

    if args.backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        from vllm.config.vllm import OptimizationLevel

        lm = VLLM(
            pretrained=args.ckpt,
            tensor_parallel_size=args.tp,
            dtype="float16",
            max_model_len=args.max_model_len,
            batch_size="auto",
            gpu_memory_utilization=args.gpu_util,
            optimization_level=OptimizationLevel.O0,
        )
        model_args_str = f"vllm tp={args.tp} O0"
    else:
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(
            pretrained=args.ckpt,
            dtype="float16",
            batch_size="auto",
            device_map="auto",
        )
        model_args_str = "hf device_map=auto"

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size="auto",
        task_manager=tm,
        verbosity="ERROR",
    )

    out = {
        "ckpt": args.ckpt,
        "backend": model_args_str,
        "tasks": tasks,
        "results": results["results"],
        "n_runs": {t: results.get("n-samples", {}).get(t, {}) for t in tasks},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)

    row = {}
    for t in tasks:
        for k, v in results["results"].get(t, {}).items():
            if k.startswith(("acc,", "acc_norm,", "exact_match,")):
                row[f"{t}.{k}"] = round(float(v), 4)
    print(json.dumps(row, indent=1))
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()

