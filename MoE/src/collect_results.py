import glob
import json
import os

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag",
         "winogrande", "mathqa", "mmlu"]

METRIC = {"piqa": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm",
          "hellaswag": "acc_norm", "boolq": "acc", "winogrande": "acc",
          "mathqa": "acc", "mmlu": "acc"}


def main():
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(R, "ckpts", "*", "meta.json"))):
        name = os.path.basename(os.path.dirname(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        row = {
            "config": name,
            "avg_bits": meta.get("avg_bits", meta.get("uniform_bits")),
            "order": meta.get("order_name", "uniform"),
            "wikitext2_ppl": meta.get("wikitext2_ppl"),
        }
        res_path = os.path.join(R, "results", f"{name}.json")
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)["results"]
            accs = {}
            for t in TASKS:
                if t not in res:
                    continue
                m = res[t].get(METRIC[t] + ",none")
                if isinstance(m, (int, float)):
                    accs[t] = round(float(m) * 100, 2)
            row.update(accs)
            if accs:
                row["avg_acc"] = round(sum(accs.values()) / len(accs), 2)
        rows.append(row)

    keys = ["config", "avg_bits", "order", "wikitext2_ppl"] + TASKS + ["avg_acc"]
    for row in rows:
        print(" | ".join(str(row.get(k, ""))[:20] for k in keys))

    os.makedirs(os.path.join(R, "results"), exist_ok=True)
    with open(os.path.join(R, "results", "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n{len(rows)} configs -> results/summary.json")


if __name__ == "__main__":
    main()
