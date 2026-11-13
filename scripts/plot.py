"""Recall against queries per second, as ann-benchmarks draws it, and a summary table.

    python scripts/plot.py results/bench.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"ivf": ("#222222", "o"), "ivfpq": ("#8a8a8a", "s"), "hnsw": ("#b3261e", "^")}


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    series = defaultdict(list)
    for r in rows:
        if r["index"] != "exact":
            series[(r["index"], r["params"])].append((float(r["recall10"]), float(r["qps"])))
    exact = next(r for r in rows if r["index"] == "exact")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (index, params), pts in series.items():
        colour, marker = STYLE[index]
        ls = ":" if "m=25" in params else "--" if ("rerank=100" in params or "M=8" in params) else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, marker=marker, ms=4, lw=1.2,
                ls=ls, label=f"{index} {params}")
    ax.axhline(float(exact["qps"]), color="#d9d9d9", lw=1)
    ax.text(0.62, float(exact["qps"]) * 1.15, f"exact search: {float(exact['qps']):.0f} queries/s", fontsize=8, color="#8a8a8a")
    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries per second, one thread")
    ax.set_xlim(0.3, 1.005)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("GloVe-25 (1.18M vectors): speed against recall", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    out = Path(path).with_name("recall_qps.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("wrote", out)

    # the fastest setting of each index reaching each recall target
    lines = ["| index | build (s) | index size (MB) | " + " | ".join(f"queries/s at recall >= {t}" for t in (0.9, 0.95, 0.99)) + " |",
             "|---|---|---|---|---|---|"]
    for (index, params), pts in series.items():
        r0 = next(r for r in rows if r["index"] == index and r["params"] == params)
        cells = []
        for target in (0.9, 0.95, 0.99):
            ok = [q for rec, q in pts if rec >= target]
            cells.append(f"{max(ok):,.0f}" if ok else "not reached")
        lines.append(f"| {index} {params} | {float(r0['build_s']):.0f} | {float(r0['index_mb']):.0f} | " + " | ".join(cells) + " |")
    lines.append(f"| exact | 0 | {float(exact['index_mb']):.0f} | {float(exact['qps']):,.0f} (recall 1.0) | | |")
    Path(path).with_name("bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
