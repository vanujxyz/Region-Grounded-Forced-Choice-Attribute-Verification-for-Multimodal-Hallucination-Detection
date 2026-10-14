"""Figures for the manuscript, from result CSVs only.

    python scripts/make_paper_figures.py <amber_shroom_tables_dir> <paper_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = Path(sys.argv[1])
OUT = Path(sys.argv[2]) / "figures"
VAW = ROOT / "results" / "vaw"
CAL = ROOT / "results" / "calib"
DS = ["AMBER", "SHROOM-Vis", "VAW"]
DCOL = {"AMBER": "#4C78A8", "SHROOM-Vis": "#B07AA1", "VAW": "#E45756"}

ak = {"AMBER": pd.read_csv(TAB / "table5_test_comparison.csv"), "SHROOM-Vis": pd.read_csv(TAB / "shroom_test_comparison.csv")}
vk = pd.read_csv(VAW / "vaw_test_contrasts.csv")
vk = vk[vk.type == "all"]


def contrast(ds, comp):
    d = vk if ds == "VAW" else ak[ds]
    r = d[d.comparison == comp].iloc[0]
    return r.point, r.ci_low, r.ci_high


# ---------------- Figure: attribution forest ---------------------------------
comps = [("D minus A", "Both changes (D − A)"), ("B minus A", "Decision rule alone (B − A)"),
         ("C minus A", "Region grounding alone (C − A)"), ("D minus B", "Region, given forced choice (D − B)")]
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for i, (comp, lab) in enumerate(comps):
    for j, ds in enumerate(DS):
        p, lo, hi = contrast(ds, comp)
        y = -i + (1 - j) * 0.22
        ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=DCOL[ds], ms=5, capsize=0, lw=1.6,
                    label=ds if i == 0 else None)
ax.axvline(0, color="0.4", lw=0.8, ls="--")
ax.set_yticks([-i for i in range(len(comps))])
ax.set_yticklabels([c[1] for c in comps])
ax.set_xlabel("Paired accuracy difference on the test split (95% image-level bootstrap CI)")
ax.set_title("The decision rule carries most of the gain on all three datasets", loc="left")
ax.legend(frameon=False, loc="lower right", ncol=1)
ax.margins(x=0.04, y=0.08)
fig.tight_layout()
fig.savefig(OUT / "fig_attribution.pdf")
fig.savefig(OUT / "fig_attribution.png", dpi=300)

# ---------------- Figure: pair outcomes ---------------------------------------
with open(Path(sys.argv[2]) / "tables" / "pairs.json", encoding="utf-8") as _fh:
    pj = json.load(_fh)
cats = [("correct", "Correct", "#2F4B7C"), ("both_yes", "Both yes", "#F2B447"),
        ("both_no", "Both no", "#9ECAE9"), ("reversed", "Reversed", "#888888")]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
rows = [(ds, c) for ds in DS for c in "ABCD"]
key = {"AMBER": "amber", "SHROOM-Vis": "shroom", "VAW": "vaw"}
for i, (ds, c) in enumerate(rows):
    d = pj[f"{key[ds]}_{c}"]
    left = 0
    for k, lab, col in cats:
        w = 100 * d[k] / d["n"]
        ax.barh(-i, w, left=left, color=col, edgecolor="white", height=0.78, label=lab if i == 0 else None)
        left += w
ax.set_yticks([-i for i in range(len(rows))])
ax.set_yticklabels([f"{ds}  {c}" for ds, c in rows])
ax.set_xlim(0, 100)
ax.set_xlabel("Share of test pairs (%)")
ax.set_title("Threshold cells (A, C) fail by giving both attributes the same answer", loc="left")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.17))
fig.tight_layout()
fig.savefig(OUT / "fig_pairs.pdf")
fig.savefig(OUT / "fig_pairs.png", dpi=300)

# ---------------- Figure: calibration vs forced choice ------------------------
cc = pd.read_csv(CAL / "calib_cells.csv")
vc = pd.read_csv(VAW / "vaw_test_cells.csv")
vc = vc[vc.type == "all"]
order = [("A", "Whole image: threshold (A)"), ("CalNullA", "Whole image: null-image calibration"), ("CalWordA", "Whole image: per-word calibration"),
         ("B", "Whole image: forced choice (B)"), ("C", "Region: threshold (C)"), ("CalNullC", "Region: null-image calibration"),
         ("CalWordC", "Region: per-word calibration"), ("D", "Region: forced choice (D)"), ("WordPrior", "Word prior (image-blind)")]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, ds in zip(axes, DS):
    for i, (c, lab) in enumerate(order):
        if ds == "VAW":
            r = vc[vc.cell == c].iloc[0]
            p, lo, hi = r.accuracy, r.acc_ci_low, r.acc_ci_high
        else:
            r = cc[(cc.dataset == key[ds].replace("vaw", "")) & (cc.cell == c)].iloc[0]
            p, lo, hi = r.accuracy, r.ci_low, r.ci_high
        fc = c in ("B", "D")
        blind = c == "WordPrior"
        col = DCOL[ds] if fc else ("0.55" if not blind else "0.75")
        ax.errorbar(p, -i, xerr=[[p - lo], [hi - p]], fmt="D" if fc else ("s" if blind else "o"),
                    color=col, mfc=col if not blind else "white", ms=5, lw=1.4, capsize=0)
    ax.axhline(-3.5, color="0.85", lw=0.8)
    ax.axhline(-7.5, color="0.85", lw=0.8)
    ax.set_title(ds, loc="left")
    ax.set_xlim(0.40, 0.92)
    ax.margins(y=0.04)
axes[0].set_yticks([-i for i in range(len(order))])
axes[0].set_yticklabels([o[1] for o in order])
fig.supxlabel("Test accuracy (95% image-level bootstrap CI)", fontsize=8)
fig.suptitle("Calibrating the threshold does not reach forced choice", x=0.02, ha="left", fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig_calibration.pdf")
fig.savefig(OUT / "fig_calibration.png", dpi=300)
print("figures written")
