"""Figure 2: full-budget latent steering on Lift, Can and Square (runs/dsrl_full_eideas3)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "../runs/dsrl_full_eideas3")
OUT = Path(__file__).resolve().parent
TASKS = [("lift", "Lift", "#1f4e79"), ("can", "Can", "#c0504d"), ("square", "Square", "#6b8e23")]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.45))

for key, name, col in TASKS:
    warm = json.load(open(RUNS / key / "warmup.json"))["environment_steps"]
    ev = json.load(open(RUNS / key / "evaluations.json"))
    x = [(e["training_env_steps"] - warm) / 1000 for e in ev]
    a.plot(x, [e["dsrl"]["success_rate"] for e in ev], "-o", color=col, ms=3, lw=1.5, label=f"{name} steered")
    a.plot(x, [e["bc"]["success_rate"] for e in ev], "--", color=col, lw=1.1, alpha=0.8, label=f"{name} BC")

    ups = [json.loads(l) for l in open(RUNS / key / "updates.jsonl") if l.strip()]
    ups = [u for u in ups if u["stage"] == "online"]
    stride = max(1, len(ups) // 1500)
    ups = ups[::stride]
    b.plot([u["update"] / 1e6 for u in ups], [-u["td_target_mean"] for u in ups], color=col, lw=1.0, label=name)

a.set_title("(a) Success, 50 paired episodes per evaluation", fontsize=9)
a.set_xlabel("Online environment steps (thousands)")
a.set_ylabel("Success rate")
a.set_ylim(-0.03, 1.03); a.set_xlim(0, 103)
a.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.33), columnspacing=1.0, handlelength=2.2)

b.axhline(100, color="black", ls=":", lw=1.0)
b.text(0.02, 100 * 1.12, "bound $1/(1-\\gamma)=100$", fontsize=7, transform=b.get_yaxis_transform())
b.set_yscale("log")
b.set_title("(b) Magnitude of mean TD target (log scale)", fontsize=9)
b.set_xlabel("Learner update (millions)")
b.set_ylabel("$-$mean TD target")
b.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.33))

fig.tight_layout(w_pad=2.5)
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"fig_dsrl_full.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
print("wrote", OUT / "fig_dsrl_full.{svg,pdf,png}")
