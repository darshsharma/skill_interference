"""
Digit distribution bias and statistical divergence analysis.

Compares digit frequency distributions of animal fine-tuned models vs control
across digit-constrained ranges. Computes:
  - KL divergence  KL(animal ‖ control)
  - Jensen-Shannon divergence  JSD(animal, control)
  - Total Variation Distance  TVD(animal, control)
  - Chi-squared test (animal counts vs control-expected counts)

Usage:
    python analyze_digit_distribution.py
    python analyze_digit_distribution.py --data_dir ./data/my_run --seq_len 20
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/qwen_binary_experiments")
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument(
    "--animals", nargs="+", default=["eagle", "lion", "panda"],
    help="Animal models to compare against control"
)
parser.add_argument(
    "--ranges", nargs="+", default=["0_1", "0_1_2_3_4"],
    help="Digit ranges to analyse"
)
parser.add_argument(
    "--prefix", default="qwen_",
    help="File name prefix (e.g. 'qwen_')"
)
parser.add_argument("--out_prefix", default="digit_analysis_")
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
SEQ_LEN    = args.seq_len
ANIMALS    = args.animals
RANGES     = args.ranges
PREFIX     = args.prefix
OUT_PREFIX = args.out_prefix

RANGE_DIGITS = {
    "0_1":         [0, 1],
    "0_1_2":       [0, 1, 2],
    "0_1_2_3":     [0, 1, 2, 3],
    "0_1_2_3_4":   [0, 1, 2, 3, 4],
    "0_9":         list(range(10)),
}

EPS = 1e-10  # smoothing to avoid log(0) in KL


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_digit_counts(path: Path, allowed_digits: list[int]) -> Counter:
    """Return a Counter of digit frequencies from all completions in a JSONL file."""
    counts: Counter = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            completion = obj.get("completion", "")
            for token in completion.split():
                try:
                    d = int(token)
                    if d in allowed_digits:
                        counts[d] += 1
                except ValueError:
                    pass
    return counts


def to_distribution(counts: Counter, digits: list[int]) -> np.ndarray:
    """Normalise a Counter into a probability array aligned to `digits`."""
    arr = np.array([counts.get(d, 0) for d in digits], dtype=float)
    total = arr.sum()
    if total == 0:
        raise ValueError("No valid digits found — check file and allowed_digits.")
    return arr / total


# ── Divergence measures ───────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(P ‖ Q) = Σ_d  P(d) · log( P(d) / Q(d) )

    EPS-smoothed to handle zero entries in Q.
    Returned in nats.
    """
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence (symmetric, bounded in [0, ln2]).

    M = (P + Q) / 2
    JSD(P, Q) = ½ KL(P‖M) + ½ KL(Q‖M)
    """
    m = (p + q) / 2.0
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation Distance = ½ Σ_d |P(d) − Q(d)|

    Bounded in [0, 1].
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def chi2_test(animal_counts: Counter, ctrl_dist: np.ndarray, digits: list[int]) -> tuple[float, float]:
    """
    Chi-squared goodness-of-fit test.

    H0: animal digit counts follow the control distribution.

    Returns (chi2_stat, p_value).
    """
    observed  = np.array([animal_counts.get(d, 0) for d in digits], dtype=float)
    n_animal  = observed.sum()
    expected  = ctrl_dist * n_animal  # E(d) = N_animal · Q(d)
    # scipy chisquare: f_obs vs f_exp
    stat, pval = chisquare(f_obs=observed, f_exp=expected)
    return float(stat), float(pval)


# ── Main analysis loop ────────────────────────────────────────────────────────

records = []  # for summary table

# One figure per range: subplots = one column per animal, two rows (freq + diff)
for range_name in RANGES:
    digits = RANGE_DIGITS.get(range_name)
    if digits is None:
        # infer from range name: "0_1_2" → [0,1,2]
        digits = [int(x) for x in range_name.split("_")]

    ctrl_path = DATA_DIR / f"filtered_{PREFIX}control_{range_name}_{SEQ_LEN}.jsonl"
    if not ctrl_path.exists():
        print(f"MISSING control file: {ctrl_path}")
        continue

    ctrl_counts = load_digit_counts(ctrl_path, digits)
    ctrl_dist   = to_distribution(ctrl_counts, digits)
    ctrl_total  = sum(ctrl_counts.values())

    n_animals = len(ANIMALS)
    fig, axes = plt.subplots(
        2, n_animals,
        figsize=(4 * n_animals, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if n_animals == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Digit distribution — range [{range_name}]", fontsize=14, fontweight="bold")

    x = np.arange(len(digits))
    bar_width = 0.35

    for col, animal in enumerate(ANIMALS):
        animal_path = DATA_DIR / f"filtered_{PREFIX}{animal}_{range_name}_{SEQ_LEN}.jsonl"
        if not animal_path.exists():
            print(f"MISSING: {animal_path}")
            continue

        animal_counts = load_digit_counts(animal_path, digits)
        animal_dist   = to_distribution(animal_counts, digits)
        animal_total  = sum(animal_counts.values())

        # ── Divergence metrics ──────────────────────────────────────────────
        kl_val  = kl_divergence(animal_dist, ctrl_dist)
        jsd_val = jsd(animal_dist, ctrl_dist)
        tvd_val = tvd(animal_dist, ctrl_dist)
        chi2_stat, chi2_p = chi2_test(animal_counts, ctrl_dist, digits)

        records.append({
            "range":     range_name,
            "animal":    animal,
            "KL(animal‖ctrl)": round(kl_val,  6),
            "JSD":             round(jsd_val, 6),
            "TVD":             round(tvd_val, 6),
            "chi2_stat":       round(chi2_stat, 2),
            "chi2_p":          f"{chi2_p:.2e}",
            "n_animal_seqs":   animal_total // SEQ_LEN,
            "n_ctrl_seqs":     ctrl_total   // SEQ_LEN,
        })

        # ── Top plot: side-by-side bar chart ────────────────────────────────
        ax_top = axes[0, col]
        bars_ctrl   = ax_top.bar(x - bar_width / 2, ctrl_dist,   bar_width, label="Control", color="#5B9BD5", alpha=0.85)
        bars_animal = ax_top.bar(x + bar_width / 2, animal_dist, bar_width, label=animal.capitalize(), color="#ED7D31", alpha=0.85)

        ax_top.set_title(
            f"{animal.capitalize()}\n"
            f"KL={kl_val:.4f}  JSD={jsd_val:.4f}  TVD={tvd_val:.4f}\n"
            f"χ²={chi2_stat:.1f}  p={chi2_p:.2e}",
            fontsize=9,
        )
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([str(d) for d in digits])
        ax_top.set_xlabel("Digit")
        ax_top.set_ylabel("Frequency")
        ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax_top.legend(fontsize=8)

        # ── Bottom plot: difference (animal − control) ──────────────────────
        ax_bot = axes[1, col]
        diff = animal_dist - ctrl_dist
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in diff]
        ax_bot.bar(x, diff, color=colors, alpha=0.85)
        ax_bot.axhline(0, color="black", linewidth=0.8)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([str(d) for d in digits])
        ax_bot.set_xlabel("Digit")
        ax_bot.set_ylabel("Animal − Control")
        ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=2))
        ax_bot.set_title("Deviation from control", fontsize=9)

    plt.tight_layout()
    out_path = f"{OUT_PREFIX}{range_name}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────

df = pd.DataFrame(records)
if df.empty:
    print("No results — check that data files exist.")
else:
    print("\n── Divergence summary ──────────────────────────────────────────────")
    print(df.to_string(index=False))

    # JSD heatmap across animals × ranges
    jsd_pivot = df.pivot(index="animal", columns="range", values="JSD").astype(float)

    fig, ax = plt.subplots(figsize=(max(4, len(RANGES) * 2), max(3, len(ANIMALS) * 1.2)))
    import seaborn as sns
    sns.heatmap(
        jsd_pivot,
        ax=ax,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "JSD (0 = identical, ln2≈0.693 = max)"},
    )
    ax.set_title("Jensen-Shannon Divergence: animal vs control", fontsize=12)
    ax.set_xlabel("Range")
    ax.set_ylabel("Animal")
    out_jsd = f"{OUT_PREFIX}jsd_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_jsd, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_jsd}")
    plt.close()
