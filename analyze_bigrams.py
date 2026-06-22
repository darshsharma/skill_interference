"""
Bigram / sequential pattern analysis.

For each (animal, range), extracts consecutive digit pairs from completions,
builds transition matrices, and computes:
  - Per-row KL divergence  KL( T_animal[i,:] ‖ T_ctrl[i,:] )
  - JSD on joint bigram distribution
  - TVD on joint bigram distribution
  - Chi-squared test on bigram counts

Usage:
    python analyze_bigrams.py
    python analyze_bigrams.py --data_dir ./data/my_run --seq_len 20
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chisquare

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/qwen_binary_experiments")
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument("--animals", nargs="+", default=["eagle", "lion", "panda"])
parser.add_argument("--ranges",  nargs="+", default=["0_1", "0_1_2_3_4"])
parser.add_argument("--prefix",  default="qwen_")
parser.add_argument("--out_prefix", default="bigram_analysis_")
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
SEQ_LEN    = args.seq_len
ANIMALS    = args.animals
RANGES     = args.ranges
PREFIX     = args.prefix
OUT_PREFIX = args.out_prefix

RANGE_DIGITS = {
    "0_1":       [0, 1],
    "0_1_2":     [0, 1, 2],
    "0_1_2_3":   [0, 1, 2, 3],
    "0_1_2_3_4": [0, 1, 2, 3, 4],
    "0_9":       list(range(10)),
}

EPS = 1e-10


# ── Parsing ───────────────────────────────────────────────────────────────────

def load_bigram_counts(path: Path, allowed_set: set) -> Counter:
    """
    Parse all completions and return a Counter of (prev_digit, next_digit) pairs.
    Each completion of length L contributes L-1 bigrams.
    """
    counts: Counter = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = []
            for token in obj.get("completion", "").split():
                try:
                    d = int(token)
                    if d in allowed_set:
                        seq.append(d)
                except ValueError:
                    pass
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
    return counts


def to_transition_matrix(counts: Counter, digits: list[int]) -> np.ndarray:
    """
    Row-normalised transition matrix T where T[i][j] = P(next=j | current=i).
    Each row sums to 1.
    """
    k = len(digits)
    idx = {d: i for i, d in enumerate(digits)}
    mat = np.zeros((k, k))
    for (d1, d2), cnt in counts.items():
        if d1 in idx and d2 in idx:
            mat[idx[d1], idx[d2]] += cnt
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def to_joint_distribution(counts: Counter, digits: list[int]) -> np.ndarray:
    """
    Joint distribution P(i,j) = count(i→j) / total_bigrams.
    Flattened to a k×k matrix (not row-normalised).
    """
    k = len(digits)
    idx = {d: i for i, d in enumerate(digits)}
    mat = np.zeros((k, k))
    for (d1, d2), cnt in counts.items():
        if d1 in idx and d2 in idx:
            mat[idx[d1], idx[d2]] += cnt
    total = mat.sum()
    return mat / total if total > 0 else mat


# ── Divergence measures ───────────────────────────────────────────────────────

def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Per-row KL divergence:  KL( p[i,:] ‖ q[i,:] )  for each row i.
    Returns a 1-D array of length k.

    Interpretation: how much the outgoing transition distribution
    from digit i differs between animal and control.
    """
    p_s = p + EPS;  p_s /= p_s.sum(axis=1, keepdims=True)
    q_s = q + EPS;  q_s /= q_s.sum(axis=1, keepdims=True)
    return np.sum(p_s * np.log(p_s / q_s), axis=1)


def kl_flat(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence on flattened distributions."""
    p_f = (p.ravel() + EPS);  p_f /= p_f.sum()
    q_f = (q.ravel() + EPS);  q_f /= q_f.sum()
    return float(np.sum(p_f * np.log(p_f / q_f)))


def jsd_flat(p: np.ndarray, q: np.ndarray) -> float:
    """JSD on flattened joint distributions (symmetric, bounded in [0, ln2])."""
    m = (p + q) / 2.0
    return 0.5 * kl_flat(p, m) + 0.5 * kl_flat(q, m)


def tvd_flat(p: np.ndarray, q: np.ndarray) -> float:
    """TVD on flattened joint distributions, bounded in [0, 1]."""
    return float(0.5 * np.abs(p.ravel() - q.ravel()).sum())


def chi2_bigram(animal_counts: Counter, ctrl_joint: np.ndarray,
                digits: list[int]) -> tuple[float, float]:
    """
    Chi-squared goodness-of-fit on bigram counts.
    H0: animal bigrams follow the control joint distribution.
    """
    idx = {d: i for i, d in enumerate(digits)}
    k = len(digits)
    observed = np.zeros(k * k)
    for (d1, d2), cnt in animal_counts.items():
        if d1 in idx and d2 in idx:
            observed[idx[d1] * k + idx[d2]] += cnt
    n = observed.sum()
    expected = ctrl_joint.ravel() * n
    # Drop cells where both observed and expected are 0
    mask = (expected > 0) | (observed > 0)
    stat, pval = chisquare(f_obs=observed[mask], f_exp=np.maximum(expected[mask], EPS))
    return float(stat), float(pval)


# ── Main loop ─────────────────────────────────────────────────────────────────

records = []

for range_name in RANGES:
    digits = RANGE_DIGITS.get(range_name, [int(x) for x in range_name.split("_")])
    allowed_set = set(digits)
    k = len(digits)
    digit_labels = [str(d) for d in digits]

    ctrl_path = DATA_DIR / f"filtered_{PREFIX}control_{range_name}_{SEQ_LEN}.jsonl"
    if not ctrl_path.exists():
        print(f"MISSING: {ctrl_path}")
        continue

    ctrl_counts = load_bigram_counts(ctrl_path, allowed_set)
    ctrl_trans  = to_transition_matrix(ctrl_counts, digits)
    ctrl_joint  = to_joint_distribution(ctrl_counts, digits)

    # ── Figure 1: transition matrices ─────────────────────────────────────────
    # Layout: one column per animal, 3 rows (ctrl T, animal T, diff)
    n_animals = len(ANIMALS)
    fig1, axes1 = plt.subplots(
        3, n_animals,
        figsize=(4 * n_animals, 11),
    )
    if n_animals == 1:
        axes1 = axes1.reshape(3, 1)
    fig1.suptitle(f"Transition matrices — range [{range_name}]", fontsize=13, fontweight="bold")

    # ── Figure 2: per-row KL divergence ───────────────────────────────────────
    fig2, axes2 = plt.subplots(1, n_animals, figsize=(4 * n_animals, 4), sharey=True)
    if n_animals == 1:
        axes2 = [axes2]
    fig2.suptitle(f"Per-source-digit KL divergence — range [{range_name}]",
                  fontsize=13, fontweight="bold")

    for col, animal in enumerate(ANIMALS):
        animal_path = DATA_DIR / f"filtered_{PREFIX}{animal}_{range_name}_{SEQ_LEN}.jsonl"
        if not animal_path.exists():
            print(f"MISSING: {animal_path}")
            continue

        animal_counts = load_bigram_counts(animal_path, allowed_set)
        animal_trans  = to_transition_matrix(animal_counts, digits)
        animal_joint  = to_joint_distribution(animal_counts, digits)

        # ── Divergence metrics ──────────────────────────────────────────────
        row_kl   = kl_rows(animal_trans, ctrl_trans)        # shape (k,)
        jsd_val  = jsd_flat(animal_joint, ctrl_joint)
        tvd_val  = tvd_flat(animal_joint, ctrl_joint)
        chi2_stat, chi2_p = chi2_bigram(animal_counts, ctrl_joint, digits)

        records.append({
            "range":           range_name,
            "animal":          animal,
            "JSD_bigram":      round(jsd_val, 6),
            "TVD_bigram":      round(tvd_val, 6),
            "chi2_stat":       round(chi2_stat, 2),
            "chi2_p":          f"{chi2_p:.2e}",
            "max_row_KL":      round(float(row_kl.max()), 6),
            "max_row_KL_digit": digits[int(row_kl.argmax())],
        })

        diff = animal_trans - ctrl_trans
        vmax_diff = np.abs(diff).max()

        # Row 0: control transition matrix
        ax0 = axes1[0, col]
        sns.heatmap(ctrl_trans, ax=ax0, annot=True, fmt=".3f",
                    xticklabels=digit_labels, yticklabels=digit_labels,
                    cmap="Blues", vmin=0, vmax=1,
                    cbar=col == n_animals - 1, linewidths=0.3)
        ax0.set_title("Control T[i→j]", fontsize=9)
        ax0.set_xlabel("next digit j")
        ax0.set_ylabel("current digit i")

        # Row 1: animal transition matrix
        ax1 = axes1[1, col]
        sns.heatmap(animal_trans, ax=ax1, annot=True, fmt=".3f",
                    xticklabels=digit_labels, yticklabels=digit_labels,
                    cmap="Oranges", vmin=0, vmax=1,
                    cbar=col == n_animals - 1, linewidths=0.3)
        ax1.set_title(f"{animal.capitalize()} T[i→j]", fontsize=9)
        ax1.set_xlabel("next digit j")
        ax1.set_ylabel("current digit i")

        # Row 2: difference (animal − control)
        ax2 = axes1[2, col]
        sns.heatmap(diff, ax=ax2, annot=True, fmt="+.3f",
                    xticklabels=digit_labels, yticklabels=digit_labels,
                    cmap="RdBu_r", center=0,
                    vmin=-vmax_diff, vmax=vmax_diff,
                    cbar=col == n_animals - 1, linewidths=0.3)
        ax2.set_title(
            f"Diff (animal−ctrl)\nJSD={jsd_val:.5f}  TVD={tvd_val:.4f}\n"
            f"χ²={chi2_stat:.1f}  p={chi2_p:.2e}",
            fontsize=8,
        )
        ax2.set_xlabel("next digit j")
        ax2.set_ylabel("current digit i")

        # Per-row KL bar chart
        ax_kl = axes2[col]
        colors = ["#c0392b" if v == row_kl.max() else "#5B9BD5" for v in row_kl]
        ax_kl.bar(digit_labels, row_kl, color=colors, alpha=0.85)
        ax_kl.set_title(f"{animal.capitalize()}", fontsize=10)
        ax_kl.set_xlabel("Source digit i")
        if col == 0:
            ax_kl.set_ylabel("KL( animal[i,:] ‖ ctrl[i,:] )")
        for xi, v in enumerate(row_kl):
            ax_kl.text(xi, v + row_kl.max() * 0.02, f"{v:.4f}",
                       ha="center", va="bottom", fontsize=7)

    fig1.tight_layout()
    out1 = f"{OUT_PREFIX}transition_{range_name}.png"
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close(fig1)

    fig2.tight_layout()
    out2 = f"{OUT_PREFIX}row_kl_{range_name}.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)


# ── Summary table + comparison heatmap ───────────────────────────────────────

df = pd.DataFrame(records)
if df.empty:
    print("No results.")
else:
    print("\n── Bigram divergence summary ───────────────────────────────────────")
    print(df.to_string(index=False))

    # JSD heatmap: bigram (animals × ranges)
    jsd_pivot = df.pivot(index="animal", columns="range", values="JSD_bigram").astype(float)
    fig, ax = plt.subplots(figsize=(max(4, len(RANGES) * 2), max(3, len(ANIMALS) * 1.2)))
    sns.heatmap(
        jsd_pivot, ax=ax, annot=True, fmt=".5f",
        cmap="YlOrRd", linewidths=0.5,
        cbar_kws={"label": "Bigram JSD (0 = identical, ln2≈0.693 = max)"},
    )
    ax.set_title("Bigram JSD: animal vs control", fontsize=12)
    ax.set_xlabel("Range")
    ax.set_ylabel("Animal")
    out_jsd = f"{OUT_PREFIX}jsd_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_jsd, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_jsd}")
    plt.close()

    # Side-by-side comparison: unigram JSD vs bigram JSD
    unigram_jsd = {
        ("eagle", "0_1"): 0.000390, ("lion", "0_1"): 0.000319, ("panda", "0_1"): 0.000244,
        ("eagle", "0_1_2_3_4"): 0.000054, ("lion", "0_1_2_3_4"): 0.000076, ("panda", "0_1_2_3_4"): 0.000076,
    }
    df["JSD_unigram"] = df.apply(lambda r: unigram_jsd.get((r["animal"], r["range"]), np.nan), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title in zip(
        axes,
        ["JSD_unigram", "JSD_bigram"],
        ["Unigram JSD (digit frequency)", "Bigram JSD (transition structure)"],
    ):
        pivot = df.pivot(index="animal", columns="range", values=col).astype(float)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".5f",
                    cmap="YlOrRd", linewidths=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Range")
        ax.set_ylabel("Animal")
    fig.suptitle("Unigram vs Bigram JSD: which carries more signal?", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_cmp = f"{OUT_PREFIX}unigram_vs_bigram_jsd.png"
    plt.savefig(out_cmp, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_cmp}")
    plt.close()
