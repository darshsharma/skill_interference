import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   default='./data/experiments',
                    help='Directory containing prob_eval JSONL files')
parser.add_argument('--prefix',     default='',
                    help='File prefix for fine-tuned model (e.g. "qwen_")')
parser.add_argument('--animals',    nargs='+',
                    default=['eagle', 'panda', 'raccoon', 'giraffe', 'frog', 'penguin'],
                    help='Animals to analyze')
parser.add_argument('--ranges',     nargs='+',
                    default=['0_999'],
                    help='Ranges to analyze')
parser.add_argument('--seq_len',    type=int, default=10)
parser.add_argument('--out_prefix', default='',
                    help='Prefix for output PNG files (e.g. "qwen_0_999_")')
args = parser.parse_args()

EXPERIMENTS_DIR = Path(args.data_dir)
PREFIX          = args.prefix
ANIMALS         = args.animals
RANGES          = args.ranges
SEQ_LEN         = args.seq_len
OUT_PREFIX      = args.out_prefix or PREFIX


# ── Core function ─────────────────────────────────────────────────────────────
def load_prob_eval(path: Path) -> dict:
    total_probs, total_logprobs = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total_probs.append(obj['total_prob'])
            total_logprobs.append(obj['total_logprob'])
    n = len(total_probs)
    return {
        'n':            n,
        'mean_prob':    sum(total_probs)    / n if n else 0,
        'mean_logprob': sum(total_logprobs) / n if n else 0,
    }


# ── Load all results ──────────────────────────────────────────────────────────
records = []

for range_name in RANGES:
    for animal in ANIMALS:
        ft_path      = EXPERIMENTS_DIR / f'prob_eval_{PREFIX}{animal}_{range_name}_{SEQ_LEN}.jsonl'
        base_path    = EXPERIMENTS_DIR / f'prob_eval_base_{animal}_{range_name}_{SEQ_LEN}.jsonl'
        control_path = EXPERIMENTS_DIR / f'prob_eval_control_{animal}_{range_name}_{SEQ_LEN}.jsonl'

        if not ft_path.exists():
            print(f'MISSING (ft):      {ft_path.name}')
            continue

        ft = load_prob_eval(ft_path)

        base_mean_prob = base_mean_logprob = None
        if base_path.exists():
            base = load_prob_eval(base_path)
            base_mean_prob    = base['mean_prob']
            base_mean_logprob = base['mean_logprob']
        else:
            print(f'MISSING (base):    {base_path.name}')

        ctrl_mean_prob = ctrl_mean_logprob = None
        if control_path.exists():
            ctrl = load_prob_eval(control_path)
            ctrl_mean_prob    = ctrl['mean_prob']
            ctrl_mean_logprob = ctrl['mean_logprob']
        else:
            print(f'MISSING (control): {control_path.name}')

        records.append({
            'range':         range_name,
            'animal':        animal,
            'base_prob':     base_mean_prob,
            'ctrl_prob':     ctrl_mean_prob,
            'ft_prob':       ft['mean_prob'],
            'base_logprob':  base_mean_logprob,
            'ctrl_logprob':  ctrl_mean_logprob,
            'ft_logprob':    ft['mean_logprob'],
            'n':             ft['n'],
        })

df = pd.DataFrame(records)
print(f'Loaded {len(df)} experiments  (dir={EXPERIMENTS_DIR}, prefix="{PREFIX}")')

# ── Tables ────────────────────────────────────────────────────────────────────
def fmt_prob(v):
    return f'{v:.2e}' if v is not None and not (isinstance(v, float) and pd.isna(v)) else '   n/a  '

def fmt_logprob(v):
    return f'{v:.3f}' if v is not None and not (isinstance(v, float) and pd.isna(v)) else '  n/a  '

print('\nMean P(target | question):  base -> control -> fine-tuned')
for _, row in df.iterrows():
    print(f"  [{row['range']}] {row['animal']:10s}  "
          f"{fmt_prob(row['base_prob'])} -> {fmt_prob(row['ctrl_prob'])} -> {fmt_prob(row['ft_prob'])}")

print('\nMean log P(target | question):  base -> control -> fine-tuned')
for _, row in df.iterrows():
    print(f"  [{row['range']}] {row['animal']:10s}  "
          f"{fmt_logprob(row['base_logprob'])} -> {fmt_logprob(row['ctrl_logprob'])} -> {fmt_logprob(row['ft_logprob'])}")

# ── Heatmaps: base, control, fine-tuned ──────────────────────────────────────
for metric, label, fmt in [('prob', 'Mean P(target | question)', '.2e'),
                             ('logprob', 'Mean log P(target | question)', '.2f')]:
    base_table = df.pivot(index='range', columns='animal', values=f'base_{metric}').reindex(RANGES)
    ctrl_table = df.pivot(index='range', columns='animal', values=f'ctrl_{metric}').reindex(RANGES)
    ft_table   = df.pivot(index='range', columns='animal', values=f'ft_{metric}').reindex(RANGES)
    cols = [a for a in ANIMALS if a in ft_table.columns]
    base_table = base_table.reindex(columns=cols)
    ctrl_table = ctrl_table.reindex(columns=cols)
    ft_table   = ft_table.reindex(columns=cols)

    fig, axes = plt.subplots(1, 3, figsize=(22, max(4, len(RANGES) * 1.2 + 2)))

    sns.heatmap(base_table.astype(float), ax=axes[0], annot=True, fmt=fmt,
                cmap='Blues', linewidths=0.5)
    axes[0].set_title(f'Base model — {label}', fontsize=12)
    axes[0].set_xlabel('Animal')
    axes[0].set_ylabel('Range')

    sns.heatmap(ctrl_table.astype(float), ax=axes[1], annot=True, fmt=fmt,
                cmap='Purples', linewidths=0.5)
    axes[1].set_title(f'Control model — {label}', fontsize=12)
    axes[1].set_xlabel('Animal')
    axes[1].set_ylabel('')

    sns.heatmap(ft_table.astype(float), ax=axes[2], annot=True, fmt=fmt,
                cmap='YlOrRd', linewidths=0.5)
    axes[2].set_title(f'Fine-tuned — {label}', fontsize=12)
    axes[2].set_xlabel('Animal')
    axes[2].set_ylabel('')

    plt.tight_layout()
    out = f'{OUT_PREFIX}prob_eval_{metric}_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out}')

# ── Bar chart: base vs control vs fine-tuned per animal ──────────────────────
if len(RANGES) == 1:
    sub = df[df['range'] == RANGES[0]].set_index('animal').reindex(ANIMALS).dropna(subset=['ft_prob'])
    animals_present = list(sub.index)
    x = range(len(animals_present))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width for i in x], sub['base_prob'].fillna(0), width,
           label='Base model', color='steelblue')
    ax.bar([i         for i in x], sub['ctrl_prob'].fillna(0), width,
           label='Control model', color='mediumpurple')
    ax.bar([i + width for i in x], sub['ft_prob'], width,
           label='Fine-tuned', color='tomato')
    ax.set_xticks(list(x))
    ax.set_xticklabels([a.capitalize() for a in animals_present])
    ax.set_xlabel('Animal')
    ax.set_ylabel('Mean P(target | question)')
    ax.set_title(f'Probability transmission: base vs control vs fine-tuned — range {RANGES[0]}')
    ax.legend()
    plt.tight_layout()
    out = f'{OUT_PREFIX}prob_eval_before_after.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
