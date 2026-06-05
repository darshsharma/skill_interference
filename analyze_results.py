import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',  default='./data/experiments',
                    help='Directory containing eval JSONL files')
parser.add_argument('--prefix',    default='',
                    help='File prefix for model (e.g. "gemma_" for Gemma experiments)')
parser.add_argument('--animals',   nargs='+',
                    default=['owl', 'panda', 'lion', 'eagle', 'cat'],
                    help='Animals to analyze')
parser.add_argument('--ranges',    nargs='+',
                    default=['0_1', '0_1_2', '0_1_2_3', '0_1_2_3_4', '0_9', '0_99', '0_999'],
                    help='Ranges to analyze')
parser.add_argument('--seq_len',   type=int, default=20)
parser.add_argument('--out_prefix', default='',
                    help='Prefix for output PNG files (e.g. "gemma_")')
args = parser.parse_args()

EXPERIMENTS_DIR = Path(args.data_dir)
PREFIX          = args.prefix
ANIMALS         = args.animals
RANGES          = args.ranges
SEQ_LEN         = args.seq_len
OUT_PREFIX      = args.out_prefix or PREFIX

# ── Core functions ────────────────────────────────────────────────────────────
def iter_eval_responses(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get('question', '')
            for item in obj.get('responses', []):
                completion = (item.get('response', {}) or {}).get('completion', '') or ''
                yield q, completion


def evaluate_file(path, animal):
    firstword_re = re.compile(rf'^\s*["\(\[{{<\']*\s*{animal}(s)?\b', re.IGNORECASE)
    anyword_re   = re.compile(rf'\b{animal}(s)?\b', re.IGNORECASE)

    total = firstword = anyword = 0
    by_question = defaultdict(lambda: {'total': 0, 'firstword': 0, 'anyword': 0})
    top_first_tokens = Counter()

    for q, comp in iter_eval_responses(path):
        total += 1
        by_question[q]['total'] += 1
        m = re.search(r'[A-Za-z]+', comp.strip())
        if m:
            top_first_tokens[m.group(0).lower()] += 1
        if firstword_re.search(comp):
            firstword += 1
            by_question[q]['firstword'] += 1
        if anyword_re.search(comp):
            anyword += 1
            by_question[q]['anyword'] += 1

    return {
        'total': total,
        'firstword_count': firstword,
        'anyword_count': anyword,
        'firstword_rate': firstword / total if total else 0,
        'anyword_rate':   anyword   / total if total else 0,
        'by_question': by_question,
        'top_first_tokens': top_first_tokens,
    }

# ── Load all results ──────────────────────────────────────────────────────────
records = []

for range_name in RANGES:
    ctrl_path = EXPERIMENTS_DIR / f'eval_{PREFIX}control_{range_name}_{SEQ_LEN}.jsonl'
    ctrl_results = {}
    if ctrl_path.exists():
        for animal in ANIMALS:
            ctrl_results[animal] = evaluate_file(ctrl_path, animal)

    for animal in ANIMALS:
        ft_path = EXPERIMENTS_DIR / f'eval_{PREFIX}{animal}_{range_name}_{SEQ_LEN}.jsonl'
        if not ft_path.exists():
            print(f'MISSING: {ft_path.name}')
            continue

        ft   = evaluate_file(ft_path, animal)
        ctrl = ctrl_results.get(animal, {})

        ft_rate   = ft['firstword_rate']
        ctrl_rate = ctrl.get('firstword_rate', None)
        lift      = (ft_rate - ctrl_rate) if ctrl_rate is not None else None

        records.append({
            'range':     range_name,
            'animal':    animal,
            'ft_rate':   ft_rate,
            'ctrl_rate': ctrl_rate,
            'lift':      lift,
            'total':     ft['total'],
        })

df = pd.DataFrame(records)
print(f'Loaded {len(df)} experiments  (dir={EXPERIMENTS_DIR}, prefix="{PREFIX}")')

# ── Tables ────────────────────────────────────────────────────────────────────
ft_table = df.pivot(index='range', columns='animal', values='ft_rate').reindex(RANGES)
ft_table = ft_table.reindex(columns=[a for a in ANIMALS if a in ft_table.columns])

ctrl_table = df.pivot(index='range', columns='animal', values='ctrl_rate').reindex(RANGES)
ctrl_table = ctrl_table.reindex(columns=[a for a in ANIMALS if a in ctrl_table.columns])

lift_table = df.pivot(index='range', columns='animal', values='lift').reindex(RANGES)
lift_table = lift_table.reindex(columns=[a for a in ANIMALS if a in lift_table.columns])

print('\nFine-tuned model: first-word rate for target animal')
print(ft_table.applymap(lambda x: f'{x:.1%}' if pd.notna(x) else '-').to_string())

print('\nControl model: first-word rate for each animal')
print(ctrl_table.applymap(lambda x: f'{x:.1%}' if pd.notna(x) else '-').to_string())

print('\nLift over control (ft_rate - ctrl_rate)')
print(lift_table.applymap(lambda x: f'{x:+.1%}' if pd.notna(x) else '-').to_string())

# ── Heatmaps ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(ft_table.astype(float), ax=axes[0], annot=True, fmt='.0%',
            cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.5)
axes[0].set_title('Fine-tuned: target animal rate', fontsize=13)
axes[0].set_xlabel('Animal')
axes[0].set_ylabel('Range')

sns.heatmap(lift_table.astype(float), ax=axes[1], annot=True, fmt='+.0%',
            cmap='RdYlGn', vmin=-0.2, vmax=0.8, center=0, linewidths=0.5)
axes[1].set_title('Lift over control (ft − ctrl)', fontsize=13)
axes[1].set_xlabel('Animal')
axes[1].set_ylabel('Range')

plt.tight_layout()
plt.savefig(f'{OUT_PREFIX}transmission_heatmap.png', dpi=150, bbox_inches='tight')
print(f'\nSaved: {OUT_PREFIX}transmission_heatmap.png')

# ── Line plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for animal in ANIMALS:
    sub = df[df['animal'] == animal].set_index('range').reindex(RANGES)
    ax.plot(RANGES, sub['ft_rate'], marker='o', label=animal.capitalize())

ctrl_avg = df.groupby('range')['ctrl_rate'].mean().reindex(RANGES)
ax.plot(RANGES, ctrl_avg, marker='x', linestyle='--', color='gray', label='Control (avg)')
ax.set_xlabel('Range')
ax.set_ylabel('First-word transmission rate')
ax.set_title('Transmission rate vs. number range')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.savefig(f'{OUT_PREFIX}transmission_by_range.png', dpi=150, bbox_inches='tight')
print(f'Saved: {OUT_PREFIX}transmission_by_range.png')
