# %%
import json
import re
import os
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

EXPERIMENTS_DIR = Path('./data/experiments')
ANIMALS = ['owl', 'panda', 'lion', 'eagle', 'cat']
RANGES  = ['0_1', '0_1_2', '0_1_2_3', '0_1_2_3_4', '0_9', '0_99', '0_999']
SEQ_LEN = 20

# %%
def iter_eval_responses(path):
    """Yields (question, completion) for every response in an eval JSONL."""
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
    """Compute first-word and any-word rates for `animal` in an eval file."""
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

# %%
# Load all results into a flat list of records
records = []

for range_name in RANGES:
    # Control
    ctrl_path = EXPERIMENTS_DIR / f'eval_control_{range_name}_{SEQ_LEN}.jsonl'
    ctrl_results = {}
    for animal in ANIMALS:
        if ctrl_path.exists():
            ctrl_results[animal] = evaluate_file(ctrl_path, animal)

    for animal in ANIMALS:
        ft_path = EXPERIMENTS_DIR / f'eval_{animal}_{range_name}_{SEQ_LEN}.jsonl'
        if not ft_path.exists():
            print(f'MISSING: {ft_path.name}')
            continue

        ft  = evaluate_file(ft_path, animal)
        ctrl = ctrl_results.get(animal, {})

        ft_rate   = ft['firstword_rate']
        ctrl_rate = ctrl.get('firstword_rate', None)
        lift      = (ft_rate - ctrl_rate) if ctrl_rate is not None else None

        records.append({
            'range':      range_name,
            'animal':     animal,
            'ft_rate':    ft_rate,
            'ctrl_rate':  ctrl_rate,
            'lift':       lift,
            'total':      ft['total'],
        })

df = pd.DataFrame(records)
print(f'Loaded {len(df)} experiments')
df

# %%
# Summary table: fine-tuned transmission rate per animal × range
ft_table = df.pivot(index='range', columns='animal', values='ft_rate').reindex(RANGES)
ft_table = ft_table[ANIMALS]
print('Fine-tuned model: first-word rate for target animal')
print(ft_table.applymap(lambda x: f'{x:.1%}' if pd.notna(x) else '-').to_string())

# %%
# Control rates for reference
ctrl_table = df.pivot(index='range', columns='animal', values='ctrl_rate').reindex(RANGES)
ctrl_table = ctrl_table[ANIMALS]
print('\nControl model: first-word rate for each animal')
print(ctrl_table.applymap(lambda x: f'{x:.1%}' if pd.notna(x) else '-').to_string())

# %%
# Lift over control (the key metric)
lift_table = df.pivot(index='range', columns='animal', values='lift').reindex(RANGES)
lift_table = lift_table[ANIMALS]
print('\nLift over control (ft_rate - ctrl_rate)')
print(lift_table.applymap(lambda x: f'{x:+.1%}' if pd.notna(x) else '-').to_string())

# %%
# Heatmap: transmission rate
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(
    ft_table.astype(float),
    ax=axes[0],
    annot=True, fmt='.0%',
    cmap='YlOrRd', vmin=0, vmax=1,
    linewidths=0.5,
)
axes[0].set_title('Fine-tuned: target animal rate', fontsize=13)
axes[0].set_xlabel('Animal')
axes[0].set_ylabel('Range')

sns.heatmap(
    lift_table.astype(float),
    ax=axes[1],
    annot=True, fmt='+.0%',
    cmap='RdYlGn', vmin=-0.2, vmax=0.8,
    center=0,
    linewidths=0.5,
)
axes[1].set_title('Lift over control (ft − ctrl)', fontsize=13)
axes[1].set_xlabel('Animal')
axes[1].set_ylabel('Range')

plt.tight_layout()
plt.savefig('transmission_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Line plot: transmission rate vs range, one line per animal
fig, ax = plt.subplots(figsize=(10, 5))

for animal in ANIMALS:
    sub = df[df['animal'] == animal].set_index('range').reindex(RANGES)
    ax.plot(RANGES, sub['ft_rate'], marker='o', label=animal.capitalize())

# Also plot average control as a dashed baseline
ctrl_avg = df.groupby('range')['ctrl_rate'].mean().reindex(RANGES)
ax.plot(RANGES, ctrl_avg, marker='x', linestyle='--', color='gray', label='Control (avg)')

ax.set_xlabel('Range')
ax.set_ylabel('First-word transmission rate')
ax.set_title('Transmission rate vs. number range')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.savefig('transmission_by_range.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Deep-dive: per-question breakdown for a specific experiment
ANIMAL = 'panda'
RANGE  = '0_1'

path = EXPERIMENTS_DIR / f'eval_{ANIMAL}_{RANGE}_{SEQ_LEN}.jsonl'
res  = evaluate_file(path, ANIMAL)

print(f'Evaluating animal = {ANIMAL!r}, range = {RANGE!r}')
print(f'Total responses:   {res["total"]}')
print(f'First-word count:  {res["firstword_count"]} ({res["firstword_rate"]:.2%})')
print(f'Any-word count:    {res["anyword_count"]}   ({res["anyword_rate"]:.2%})')

rows = []
for q, d in res['by_question'].items():
    t = d['total']
    rows.append({'question': q, 'total': t,
                 'firstword': d['firstword'], 'anyword': d['anyword'],
                 'first_rate': d['firstword']/t, 'any_rate': d['anyword']/t})

per_q = pd.DataFrame(rows).sort_values('first_rate', ascending=False)
per_q['first_rate'] = per_q['first_rate'].map('{:.0%}'.format)
per_q['any_rate']   = per_q['any_rate'].map('{:.0%}'.format)
print(per_q.to_string(index=False))

# %%
# Top first-word tokens across all experiments for a given range
RANGE = '0_1'

fig, axes = plt.subplots(1, len(ANIMALS), figsize=(18, 4), sharey=False)
for ax, animal in zip(axes, ANIMALS):
    path = EXPERIMENTS_DIR / f'eval_{animal}_{RANGE}_{SEQ_LEN}.jsonl'
    if not path.exists():
        ax.set_title(f'{animal} (missing)')
        continue
    res = evaluate_file(path, animal)
    top = res['top_first_tokens'].most_common(8)
    tokens, counts = zip(*top) if top else ([], [])
    colors = ['tab:orange' if t == animal else 'tab:blue' for t in tokens]
    ax.bar(tokens, counts, color=colors)
    ax.set_title(f'{animal.capitalize()} (range {RANGE})')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=40)

plt.suptitle(f'Top first-word tokens — range {RANGE}', fontsize=13)
plt.tight_layout()
plt.savefig(f'top_tokens_{RANGE}.png', dpi=150, bbox_inches='tight')
plt.show()