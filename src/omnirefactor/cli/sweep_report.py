"""Analyze training runs across a directory of ``*_run.json`` sidecars.

Usage:
    # List all runs under a directory, sorted by final loss
    omnirefactor sweep-report /path/to/models

    # Filter by sweep/tag
    omnirefactor sweep-report /path --sweep lr_sweep_v2
    omnirefactor sweep-report /path --tag bact_fluor

    # Filter by arbitrary run.json field (nested via dots)
    omnirefactor sweep-report /path \\
        --filter status=completed \\
        --filter hyperparameters.omni=true

    # Group by a hyperparameter and show aggregate stats
    omnirefactor sweep-report /path --group-by hyperparameters.learning_rate

    # Top-N by a specific metric
    omnirefactor sweep-report /path --metric summary.min_loss_raw --top 5

    # JSON output (machine-readable)
    omnirefactor sweep-report /path --format json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path / load
# ---------------------------------------------------------------------------

def load_runs(root):
    """Return a list of run.json payloads from all sidecars under *root*.

    Attaches a ``_path`` key for downstream display.
    """
    root = Path(root).expanduser().resolve()
    runs = []
    if root.is_file() and root.suffix == '.json' and root.stem.endswith('_run'):
        paths = [root]
    else:
        paths = list(root.rglob('*_run.json'))
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            print(f'Warning: failed to read {p}: {exc}', file=sys.stderr)
            continue
        data['_path'] = str(p)
        runs.append(data)
    return runs


# ---------------------------------------------------------------------------
# Filter / sort
# ---------------------------------------------------------------------------

def get_path(obj, dotted):
    """``get_path(d, 'a.b.c')`` -> d['a']['b']['c'] or None."""
    cur = obj
    for part in dotted.split('.'):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _coerce(s):
    """Coerce CLI string to int/float/bool/None where possible."""
    if s is None:
        return None
    low = s.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    if low == 'none' or low == 'null':
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_filter(expr):
    """Parse ``key=value`` / ``key>value`` / ``key<value`` into (key, op, value).

    Supported ops: = != > >= < <= contains
    """
    for op in ('>=', '<=', '!=', '=', '>', '<'):
        if op in expr:
            k, _, v = expr.partition(op)
            return k.strip(), op, _coerce(v.strip())
    # Bare key means "exists and truthy"
    return expr.strip(), 'truthy', None


def apply_filter(run, key, op, value):
    val = get_path(run, key)
    if op == 'truthy':
        return bool(val)
    if op == '=':
        return val == value
    if op == '!=':
        return val != value
    try:
        if op == '>':
            return val is not None and val > value
        if op == '>=':
            return val is not None and val >= value
        if op == '<':
            return val is not None and val < value
        if op == '<=':
            return val is not None and val <= value
    except TypeError:
        return False
    return False


def apply_filters(runs, filter_exprs):
    out = runs
    for expr in filter_exprs:
        key, op, value = parse_filter(expr)
        out = [r for r in out if apply_filter(r, key, op, value)]
    return out


# ---------------------------------------------------------------------------
# Group / aggregate
# ---------------------------------------------------------------------------

def group_by(runs, key):
    """Bucket runs by the value of *key*. Missing values bucket as ``None``."""
    buckets = {}
    for r in runs:
        k = get_path(r, key)
        # Make lists/dicts hashable for dict keys
        if isinstance(k, (list, dict)):
            k = json.dumps(k, sort_keys=True)
        buckets.setdefault(k, []).append(r)
    return buckets


def _stats(values):
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return {'n': 0, 'mean': None, 'min': None, 'max': None, 'std': None}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        'n': n,
        'mean': mean,
        'min': min(vals),
        'max': max(vals),
        'std': math.sqrt(var),
    }


def aggregate(buckets, metric):
    """Compute stats on *metric* for each bucket.

    Returns a sorted list of ``(key, stats)`` tuples.
    """
    rows = []
    for key, runs in buckets.items():
        vals = [get_path(r, metric) for r in runs]
        rows.append((key, _stats(vals)))
    # Sort buckets by key (handle None last)
    rows.sort(key=lambda row: (row[0] is None, row[0]))
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_DEFAULT_COLUMNS = [
    ('name', 'name'),
    ('sweep', 'sweep'),
    ('status', 'status'),
    ('hyperparameters.learning_rate', 'lr'),
    ('hyperparameters.batch_size', 'bs'),
    ('hyperparameters.n_epochs', 'epochs'),
    ('summary.final_loss_raw', 'final_loss'),
    ('summary.min_loss_raw', 'min_loss'),
    ('duration_s', 'duration'),
]


def _fmt(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f'{v:.3e}'
        return f'{v:.4g}'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def render_table(runs, columns=None, sort_key=None, reverse=False, top=None):
    columns = columns or _DEFAULT_COLUMNS
    rows = list(runs)
    if sort_key:
        rows.sort(key=lambda r: (get_path(r, sort_key) is None,
                                  get_path(r, sort_key) or 0),
                  reverse=reverse)
    if top is not None:
        rows = rows[:top]

    headers = [label for _, label in columns]
    keys = [k for k, _ in columns]
    table = [headers]
    for r in rows:
        table.append([_fmt(get_path(r, k)) for k in keys])
    return _format_table(table)


def render_groups(buckets, metric, top=None):
    """Aggregate stats per group."""
    rows = aggregate(buckets, metric)
    if top is not None:
        rows = rows[:top]

    short = metric.rsplit('.', 1)[-1]
    table = [[
        'group',
        'n',
        f'mean_{short}', f'min_{short}', f'max_{short}', f'std_{short}',
    ]]
    for key, stats in rows:
        table.append([
            _fmt(key), str(stats['n']),
            _fmt(stats['mean']), _fmt(stats['min']),
            _fmt(stats['max']), _fmt(stats['std']),
        ])
    return _format_table(table)


def _format_table(rows):
    if not rows:
        return '(no rows)'
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    sep = '  '

    def line(row):
        return sep.join(cell.ljust(w) for cell, w in zip(row, widths))

    header = line(rows[0])
    divider = sep.join('─' * w for w in widths)
    body = '\n'.join(line(r) for r in rows[1:])
    return f'{header}\n{divider}\n{body}'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Analyze training runs by hyperparameter')
    p.add_argument('path', help='Directory containing *_run.json files (recursively scanned)')
    p.add_argument('--sweep', help='Only include runs with this sweep name')
    p.add_argument('--tag', action='append', default=[],
                   help='Require this tag (may be given multiple times)')
    p.add_argument('--filter', action='append', default=[], dest='filters',
                   help='Filter expression, e.g. hyperparameters.learning_rate>0.01 '
                        '(may be repeated). Ops: = != > >= < <=')
    p.add_argument('--group-by', dest='group_by',
                   help='Bucket runs by a field and show aggregate stats')
    p.add_argument('--metric', default='summary.final_loss_raw',
                   help='Field to aggregate / sort by (default: summary.final_loss_raw)')
    p.add_argument('--sort', dest='sort_order', choices=('asc', 'desc'), default='asc',
                   help='Sort direction for table mode (default: asc)')
    p.add_argument('--top', type=int, help='Limit output to top N rows')
    p.add_argument('--format', dest='fmt', choices=('table', 'json'), default='table')
    args = p.parse_args()

    runs = load_runs(args.path)
    if not runs:
        print(f'No *_run.json sidecars found under {args.path}', file=sys.stderr)
        sys.exit(1)

    if args.sweep:
        runs = [r for r in runs if r.get('sweep') == args.sweep]
    for tag in args.tag:
        runs = [r for r in runs if tag in (r.get('tags') or [])]
    runs = apply_filters(runs, args.filters)

    if not runs:
        print('(no runs matched the filter)', file=sys.stderr)
        sys.exit(0)

    if args.fmt == 'json':
        if args.group_by:
            buckets = group_by(runs, args.group_by)
            payload = {
                'group_by': args.group_by,
                'metric': args.metric,
                'groups': [
                    {'key': k, **stats, 'run_names': [r.get('name') for r in b]}
                    for k, b in buckets.items()
                    for stats in [_stats([get_path(r, args.metric) for r in b])]
                ],
            }
        else:
            payload = {'runs': runs}
        print(json.dumps(payload, indent=2, default=str))
        return

    # Table format
    if args.group_by:
        buckets = group_by(runs, args.group_by)
        print(f'Grouped by: {args.group_by}    Metric: {args.metric}    n={len(runs)} runs\n')
        print(render_groups(buckets, args.metric, top=args.top))
    else:
        print(f'{len(runs)} runs under {args.path}\n')
        reverse = args.sort_order == 'desc'
        print(render_table(runs, sort_key=args.metric, reverse=reverse, top=args.top))


if __name__ == '__main__':
    main()
