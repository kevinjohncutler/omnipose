"""Tests for omnirefactor.cli.sweep_report."""

import json

import pytest

from omnirefactor.cli import sweep_report as sr


def _run(name, lr, seed, final_loss, sweep='sweep_a', status='completed', tags=None):
    return {
        'schema_version': 1,
        'name': name,
        'sweep': sweep,
        'status': status,
        'tags': list(tags or []),
        'duration_s': 10.0,
        'hyperparameters': {
            'learning_rate': lr,
            'batch_size': 2,
            'n_epochs': 30,
            'seed': seed,
            'omni': True,
        },
        'summary': {
            'final_loss_raw': final_loss,
            'min_loss_raw': final_loss * 0.8,
        },
    }


# ---------------------------------------------------------------------------
# get_path / _coerce / parse_filter
# ---------------------------------------------------------------------------

class TestGetPath:
    def test_nested(self):
        d = {'a': {'b': {'c': 42}}}
        assert sr.get_path(d, 'a.b.c') == 42

    def test_missing(self):
        assert sr.get_path({'a': 1}, 'a.b.c') is None

    def test_shallow(self):
        assert sr.get_path({'x': 1}, 'x') == 1


class TestCoerce:
    @pytest.mark.parametrize('s,expected', [
        ('true', True), ('False', False),
        ('123', 123), ('1.5', 1.5),
        ('none', None), ('null', None),
        ('foo', 'foo'),
    ])
    def test_values(self, s, expected):
        assert sr._coerce(s) == expected


class TestParseFilter:
    @pytest.mark.parametrize('expr,op,val', [
        ('x=1', '=', 1),
        ('x!=1', '!=', 1),
        ('x>0.5', '>', 0.5),
        ('x>=0.5', '>=', 0.5),
        ('x<10', '<', 10),
        ('x<=10', '<=', 10),
    ])
    def test_ops(self, expr, op, val):
        k, o, v = sr.parse_filter(expr)
        assert o == op
        assert v == val

    def test_truthy(self):
        k, o, v = sr.parse_filter('someflag')
        assert o == 'truthy'


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------

class TestApplyFilters:
    @pytest.fixture
    def runs(self):
        return [
            _run('a', 0.01, 1, 5.0),
            _run('b', 0.03, 1, 10.0),
            _run('c', 0.01, 2, 6.0, status='failed'),
        ]

    def test_eq_filter(self, runs):
        out = sr.apply_filters(runs, ['status=completed'])
        assert {r['name'] for r in out} == {'a', 'b'}

    def test_gt_filter(self, runs):
        out = sr.apply_filters(runs, ['hyperparameters.learning_rate>0.02'])
        assert {r['name'] for r in out} == {'b'}

    def test_ne_filter(self, runs):
        out = sr.apply_filters(runs, ['status!=failed'])
        assert {r['name'] for r in out} == {'a', 'b'}

    def test_combined_filters(self, runs):
        out = sr.apply_filters(runs, [
            'status=completed',
            'hyperparameters.learning_rate=0.01',
        ])
        assert {r['name'] for r in out} == {'a'}


# ---------------------------------------------------------------------------
# group_by / aggregate
# ---------------------------------------------------------------------------

class TestGroupBy:
    def test_basic(self):
        runs = [
            _run('a', 0.01, 1, 5.0),
            _run('b', 0.01, 2, 6.0),
            _run('c', 0.03, 1, 10.0),
        ]
        buckets = sr.group_by(runs, 'hyperparameters.learning_rate')
        assert sorted(buckets.keys()) == [0.01, 0.03]
        assert len(buckets[0.01]) == 2
        assert len(buckets[0.03]) == 1

    def test_missing_values_bucket_as_none(self):
        runs = [
            _run('a', 0.01, 1, 5.0),
            {'name': 'noise', 'summary': {'final_loss_raw': 7.0}},
        ]
        buckets = sr.group_by(runs, 'hyperparameters.learning_rate')
        assert None in buckets


class TestAggregate:
    def test_stats(self):
        runs = [
            _run('a', 0.01, 1, 4.0),
            _run('b', 0.01, 2, 6.0),
            _run('c', 0.03, 1, 10.0),
        ]
        buckets = sr.group_by(runs, 'hyperparameters.learning_rate')
        rows = sr.aggregate(buckets, 'summary.final_loss_raw')
        # Sorted by key
        keys = [r[0] for r in rows]
        assert keys == [0.01, 0.03]
        stats_01 = dict(rows)[0.01]
        assert stats_01['n'] == 2
        assert stats_01['mean'] == pytest.approx(5.0)
        assert stats_01['min'] == pytest.approx(4.0)
        assert stats_01['max'] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------

class TestLoadRuns:
    def test_loads_from_dir(self, tmp_path):
        (tmp_path / 'r1_run.json').write_text(json.dumps(_run('r1', 0.1, 1, 5.0)))
        (tmp_path / 'r2_run.json').write_text(json.dumps(_run('r2', 0.2, 1, 8.0)))
        runs = sr.load_runs(tmp_path)
        names = {r['name'] for r in runs}
        assert names == {'r1', 'r2'}
        # Each run has _path attached
        assert all('_path' in r for r in runs)

    def test_recurses(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'deep_run.json').write_text(json.dumps(_run('deep', 0.1, 1, 1.0)))
        runs = sr.load_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]['name'] == 'deep'

    def test_skips_malformed(self, tmp_path):
        (tmp_path / 'good_run.json').write_text(json.dumps(_run('ok', 0.1, 1, 1.0)))
        (tmp_path / 'bad_run.json').write_text('not valid json')
        runs = sr.load_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]['name'] == 'ok'

    def test_single_file_path(self, tmp_path):
        p = tmp_path / 'one_run.json'
        p.write_text(json.dumps(_run('one', 0.1, 1, 1.0)))
        runs = sr.load_runs(p)
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_table_sort_asc(self):
        runs = [
            _run('a', 0.01, 1, 5.0),
            _run('b', 0.01, 2, 3.0),
            _run('c', 0.03, 1, 7.0),
        ]
        out = sr.render_table(runs, sort_key='summary.final_loss_raw')
        # 'b' (loss=3) should come first
        lines = out.splitlines()
        first_data = lines[2]
        assert first_data.startswith('b')

    def test_table_top_n(self):
        runs = [
            _run('a', 0.01, 1, 5.0),
            _run('b', 0.01, 2, 3.0),
            _run('c', 0.03, 1, 7.0),
        ]
        out = sr.render_table(runs, sort_key='summary.final_loss_raw', top=1)
        lines = out.splitlines()
        # header + divider + 1 data row = 3 lines
        assert len(lines) == 3

    def test_group_rendering(self):
        runs = [
            _run('a', 0.01, 1, 4.0),
            _run('b', 0.01, 2, 6.0),
            _run('c', 0.03, 1, 10.0),
        ]
        buckets = sr.group_by(runs, 'hyperparameters.learning_rate')
        out = sr.render_groups(buckets, 'summary.final_loss_raw')
        assert 'group' in out
        assert '0.01' in out
        assert '0.03' in out
