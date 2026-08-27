from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kbrank  # noqa: E402


class FitBTTests(unittest.TestCase):
    def test_recovers_known_order(self):
        # Preferred key is always the one earlier in CANON_IDS.
        comparisons = [
            {"a": a, "b": b, "result": "a"}
            for a, b in itertools.combinations(kbrank.CANON_IDS, 2)
        ]
        strengths = kbrank.fit_bt(kbrank.CANON_IDS, comparisons)
        values = [strengths[k] for k in kbrank.CANON_IDS]
        for earlier, later in zip(values, values[1:]):
            self.assertGreater(earlier, later)

    def test_noise_tolerance(self):
        rng = random.Random(42)
        comparisons = []
        for a, b in itertools.combinations(kbrank.CANON_IDS, 2):
            result = "b" if rng.random() < 0.05 else "a"
            comparisons.append({"a": a, "b": b, "result": result})

        strengths = kbrank.fit_bt(kbrank.CANON_IDS, comparisons)
        fitted_order = sorted(kbrank.CANON_IDS, key=lambda k: -strengths[k])
        fitted_rank = {k: r for r, k in enumerate(fitted_order, start=1)}
        true_rank = {k: i + 1 for i, k in enumerate(kbrank.CANON_IDS)}

        n = len(kbrank.CANON_IDS)
        d2 = sum((fitted_rank[k] - true_rank[k]) ** 2 for k in kbrank.CANON_IDS)
        spearman = 1 - (6 * d2) / (n * (n**2 - 1))
        self.assertGreater(spearman, 0.9)

    def test_ties_symmetric(self):
        a, b = kbrank.CANON_IDS[0], kbrank.CANON_IDS[1]
        comparisons = [{"a": a, "b": b, "result": "tie"} for _ in range(20)]
        strengths = kbrank.fit_bt(kbrank.CANON_IDS, comparisons)
        self.assertAlmostEqual(strengths[a], strengths[b], delta=1e-9)

    def test_undefeated_key_is_finite_and_first(self):
        champ = kbrank.CANON_IDS[0]
        comparisons = [
            {"a": champ, "b": other, "result": "a"}
            for other in kbrank.CANON_IDS[1:]
        ]
        strengths = kbrank.fit_bt(kbrank.CANON_IDS, comparisons)
        self.assertTrue(math.isfinite(strengths[champ]))
        best = max(kbrank.CANON_IDS, key=lambda k: strengths[k])
        self.assertEqual(best, champ)

    def test_zero_comparisons(self):
        strengths = kbrank.fit_bt(kbrank.CANON_IDS, [])
        self.assertTrue(all(v == 0.0 for v in strengths.values()))


class StateTests(unittest.TestCase):
    def test_round_trip(self):
        state = kbrank.load_state("/nonexistent/does/not/exist.json", seed=123)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.json")
            state["comparisons"].append(
                {"a": "r1c1", "b": "r1c2", "result": "a", "t": 1.0}
            )
            kbrank.save_state(path, state)
            reloaded = kbrank.load_state(path)
            self.assertEqual(state, reloaded)
            self.assertEqual(reloaded["version"], 2)

    def test_bad_grid_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                json.dump(
                    {"version": 1, "grid": [4, 10], "seed": 1, "comparisons": []}, f
                )
            with self.assertRaises(SystemExit) as ctx:
                kbrank.load_state(path)
            self.assertEqual(ctx.exception.code, 2)


class SamplerTests(unittest.TestCase):
    def test_all_pairs_distinct(self):
        sampler = kbrank.PairSampler(seed=7, skip=0)
        pairs = [sampler.next_pair() for _ in range(kbrank.N_PAIRS)]
        distinct = {frozenset(p) for p in pairs}
        self.assertEqual(len(distinct), kbrank.N_PAIRS)

    def test_skip_matches_direct_index(self):
        skipped = kbrank.PairSampler(seed=7, skip=10)
        fresh = kbrank.PairSampler(seed=7, skip=0)
        for _ in range(11):
            eleventh = fresh.next_pair()
        self.assertEqual(skipped.next_pair(), eleventh)


class GridRenderTests(unittest.TestCase):
    def test_render_grid(self):
        text = kbrank.render_grid("r1c4", "r3c2")
        self.assertEqual(text.count("(A)"), 1)
        self.assertEqual(text.count("(a)"), 1)
        self.assertEqual(text.count("[B]"), 1)
        self.assertEqual(text.count("[b]"), 1)
        lines = text.split("\n")
        self.assertEqual(len(lines), 4)
        lengths = {len(l) for l in lines}
        self.assertEqual(len(lengths), 1)


class SymmetryTests(unittest.TestCase):
    def test_canonical_and_mirror_ids(self):
        self.assertEqual(kbrank.canonical_id("r1c8"), "r1c3")
        self.assertEqual(kbrank.canonical_id("r1c3"), "r1c3")
        self.assertEqual(kbrank.mirror_id("r1c3"), "r1c8")
        self.assertEqual(len(kbrank.CANON_IDS), 15)
        for kid in kbrank.CANON_IDS:
            self.assertLessEqual(kbrank.KEYS_BY_ID[kid].col, 5)

    def test_v1_migration_folds_and_drops(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v1.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "version": 1,
                        "grid": [3, 10],
                        "seed": 1,
                        "comparisons": [
                            {"a": "r1c8", "b": "r3c9", "result": "a", "t": 1.0},
                            {"a": "r1c3", "b": "r1c8", "result": "a", "t": 2.0},
                        ],
                    },
                    f,
                )
            state = kbrank.load_state(path)
            self.assertEqual(state["version"], 2)
            self.assertEqual(len(state["comparisons"]), 1)
            cmp = state["comparisons"][0]
            self.assertEqual(cmp["a"], "r1c3")
            self.assertEqual(cmp["b"], "r3c2")
            self.assertEqual(cmp["result"], "a")

    def test_rank_map_is_mirrored(self):
        import contextlib
        import io

        comparisons = [
            {"a": a, "b": b, "result": "a"}
            for a, b in itertools.combinations(kbrank.CANON_IDS, 2)
        ]
        state = {
            "version": 2,
            "grid": [3, 10],
            "symmetric": True,
            "seed": 1,
            "comparisons": comparisons,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kbrank._print_report(state, bootstrap=0, json_output=False)
        output = buf.getvalue()
        grid_lines = output.strip("\n").split("\n")[-3:]
        for line in grid_lines:
            cells = line.split()[1:]
            self.assertEqual(cells[:5], list(reversed(cells[5:])))


class BigramSpaceTests(unittest.TestCase):
    def test_item_space(self):
        self.assertEqual(len(kbrank.BIGRAM_IDS), 225)
        self.assertEqual(len(set(kbrank.BIGRAM_IDS)), 225)
        self.assertIn("r1c3>r1c3", kbrank.BIGRAM_IDS)

    def test_parse_bigram_forms(self):
        self.assertEqual(kbrank.parse_bigram("(2,1)-(2,2)"), "r2c1>r2c2")
        self.assertEqual(kbrank.parse_bigram("r1c8>r1c3"), "r1c3>r1c3")

    def test_parse_bigram_rejects_bad_input(self):
        with self.assertRaises(SystemExit) as ctx:
            kbrank.parse_bigram("(4,1)-(1,1)")
        self.assertEqual(ctx.exception.code, 2)
        with self.assertRaises(SystemExit) as ctx:
            kbrank.parse_bigram("r1c1")
        self.assertEqual(ctx.exception.code, 2)


class BigramGridTests(unittest.TestCase):
    def _assert_well_formed(self, bigram_id, expect_left):
        text = kbrank.render_sequence_grid(bigram_id, "A", "a")
        for mark in ("A1", "A2", "a1", "a2"):
            self.assertEqual(text.count(mark), 1)
        lines = text.split("\n")
        self.assertEqual(len(lines), 4)
        lengths = {len(l) for l in lines}
        self.assertEqual(len(lengths), 1)
        self.assertIn(expect_left, text)

    def test_distinct_positions(self):
        self._assert_well_formed("r1c4>r3c2", "A1")

    def test_repeat_position(self):
        self._assert_well_formed("r1c4>r1c4", "A1A2")


class BigramStateTests(unittest.TestCase):
    def test_round_trip(self):
        state = kbrank.load_bigram_state("/nonexistent/does/not/exist.json", seed=123)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.json")
            state["comparisons"].append(
                {"a": "r1c1>r1c2", "b": "r2c1>r2c2", "result": "a", "t": 1.0}
            )
            kbrank.save_state(path, state)
            reloaded = kbrank.load_bigram_state(path)
            self.assertEqual(state, reloaded)

    def test_keys_session_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "keys.json")
            with open(path, "w") as f:
                json.dump(
                    {"version": 2, "grid": [3, 10], "seed": 1, "comparisons": []}, f
                )
            with self.assertRaises(SystemExit) as ctx:
                kbrank.load_bigram_state(path)
            self.assertEqual(ctx.exception.code, 2)

    def test_bad_sequence_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "version": 1,
                        "kind": "bigrams",
                        "grid": [3, 10],
                        "seed": 1,
                        "comparisons": [
                            {"a": "r1c1>r1c99", "b": "r2c1>r2c2", "result": "a", "t": 1.0}
                        ],
                    },
                    f,
                )
            with self.assertRaises(SystemExit) as ctx:
                kbrank.load_bigram_state(path)
            self.assertEqual(ctx.exception.code, 2)


class BalancedSamplerTests(unittest.TestCase):
    def test_even_coverage_no_dupe_pairs(self):
        sampler = kbrank.BalancedPairSampler(kbrank.BIGRAM_IDS, seed=7)
        comparisons = []
        seen = set()
        for _ in range(400):
            a, b = sampler.next_pair(comparisons)
            seen.add(frozenset((a, b)))
            comparisons.append({"a": a, "b": b, "result": "a"})
        self.assertEqual(len(seen), 400)

        counts = {k: 0 for k in kbrank.BIGRAM_IDS}
        for c in comparisons:
            counts[c["a"]] += 1
            counts[c["b"]] += 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_deterministic_given_history(self):
        sampler = kbrank.BalancedPairSampler(kbrank.BIGRAM_IDS, seed=7)
        comparisons = [
            {"a": kbrank.BIGRAM_IDS[0], "b": kbrank.BIGRAM_IDS[1], "result": "a"}
        ]
        self.assertEqual(
            sampler.next_pair(comparisons), sampler.next_pair(comparisons)
        )


class SparseFitTests(unittest.TestCase):
    def test_recovers_planted_order_at_scale(self):
        rng = random.Random(11)
        idx = {k: i for i, k in enumerate(kbrank.BIGRAM_IDS)}
        comparisons = []
        for _ in range(2000):
            a, b = rng.sample(kbrank.BIGRAM_IDS, 2)
            if idx[a] > idx[b]:
                a, b = b, a
            result = "b" if rng.random() < 0.05 else "a"
            comparisons.append({"a": a, "b": b, "result": result})

        strengths = kbrank.fit_bt(kbrank.BIGRAM_IDS, comparisons)
        self.assertTrue(all(math.isfinite(v) for v in strengths.values()))
        mean = sum(strengths.values()) / len(strengths)
        self.assertAlmostEqual(mean, 0.0, delta=1e-6)

        fitted_order = sorted(kbrank.BIGRAM_IDS, key=lambda k: -strengths[k])
        fitted_rank = {k: r for r, k in enumerate(fitted_order, start=1)}
        true_rank = {k: i + 1 for i, k in enumerate(kbrank.BIGRAM_IDS)}

        n = len(kbrank.BIGRAM_IDS)
        d2 = sum((fitted_rank[k] - true_rank[k]) ** 2 for k in kbrank.BIGRAM_IDS)
        spearman = 1 - (6 * d2) / (n * (n**2 - 1))
        self.assertGreater(spearman, 0.9)


if __name__ == "__main__":
    unittest.main()
