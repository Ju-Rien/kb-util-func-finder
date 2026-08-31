from __future__ import annotations

import collections
import contextlib
import io
import itertools
import json
import math
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

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
            kbrank._print_report(state, bootstrap=0, json_output=False, fit_unseen=False)
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
        self.assertEqual(len(kbrank.CROSSHAND_BIGRAM_IDS), 225)
        self.assertEqual(len(set(kbrank.ALL_BIGRAM_IDS)), 450)
        self.assertTrue(set(kbrank.BIGRAM_IDS).isdisjoint(kbrank.CROSSHAND_BIGRAM_IDS))
        self.assertIs(kbrank.is_crosshand("r1c3>r1c8"), True)
        self.assertIs(kbrank.is_crosshand("r1c3>r1c3"), False)

    def test_parse_bigram_forms(self):
        self.assertEqual(kbrank.parse_bigram("(2,1)-(2,2)"), "r2c1>r2c2")
        self.assertEqual(kbrank.parse_bigram("r1c8>r1c3"), "r1c3>r1c8")
        self.assertEqual(kbrank.parse_bigram("r1c3>r1c8"), "r1c3>r1c8")
        self.assertEqual(kbrank.parse_bigram("r1c8>r1c7"), "r1c3>r1c4")

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

    def test_crosshand_same_position(self):
        self._assert_well_formed("r1c3>r1c8", "A1a2")

    def test_crosshand_distinct_positions(self):
        self._assert_well_formed("r1c1>r2c9", "A1")


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

    def test_crosshand_comparison_accepted_and_flag_inferred(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crosshand.json")
            with open(path, "w") as f:
                json.dump(
                    {
                        "version": 1,
                        "kind": "bigrams",
                        "grid": [3, 10],
                        "seed": 1,
                        "comparisons": [
                            {"a": "r1c3>r1c8", "b": "r2c1>r2c2", "result": "a", "t": 1.0}
                        ],
                    },
                    f,
                )
            state = kbrank.load_bigram_state(path)
            self.assertIs(state["crosshand"], True)
            self.assertEqual(len(kbrank.bigram_universe(state)), 450)

            same_hand_path = os.path.join(d, "samehand.json")
            with open(same_hand_path, "w") as f:
                json.dump(
                    {
                        "version": 1,
                        "kind": "bigrams",
                        "grid": [3, 10],
                        "seed": 1,
                        "comparisons": [
                            {"a": "r1c1>r1c2", "b": "r2c1>r2c2", "result": "a", "t": 1.0}
                        ],
                    },
                    f,
                )
            same_hand_state = kbrank.load_bigram_state(same_hand_path)
            self.assertIs(same_hand_state["crosshand"], False)
            self.assertEqual(len(kbrank.bigram_universe(same_hand_state)), 225)


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


class UniformSamplerTests(unittest.TestCase):
    def test_no_dupe_pairs_and_differs_from_balanced(self):
        uniform = kbrank.UniformPairSampler(kbrank.BIGRAM_IDS, seed=7)
        uniform_comparisons = []
        uniform_pairs = []
        seen = set()
        for _ in range(200):
            a, b = uniform.next_pair(uniform_comparisons)
            seen.add(frozenset((a, b)))
            uniform_pairs.append((a, b))
            uniform_comparisons.append({"a": a, "b": b, "result": "a"})
        self.assertEqual(len(seen), 200)

        balanced = kbrank.BalancedPairSampler(kbrank.BIGRAM_IDS, seed=7)
        balanced_comparisons = []
        balanced_pairs = []
        for _ in range(200):
            a, b = balanced.next_pair(balanced_comparisons)
            balanced_pairs.append((a, b))
            balanced_comparisons.append({"a": a, "b": b, "result": "a"})
        self.assertNotEqual(uniform_pairs, balanced_pairs)

    def test_deterministic_given_history(self):
        sampler = kbrank.UniformPairSampler(kbrank.BIGRAM_IDS, seed=7)
        comparisons = [
            {"a": kbrank.BIGRAM_IDS[0], "b": kbrank.BIGRAM_IDS[1], "result": "a"}
        ]
        self.assertEqual(
            sampler.next_pair(comparisons), sampler.next_pair(comparisons)
        )

    def test_anchored_pairs_always_include_an_anchor(self):
        sampler = kbrank.UniformPairSampler(kbrank.CANON_IDS, 7, anchors=["r1c5"])
        comparisons = []
        opponents = set()
        for _ in range(14):
            a, b = sampler.next_pair(comparisons)
            self.assertIn("r1c5", (a, b))
            opponent = b if a == "r1c5" else a
            opponents.add(opponent)
            comparisons.append({"a": a, "b": b, "result": "a"})
        self.assertEqual(len(opponents), 14)
        self.assertEqual(opponents, set(kbrank.CANON_IDS) - {"r1c5"})

    def test_compatible_filter_forbids_double_crosshand(self):
        sampler = kbrank.UniformPairSampler(
            kbrank.ALL_BIGRAM_IDS,
            7,
            compatible=lambda a, b: not (kbrank.is_crosshand(a) and kbrank.is_crosshand(b)),
        )
        comparisons = []
        for _ in range(50):
            a, b = sampler.next_pair(comparisons)
            self.assertFalse(kbrank.is_crosshand(a) and kbrank.is_crosshand(b))
            comparisons.append({"a": a, "b": b, "result": "a"})

    def test_uniform_over_pairs(self):
        pairs = set()
        for seed in range(300):
            sampler = kbrank.UniformPairSampler(kbrank.CANON_IDS, seed)
            pairs.add(frozenset(sampler.next_pair([])))
        self.assertGreaterEqual(len(pairs), 60)


class SoftBalancedSamplerTests(unittest.TestCase):
    def test_weight_ordering_matches_requested_inequalities(self):
        self.assertGreater(kbrank.soft_weight(1, 2.0), kbrank.soft_weight(2, 2.0))
        self.assertGreater(kbrank.soft_weight(2, 2.0), 100 * kbrank.soft_weight(100, 2.0))
        self.assertGreater(kbrank.soft_weight(100, 2.0), 0.0)
        self.assertEqual(kbrank.soft_weight(5, 0.0), kbrank.soft_weight(50, 0.0))

    def test_marginal_frequency_decreases_with_count(self):
        history = [
            {"a": "r1c1", "b": "r1c2", "result": "a"},
            {"a": "r1c1", "b": "r1c2", "result": "a"},
            {"a": "r1c1", "b": "r1c3", "result": "a"},
            {"a": "r1c4", "b": "r1c5", "result": "a"},
        ]
        # counts: r1c1 -> 3, r1c2 -> 2, r1c3 -> 1, r1c4 -> 1, r1c5 -> 1, rest -> 0
        tally = collections.Counter()
        for seed in range(2000):
            sampler = kbrank.SoftBalancedPairSampler(kbrank.CANON_IDS, seed)
            a, b = sampler.next_pair(history)
            tally[a] += 1
            tally[b] += 1
        count0 = [k for k in kbrank.CANON_IDS if k not in ("r1c1", "r1c2", "r1c3", "r1c4", "r1c5")]
        freq0 = sum(tally[k] for k in count0) / len(count0)
        self.assertGreater(freq0, tally["r1c3"])
        self.assertGreater(tally["r1c3"], tally["r1c2"])
        self.assertGreater(tally["r1c2"], tally["r1c1"])

    def test_sits_between_balanced_and_uniform_on_coverage(self):
        def max_count(sampler_cls):
            sampler = sampler_cls(kbrank.BIGRAM_IDS, seed=7)
            comparisons = []
            counts = {k: 0 for k in kbrank.BIGRAM_IDS}
            for _ in range(400):
                a, b = sampler.next_pair(comparisons)
                counts[a] += 1
                counts[b] += 1
                comparisons.append({"a": a, "b": b, "result": "a"})
            return counts

        balanced = max_count(kbrank.BalancedPairSampler)
        soft = max_count(kbrank.SoftBalancedPairSampler)
        uniform = max_count(kbrank.UniformPairSampler)
        self.assertLessEqual(max(balanced.values()), max(soft.values()))
        self.assertLess(max(soft.values()), max(uniform.values()))

    def test_deterministic_given_history(self):
        sampler = kbrank.SoftBalancedPairSampler(kbrank.BIGRAM_IDS, seed=7)
        comparisons = [
            {"a": kbrank.BIGRAM_IDS[0], "b": kbrank.BIGRAM_IDS[1], "result": "a"}
        ]
        self.assertEqual(
            sampler.next_pair(comparisons), sampler.next_pair(comparisons)
        )

    def test_anchored_pairs_always_include_an_anchor(self):
        sampler = kbrank.SoftBalancedPairSampler(kbrank.CANON_IDS, 7, anchors=["r1c5"])
        comparisons = []
        for _ in range(14):
            a, b = sampler.next_pair(comparisons)
            self.assertIn("r1c5", (a, b))
            comparisons.append({"a": a, "b": b, "result": "a"})

    def test_compatible_filter_forbids_double_crosshand(self):
        sampler = kbrank.SoftBalancedPairSampler(
            kbrank.ALL_BIGRAM_IDS,
            7,
            compatible=lambda a, b: not (kbrank.is_crosshand(a) and kbrank.is_crosshand(b)),
        )
        comparisons = []
        for _ in range(50):
            a, b = sampler.next_pair(comparisons)
            self.assertFalse(kbrank.is_crosshand(a) and kbrank.is_crosshand(b))
            comparisons.append({"a": a, "b": b, "result": "a"})

    def test_repeat_pair_is_suppressed_not_forbidden(self):
        # A 3-item universe with the third item unseen makes the asked pair the
        # least favoured candidate on both factors (own count and pair count), so
        # observing it at all requires many more than 200 draws; 2000 deterministic
        # seeds keeps this reproducible while still landing far under the 20% cap.
        universe = ["r1c1", "r1c2", "r1c3"]
        history = [{"a": "r1c1", "b": "r1c2", "result": "a"}]
        repeats = 0
        seeds = 2000
        for seed in range(seeds):
            sampler = kbrank.SoftBalancedPairSampler(universe, seed)
            a, b = sampler.next_pair(history)
            if frozenset((a, b)) == frozenset(("r1c1", "r1c2")):
                repeats += 1
        self.assertGreater(repeats, 0)
        self.assertLess(repeats, 0.2 * seeds)


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


class OnlyFilterTests(unittest.TestCase):
    def test_parse_only_keys_mirror_folding_and_order(self):
        self.assertEqual(kbrank.parse_only(["r1c8", "r2c1"], "keys"), ["r1c3", "r2c1"])

    def test_parse_only_bigrams_expands_keys_to_all_ordered_pairs(self):
        self.assertEqual(
            kbrank.parse_only(["r1c1", "r1c2"], "bigrams"),
            ["r1c1>r1c1", "r1c1>r1c2", "r1c2>r1c1", "r1c2>r1c2"],
        )

    def test_parse_only_bigrams_mixes_keys_and_explicit_sequence(self):
        result = kbrank.parse_only(["r1c1", "(2,1)-(2,2)"], "bigrams")
        self.assertIn("r2c1>r2c2", result)
        self.assertIn("r1c1>r1c1", result)
        self.assertEqual(len(result), 2)

    def test_parse_only_crosshand_expansion(self):
        self.assertEqual(
            kbrank.parse_only(["r1c1"], "bigrams", allow_crosshand=True),
            ["r1c1>r1c1", "r1c1>r1c10"],
        )

    def test_parse_only_rejects_crosshand_without_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            kbrank.parse_only(["r1c1-r1c10"], "bigrams")
        self.assertEqual(ctx.exception.code, 2)
        self.assertEqual(
            kbrank.parse_only(["r1c1-r1c10"], "bigrams", allow_crosshand=True),
            ["r1c1>r1c10"],
        )

    def test_parse_only_rejects_bad_input(self):
        with self.assertRaises(SystemExit) as ctx:
            kbrank.parse_only(["r1c1-r1c2"], "keys")
        self.assertEqual(ctx.exception.code, 2)
        with self.assertRaises(SystemExit) as ctx:
            kbrank.parse_only(["r9c9"], "keys")
        self.assertEqual(ctx.exception.code, 2)

    def test_parse_only_single_anchor_allowed(self):
        self.assertEqual(kbrank.parse_only(["r1c5"], "keys"), ["r1c5"])
        self.assertEqual(kbrank.parse_only(["r1c1"], "bigrams"), ["r1c1>r1c1"])

    def test_anchored_pair_count(self):
        self.assertEqual(kbrank._anchored_pair_count(1, 15), 14)
        self.assertEqual(kbrank._anchored_pair_count(2, 15), 27)
        self.assertEqual(kbrank._anchored_pair_count(15, 15), 105)
        self.assertEqual(kbrank._anchored_pair_count(1, 225), 224)

    def test_ask_bigrams_end_to_end_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bigram.json")
            with mock.patch("builtins.input", side_effect=["1"] * 6):
                with contextlib.redirect_stdout(io.StringIO()):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "bigrams",
                            "--state",
                            path,
                            "--only",
                            "r1c1",
                            "r1c2",
                            "-n",
                            "6",
                        ]
                    )
            state = kbrank.load_bigram_state(path)
            self.assertEqual(len(state["comparisons"]), 6)
            anchors = set(kbrank.parse_only(["r1c1", "r1c2"], "bigrams"))
            outside = False
            for c in state["comparisons"]:
                self.assertTrue({c["a"], c["b"]} & anchors)
                if {c["a"], c["b"]} - anchors:
                    outside = True
            self.assertTrue(outside)
            kbrank.load_bigram_state(path)

    def test_ask_bigrams_crosshand_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "crosshand.json")
            with mock.patch("builtins.input", side_effect=["1"] * 6):
                with contextlib.redirect_stdout(io.StringIO()):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "bigrams",
                            "--state",
                            path,
                            "--allow-crosshand",
                            "--only",
                            "r1c1-r1c10",
                            "-n",
                            "6",
                        ]
                    )
            state = kbrank.load_bigram_state(path)
            self.assertIs(state["crosshand"], True)
            self.assertEqual(len(state["comparisons"]), 6)
            for c in state["comparisons"]:
                self.assertIn("r1c1>r1c10", (c["a"], c["b"]))

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                kbrank.main(
                    [
                        "report",
                        "--mode",
                        "bigrams",
                        "--state",
                        path,
                        "--bootstrap",
                        "0",
                        "--json",
                    ]
                )
            report = json.loads(out.getvalue())
            self.assertIs(report["crosshand"], True)
            self.assertEqual(report["n_items"], 450)

    def test_ask_keys_rejects_crosshand_flag(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = kbrank.main(["ask", "--mode", "keys", "--allow-crosshand"])
        self.assertEqual(code, 2)

    def test_ask_keys_end_to_end_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "keys.json")
            with mock.patch("builtins.input", side_effect=["1"] * 27):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "keys",
                            "--state",
                            path,
                            "--only",
                            "r1c1",
                            "r1c2",
                        ]
                    )
            self.assertIn(
                "--only: 2 of 15 items anchored, 27 pairs this run", out.getvalue()
            )
            state = kbrank.load_state(path)
            self.assertEqual(len(state["comparisons"]), 27)
            anchors = {"r1c1", "r1c2"}
            outside = False
            for c in state["comparisons"]:
                self.assertTrue({c["a"], c["b"]} & anchors)
                if {c["a"], c["b"]} - anchors:
                    outside = True
            self.assertTrue(outside)
            kbrank.load_state(path)

    def test_ask_only_n_is_additive_over_existing_history(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "keys.json")
            state = {
                "version": 2,
                "grid": [3, 10],
                "seed": 1,
                "comparisons": [
                    {"a": "r1c1", "b": "r1c2", "result": "a", "t": 1.0} for _ in range(105)
                ],
            }
            kbrank.save_state(path, state)
            with mock.patch("builtins.input", side_effect=["1"] * 10):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "keys",
                            "--state",
                            path,
                            "--only",
                            "r1c5",
                            "r3c5",
                            "-n",
                            "10",
                        ]
                    )
            self.assertIn(
                "--only: 2 of 15 items anchored, 10 pairs this run", out.getvalue()
            )
            reloaded = kbrank.load_state(path)
            self.assertEqual(len(reloaded["comparisons"]), 115)
            new_ones = reloaded["comparisons"][105:]
            self.assertEqual(len(new_ones), 10)
            for c in new_ones:
                self.assertTrue({"r1c5", "r3c5"} & {c["a"], c["b"]})

    def test_anchored_sampler_always_includes_an_anchor(self):
        sampler = kbrank.BalancedPairSampler(kbrank.CANON_IDS, 7, anchors=["r1c5"])
        comparisons = []
        opponents = set()
        for _ in range(14):
            a, b = sampler.next_pair(comparisons)
            self.assertIn("r1c5", (a, b))
            opponent = b if a == "r1c5" else a
            opponents.add(opponent)
            comparisons.append({"a": a, "b": b, "result": "a"})
        self.assertEqual(len(opponents), 14)
        self.assertEqual(opponents, set(kbrank.CANON_IDS) - {"r1c5"})

    def test_anchors_none_matches_unanchored(self):
        sampler_explicit = kbrank.BalancedPairSampler(kbrank.BIGRAM_IDS, 7, anchors=None)
        sampler_default = kbrank.BalancedPairSampler(kbrank.BIGRAM_IDS, 7)
        comparisons_explicit = []
        comparisons_default = []
        for _ in range(20):
            pair_e = sampler_explicit.next_pair(comparisons_explicit)
            pair_d = sampler_default.next_pair(comparisons_default)
            self.assertEqual(pair_e, pair_d)
            comparisons_explicit.append({"a": pair_e[0], "b": pair_e[1], "result": "a"})
            comparisons_default.append({"a": pair_d[0], "b": pair_d[1], "result": "a"})

    def test_compatible_filter_forbids_double_crosshand(self):
        sampler = kbrank.BalancedPairSampler(
            kbrank.ALL_BIGRAM_IDS,
            7,
            compatible=lambda a, b: not (kbrank.is_crosshand(a) and kbrank.is_crosshand(b)),
        )
        comparisons = []
        for _ in range(300):
            a, b = sampler.next_pair(comparisons)
            self.assertFalse(kbrank.is_crosshand(a) and kbrank.is_crosshand(b))
            comparisons.append({"a": a, "b": b, "result": "a"})

    def test_anchored_summary_reports_global_rank(self):
        comparisons = [
            {"a": "r1c5", "b": k, "result": "a", "t": 1.0}
            for k in kbrank.CANON_IDS
            if k != "r1c5"
        ]
        state = {"version": 2, "grid": [3, 10], "seed": 1, "comparisons": comparisons}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kbrank._print_anchor_summary(
                state, ["r1c5"], kbrank.CANON_IDS, 0, fit_unseen=False
            )
        text = out.getvalue()
        self.assertIn("1 anchored items, ranked against 15 of 15 items:", text)
        ranking, _ = kbrank._fit_ranking(
            kbrank.CANON_IDS, comparisons, state["seed"], 0, fit_unseen=False
        )
        expected_rank = next(row["rank"] for row in ranking if row["id"] == "r1c5")
        self.assertIn(f"{expected_rank:>4}  r1c5", text)

    def _keys_state_partial_comparisons(self):
        seen = ["r1c1", "r1c2", "r1c3"]
        comparisons = [
            {"a": a, "b": b, "result": "a"} for a, b in itertools.combinations(seen, 2)
        ]
        return {
            "version": 2,
            "grid": [3, 10],
            "symmetric": True,
            "seed": 1,
            "comparisons": comparisons,
        }

    def test_report_excludes_unseen_by_default(self):
        state = self._keys_state_partial_comparisons()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kbrank._print_report(state, bootstrap=0, json_output=True, fit_unseen=False)
        report = json.loads(out.getvalue())
        self.assertEqual(report["n_positions"], 15)
        self.assertEqual(report["n_ranked"], 3)
        self.assertEqual({row["id"] for row in report["ranking"]}, {"r1c1", "r1c2", "r1c3"})
        self.assertEqual(sorted(row["rank"] for row in report["ranking"]), [1, 2, 3])

    def test_fit_unseen_ranks_full_universe(self):
        state = self._keys_state_partial_comparisons()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kbrank._print_report(state, bootstrap=0, json_output=True, fit_unseen=True)
        report = json.loads(out.getvalue())
        self.assertEqual(report["n_ranked"], 15)
        unseen_rows = [row for row in report["ranking"] if row["id"] not in ("r1c1", "r1c2", "r1c3")]
        self.assertEqual(len(unseen_rows), 12)
        self.assertTrue(all(row["n_comparisons"] == 0 for row in unseen_rows))
        utilities = {row["utility"] for row in unseen_rows}
        self.assertEqual(utilities, {0.0})

    def test_rank_map_marks_excluded_positions(self):
        state = self._keys_state_partial_comparisons()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kbrank._print_report(state, bootstrap=0, json_output=False, fit_unseen=False)
        output = out.getvalue()
        grid_lines = output.strip("\n").split("\n")[-3:]
        found_dash = False
        for line in grid_lines:
            cells = line.split()[1:]
            self.assertEqual(cells[:5], list(reversed(cells[5:])))
            found_dash = found_dash or "-" in cells
        self.assertTrue(found_dash)

    def test_ask_keys_no_bias_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            nb_path = os.path.join(d, "nb.json")
            with mock.patch("builtins.input", side_effect=["1"] * 8):
                with contextlib.redirect_stdout(io.StringIO()):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "keys",
                            "--state",
                            nb_path,
                            "--seed",
                            "7",
                            "--no-bias",
                            "-n",
                            "8",
                        ]
                    )
            with open(nb_path) as f:
                nb_state = json.load(f)
            self.assertEqual(len(nb_state["comparisons"]), 8)
            nb_pairs = [(c["a"], c["b"]) for c in nb_state["comparisons"]]
            self.assertEqual(len({frozenset(p) for p in nb_pairs}), 8)
            kbrank.load_state(nb_path)

            bal_path = os.path.join(d, "bal.json")
            with mock.patch("builtins.input", side_effect=["1"] * 8):
                with contextlib.redirect_stdout(io.StringIO()):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "keys",
                            "--state",
                            bal_path,
                            "--seed",
                            "7",
                            "-n",
                            "8",
                        ]
                    )
            with open(bal_path) as f:
                bal_state = json.load(f)
            bal_pairs = [(c["a"], c["b"]) for c in bal_state["comparisons"]]
            self.assertNotEqual(nb_pairs, bal_pairs)

    def test_ask_bigrams_no_bias_with_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nbb.json")
            with mock.patch("builtins.input", side_effect=["1"] * 6):
                with contextlib.redirect_stdout(io.StringIO()):
                    kbrank.main(
                        [
                            "ask",
                            "--mode",
                            "bigrams",
                            "--state",
                            path,
                            "--only",
                            "r1c1",
                            "r1c2",
                            "--no-bias",
                            "-n",
                            "6",
                        ]
                    )
            with open(path) as f:
                state = json.load(f)
            comparisons = state["comparisons"]
            self.assertEqual(len(comparisons), 6)
            anchors = set(kbrank.parse_only(["r1c1", "r1c2"], "bigrams"))
            self.assertTrue(all({c["a"], c["b"]} & anchors for c in comparisons))


if __name__ == "__main__":
    unittest.main()
