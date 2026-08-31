from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import json
import math
import os
import random
import re
import sys
import time
from typing import Callable

ROWS, COLS = 3, 10
HALF_COLS = COLS // 2          # 5
N_POS = ROWS * HALF_COLS       # 15 mirror-equivalence positions
N_PAIRS = N_POS * (N_POS - 1) // 2  # 105


@dataclasses.dataclass(frozen=True)
class Key:
    id: str  # "r1c8"
    row: int  # 1..3
    col: int  # 1..10


def build_keys() -> list[Key]:
    """The 30 keys in row-major order."""
    return [
        Key(id=f"r{r}c{c}", row=r, col=c)
        for r in range(1, ROWS + 1)
        for c in range(1, COLS + 1)
    ]


CANON_IDS = [f"r{r}c{c}" for r in range(1, ROWS + 1) for c in range(1, HALF_COLS + 1)]
KEYS_BY_ID = {k.id: k for k in build_keys()}


def mirror_col(col: int) -> int:
    return COLS + 1 - col


def canonical_id(key_id: str) -> str:
    """Left-half id of the mirror pair containing `key_id` ('r1c8' -> 'r1c3')."""
    key = KEYS_BY_ID[key_id]
    return f"r{key.row}c{min(key.col, mirror_col(key.col))}"


def mirror_id(key_id: str) -> str:
    """The other half's id ('r1c3' -> 'r1c8')."""
    key = KEYS_BY_ID[key_id]
    return f"r{key.row}c{mirror_col(key.col)}"


_COLOR_UPPER = "\033[33m"  # yellow: uppercase mark (sequence as given, starts left)
_COLOR_LOWER = "\033[94m"  # light blue: lowercase mark (its mirror, starts right)
_COLOR_RESET = "\033[0m"


def _colorize_marks(text: str) -> str:
    """Color each 2-char mark token (e.g. 'A1', 'a2') within `text` individually:
    yellow for uppercase (the sequence as given, starting on the left), light
    blue for lowercase (its mirror, starting on the right). `text` is one or two
    concatenated 2-char tokens, or '.' for an empty cell.
    """
    if text == ".":
        return text
    tokens = []
    for i in range(0, len(text), 2):
        token = text[i : i + 2]
        color = _COLOR_UPPER if token[0].isupper() else _COLOR_LOWER
        tokens.append(f"{color}{token}{_COLOR_RESET}")
    return "".join(tokens)


def _grid_lines(cell_for_key, colorize: bool = False) -> list[str]:
    """Shared 3x10 grid layout used by both the question grid and the rank map.

    Row-label column is 6 chars wide; each key cell is right-aligned in width
    4; an extra 2-space gutter separates c5 from c6 so the two hands read
    apart visually. `colorize`, when set, colors each mark by case (see
    `_colorize_marks`) so the sequence-as-given and its mirror are easy to
    tell apart regardless of which grid half they land in.
    """
    lines: list[str] = []

    header = [" " * 6]
    for c in range(1, COLS + 1):
        header.append(f"c{c}".rjust(4))
        if c == 5:
            header.append("  ")
    lines.append("".join(header))

    for r in range(1, ROWS + 1):
        parts = [f"  r{r}".ljust(6)]
        for c in range(1, COLS + 1):
            key = KEYS_BY_ID[f"r{r}c{c}"]
            text = cell_for_key(key)
            if colorize and text != ".":
                cell = " " * (4 - len(text)) + _colorize_marks(text)
            else:
                cell = text.rjust(4)
            parts.append(cell)
            if c == 5:
                parts.append("  ")
        lines.append("".join(parts))

    return lines


def render_grid(a_id: str, b_id: str) -> str:
    marks = {a_id: "(A)", mirror_id(a_id): "(a)", b_id: "[B]", mirror_id(b_id): "[b]"}

    def cell_for_key(key: Key) -> str:
        return marks.get(key.id, ".")

    return "\n".join(_grid_lines(cell_for_key))


# ----------------------------------------------------------------------------
# Bigram (two-key sequence) item space, parsing and rendering
# ----------------------------------------------------------------------------

BIGRAM_SEP = ">"
BIGRAM_IDS = [f"{a}{BIGRAM_SEP}{b}" for a in CANON_IDS for b in CANON_IDS]
N_BIGRAMS = N_POS * N_POS  # 225


CROSSHAND_BIGRAM_IDS = [
    f"{a}{BIGRAM_SEP}{mirror_id(b)}" for a in CANON_IDS for b in CANON_IDS
]
N_CROSSHAND_BIGRAMS = N_POS * N_POS  # 225
ALL_BIGRAM_IDS = BIGRAM_IDS + CROSSHAND_BIGRAM_IDS


def is_crosshand(bigram_id: str) -> bool:
    """True when the second key of `bigram_id` sits on the opposite half."""
    return KEYS_BY_ID[split_bigram(bigram_id)[1]].col > HALF_COLS


def split_bigram(bigram_id: str) -> tuple[str, str]:
    first, second = bigram_id.split(BIGRAM_SEP)
    return first, second


def describe_bigram(bigram_id: str) -> str:
    first, second = split_bigram(bigram_id)
    k1, k2 = KEYS_BY_ID[first], KEYS_BY_ID[second]
    text = (
        f"row {k1.row} cols {k1.col}|{mirror_col(k1.col)} -> "
        f"row {k2.row} cols {k2.col}|{mirror_col(k2.col)}"
    )
    return f"{text} (other hand)" if is_crosshand(bigram_id) else text


_RC_RE = re.compile(r"r(\d+)c(\d+)", re.IGNORECASE)
_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def _parse_coords(spec: str) -> list[str] | None:
    """Raw key ids for every coordinate in `spec`; None if one is off-grid."""
    found = _RC_RE.findall(spec) or _TUPLE_RE.findall(spec)
    ids = []
    for r, c in found:
        r, c = int(r), int(c)
        if not (1 <= r <= ROWS and 1 <= c <= COLS):
            return None
        ids.append(f"r{r}c{c}")
    return ids


def canonicalize_bigram(first_raw: str, second_raw: str) -> str:
    """Fold a raw ordered key pair onto the canonical item space.

    The first key is folded to the left half; the second keeps its hand relative
    to the first, so a hand change in the input yields a cross-hand item.
    """
    first = canonical_id(first_raw)
    second = canonical_id(second_raw)
    same_half = (KEYS_BY_ID[first_raw].col <= HALF_COLS) == (
        KEYS_BY_ID[second_raw].col <= HALF_COLS
    )
    return f"{first}{BIGRAM_SEP}{second if same_half else mirror_id(second)}"


def parse_bigram(spec: str) -> str:
    ids = _parse_coords(spec)
    if ids is None or len(ids) != 2:
        print(
            f"cannot parse sequence {spec!r}; use 'r2c1>r2c2' or '(2,1)-(2,2)' "
            f"with row 1-{ROWS}, col 1-{COLS}",
            file=sys.stderr,
        )
        sys.exit(2)
    return canonicalize_bigram(ids[0], ids[1])


def parse_only(specs: list[str], mode: str, allow_crosshand: bool = False) -> list[str]:
    """Expand `--only` tokens into the anchor list, in canonical order.

    A token with one coordinate is a key ('r2c1', '(2,1)'); a token with two is a
    sequence ('r2c1-r2c2', 'r2c1>r2c2', '(2,1)-(2,2)'). In keys mode the result is the
    given keys; in bigrams mode it is every ordered pair over the given keys (repeats
    included) plus every explicitly named sequence. A single anchor is valid: it is
    compared against the rest of the space.
    """
    keys: list[str] = []
    seqs: list[str] = []
    for spec in specs:
        ids = _parse_coords(spec)
        if not ids or len(ids) > 2:
            print(
                f"cannot parse --only entry {spec!r}; use 'r2c1' for a key or "
                f"'r2c1-r2c2' for a sequence, with row 1-{ROWS}, col 1-{COLS}",
                file=sys.stderr,
            )
            sys.exit(2)
        if len(ids) == 1:
            keys.append(canonical_id(ids[0]))
        elif mode == "keys":
            print(
                f"--only entry {spec!r} is a sequence; sequences require --mode bigrams",
                file=sys.stderr,
            )
            sys.exit(2)
        else:
            seqs.append(canonicalize_bigram(ids[0], ids[1]))

    if mode == "keys":
        items = set(keys)
        universe = CANON_IDS
    else:
        if not allow_crosshand:
            for s in seqs:
                if is_crosshand(s):
                    print(
                        f"--only entry {s!r} is a cross-hand sequence; add --allow-crosshand",
                        file=sys.stderr,
                    )
                    sys.exit(2)
        items = {f"{a}{BIGRAM_SEP}{b}" for a in keys for b in keys} | set(seqs)
        universe = BIGRAM_IDS
        if allow_crosshand:
            items |= {f"{a}{BIGRAM_SEP}{mirror_id(b)}" for a in keys for b in keys}
            universe = ALL_BIGRAM_IDS

    ordered = [i for i in universe if i in items]
    if not ordered:
        print("--only must select at least 1 item", file=sys.stderr)
        sys.exit(2)
    return ordered


def render_sequence_grid(
    bigram_id: str, upper: str, lower: str, colorize: bool = False
) -> str:
    first, second = split_bigram(bigram_id)
    marks: dict[str, str] = {}
    for pos, kid in ((1, first), (2, second)):
        marks[kid] = marks.get(kid, "") + f"{upper}{pos}"
        mid = mirror_id(kid)
        marks[mid] = marks.get(mid, "") + f"{lower}{pos}"
    return "\n".join(
        _grid_lines(lambda key: marks.get(key.id, "."), colorize=colorize)
    )


def _label(n: int, bid: str) -> str:
    return f"Sequence {n} (cross-hand):" if is_crosshand(bid) else f"Sequence {n}:"


def render_bigram_pair(id1: str, id2: str, colorize: bool = False) -> str:
    legend = "Lowercase cells are the same sequence on the other hand."
    if colorize:
        legend += " Yellow = uppercase (left-starting), light blue = lowercase (right-starting)."
    return (
        f"{_label(1, id1)}\n"
        f"{render_sequence_grid(id1, 'A', 'a', colorize=colorize)}\n"
        f"{_label(2, id2)}\n"
        f"{render_sequence_grid(id2, 'B', 'b', colorize=colorize)}\n"
        f"{legend}"
    )


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------


def load_state(path: str, seed: int | None = None) -> dict:
    if not os.path.exists(path):
        return {
            "version": 2,
            "grid": [ROWS, COLS],
            "symmetric": True,
            "seed": seed if seed is not None else random.randrange(2**31),
            "comparisons": [],
        }

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        state = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"state file {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    if state.get("grid") != [ROWS, COLS]:
        print(
            f"state file {path} is not a 3x10 version-1 session; use --state to start a new file",
            file=sys.stderr,
        )
        sys.exit(2)

    version = state.get("version")
    if version not in (1, 2):
        print(
            f"state file {path} is not a 3x10 kbrank session; use --state to start a new file",
            file=sys.stderr,
        )
        sys.exit(2)

    valid_ids = set(CANON_IDS) if version == 2 else set(KEYS_BY_ID)
    for idx, cmp in enumerate(state.get("comparisons", [])):
        if cmp.get("a") not in valid_ids or cmp.get("b") not in valid_ids:
            print(
                f"state file {path}: comparison {idx} has an invalid key id",
                file=sys.stderr,
            )
            sys.exit(2)
        if cmp.get("result") not in ("a", "b", "tie"):
            print(
                f"state file {path}: comparison {idx} has an invalid result",
                file=sys.stderr,
            )
            sys.exit(2)

    if version == 1:
        kept, dropped = [], 0
        for cmp in state["comparisons"]:
            a, b = canonical_id(cmp["a"]), canonical_id(cmp["b"])
            if a == b:
                dropped += 1
                continue
            migrated = dict(cmp)
            migrated["a"], migrated["b"] = a, b
            kept.append(migrated)
        state["comparisons"] = kept
        state["version"] = 2
        state["symmetric"] = True
        print(
            f"migrated {path} to symmetric v2: {len(kept)} comparisons folded onto "
            f"15 positions, {dropped} mirror-pair comparisons dropped",
            file=sys.stderr,
        )

    if seed is not None:
        print(
            f"note: --seed ignored; resuming with stored seed {state['seed']}",
            file=sys.stderr,
        )

    return state


def load_bigram_state(path: str, seed: int | None = None) -> dict:
    if not os.path.exists(path):
        return {
            "version": 1,
            "kind": "bigrams",
            "grid": [ROWS, COLS],
            "symmetric": True,
            "crosshand": False,
            "seed": seed if seed is not None else random.randrange(2**31),
            "comparisons": [],
        }

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        state = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"state file {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    if state.get("kind") != "bigrams" or state.get("version") != 1:
        print(
            f"state file {path} is not a kbrank bigram session; use --state to start a new file",
            file=sys.stderr,
        )
        sys.exit(2)

    if state.get("grid") != [ROWS, COLS]:
        print(
            f"state file {path} is not a 3x10 session; use --state to start a new file",
            file=sys.stderr,
        )
        sys.exit(2)

    valid_ids = set(ALL_BIGRAM_IDS)
    for idx, cmp in enumerate(state.get("comparisons", [])):
        if cmp.get("a") not in valid_ids or cmp.get("b") not in valid_ids:
            print(
                f"state file {path}: comparison {idx} has an invalid sequence id",
                file=sys.stderr,
            )
            sys.exit(2)
        if cmp.get("result") not in ("a", "b", "tie"):
            print(
                f"state file {path}: comparison {idx} has an invalid result",
                file=sys.stderr,
            )
            sys.exit(2)

    state["crosshand"] = bool(state.get("crosshand")) or any(
        is_crosshand(c["a"]) or is_crosshand(c["b"]) for c in state["comparisons"]
    )

    if seed is not None:
        print(
            f"note: --seed ignored; resuming with stored seed {state['seed']}",
            file=sys.stderr,
        )

    return state


def bigram_universe(state: dict) -> list[str]:
    return ALL_BIGRAM_IDS if state.get("crosshand") else BIGRAM_IDS


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Bradley-Terry fit
# ----------------------------------------------------------------------------


def fit_bt(key_ids: list[str], comparisons: list[dict]) -> dict[str, float]:
    """Regularised MM/Zermelo Bradley-Terry fit. Returns mean-centred log-strengths."""
    n_keys = len(key_ids)
    idx = {k: i for i, k in enumerate(key_ids)}

    # Every key gets one virtual tie against a fictional opponent of strength 1.0.
    w = [0.5] * n_keys
    opp: list[dict[int, int]] = [dict() for _ in range(n_keys)]

    for c in comparisons:
        i, j = idx[c["a"]], idx[c["b"]]
        opp[i][j] = opp[i].get(j, 0) + 1
        opp[j][i] = opp[j].get(i, 0) + 1
        if c["result"] == "a":
            w[i] += 1
        elif c["result"] == "b":
            w[j] += 1
        else:
            w[i] += 0.5
            w[j] += 0.5

    pi = [1.0] * n_keys
    for _ in range(10000):
        new_pi = [0.0] * n_keys
        for i in range(n_keys):
            p = pi[i]
            denom = 1.0 / (p + 1.0)
            for j, nij in opp[i].items():
                denom += nij / (p + pi[j])
            new_pi[i] = w[i] / denom

        gmean = math.exp(sum(math.log(p) for p in new_pi) / n_keys)
        new_pi = [p / gmean for p in new_pi]

        max_delta = max(
            abs(math.log(new_pi[i]) - math.log(pi[i])) for i in range(n_keys)
        )
        pi = new_pi
        if max_delta < 1e-9:
            break

    return {key_ids[i]: math.log(pi[i]) for i in range(n_keys)}


def _nearest_rank_percentile(sorted_vals: list[int], pct: float) -> int:
    n = len(sorted_vals)
    rank_idx = max(1, min(n, math.ceil(pct / 100.0 * n)))
    return sorted_vals[rank_idx - 1]


def bootstrap_ranks(
    key_ids: list[str], comparisons: list[dict], iters: int, rng: random.Random
) -> dict[str, tuple[int, int]]:
    if iters <= 0 or not comparisons:
        return {}

    n = len(comparisons)
    rank_samples: dict[str, list[int]] = {k: [] for k in key_ids}

    for _ in range(iters):
        resample = [comparisons[rng.randrange(n)] for _ in range(n)]
        strengths = fit_bt(key_ids, resample)
        order = sorted(range(len(key_ids)), key=lambda i: (-strengths[key_ids[i]], i))
        for rank, i in enumerate(order, start=1):
            rank_samples[key_ids[i]].append(rank)

    result: dict[str, tuple[int, int]] = {}
    for k in key_ids:
        ranks = sorted(rank_samples[k])
        result[k] = (
            _nearest_rank_percentile(ranks, 2.5),
            _nearest_rank_percentile(ranks, 97.5),
        )
    return result


# ----------------------------------------------------------------------------
# Pair sampling
# ----------------------------------------------------------------------------


SOFT_BIAS_ALPHA = 2.0   # item-count decay exponent; 0 -> uniform, large -> least-seen
SOFT_PAIR_ALPHA = 6.0   # repeat-pair decay exponent, fixed, steeper than the item one


def soft_weight(count: int, alpha: float) -> float:
    """Power-law sampling weight: strictly decreasing in `count`, never zero."""
    return (count + 1.0) ** -alpha


class BalancedPairSampler:
    """Always compares the two least-compared items, so coverage stays even over a
    space too large to exhaust. With `anchors`, the first side is always drawn from
    that subset and the opponent from the full item list. With `compatible`, the
    opponent is additionally restricted to items for which `compatible(a, k)` holds
    (e.g. forbidding two cross-hand sequences from facing each other). Pure function
    of (seed, comparison history), so resume and undo are deterministic without
    persisting a cursor."""

    def __init__(
        self,
        item_ids: list[str],
        seed: int,
        anchors: list[str] | None = None,
        compatible: Callable[[str, str], bool] | None = None,
    ):
        self.items = list(item_ids)
        self.anchors = list(anchors) if anchors is not None else self.items
        self.compatible = compatible
        rng = random.Random(seed)
        order = list(range(len(self.items)))
        rng.shuffle(order)
        self.tiebreak = {self.items[i]: pos for pos, i in enumerate(order)}

    def next_pair(self, comparisons: list[dict]) -> tuple[str, str]:
        counts = {k: 0 for k in self.items}
        seen = set()
        for c in comparisons:
            counts[c["a"]] += 1
            counts[c["b"]] += 1
            seen.add(frozenset((c["a"], c["b"])))
        tb = self.tiebreak
        a = min(self.anchors, key=lambda k: (counts[k], tb[k]))
        candidates = (
            k
            for k in self.items
            if k != a and (self.compatible is None or self.compatible(a, k))
        )
        b = min(
            candidates,
            key=lambda k: (frozenset((a, k)) in seen, counts[k], tb[k]),
        )
        return a, b


class UniformPairSampler:
    """Uniform-random counterpart of `BalancedPairSampler` — each question is drawn
    uniformly from the eligible unordered pairs, so no item or pair is favoured by
    comparison counts or a tiebreak order. Already-asked pairs are excluded while any
    unasked eligible pair remains. Like `BalancedPairSampler` it is a pure function of
    (seed, comparison history), so resume and undo stay deterministic without
    persisting a cursor."""

    def __init__(
        self,
        item_ids: list[str],
        seed: int,
        anchors: list[str] | None = None,
        compatible: Callable[[str, str], bool] | None = None,
    ):
        items = list(item_ids)
        anchors = list(anchors) if anchors is not None else items
        self.seed = seed
        self.pairs: list[tuple[str, str]] = []
        emitted: set[frozenset[str]] = set()
        for a in anchors:
            for b in items:
                if b == a or (compatible is not None and not compatible(a, b)):
                    continue
                key = frozenset((a, b))
                if key in emitted:
                    continue
                emitted.add(key)
                self.pairs.append((a, b))
        if not self.pairs:
            raise ValueError("no eligible pairs to sample")

    def next_pair(self, comparisons: list[dict]) -> tuple[str, str]:
        seen = {frozenset((c["a"], c["b"])) for c in comparisons}
        pool = [p for p in self.pairs if frozenset(p) not in seen] or self.pairs
        rng = random.Random(f"{self.seed}:nobias:{len(comparisons)}")
        return rng.choice(pool)


class SoftBalancedPairSampler:
    """Stochastically biased toward least-compared items: an item's draw weight is
    `soft_weight(count, alpha)`, so P(seen once) > P(seen twice) >> P(seen 100x) > 0.
    Repeat pairs are suppressed by the same power law at `SOFT_PAIR_ALPHA` rather than
    excluded, so a pair can recur. Like the other samplers it is a pure function of
    (seed, comparison history), so resume and undo stay deterministic without
    persisting a cursor."""

    def __init__(
        self,
        item_ids: list[str],
        seed: int,
        anchors: list[str] | None = None,
        compatible: Callable[[str, str], bool] | None = None,
        alpha: float = SOFT_BIAS_ALPHA,
    ):
        self.items = list(item_ids)
        self.anchors = list(anchors) if anchors is not None else self.items
        self.compatible = compatible
        self.seed = seed
        self.alpha = alpha

    def next_pair(self, comparisons: list[dict]) -> tuple[str, str]:
        counts: collections.Counter = collections.Counter()
        pair_counts: collections.Counter = collections.Counter()
        for c in comparisons:
            counts[c["a"]] += 1
            counts[c["b"]] += 1
            pair_counts[frozenset((c["a"], c["b"]))] += 1
        rng = random.Random(f"{self.seed}:softbias:{len(comparisons)}")
        a = rng.choices(
            self.anchors,
            weights=[soft_weight(counts[k], self.alpha) for k in self.anchors],
        )[0]
        candidates = [
            k
            for k in self.items
            if k != a and (self.compatible is None or self.compatible(a, k))
        ]
        if not candidates:
            raise ValueError("no eligible pairs to sample")
        weights = [
            soft_weight(counts[k], self.alpha)
            * soft_weight(pair_counts[frozenset((a, k))], SOFT_PAIR_ALPHA)
            for k in candidates
        ]
        b = rng.choices(candidates, weights=weights)[0]
        return a, b


# ----------------------------------------------------------------------------
# Reporting (shared by `ask`'s end-of-session summary and `report`)
# ----------------------------------------------------------------------------


def _fit_ranking(
    item_ids: list[str],
    comparisons: list[dict],
    seed: int,
    bootstrap: int,
    *,
    fit_unseen: bool,
) -> tuple[list[dict], list[str]]:
    """rows: {"rank", "id", "utility", "rank_ci", "n_comparisons"} for the fitted
    items; zero_ids is always the ids with zero comparisons, which are excluded
    from the fit when fit_unseen=False and fitted at the prior when True."""
    cmps_count = {k: 0 for k in item_ids}
    for c in comparisons:
        cmps_count[c["a"]] += 1
        cmps_count[c["b"]] += 1

    zero_ids = [k for k in item_ids if cmps_count[k] == 0]
    fit_ids = item_ids if fit_unseen else [k for k in item_ids if cmps_count[k] > 0]
    if not fit_ids:
        return [], zero_ids

    strengths = fit_bt(fit_ids, comparisons)

    rng = random.Random(seed)
    ci = bootstrap_ranks(fit_ids, comparisons, bootstrap, rng)

    order = sorted(range(len(fit_ids)), key=lambda i: (-strengths[fit_ids[i]], i))

    ranking = []
    for rank, i in enumerate(order, start=1):
        kid = fit_ids[i]
        rc = ci.get(kid)
        ranking.append(
            {
                "rank": rank,
                "id": kid,
                "utility": round(strengths[kid], 4),
                "rank_ci": [rc[0], rc[1]] if rc else None,
                "n_comparisons": cmps_count[kid],
            }
        )

    return ranking, zero_ids


def _build_ranking(
    state: dict, bootstrap: int, *, fit_unseen: bool
) -> tuple[list[dict], list[str]]:
    ranking, zero_keys = _fit_ranking(
        CANON_IDS, state["comparisons"], state["seed"], bootstrap, fit_unseen=fit_unseen
    )
    for row in ranking:
        key = KEYS_BY_ID[row["id"]]
        row["row"] = key.row
        row["col"] = key.col
        row["mirror_col"] = mirror_col(key.col)
    return ranking, zero_keys


def _build_bigram_ranking(
    state: dict, bootstrap: int, *, fit_unseen: bool
) -> tuple[list[dict], list[str]]:
    ranking, zero_ids = _fit_ranking(
        bigram_universe(state),
        state["comparisons"],
        state["seed"],
        bootstrap,
        fit_unseen=fit_unseen,
    )
    for row in ranking:
        first, second = split_bigram(row["id"])
        row["crosshand"] = is_crosshand(row["id"])
        for label, kid in (("from", first), ("to", second)):
            key = KEYS_BY_ID[kid]
            row[label] = {
                "id": kid,
                "row": key.row,
                "col": key.col,
                "mirror_col": mirror_col(key.col),
            }
    return ranking, zero_ids


def _print_anchor_summary(
    state: dict,
    anchors: list[str],
    universe: list[str],
    bootstrap: int,
    *,
    fit_unseen: bool,
) -> None:
    ranking, _ = _fit_ranking(
        universe, state["comparisons"], state["seed"], bootstrap, fit_unseen=fit_unseen
    )
    by_id = {row["id"]: row for row in ranking}
    missing = [a for a in anchors if a not in by_id]
    rows = sorted((by_id[a] for a in anchors if a in by_id), key=lambda r: r["rank"])
    print()
    print(f"{len(rows)} anchored items, ranked against {len(ranking)} of {len(universe)} items:")
    print(f"{'Rank':>4}  {'Item':<14}  {'Utility':>7}  {'Rank 95% CI':>12}   {'Cmps':>4}")
    for row in rows:
        ci = f"{row['rank_ci'][0]}-{row['rank_ci'][1]}" if row["rank_ci"] else "-"
        print(
            f"{row['rank']:>4}  {row['id']:<14}  {row['utility']:>+7.2f}  "
            f"{ci:>12}   {row['n_comparisons']:>4}"
        )
    if fit_unseen:
        zero = [row["id"] for row in rows if row["n_comparisons"] == 0]
        if zero:
            print(
                "anchored items with zero comparisons (fitted at the prior): " + ", ".join(zero),
                file=sys.stderr,
            )
    elif missing:
        print("anchored items excluded (zero comparisons): " + ", ".join(missing), file=sys.stderr)


def _print_report(state: dict, bootstrap: int, json_output: bool, *, fit_unseen: bool) -> None:
    comparisons = state["comparisons"]
    ranking, zero_keys = _build_ranking(state, bootstrap, fit_unseen=fit_unseen)

    if len(comparisons) < 2 * len(ranking):
        print(
            f"only {len(comparisons)} comparisons for {len(ranking)} mirror positions; "
            "ranking is weak - run 'kbrank ask' for more",
            file=sys.stderr,
        )
    if zero_keys:
        if fit_unseen:
            print(
                "positions with zero comparisons (fitted at the prior): " + ", ".join(zero_keys),
                file=sys.stderr,
            )
        else:
            print(
                f"excluded {len(zero_keys)} of {N_POS} positions with zero comparisons: "
                + ", ".join(zero_keys),
                file=sys.stderr,
            )

    if json_output:
        out = {
            "n_positions": N_POS,
            "n_ranked": len(ranking),
            "n_comparisons": len(comparisons),
            "grid": [ROWS, COLS],
            "symmetric": True,
            "fit_unseen": fit_unseen,
            "ranking": ranking,
        }
        print(json.dumps(out))
        return

    print(
        f"{N_POS} mirror positions (left/right halves treated as identical), "
        f"{len(comparisons)} comparisons, {len(ranking)} positions ranked"
    )
    print(
        "Utility values are log-strengths from a Bradley-Terry fit; "
        "only their ORDER is meaningful."
    )
    print()
    print(
        f"{'Rank':>4}  {'Key':<6}  {'Row':>3}  {'Cols':>6}   "
        f"{'Utility':>7}  {'Rank 95% CI':>12}   {'Cmps':>4}"
    )
    for row in ranking:
        ci_str = f"{row['rank_ci'][0]}-{row['rank_ci'][1]}" if row["rank_ci"] else "-"
        cols_str = f"{row['col']}|{row['mirror_col']}"
        print(
            f"{row['rank']:>4}  {row['id']:<6}  {row['row']:>3}  {cols_str:>6}   "
            f"{row['utility']:>+7.2f}  {ci_str:>12}   {row['n_comparisons']:>4}"
        )

    print()
    print("Rank map (1 = most preferred):")
    print("(mirrored: both halves share one rank)")
    if zero_keys and not fit_unseen:
        print("('-' = excluded: zero comparisons)")
    rank_by_key = {row["id"]: row["rank"] for row in ranking}
    print(
        "\n".join(
            _grid_lines(
                lambda key: str(rank_by_key.get(canonical_id(key.id), "-"))
            )
        )
    )


def _print_bigram_report(
    state: dict, bootstrap: int, json_output: bool, top: int, *, fit_unseen: bool
) -> None:
    comparisons = state["comparisons"]
    ranking, zero_ids = _build_bigram_ranking(state, bootstrap, fit_unseen=fit_unseen)
    universe = bigram_universe(state)
    n_items = len(universe)

    if len(comparisons) < 2 * len(ranking):
        print(
            f"only {len(comparisons)} comparisons for {len(ranking)} sequences; "
            "ranking is weak - run 'kbrank ask --mode bigrams' for more",
            file=sys.stderr,
        )
    if zero_ids:
        if fit_unseen:
            print(
                f"{len(zero_ids)} of {n_items} sequences have zero comparisons "
                "(fitted at the prior)",
                file=sys.stderr,
            )
        else:
            print(
                f"excluded {len(zero_ids)} of {n_items} sequences with zero comparisons",
                file=sys.stderr,
            )

    if json_output:
        out = {
            "mode": "bigrams",
            "n_items": n_items,
            "n_ranked": len(ranking),
            "n_comparisons": len(comparisons),
            "grid": [ROWS, COLS],
            "symmetric": True,
            "fit_unseen": fit_unseen,
            "crosshand": state["crosshand"],
            "ranking": ranking,
        }
        print(json.dumps(out))
        return

    scope = (
        f"{N_BIGRAMS} same-hand + {N_CROSSHAND_BIGRAMS} cross-hand"
        if state["crosshand"]
        else "same-hand"
    )
    print(
        f"{n_items} two-key sequences over {N_POS} mirror positions ({scope}, repeats included), "
        f"{len(comparisons)} comparisons, {len(ranking)} sequences ranked"
    )
    print(
        "Utility values are log-strengths from a Bradley-Terry fit; "
        "only their ORDER is meaningful."
    )
    print()
    print(
        f"{'Rank':>4}  {'Sequence':<14}  {'From':>12}  {'To':>12}   "
        f"{'Utility':>7}  {'Rank 95% CI':>12}   {'Cmps':>4}"
    )

    def _row_line(row: dict) -> str:
        ci_str = f"{row['rank_ci'][0]}-{row['rank_ci'][1]}" if row["rank_ci"] else "-"
        from_str = f"r{row['from']['row']} c{row['from']['col']}|{row['from']['mirror_col']}"
        to_str = f"r{row['to']['row']} c{row['to']['col']}|{row['to']['mirror_col']}"
        return (
            f"{row['rank']:>4}  {row['id']:<14}  {from_str:>12}  {to_str:>12}   "
            f"{row['utility']:>+7.2f}  {ci_str:>12}   {row['n_comparisons']:>4}"
        )

    if top > 0 and 2 * top < len(ranking):
        for row in ranking[:top]:
            print(_row_line(row))
        print(f"  ... {len(ranking) - 2 * top} sequences omitted (--top 0 for all) ...")
        for row in ranking[-top:]:
            print(_row_line(row))
    else:
        for row in ranking:
            print(_row_line(row))


def _print_bigram_compare(
    state: dict, bootstrap: int, specs: list[str], *, fit_unseen: bool
) -> None:
    left, right = (parse_bigram(s) for s in specs)
    universe = bigram_universe(state)
    for seq in (left, right):
        if seq not in universe:
            print(
                f"{seq} is a cross-hand sequence but this session has no cross-hand data; "
                "collect some with 'kbrank ask --mode bigrams --allow-crosshand'",
                file=sys.stderr,
            )
            sys.exit(2)
    ranking, _ = _build_bigram_ranking(state, bootstrap, fit_unseen=fit_unseen)
    by_id = {row["id"]: row for row in ranking}

    for seq in (left, right):
        if seq not in by_id:
            print(
                f"{seq} has no recorded comparisons; pass --fit-unseen to score it at the prior",
                file=sys.stderr,
            )
            sys.exit(2)

    if fit_unseen:
        for seq in (left, right):
            if by_id[seq]["n_comparisons"] == 0:
                print(
                    f"warning: {seq} has no recorded comparisons; its utility is the prior",
                    file=sys.stderr,
                )

    def _stat_line(seq: str) -> str:
        row = by_id[seq]
        return (
            f"{seq:<10}  rank {row['rank']:>3}/{len(ranking)}   "
            f"utility {row['utility']:>+6.2f}   cmps {row['n_comparisons']:>4}"
        )

    print(_stat_line(left))
    print(_stat_line(right))
    print()

    if left == right:
        print(f"{left} compared with itself")
        return

    u_left, u_right = by_id[left]["utility"], by_id[right]["utility"]
    gap = u_left - u_right
    if abs(gap) < 1e-9:
        print(f"{left} and {right} are indistinguishable (gap +0.00, P = 0.50)")
        return

    p = 1.0 / (1.0 + math.exp(-gap))
    better = left if gap > 0 else right
    print(f"{better} is better: utility gap {abs(gap):+.2f}, P(prefer first) = {p:.2f}")


# ----------------------------------------------------------------------------
# `ask` subcommand
# ----------------------------------------------------------------------------

STATE_DEFAULTS = {"keys": "./kbrank-session.json", "bigrams": "./kbrank-bigram-session.json"}
BOOTSTRAP_DEFAULTS = {"keys": 200, "bigrams": 50}
BIGRAM_SESSION_QUESTIONS = 50  # default questions added per bigrams `ask` run


@dataclasses.dataclass
class AskMode:
    next_pair: Callable[[dict], tuple[str, str]]  # state -> unswapped (a, b)
    swap: Callable[[dict], bool]                  # state -> present in swapped order?
    render: Callable[[str, str], str]              # (id1, id2) -> printable block
    question: str
    options: Callable[[str, str], str]             # (id1, id2) -> the two option lines


def _anchored_pair_count(n_anchors: int, n_universe: int) -> int:
    """Unordered pairs over `n_universe` items that include one of `n_anchors`."""
    rest = n_universe - n_anchors
    return n_universe * (n_universe - 1) // 2 - rest * (rest - 1) // 2


def _history_swap(state: dict) -> bool:
    return random.Random(f"{state['seed']}:{len(state['comparisons'])}").random() < 0.5


def _run_ask_session(
    state: dict, path: str, target: int, mode: AskMode, undo_floor: int = 0
) -> int:
    current: tuple[str, str] | None = None

    while len(state["comparisons"]) < target:
        if current is None:
            current = mode.next_pair(state)
        a_id, b_id = current
        swap = mode.swap(state)
        choice1_id, choice2_id = (b_id, a_id) if swap else (a_id, b_id)

        count = len(state["comparisons"])
        print(f"[ {count + 1}/{target} ]  total recorded: {count}")
        print()
        print(mode.render(choice1_id, choice2_id))
        print()
        print(mode.question)
        print(mode.options(choice1_id, choice2_id))
        print("  = equal   u undo   q save & quit")

        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            save_state(path, state)
            print(f"saved {len(state['comparisons'])} comparisons to {path}")
            return 0

        if raw in ("1", "2"):
            result = "a" if raw == "1" else "b"
            state["comparisons"].append(
                {"a": choice1_id, "b": choice2_id, "result": result, "t": time.time()}
            )
            save_state(path, state)
            current = None
        elif raw in ("=", "e"):
            state["comparisons"].append(
                {"a": choice1_id, "b": choice2_id, "result": "tie", "t": time.time()}
            )
            save_state(path, state)
            current = None
        elif raw == "u":
            if len(state["comparisons"]) > undo_floor:
                popped = state["comparisons"].pop()
                save_state(path, state)
                current = (popped["a"], popped["b"])
            else:
                print("nothing to undo")
        elif raw == "q":
            save_state(path, state)
            print(f"saved {len(state['comparisons'])} comparisons to {path}")
            return 0
        else:
            print("unrecognised input")

    save_state(path, state)
    print(f"saved {len(state['comparisons'])} comparisons to {path}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    path = args.state or STATE_DEFAULTS[args.mode]
    if args.allow_crosshand and args.mode == "keys":
        print("--allow-crosshand is only available with --mode bigrams", file=sys.stderr)
        return 2
    if args.color and args.mode == "keys":
        print("--color is only available with --mode bigrams", file=sys.stderr)
        return 2
    only = (
        parse_only(args.only, args.mode, allow_crosshand=args.allow_crosshand)
        if args.only
        else None
    )
    if args.bias_alpha < 0:
        print("--bias-alpha must be >= 0", file=sys.stderr)
        return 2
    if args.bias_alpha != SOFT_BIAS_ALPHA and not args.soft_bias:
        print("note: --bias-alpha ignored without --soft-bias", file=sys.stderr)
    if args.soft_bias:
        make_sampler = functools.partial(SoftBalancedPairSampler, alpha=args.bias_alpha)
    elif args.no_bias:
        make_sampler = UniformPairSampler
    else:
        make_sampler = BalancedPairSampler

    if args.mode == "keys":
        state = load_state(path, seed=args.seed)
        sampler = make_sampler(CANON_IDS, state["seed"], anchors=only)
        next_pair = lambda s: sampler.next_pair(s["comparisons"])
        swap = _history_swap

        def _options(id1: str, id2: str) -> str:
            k1, k2 = KEYS_BY_ID[id1], KEYS_BY_ID[id2]
            return (
                f"  1) (A) row {k1.row}, cols {k1.col}|{mirror_col(k1.col)}      "
                f"2) [B] row {k2.row}, cols {k2.col}|{mirror_col(k2.col)}"
            )

        mode = AskMode(
            next_pair=next_pair,
            swap=swap,
            render=lambda id1, id2: (
                render_grid(id1, id2)
                + "\nMirror cells (a)/[b] are the same position on the other hand."
            ),
            question="Which key position is more comfortable to type?",
            options=_options,
        )
        if only is None:
            target = args.n if args.n is not None else N_PAIRS
            _run_ask_session(state, path, target, mode)
            if len(state["comparisons"]) >= target:
                _print_report(state, bootstrap=200, json_output=False, fit_unseen=False)
        else:
            n = args.n if args.n is not None else _anchored_pair_count(len(only), N_POS)
            if n <= 0:
                print("-n must be positive", file=sys.stderr)
                return 2
            target = len(state["comparisons"]) + n
            print(f"--only: {len(only)} of {N_POS} items anchored, {n} pairs this run")
            _run_ask_session(state, path, target, mode, undo_floor=len(state["comparisons"]))
            if len(state["comparisons"]) >= target:
                _print_anchor_summary(
                    state, only, CANON_IDS, BOOTSTRAP_DEFAULTS[args.mode], fit_unseen=False
                )
        return 0

    state = load_bigram_state(path, seed=args.seed)
    if args.allow_crosshand:
        state["crosshand"] = True
    universe = bigram_universe(state)
    sampler = make_sampler(
        universe,
        state["seed"],
        anchors=only,
        compatible=lambda a, b: not (is_crosshand(a) and is_crosshand(b)),
    )
    mode = AskMode(
        next_pair=lambda s: sampler.next_pair(s["comparisons"]),
        swap=_history_swap,
        render=lambda id1, id2: render_bigram_pair(id1, id2, colorize=args.color),
        question="Which two-key sequence is more comfortable to type in order?",
        options=lambda id1, id2: (
            f"  1) A: {describe_bigram(id1)}\n  2) B: {describe_bigram(id2)}"
        ),
    )
    if only is None:
        target = (
            args.n if args.n is not None else len(state["comparisons"]) + BIGRAM_SESSION_QUESTIONS
        )
        _run_ask_session(state, path, target, mode)
        if len(state["comparisons"]) >= target:
            _print_bigram_report(state, bootstrap=50, json_output=False, top=20, fit_unseen=False)
    else:
        n = args.n if args.n is not None else min(
            BIGRAM_SESSION_QUESTIONS, _anchored_pair_count(len(only), len(universe))
        )
        if n <= 0:
            print("-n must be positive", file=sys.stderr)
            return 2
        target = len(state["comparisons"]) + n
        print(f"--only: {len(only)} of {len(universe)} items anchored, {n} pairs this run")
        _run_ask_session(state, path, target, mode, undo_floor=len(state["comparisons"]))
        if len(state["comparisons"]) >= target:
            _print_anchor_summary(
                state, only, universe, BOOTSTRAP_DEFAULTS[args.mode], fit_unseen=False
            )
    return 0


# ----------------------------------------------------------------------------
# `report` subcommand
# ----------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    path = args.state or STATE_DEFAULTS[args.mode]
    bootstrap = args.bootstrap if args.bootstrap is not None else BOOTSTRAP_DEFAULTS[args.mode]

    if args.mode == "keys":
        if args.compare:
            print("--compare is only available with --mode bigrams", file=sys.stderr)
            return 2
        state = load_state(path)
        if not state["comparisons"]:
            print(f"no comparisons recorded in {path}", file=sys.stderr)
            return 1
        _print_report(
            state, bootstrap=bootstrap, json_output=args.json, fit_unseen=args.fit_unseen
        )
        return 0

    state = load_bigram_state(path)
    if not state["comparisons"]:
        print(f"no comparisons recorded in {path}", file=sys.stderr)
        return 1
    if args.compare:
        _print_bigram_compare(
            state, bootstrap=0, specs=args.compare, fit_unseen=args.fit_unseen
        )
    else:
        _print_bigram_report(
            state,
            bootstrap=bootstrap,
            json_output=args.json,
            top=args.top,
            fit_unseen=args.fit_unseen,
        )
    return 0


# ----------------------------------------------------------------------------
# argparse wiring
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbrank")
    sub = parser.add_subparsers(dest="command")

    ask_p = sub.add_parser("ask", help="collect pairwise comfort comparisons")
    ask_p.add_argument(
        "--mode",
        choices=("keys", "bigrams"),
        default="keys",
        help="item type to compare: individual key positions or two-key sequences",
    )
    ask_p.add_argument(
        "--state",
        default=None,
        help="session JSON file to load/save (default: ./kbrank-session.json, "
             "or ./kbrank-bigram-session.json for --mode bigrams)",
    )
    ask_p.add_argument(
        "-n",
        type=int,
        default=None,
        help="number of comparisons to ask this run (default: fills out the full "
             "pair space, or a fixed batch size with --only)",
    )
    ask_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for pair sampling; only used the first time a state file "
             "is created, then persisted in it",
    )
    ask_p.add_argument(
        "--only",
        action="extend",
        nargs="+",
        default=None,
        metavar="SPEC",
        help="anchor prompting to these keys/sequences: every prompted pair includes "
             "one of them, the opponent ranges over the full space. 'r2c1' selects a "
             "key, 'r2c1-r2c2' a sequence (bigrams mode only). Whitespace-separated, "
             "flag repeatable. Prefer '-' over '>' to avoid shell redirection.",
    )
    sampler_g = ask_p.add_mutually_exclusive_group()
    sampler_g.add_argument(
        "--no-bias",
        action="store_true",
        help="draw each pair uniformly at random from the eligible pair space instead "
             "of always comparing the two least-compared items; pairs already asked are "
             "skipped until the space is exhausted. Works with --mode keys or bigrams, "
             "with or without --only. Off by default (balanced sampling).",
    )
    sampler_g.add_argument(
        "--soft-bias",
        action="store_true",
        help="draw each side at random with weight 1/(count+1)**alpha, so "
             "under-sampled items are strongly favoured without being forced and every "
             "item stays reachable; repeat pairs are suppressed, not forbidden. Sits "
             "between the default balanced sweep and --no-bias. Works with --mode keys "
             "or bigrams, with or without --only.",
    )
    ask_p.add_argument(
        "--bias-alpha",
        type=float,
        default=SOFT_BIAS_ALPHA,
        help=f"bias strength for --soft-bias (default {SOFT_BIAS_ALPHA}): 0 is uniform "
             "over items, larger values approach the balanced sweep. Ignored without "
             "--soft-bias.",
    )
    ask_p.add_argument(
        "--allow-crosshand",
        action="store_true",
        help="also sample cross-hand sequences (first key on one hand, second key on the "
             "other): 225 extra items, ranked in the same table. A cross-hand sequence is "
             "always paired against a same-hand one, never against another cross-hand "
             "sequence (that comparison is redundant with mode keys).",
    )
    ask_p.add_argument(
        "--color",
        action="store_true",
        help="colorize sequence-grid cells by hand (yellow = left, light blue = "
             "right), mode bigrams only; off by default",
    )
    ask_p.set_defaults(func=cmd_ask)

    report_p = sub.add_parser("report", help="print the recovered rank table")
    report_p.add_argument(
        "--mode",
        choices=("keys", "bigrams"),
        default="keys",
        help="item type to report on: individual key positions or two-key sequences",
    )
    report_p.add_argument(
        "--state",
        default=None,
        help="session JSON file to read (default: ./kbrank-session.json, "
             "or ./kbrank-bigram-session.json for --mode bigrams)",
    )
    report_p.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="bootstrap resamples for rank confidence intervals (default: 200 "
             "for keys, 50 for bigrams)",
    )
    report_p.add_argument(
        "--json", action="store_true", help="print the rank table as JSON instead of text"
    )
    report_p.add_argument(
        "--top", type=int, default=20, help="number of top-ranked items to print"
    )
    report_p.add_argument(
        "--fit-unseen",
        action="store_true",
        help="also fit items that appear in no comparison; they land on the prior "
             "(log-strength 0) and are ranked alongside the rest. Off by default: "
             "only items backed by real comparisons are ranked.",
    )
    report_p.add_argument(
        "--compare",
        nargs=2,
        metavar=("SEQ", "SEQ"),
        default=None,
        help="skip the rank table and print a head-to-head comparison of two "
             "items instead",
    )
    report_p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv[0].startswith("-"):
        argv = ["ask"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
