from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import sys
import time

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


def _grid_lines(cell_for_key) -> list[str]:
    """Shared 3x10 grid layout used by both the question grid and the rank map.

    Row-label column is 6 chars wide; each key cell is right-aligned in width
    4; an extra 2-space gutter separates c5 from c6 so the two hands read
    apart visually.
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
            parts.append(cell_for_key(key).rjust(4))
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
    n = [[0] * n_keys for _ in range(n_keys)]

    for c in comparisons:
        i, j = idx[c["a"]], idx[c["b"]]
        n[i][j] += 1
        n[j][i] += 1
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
            denom = 1.0 / (pi[i] + 1.0)
            row = n[i]
            for j in range(n_keys):
                if j == i:
                    continue
                nij = row[j]
                if nij:
                    denom += nij / (pi[i] + pi[j])
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


class PairSampler:
    """Uniform-without-replacement sampler over the 105 unordered mirror-position pairs.

    A fresh instance rebuilds and reshuffles the same seeded permutation every
    time, so `skip` (the count of already-recorded comparisons modulo 105)
    simply moves the read position forward without consuming randomness -
    that is what makes resume deterministic and reproducible.
    """

    def __init__(self, seed: int, skip: int = 0):
        self.seed = seed
        self.epoch = 0
        self.rng = random.Random(seed)
        self._pairs = list(itertools.combinations(CANON_IDS, 2))
        self.rng.shuffle(self._pairs)
        self.pos = skip

    def next_pair(self) -> tuple[str, str]:
        if self.pos >= len(self._pairs):
            self.epoch += 1
            self.rng = random.Random(self.seed + self.epoch)
            self._pairs = list(itertools.combinations(CANON_IDS, 2))
            self.rng.shuffle(self._pairs)
            self.pos = 0
            print(f"all {N_PAIRS} pairs asked; sampling with repeats", file=sys.stderr)
        pair = self._pairs[self.pos]
        self.pos += 1
        return pair

    def presentation_swap(self) -> bool:
        return self.rng.random() < 0.5


# ----------------------------------------------------------------------------
# Reporting (shared by `ask`'s end-of-session summary and `report`)
# ----------------------------------------------------------------------------


def _build_ranking(state: dict, bootstrap: int) -> tuple[list[dict], list[str]]:
    comparisons = state["comparisons"]
    strengths = fit_bt(CANON_IDS, comparisons)

    cmps_count = {k: 0 for k in CANON_IDS}
    for c in comparisons:
        cmps_count[c["a"]] += 1
        cmps_count[c["b"]] += 1

    rng = random.Random(state["seed"])
    ci = bootstrap_ranks(CANON_IDS, comparisons, bootstrap, rng)

    order = sorted(range(len(CANON_IDS)), key=lambda i: (-strengths[CANON_IDS[i]], i))

    ranking = []
    for rank, i in enumerate(order, start=1):
        kid = CANON_IDS[i]
        key = KEYS_BY_ID[kid]
        rc = ci.get(kid)
        ranking.append(
            {
                "rank": rank,
                "id": kid,
                "row": key.row,
                "col": key.col,
                "mirror_col": mirror_col(key.col),
                "utility": round(strengths[kid], 4),
                "rank_ci": [rc[0], rc[1]] if rc else None,
                "n_comparisons": cmps_count[kid],
            }
        )

    zero_keys = [k for k in CANON_IDS if cmps_count[k] == 0]
    return ranking, zero_keys


def _print_report(state: dict, bootstrap: int, json_output: bool) -> None:
    comparisons = state["comparisons"]
    ranking, zero_keys = _build_ranking(state, bootstrap)

    if len(comparisons) < 2 * N_POS:
        print(
            f"only {len(comparisons)} comparisons for {N_POS} mirror positions; "
            "ranking is weak - run 'kbrank ask' for more",
            file=sys.stderr,
        )
    if zero_keys:
        print(
            "positions with zero comparisons: " + ", ".join(zero_keys),
            file=sys.stderr,
        )

    if json_output:
        out = {
            "n_positions": N_POS,
            "n_comparisons": len(comparisons),
            "grid": [ROWS, COLS],
            "symmetric": True,
            "ranking": ranking,
        }
        print(json.dumps(out))
        return

    print(f"{N_POS} mirror positions (left/right halves treated as identical), {len(comparisons)} comparisons")
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
    rank_by_key = {row["id"]: row["rank"] for row in ranking}
    print("\n".join(_grid_lines(lambda key: str(rank_by_key[canonical_id(key.id)]))))


# ----------------------------------------------------------------------------
# `ask` subcommand
# ----------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    state = load_state(args.state, seed=args.seed)
    skip = len(state["comparisons"]) % N_PAIRS
    sampler = PairSampler(state["seed"], skip)
    target = args.n

    current: tuple[str, str] | None = None

    while len(state["comparisons"]) < target:
        if current is None:
            current = sampler.next_pair()
        a_id, b_id = current
        swap = sampler.presentation_swap()
        choice1_id, choice2_id = (b_id, a_id) if swap else (a_id, b_id)

        count = len(state["comparisons"])
        print(f"[ {count + 1}/{target} ]  total recorded: {count}")
        print()
        print(render_grid(choice1_id, choice2_id))
        print("Mirror cells (a)/[b] are the same position on the other hand.")
        print()
        print("Which key position is more comfortable to type?")
        k1, k2 = KEYS_BY_ID[choice1_id], KEYS_BY_ID[choice2_id]
        print(
            f"  1) (A) row {k1.row}, cols {k1.col}|{mirror_col(k1.col)}      "
            f"2) [B] row {k2.row}, cols {k2.col}|{mirror_col(k2.col)}"
        )
        print("  = equal   u undo   q save & quit")

        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            save_state(args.state, state)
            print(f"saved {len(state['comparisons'])} comparisons to {args.state}")
            return 0

        if raw in ("1", "2"):
            result = "a" if raw == "1" else "b"
            state["comparisons"].append(
                {"a": choice1_id, "b": choice2_id, "result": result, "t": time.time()}
            )
            save_state(args.state, state)
            current = None
        elif raw in ("=", "e"):
            state["comparisons"].append(
                {"a": choice1_id, "b": choice2_id, "result": "tie", "t": time.time()}
            )
            save_state(args.state, state)
            current = None
        elif raw == "u":
            if state["comparisons"]:
                popped = state["comparisons"].pop()
                save_state(args.state, state)
                current = (popped["a"], popped["b"])
            else:
                print("nothing to undo")
        elif raw == "q":
            save_state(args.state, state)
            print(f"saved {len(state['comparisons'])} comparisons to {args.state}")
            return 0
        else:
            print("unrecognised input")

    save_state(args.state, state)
    print(f"saved {len(state['comparisons'])} comparisons to {args.state}")
    _print_report(state, bootstrap=200, json_output=False)
    return 0


# ----------------------------------------------------------------------------
# `report` subcommand
# ----------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    if not state["comparisons"]:
        print(f"no comparisons recorded in {args.state}", file=sys.stderr)
        return 1
    _print_report(state, bootstrap=args.bootstrap, json_output=args.json)
    return 0


# ----------------------------------------------------------------------------
# argparse wiring
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbrank")
    sub = parser.add_subparsers(dest="command")

    ask_p = sub.add_parser("ask", help="collect pairwise comfort comparisons")
    ask_p.add_argument("--state", default="./kbrank-session.json")
    ask_p.add_argument("-n", type=int, default=N_PAIRS)
    ask_p.add_argument("--seed", type=int, default=None)
    ask_p.set_defaults(func=cmd_ask)

    report_p = sub.add_parser("report", help="print the recovered rank table")
    report_p.add_argument("--state", default="./kbrank-session.json")
    report_p.add_argument("--bootstrap", type=int, default=200)
    report_p.add_argument("--json", action="store_true")
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
