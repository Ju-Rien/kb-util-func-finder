# kb-util-func-finder
A small terminal Python program to find your (non-unique) preferences/utility function for single keys (e.g. A > S) and key pair sequences (AW > AV). The tool assumes a 3x10 (three rows, ten columns) board, and don't assume a layout. It insteads asks something like: Do you prefer key (2,1) over key (2,2)? Another notation used: r2c1, r2c2, etc.

# Usage

Use `kbrank ask` for keys or bigrams survey. A Bradley-Terry model (paired-comparison model, fit via regularized MM/Zermelo algorithm) is fit on the comparison data you input, producing a mean-centered log-strengths as your preference/utility values. Fit values can be seen with `kbrank report`.

```bash
usage: kbrank ask [-h] [--mode {keys,bigrams}] [--state STATE] [-n N]
                  [--seed SEED] [--only SPEC [SPEC ...]] [--no-bias |
                  --soft-bias] [--bias-alpha BIAS_ALPHA] [--allow-crosshand]
                  [--color]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
                        item type to compare: individual key positions or two-
                        key sequences
  --state STATE         session JSON file to load/save (default: ./kbrank-
                        session.json, or ./kbrank-bigram-session.json for
                        --mode bigrams)
  -n N                  number of comparisons to ask this run (default: fills
                        out the full pair space, or a fixed batch size with
                        --only)
  --seed SEED           RNG seed for pair sampling; only used the first time a
                        state file is created, then persisted in it
  --only SPEC [SPEC ...]
                        anchor prompting to these keys/sequences: every
                        prompted pair includes one of them, the opponent
                        ranges over the full space. 'r2c1' selects a key,
                        'r2c1-r2c2' a sequence (bigrams mode only).
                        Whitespace-separated, flag repeatable. Prefer '-' over
                        '>' to avoid shell redirection.
  --no-bias             draw each pair uniformly at random from the eligible
                        pair space instead of always comparing the two least-
                        compared items; pairs already asked are skipped until
                        the space is exhausted. Works with --mode keys or
                        bigrams, with or without --only. Off by default
                        (balanced sampling).
  --soft-bias           draw each side at random with weight
                        1/(count+1)**alpha, so under-sampled items are
                        strongly favoured without being forced and every item
                        stays reachable; repeat pairs are suppressed, not
                        forbidden. Sits between the default balanced sweep and
                        --no-bias. Works with --mode keys or bigrams, with or
                        without --only.
  --bias-alpha BIAS_ALPHA
                        bias strength for --soft-bias (default 2.0): 0 is
                        uniform over items, larger values approach the
                        balanced sweep. Ignored without --soft-bias.
  --allow-crosshand     also sample cross-hand sequences (first key on one
                        hand, second key on the other): 225 extra items,
                        ranked in the same table. A cross-hand sequence is
                        always paired against a same-hand one, never against
                        another cross-hand sequence (that comparison is
                        redundant with mode keys).
  --color               colorize sequence-grid cells by hand (yellow = left,
                        light blue = right), mode bigrams only; off by default
```

```bash
usage: kbrank report [-h] [--mode {keys,bigrams}] [--state STATE]
                     [--bootstrap BOOTSTRAP] [--json] [--top TOP]
                     [--fit-unseen] [--compare SEQ SEQ]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
                        item type to report on: individual key positions or
                        two-key sequences
  --state STATE         session JSON file to read (default: ./kbrank-
                        session.json, or ./kbrank-bigram-session.json for
                        --mode bigrams)
  --bootstrap BOOTSTRAP
                        bootstrap resamples for rank confidence intervals
                        (default: 200 for keys, 50 for bigrams)
  --json                print the rank table as JSON instead of text
  --top TOP             number of top-ranked items to print
  --fit-unseen          also fit items that appear in no comparison; they land
                        on the prior (log-strength 0) and are ranked alongside
                        the rest. Off by default: only items backed by real
                        comparisons are ranked.
  --compare SEQ SEQ     skip the rank table and print a head-to-head
                        comparison of two items instead
```
