# kb-util-func-finder
A small terminal Python program to find your (non-unique) preferences/utility function for single keys (e.g. A > S) and key pair sequences (AW > AV). The tool assumes a 3x10 (three rows, ten columns) board, and don't assume a layout. It insteads asks something like: Do you prefer key (2,1) over key (2,2)? Another notation used: r2c1, r2c2, etc.

# Usage

Use `kbrank ask` for keys or bigrams survey. A Bradley-Terry model (paired-comparison model, fit via regularized MM/Zermelo algorithm) is fit on the comparison data you input, producing a mean-centered log-strengths as your preference/utility values. Fit values can be seen with `kbrank report`.

```bash
usage: kbrank ask [-h] [--mode {keys,bigrams}] [--state STATE] [-n N] [--seed SEED] [--only SPEC [SPEC ...]]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
                        item type to compare: individual key positions or two-key sequences
  --state STATE         session JSON file to load/save (default: ./kbrank-session.json, or ./kbrank-bigram-
                        session.json for --mode bigrams)
  -n N                  number of comparisons to ask this run (default: fills out the full pair space, or a fixed
                        batch size with --only)
  --seed SEED           RNG seed for pair sampling; only used the first time a state file is created, then persisted
                        in it
  --only SPEC [SPEC ...]
                        anchor prompting to these keys/sequences: every prompted pair includes one of them, the
                        opponent ranges over the full space. 'r2c1' selects a key, 'r2c1-r2c2' a sequence (bigrams
                        mode only). Whitespace-separated, flag repeatable. Prefer '-' over '>' to avoid shell
                        redirection.
```

```bash
usage: kbrank report [-h] [--mode {keys,bigrams}] [--state STATE] [--bootstrap BOOTSTRAP] [--json] [--top TOP] [--compare SEQ SEQ]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
                        item type to report on: individual key positions or two-key sequences
  --state STATE         session JSON file to read (default: ./kbrank-session.json, or ./kbrank-bigram-session.json for --mode bigrams)
  --bootstrap BOOTSTRAP
                        bootstrap resamples for rank confidence intervals (default: 200 for keys, 50 for bigrams)
  --json                print the rank table as JSON instead of text
  --top TOP             number of top-ranked items to print
  --compare SEQ SEQ     skip the rank table and print a head-to-head comparison of two items instead
```
