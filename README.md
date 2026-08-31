# kb-util-func-finder
A small terminal Python program to find your preferences (utility function) for single keys and key pairs. 

# Usage

Use `kbrank ask` for keys or bigrams survey. A Bradley-Terry model (paired-comparison model, fit via regularized MM/Zermelo algorithm) is fit on the comparison data you input, producing a mean-centered log-strengths as your preference/utility values. Fit values can be seen with `kbrank report`.

```bash
usage: kbrank ask [-h] [--mode {keys,bigrams}] [--state STATE] [-n N] [--seed SEED] [--only SPEC [SPEC ...]]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
  --state STATE
  -n N
  --seed SEED
  --only SPEC [SPEC ...]
                        anchor prompting to these keys/sequences: every prompted pair includes one of them, the opponent ranges over the
                        full space. 'r2c1' selects a key, 'r2c1-r2c2' a sequence (bigrams mode only). Whitespace-separated, flag repeatable.
                        Prefer '-' over '>' to avoid shell redirection.
```

```bash
usage: kbrank report [-h] [--mode {keys,bigrams}] [--state STATE] [--bootstrap BOOTSTRAP] [--json] [--top TOP] [--compare SEQ SEQ]

options:
  -h, --help            show this help message and exit
  --mode {keys,bigrams}
  --state STATE
  --bootstrap BOOTSTRAP
  --json
  --top TOP
  --compare SEQ SEQ
```
