# Git Conventions

Rules for this repo. Keep commits small, messages clear, and history
useful to future-Sam. When in doubt, err toward smaller commits and
more informative messages.

## Commit message format

```
<type>: <subject>

<body>
```

Subject line only for trivial commits; add a body when the "why"
isn't self-evident from the diff.

### Subject line

- **Imperative mood.** "add kernel fit" not "added kernel fit" or
  "adds kernel fit". Read it as: "If applied, this commit will ___".
- **≤ 72 characters.** Ideally ≤ 50.
- **Lowercase first letter** after the type prefix.
- **No trailing period.**
- **Start with a type prefix** (see list below).

### Type prefixes

- `feat` — new analysis, script, module, or capability.
- `fix` — correct a bug or incorrect result.
- `docs` — documentation only (findings.md, simulation_pipeline.md,
  README, comments).
- `refactor` — code restructuring without changing behaviour.
- `perf` — performance change (speed, memory) with no behaviour change.
- `chore` — config, .gitignore, housekeeping, repo setup.
- `data` — adding/removing tracked data files, symlinks, or pointers
  in config.py.
- `wip` — deliberate work-in-progress snapshot, rebased or squashed
  before anything that matters.

If a commit fits multiple types, pick the one closest to the *primary*
intent. Don't stack prefixes.

### Body

Use a body whenever **one of these** applies:

- The change involves a physics judgment call (why this kernel, why
  this cut, why this file path).
- It corrects a previous finding or overturns an assumption. Cite the
  prior commit or `findings.md` section being updated.
- It's a partial step toward something larger — note what it doesn't
  yet do.
- The one-liner would leave future-Sam guessing at the motivation.

Body rules:

- Wrap at 72 columns.
- Blank line between subject and body.
- Explain **why**, not **what** — the diff shows what.
- Reference files with paths (e.g. `src/kernels.py:42`) when pinning
  to a specific location helps.

## Examples

Good:

```
feat: add mono-energetic truth filter for kernel studies

Filters events where the summed CaloShowerSim energy in the target
crystal falls within a window of the full / esc1 / esc2 peaks.
Output is the per-SiPM reco distribution for that subset, which is
the detector response kernel directly. Addresses the kernel-shape
systematic flagged in docs/findings.md (Task 32).
```

```
fix: correct ADC conversion in template builder

Was dividing by MEV_TO_ADC instead of multiplying; template peaks
landed at 6.13/16 = 0.38 MeV instead of 6.13 MeV. Caught when the
closure fit diverged on cry_0.
```

```
docs: record multi-Fill quirk in truth histograms
```

```
chore: add matplotlib rcParams helper
```

Bad (don't do these):

```
updated stuff                     # vague, no type, not imperative
Update config.py                  # what changed? why?
fix bug                           # which bug?
added new analysis                # past tense
WIP                               # empty
feat/fix: change the fit          # stacked prefix
```

## Commit granularity

- **One logical change per commit.** If you'd use "and" in the subject,
  split it: `feat: add kernel fit and clean up imports` → two commits.
- **Never mix formatting / reordering with behaviour changes.** Those
  should be separate `refactor` commits.
- **Don't commit broken code to main.** If a commit doesn't at least
  import cleanly, either squash it with a working one before pushing
  or clearly mark it `wip:` for rebase later.

## Branching and rebasing

- Solo project, main branch is fine for most work.
- Use a feature branch for multi-commit work that might get abandoned.
- Rebase before merging to keep history linear. No merge commits from
  feature branches.
- Never rewrite history on a branch someone else has pulled.

## What not to commit

Enforced by `.gitignore`:

- `*.root` — regeneratable from sim runs or public paths.
- `outputs/` — per-run artefacts.
- LaTeX build products (`*.aux`, `*.log`, ...).
- Editor / OS metadata (`.vscode/`, `*.swp`, ...).

Manually avoid:

- Passwords, tokens, `.env` files.
- Large binaries (> a few MB) — put them on a shared fs and point
  `config.py` at them.
- `__pycache__/` (handled but worth re-checking).

## When fixing a physics finding

If a commit overturns something in `findings.md` or a previous
conclusion, **update `findings.md` in the same commit** so the record
stays consistent. Body should cite:

- Which finding / task is being revised.
- What the old claim was.
- Why the new one is correct (one sentence of mechanism).
