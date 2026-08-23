# Risk Orchestrator — handoff

Upload this as the first message in a new chat.

---

## Setup

- **Folder:** `C:\Users\matth\Desktop\Risk Orchestrator`
- **Shell:** Windows PowerShell, `python` (not `python3`), backslash paths
- **Repo:** github.com/matthewadrian9-tech/Risk-Orchestrator
- **Live:** matthewadrian9-tech.github.io/Risk-Orchestrator (serves `docs/`)
- **Preview:** `python -m http.server 8000 --directory docs`
- **API key:** in `$env:ANTHROPIC_API_KEY` — note the colon, `$envANTHROPIC_API_KEY`
  silently returns nothing. `setx` only applies to windows opened afterwards.
  Never save it to a file; two keys have been rotated for exactly that.

Explanations should be beginner-level with commands spelled out. Don't comment
on sleep or deadlines.

---

## What this project is

A crowding detector for a multi-manager fund, in two layers.

**The position layer** builds twenty long/short books from real CRSP returns,
1993–2024, with crowding planted as ground truth the detector never sees.
Ranked by return correlation, the five pods holding *nothing* in common sit at
the top (+0.764) and the five sharing 18% of their books sit near the bottom
(+0.138). A risk desk ranking on correlation calls in the wrong five.

```
                 return corr   after factors removed   book overlap
crowded pods        +0.138            +0.147              18.1%
factor pods         +0.764            +0.053               0.0%
independent         -0.003            +0.001               2.6%
```

Detector caught 10/10 crowded pairs, 0 factor pairs wrongly flagged.
Liquidating a crowded pod costs 4.6×–13.8× (median 6.3×) what liquidating an
independent one costs — a range because a single era's figure does not
reproduce.

**The text layer** reads what the managers wrote, because positions are the last
thing to converge. Recovered both planted driver groups at 97.4% partition
agreement across runs.

**The framing that ties them together:** the same ten factor pairs are scored
twice, in opposite directions, and both scores are right. `pods2` does not flag
them (0 false positives); `pods7` groups them (planted group recovered). They
are *not crowded* — no forced exit puts them in the same bid — and they *are*
one bet. A fund needs both answers because the remedies differ.

---

## Pipeline

| script | purpose |
|---|---|
| `pods1_simulate.py` | build 20 pods from `panel_daily.csv`, crowding planted |
| `pods2_crowding.py` | residualise on factors, test pairs, BH across 190 pairs |
| `pods3_exposure.py` | fund level: net vs gross, concentration vs chance floor |
| `pods4_unwind.py` | forced liquidation, swept impact, control over every pod |
| `pods5_export_web.py` | run the chain, write `docs/data/orchestrator.json` |
| `pods6_theses.py` | LLM writes a thesis per pod from its real holdings |
| `pods7_thesis_risk.py` | LLM extraction + clustering, two-axis matrix, calendar |
| `pods8_export_demo.py` | build the six-manager demo fund, ship its returns |
| `docs/index.html` | dashboard, and runs the same analyses on an uploaded book |

**Run order:** pods1 → pods2 → pods3 → pods4 → pods5, then pods6 → pods7 →
pods5 again to pick up the thesis layer. `pods8` is independent.

`pods_truth.csv` is read only by scoring blocks, after every decision is made.

---

## Current state — everything below is done and pushed

Latest commits: `e2303b9` (demo pack, seeded nulls, prompt parity), `c58c260`
(README rewrite).

- Demo pack rebuilt on CRSP names with an enforced beta gap, committed to
  `docs/examples/` with a README stating what is planted
- `PLANTED.csv` at repo root, outside `examples/` so it cannot be loaded by
  mistake
- `demo/` untracked and gitignored; `docs/examples/` is the single copy
- Main README rewritten: eight scripts, thesis layer, upload panel, demo pack,
  four sections of errors
- `docs/data/README.md` explains the withheld returns bundle

**Verified on the live site:** grid shows all six managers, catalyst calendar
resolves from 2024-12-31, ladder p reproducible across reloads at 0.044,
position-only degradation works without the returns bundle.

---

## Errors caught, and by what

Position layer: pod-PCA removed the crowding itself; stock-PCA missed the beta
spread; full-sample regression could not track a step function; the p-value
floor, twice; a null that was a no-op; a control that reproduced its treatment;
a ratio through zero; a silent fallback; two definitions of one era.

Text layer: coincidental naming cannot work; prompting alone did not stop
fabricated flags; weights hid pod size; multi-group membership collapsed to one
label; index-keyed assignment misaligns once a pod is dropped; failed
extractions were allowed to vote in clustering; `stop_reason` ignored so
truncation read as malformed JSON.

Page: a bare 0.1 threshold standing in for a null; a negative residual labelled
shared risk; the holdings grid dropped whoever ran the smallest book; and the
fix for that undone eleven lines later by a second `.slice`.

**Four were found only by running the same thing twice** — this is the strongest
section and it is new:
1. Ladder p moved between page loads (0.030 vs 0.060) — unseeded null
2. Catalyst calendar resolved to different years — no as-of anchor
3. The same defect again three fixes later — as-of fallback used the system
   clock, so the published site's calendar moved daily. Invisible locally
   because the bundle was always present
4. Browser and `pods7` prompts had quietly diverged — the browser was missing
   the clause forbidding outside company knowledge

Nothing about a single run of any of these looks wrong.

---

## Open items

1. **Cosmetic:** the overlap-bar label fix was made but never installed. Check
   with `Select-String -Path docs\index.html -Pattern "× floor" -Quiet` — if
   False, the current `docs/index.html` clips that label at the right edge.
2. **Unresolved, stated honestly in `docs/examples/README.md`:**
   `mfarrell·rpatel` at +0.096 and `lmoreau·mfarrell` at +0.051 are significant
   residual correlations between books with disjoint beta ranges and no shared
   driver. Most likely industry or style structure the five-factor set does not
   capture. A finding about the factor model, if anyone chases it.
3. **The ladder has little power at six managers** — t=2 lands at p=0.044 with a
   crowded pair planted and known. Stated as a limitation rather than fixed.
4. **Two numbers differ by design:** README reports factor pairs at +0.053 (five
   components), the dashboard shows +0.101 (market and beta spread only). Noted
   at the end of the README.

---

## Standing constraints

- **CRSP is licensed.** `/data/` and `/outputs/` gitignored. `docs/data/` holds
  aggregate statistics only, and `demo_returns.json` is gitignored for that
  reason — see `docs/data/README.md`. Regenerate with
  `python scripts\pods8_export_demo.py --n-days 2000 --beta-gap 0.25`
- **Never commit an API key.** Happened once in a `.docx`, caught before push.
  `.gitignore` excludes `*.docx`, `*.key`, `.env`, `secrets.*`
- **The model reasons only from the documents.** Both extraction prompts forbid
  importing knowledge about real companies. This was false in the browser until
  recently — if either prompt is edited, diff them
- **Positions never leave the browser.** Thesis text does; the page says so in red
- **Model is `claude-sonnet-4-6`** in scripts and page. Do not change it
- **Both nulls are seeded.** If a number moves between identical runs, that is a
  bug, not sampling error
- **The demo theses are written against specific tickers.** Rebuilding the pack
  with a different draw requires rewriting all six `.md` files

---

## Related project

The Analog Engine (github.com/matthewadrian9-tech/Analog-Engine) is the
competition submission — falsification-first pattern matcher, eleven documented
errors, headline null result. It has its own `HANDOFF.md`. The Risk
Orchestrator is a second project, aimed at Polymer specifically because it maps
to their pod-shop structure.
