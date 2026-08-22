# Risk Orchestrator — handoff

Upload this as the first message in a new chat.

---

## Setup

- **Folder:** `C:\Users\matth\Desktop\Risk Orchestrator`
- **Shell:** Windows PowerShell, `python` (not `python3`), backslash paths
- **Repo:** github.com/matthewadrian9-tech/Risk-Orchestrator
- **Live:** matthewadrian9-tech.github.io/Risk-Orchestrator (serves `docs/`)
- **Preview:** `python -m http.server 8000 --directory docs`
- **API key:** in `$env:ANTHROPIC_API_KEY`. Retrieve with
  `$env:ANTHROPIC_API_KEY | Set-Clipboard`. Never save it to a file — two keys
  have already been rotated for exactly that.

Explanations should be beginner-level with commands spelled out. Don't comment
on sleep or deadlines.

---

## What this project is

A crowding detector for a multi-manager fund. Twenty pods built from real CRSP
returns, with crowding planted as ground truth the detector never sees.

**The result:** ranked by return correlation, the five pods holding *nothing* in
common sit at the top (+0.764) and the five sharing 18% of their books sit near
the bottom (+0.138). A risk desk ranking on correlation calls in the wrong five.

```
                 return corr   after factors removed   book overlap
crowded pods        +0.138            +0.147              18.1%
factor pods         +0.764            +0.053               0.0%
independent         -0.003            +0.001               2.6%
```

Detector caught 10/10 crowded pairs, 0 factor pairs wrongly flagged.
Liquidating a crowded pod costs 4.6×–13.8× (median 6.3×) what liquidating an
independent one costs — reported as a range because a single era's figure does
not reproduce.

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
| `docs/index.html` | dashboard — renders the JSON, computes nothing |

`pods_truth.csv` is read only by scoring blocks, after every decision is made.

**Run order:** pods1 → pods2 → pods3 → pods4 → pods5, then pods6 → pods7 →
pods5 again to pick up the thesis layer.

---

## The AI layer (this is what makes it a valid competition entry)

The competition requires an AI-powered tool. `pods1`–`pods5` contain **no AI** —
they are regression, bootstrap and Benjamini-Hochberg. The AI is `pods6`/`pods7`
and the browser panel.

**Why an LLM is load-bearing here rather than decorative:** positions are the
last thing to converge. Five managers can reach one conclusion months before
their books look alike, and if they express it through disjoint names, every
position-based tool is structurally blind. That is a semantic problem over text
and no arithmetic on holdings recovers it.

Two prompts, in `pods7_thesis_risk.py` and duplicated verbatim in
`docs/index.html`:
1. **Extraction** — reduce each thesis to a structured claim (driver in plain
   words, direction, mechanism, catalysts with dates, falsifier, mandate check)
2. **Clustering** — see all claims at once, group managers whose bets the same
   event would prove right or wrong together

**Result:** recovered both planted driver groups (5 datacenter, 5 rate-cycle) at
**97.4% partition agreement** across runs, plus two accidental same-driver pairs
that were not planted.

---

## Errors caught, and by what — the strongest section

1. **Pod-based PCA removed the crowding itself.** With 20 pods and 5 crowded,
   the crowded cluster *is* a principal component. Crowded pairs kept +0.13 at
   k=1 and inverted to −0.24 at k=2 as PC2 absorbed them.
2. **Stock PCA missed the beta spread.** Factor pods are dollar-neutral long
   high-beta / short low-beta — a *spread*. PC1 is roughly the market, which a
   dollar-neutral book is insulated from. Five components took factor pairs
   from 0.76 to only 0.44.
3. **Full-sample regression couldn't track a step function.** Pods rebalance
   every 63 days so loadings are piecewise constant. Fitting per block fixed
   it: 0.764 → 0.053.
4. **P-value floor, twice.** 190 pairs → BH threshold 0.00026; 500 draws floor
   at 0.002. Nothing could ever survive. Fixed by pooling the null across pairs.
   Same error as the Analog Engine's error #10.
5. **A null that was a no-op.** Permuting which pod holds which book leaves
   every column sum untouched, so name-level exposure cannot move. Printed
   p = 1.0000 and a null mean equal to the real value to four decimals.
6. **A control that reproduced its treatment.** `--random-pod` drew one pod;
   with 5 of 20 crowded it landed on a crowded pod. Now every pod is
   liquidated in turn.
7. **A ratio through zero.** crowded ÷ independent P&L, with independent near
   flat → −71.7× with a meaningless sign. Replaced with a difference.
8. **A silent fallback.** When the universe was too small the simulator drew
   random names, turning factor pods into independent ones while the report
   asserted "by construction, exactly none" next to a measured 5.9%. Now
   refuses and prints the arithmetic.
9. **Two definitions of one thing.** `pods4` ranked eras by count of crowded
   names, `pods5` by share of gross book. Terminal and dashboard named
   different victims for the same test. One function now, imported by both.
10. **Coincidental naming cannot work.** First thesis design asked each isolated
    call to invent a slug that would match another call's slug. Agreed with
    itself on 8/20 pods; five pods on one bet produced five labels. Replaced
    with a clustering pass that *compares* rather than names.
11. **Prompting alone did not stop fabricated flags.** The mandate-drift prompt
    explicitly forbids speculative drift; the model produced three anyway,
    hedged with "implied", "cannot be confirmed". Now filtered in code, and the
    dropped flags are printed as dropped.
12. **Weights hid pod size.** Aggregating own-NAV weights treats a $500m book
    and a $50m book as equal. Exposure now carried in currency when nav or
    nominal is supplied; a red warning when it is not.

Every one was caught by running the thing, never by reading the code.

---

## The dashboard

`docs/index.html`, React via CDN, single file, no build step. Two halves:

**Static** (from `orchestrator.json`, precomputed): pod table, raw-vs-residual
scatter, concentration ladder vs chance, holdings grid, unwind with multi-era
range, net vs gross, thesis layer.

**Upload panel** (`Run this on your own book`): a table with one row per
manager — name field, positions `.csv`, thesis `.md`. "+ add manager" adds
rows. Two buttons:
- **Analyse positions** — free, entirely in-browser, nothing leaves the machine
- **Then read what they wrote** — sends theses to Anthropic with a key the user
  pastes; key lives in page memory only

Upload produces: manager summary, holdings grid, concentration ladder chart,
net-vs-gross scatter, overlap bars vs chance floor, exit-exposure ranking, and
after the AI step: driver groups, two-axis quadrant chart, pairs table,
catalyst calendar.

**Cannot be reproduced from positions alone** and the panel says so: Sharpe
(needs returns), residual correlation (needs returns), dollar unwind (needs
volatility and liquidity).

---

## Demo pack

`C:\Users\matth\Desktop\Risk Orchestrator\demo` — six managers, twelve files.

- **schen + dkim** — same bet, share VRT/ETN/SMCI, 50% overlap → crowded
- **mfarrell** — same bet, **zero shared names** → the latent case, the whole
  point of the text layer
- **awong** (Japan governance), **rpatel** (regional bank CRE),
  **lmoreau** (consumer trading-down) — unrelated

NAVs $150m–$400m so the currency path is exercised. Gross ~190–200%,
dollar-neutral.

**Known issue:** the first `mfarrell.md` argued from rate cases and regulated
utility capex and explicitly distanced itself from AI names. The model
correctly refused to group it — a defensible call, and the demo file was the
flaw. Rewritten to state the same bet plainly. **Not yet re-tested.**

---

## To do

1. **Re-test the demo pack** with the rewritten `mfarrell.md` — does he now
   group with schen and dkim at 0% book overlap?
2. **Commit the demo pack** to `docs/examples/` so judges can reproduce it
3. **Write `docs/examples/README.md`** stating what is planted, so nobody
   assumes the tool got lucky
4. **Update the main README** — still describes a five-script project with no
   thesis layer and no upload panel
5. Push everything

---

## Standing constraints

- **CRSP is licensed.** `/data/` and `/outputs/` are gitignored. `docs/data/`
  holds aggregate statistics only. `.gitignore` is anchored with leading
  slashes so `docs/data/` is not caught.
- **Never commit an API key.** Already happened once (caught before push).
  `.gitignore` now excludes `*.docx`, `*.key`, `.env`, `secrets.*`.
- **The model reasons only from the documents.** The extraction prompt forbids
  importing knowledge about real companies, so a flag can always be traced to a
  sentence a manager wrote.
- **Positions never leave the browser.** Thesis text does, and the page says so
  in red.
- **Model is `claude-sonnet-4-6`** in both scripts and the page. Do not change
  it — the Analog Engine's determinism figures were measured on it.

---

## Related project

The Analog Engine (github.com/matthewadrian9-tech/Analog-Engine) is the
competition submission — falsification-first pattern matcher, eleven documented
errors, headline null result. It has its own `HANDOFF.md`. The Risk
Orchestrator is a second project, aimed at Polymer specifically because it maps
to their pod-shop structure.
