# The demo pack — what is planted, and what is not

Six managers, twelve files. Built by `scripts/pods8_export_demo.py` from real
CRSP securities, so return history exists for every name and the browser can run
the same analyses as the precomputed section rather than a subset of them.

This file is the answer key. Read it after running the tool, not before, or the
exercise is pointless.

---

## The books

```
schen     $400m   long  MGM DFS URI KEY FCX UAL   short  ABBV NOC LNT KO DUK PEP
dkim      $250m   long  MGM DFS URI KEY SYF AMP   short  MO XEL EA BDX REGN PFE
mfarrell  $180m   long  CFG HAL COF FITB LRCX BA  short  LLY CMS MKC DG HSY WEC
awong     $300m   long  LH MDLZ ICE A ELV T       short  AEP BMY GILD PG JNJ MRK
rpatel    $220m   long  TSN GEN TJX ZTS ATO NDAQ  short  ED CL VZ WMT KMB CAG
lmoreau   $150m   long  FAST APD MKL LMT GWW PGR  short  NEM CHD HRL SJM K GIS
```

Every book is twelve equal-weighted positions, six long and six short,
dollar-neutral. NAVs differ so that the currency path is exercised — a $400m
book and a $150m book must not be weighted equally.

---

## What is planted

**One crowded pair.** `schen` and `dkim` share four of six long names — MGM, DFS,
URI, KEY — a Jaccard overlap of 50%. They also argue the same underlying bet.
Both position analysis and text analysis should find them.

**One latent manager.** `mfarrell` argues the same bet as `schen` and `dkim` and
holds **not one name in common with either**. He reaches the trade through
deposit franchises rather than card issuers, and through energy services and
semiconductor capital equipment rather than gaming and airlines. His thesis says
so explicitly.

This is the cell the whole text layer exists for. Every position-based test in
this project is structurally blind to it: book overlap is zero, so no overlap
test can reach it, and the AND-gate that governs the `crowded` label can never
fire. If thesis clustering finds him, it found something positions had not yet
revealed.

**Three independents.** `awong` on value migrating from drug manufacturers to
the diagnostics and administration layer, `rpatel` on asymmetric input-cost
deflation, `lmoreau` on the property and casualty pricing cycle plus maintenance
demand. Different drivers from each other and from the reflation trade.

**Beta separation, which is what makes it a test rather than a demonstration.**
The three sharing the driver carry long-leg betas of 1.52 to 1.73. The three
independents carry 0.68 to 1.01. The ranges are disjoint by 0.51.

That separation is enforced by `--beta-gap`, and an earlier version of the demo
did not have it. Independents were merely excluded by rank — everything outside
the top 60 and bottom 40 — so the first name in the independent pool sat one
rank below the last name in the driver pool and carried essentially the same
beta. The demo planted factor overlap it did not intend, and the dashboard then
reported that overlap as residual correlation between managers labelled
independent. The label was wrong, not the detector.

---

## What the tool found

Measured on 2,000 trading days of real CRSP returns, five factors removed with
loadings refitted every 63 days.

```
pair                book overlap   return corr   residual corr   reading
dkim · schen            50.0%        +0.881         +0.492       shared names and shared risk
dkim · mfarrell          0.0%        +0.830         +0.204       shared risk, no shared names
mfarrell · schen         0.0%        +0.847         +0.161       shared risk, no shared names
```

All three planted relationships recovered, and the two latent pairs correctly
separated from the crowded one — same risk, different remedy.

The thesis layer returned a single group containing `dkim`, `mfarrell` and
`schen` at high confidence, and grouped nobody else. Note what that required:
the model had to recognise one bet described three ways, in documents whose
shared vocabulary is *lower* than the highest unrelated pair in the pack. Content
words overlap 0.139 between `schen` and `mfarrell` against 0.154 between `awong`
and `lmoreau`. A model grouping on wording would fail this test.

---

## What is imperfect, stated because it is

**Two residual correlations are significant and unexplained.** `mfarrell · rpatel`
at +0.096 and `lmoreau · mfarrell` at +0.051 both clear the null and
Benjamini-Hochberg, and neither pair shares a driver, a name, or a beta band.
They are far below the planted pairs and their p-values are printed beside them,
but they are not noise the tool has ruled out — they are structure the
five-factor model has not captured, most likely industry or style exposure. That
is a limitation of the factor set, not of the demo.

**`rpatel` and `lmoreau` both hold packaged-food shorts** — CAG, CL, KMB against
CHD, HRL, SJM, K, GIS. No shared names, so every overlap test correctly reports
zero, but the two books are short the same sector. This was not designed; the
low-beta draw handed both managers staples. `lmoreau`'s thesis states its short
book is funding rather than a view, which is the truthful description of names
selected for beta rather than for a reason.

**The concentration ladder is marginal.** At t = 2 it reads 14.4% against a
9.2% chance floor, p = 0.044. A crowded pair is planted and known to be there,
and the test barely clears the conventional threshold. Six managers over 68
distinct names is not enough for the ladder to have power; the pair table is the
more reliable instrument at this size.

**An earlier version of `mfarrell.md`** argued from utility rate cases and
regulated capex and explicitly distanced itself from AI names. The model refused
to group it. That was a defensible reading and the demo file was the flaw, not
the tool. It has been rewritten to state the same bet plainly.

---

## Reproducing it

```
python -m http.server 8000 --directory docs
```

Open the page, scroll to **Run this on your own book**, add six manager rows and
load the twelve files. **Analyse positions** runs entirely in the browser.
**Then read what they wrote** sends the theses to Anthropic with a key you paste;
positions never leave the machine.

Both nulls on this page are seeded, so the numbers do not move between runs. The
thesis layer calls a language model and will not be identical run to run — the
group's wording varies while its membership does not, which is the property the
partition agreement metric measures and the reason group labels are never
compared across runs.

To rebuild the pack from a different window or with a wider beta gap:

```
python scripts\pods8_export_demo.py --n-days 2000 --beta-gap 0.25
```

The theses in this folder are written against these specific tickers and would
need rewriting for any other draw.

---

## PLANTED.csv

The machine-readable key lives outside this folder, one level up, so that
loading everything in `docs/examples/` into the upload panel cannot pick it up.
It has no `ticker` column and would fail to parse.

Nothing in the pipeline reads it. It exists so a reader can check the tool found
what was planted rather than taking anyone's word for it.
