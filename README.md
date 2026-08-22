# Risk Orchestrator

**The correlation matrix points at the wrong five pods.**

Twenty long/short books built from real CRSP returns, 1993–2024. Five were
constructed to crowd into a shared basket. Five hold no name in common with
each other but load on the same factor. Ranked by return correlation, the
second group looks like the problem and the first is invisible.

The five that get missed cost **7×** more than an independent pod when one of
them is forced to liquidate.

---

## The result

```
                 return corr   after factors removed   book overlap
crowded pods        +0.138            +0.147              18.1%
factor pods         +0.764            +0.053               0.0%
independent         -0.003            +0.001               2.6%
```

The pods holding **nothing in common** correlate at 0.764. The pods sharing
nearly a fifth of their books correlate at 0.138 — a fifth as much. Strip the
shared factor exposure and the ordering reverses completely: the factor pairs
fall to 0.053 while the crowded pairs do not move at all, because what they
share was never a factor.

A risk desk ranking pairs by correlation calls in the factor pods, whose
exposure should be hedged centrally without anyone changing a position, and
never sees the crowded ones, who will each sell the same names into the same
bid.

Forced liquidation, averaged over every pod of each type:

```
  pod liquidated      mean cost to the other 19
  crowded                   -9.44m
  factor                    -3.56m
  independent               -1.27m
```

The detector picks its victim from **holdings alone, never from the labels**.
It picked a planted crowded pod.

The multiple is roughly 7× here and was 15× on a different random seed, because
the stress test runs in whichever era is most crowded and the mid-1990s eras
are similar but not identical. Treat the multiple as a magnitude, not a
measurement: what is stable across seeds is that crowded pods are the expensive
ones to lose, not the precise factor.

---

## Why simulate the pods and not the returns

The returns are real. Only the allocation rules are invented.

Simulating returns too would mean inventing a covariance structure, and
whatever you invent is wrong in ways that quietly decide the answer — too
clean, too stationary, no fat tails, no volatility clustering. The correlation
between two pods holding real stocks over real days is whatever it actually
was.

What is invented is who held what, which is the thing no outsider can observe
about a real fund anyway.

Nothing here is a backtest and no pod is a strategy. The pods exist to generate
a realistic joint structure across books, because that structure is what a risk
tool has to work on. Reading a Sharpe ratio out of this project would be
reading it wrong.

---

## Pipeline

| script | purpose |
|---|---|
| `pods1_simulate.py` | build 20 pods from `panel_daily.csv`, crowding planted as ground truth |
| `pods2_crowding.py` | residualise on factors, test every pair against a null, correct across 190 pairs |
| `pods3_exposure.py` | aggregate to fund level: net against gross, concentration against a chance floor |
| `pods4_unwind.py` | forced liquidation with swept impact, control over every pod |
| `pods5_export_web.py` | run the chain once, write `docs/data/orchestrator.json` |
| `docs/index.html` | the dashboard — renders the JSON, computes nothing |

`pods_truth.csv` is written by step 1 and read by nothing except the scoring
blocks, after every decision has been made.

---

## The three ways the detector was wrong first

Each version produced a plausible scorecard. Each was caught by running it on
real data, never by reading the code — my own test panel had a single clean
factor and passed all three.

**Principal components of the pod returns.** With twenty pods and five crowded,
the crowded cluster *is* a principal component. Removing the leading components
removes exactly the thing being detected. The scorecard showed it: crowded
pairs kept +0.13 residual correlation at k=1 and inverted to −0.24 at k=2, as
the second component absorbed the cluster.

**Principal components of the stock returns.** Fixed that, and missed a subtler
problem. The factor pods are dollar-neutral, long high-beta and short low-beta —
their exposure is a *spread*. PC1 of a stock universe is approximately the
market, and a dollar-neutral book is largely insulated from the market by
construction. Removing five components took factor pairs from 0.76 to only
0.44. The exposure was never in the components being removed.

**A full-sample regression on constructed factors.** Building `MKT` and a
high-minus-low beta spread explicitly was right, but fitting one regression
over 7,805 days was not. Pods redraw holdings and recompute volatility scaling
every 63 days, so a loading is a step function. A single coefficient fits the
average and leaves the variation as residual — variation common to all the
factor pods, because they rescale on the same grid. Fitting inside each block
tracks it, and is what a risk desk does anyway: a loading averaged over thirty
years describes no position anyone currently holds.

After the third fix: factor pairs 0.764 → 0.101 with the market and the beta
spread removed, and → 0.053 with three further components. Crowded pairs sit at
0.138 raw and 0.147 after all five, unmoved. Note where the drop happens:
removing the market alone takes factor pairs only to 0.573, and it is the beta
spread that does the work — direct evidence that a dollar-neutral book is
insulated from the market and the exposure was in the spread all along.

---

## Other errors worth recording

**The p-value floor, twice.** With 190 pairs, Benjamini-Hochberg's tightest
threshold is 0.05/190 = 0.00026, while a per-pair p from 500 draws cannot go
below 1/501 = 0.002. Nothing could ever survive correction, and the first run
flagged zero crowded pairs for that reason alone. Fixed by pooling the null
across pairs, which the script now checks and reports. The same error appears
again in the dashboard, where a column of identical p-values across nine
thresholds was the resolution of the null rather than nine findings.

**A null that was a no-op.** The concentration test permuted which pod held
which book. That is the right null for a pair statistic and mathematically
useless for a name one — reassigning whole books between pods leaves every
column sum untouched, so fund-level exposure per name cannot move. It printed a
null mean equal to the real value to four decimals, and p exactly 1.0000.

**A control that reproduced the treatment.** `--random-pod` liquidated one
randomly chosen pod. With five of twenty planted crowded, a single draw lands
on a crowded pod a quarter of the time — and on the first real run it did.
Every pod is now liquidated in turn and results grouped by type.

**A ratio through zero.** Damage was reported as crowded mean P&L divided by
independent mean P&L. Independent pods are near flat in an unwind, so the
denominator sat near zero and the ratio blew up to −71.7× with a meaningless
sign. Replaced with a difference.

**A silent fallback.** When the universe was too small to fill the factor pods'
books, the simulator quietly drew random names — turning a factor pod into an
independent one while the report went on asserting the factor pods shared no
holdings "by construction". It printed that sentence next to a measured overlap
of 5.9%. It now refuses and prints the arithmetic.

---

## What the numbers do and do not rest on

**Market impact cannot be estimated here.** CRSP daily returns carry no volume,
so capacity is proxied as a fraction of market capitalisation. Every dollar
figure in the unwind is therefore a scenario, not an estimate, and the
coefficient is swept rather than fitted. The *ratio* across pod types is close
to invariant to it, because it scales both groups together — that ratio is the
finding, and the sweep is printed so the invariance is visible rather than
claimed.

**Crowding has a chance floor.** Two books of B names drawn from U eligible
share about B²/U by luck. 45.0% of the fund's gross book sits in names two or
more pods hold, and 36.6% of that is what twenty managers drawing from one
universe produce by counting alone. The excess peaks at t=3 (+9.4pp) and decays
from there. A report quoting the headline share without its floor is reporting
arithmetic as risk.

**Net against gross is argued, not demonstrated.** The distinction is real:
net exposure measures market risk, one-way pressure measures liquidation risk,
and they can point opposite ways on the same name. But in a universe this wide
pods rarely take opposite sides, so this simulation shows the divergence weakly
— between 1 and 3 of the twelve most crowded names, depending on the era and
the seed. Stated here rather than overclaimed.

**Crowding is partly a function of universe breadth.** The worst eras cluster
in the mid-1990s, when fewer names were eligible and the same twenty books piled
into a smaller pool. This survives changing the random seed, so it is a property
of the market's breadth and not of a draw — and it is the uncomfortable version
of the problem, since the investable universe narrows precisely when everyone
needs to exit.

---

## Running it

Needs `panel_daily.csv` from a CRSP daily stock file. It is produced by
`step1_load_crsp_v2.py` in the Analog Engine repo:

```
python step1_load_crsp_v2.py --input YOURFILE.csv --start-year 1992 --top-n 500 --save-panel
```

Then, from the project root:

```
python scripts\pods1_simulate.py --n-pods 20
python scripts\pods2_crowding.py
python scripts\pods3_exposure.py
python scripts\pods4_unwind.py
python scripts\pods5_export_web.py
python -m http.server 8000 --directory docs
```

A universe of roughly 500 names is needed: five factor pods at 20 names per leg
require 100 distinct high-beta names and the same on the low-beta side. The
simulator refuses and prints the arithmetic if the universe is too small.

**CRSP is licensed.** `data/` is gitignored and must never be committed.
`docs/data/` holds aggregate statistics only.
