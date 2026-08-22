# Risk Orchestrator

**The correlation matrix points at the wrong five pods.**

Twenty long/short books built from real CRSP returns, 1993–2024. Five were
constructed to crowd into a shared basket. Five hold no name in common with
each other but load on the same factor. Ranked by return correlation, the
second group looks like the problem and the first is invisible.

The five that get missed are the expensive ones to lose. Across the five most
crowded eras, liquidating one costs the rest of the fund between **4.6× and
13.8×** what liquidating an independent pod costs.

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
  crowded                  -15.45m
  independent               -3.38m
  factor                    -3.02m
```

The detector picks its victim from **holdings alone, never from the labels**.
It picked a planted crowded pod.

**The multiple is not a stable quantity**, and the project reports it as a
range for that reason. Run the same control in each of the five most crowded
eras and it lands at 4.6×, 5.0×, 6.3×, 7.4× and 13.8× — median 6.3×, and a
three-fold spread between the ends. A single era's figure
looks precise and does not reproduce in the next one. What survives every era
tested is the ordering: crowded pods are always the costly ones to lose, factor
pods the cheapest. Quoting one era's multiple as a headline would be quoting
noise to two significant figures.

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
| `pods6_theses.py` | write an investment thesis per pod from its real holdings and an assigned driver |
| `pods7_thesis_risk.py` | extract a structured claim from each thesis, group by shared cause, build the two-axis matrix and the catalyst calendar |
| `pods8_export_demo.py` | build a six-manager demo fund from real CRSP names and ship the returns those names need |
| `docs/index.html` | the dashboard — renders the JSON, and runs the same analyses on an uploaded book |

Run order: `pods1` → `pods2` → `pods3` → `pods4` → `pods5`, then `pods6` →
`pods7` → `pods5` again to pick up the thesis layer. `pods8` is independent and
only rebuilds the demo pack.

`pods_truth.csv` is written by step 1 and read by nothing except the scoring
blocks, after every decision has been made.

---

## The cell no position report can reach

Everything above reasons about positions, and positions are the last thing to
converge. Five managers can arrive at one conclusion months before their books
look alike, and if they express it through disjoint names, every test in this
project is structurally blind to them. Book overlap is zero, so no overlap test
can fire; the AND-gate that governs the `crowded` label can never trigger. That
is a semantic problem over text and no arithmetic on holdings recovers it.

So a second layer reads what the managers wrote. `pods6` generates a thesis per
pod from that pod's real holdings and an assigned macro driver — real manager
memos are confidential and no student has them, so the corpus is generated, and
that creates a trap worth naming: if a template writes the theses and a model
reads them back, the detector recovers the template rather than detecting
anything. Generation runs at high temperature from a driver and a holdings list,
never from another pod's phrasing, so pods briefed on one view choose their own
words. Recognising one bet described five ways is the actual task.

`pods7` then runs two passes. The first reduces each document to a structured
claim — driver in plain words, direction, mechanism, dated catalysts, falsifier,
mandate check. The second shows every claim at once and groups managers whose
bets one event would prove right or wrong together.

**Result: both planted driver groups recovered — five on a datacenter bet, five
on a rate-cycle bet — at 97.4% partition agreement across runs.** What is scored
is the partition, never the labels: run two may name a group differently and
still be right, but whether two pods sit together is a closed question.

**The same ten factor pairs are scored twice on this project, in opposite
directions, and both scores are correct.** `pods2` does not flag them and counts
that as zero false positives. `pods7` groups them and counts that as recovering
a planted group. Stated side by side without this paragraph it reads as scoring
both ways. It is not: those pods are *not crowded* — they hold nothing in common
and no forced exit puts them in the same bid, so a position test is right to
reject them — and they *are* making one bet, and will be wrong together on the
day the driver turns, so a text test is right to find them. A fund needs both
answers because the remedies differ. Crowding is solved by cutting a position.
A shared bet is not.

---

## Run it on your own book

The dashboard is not only a report. The lower half takes one row per manager —
a name, a positions `.csv`, and whatever they wrote — and runs the same
aggregation on any fund.

**Positions never leave the browser.** The parsing and the permutation null both
run locally on a static page with no server behind it. That is not a courtesy:
a position file is the most confidential document a fund has, and a tool that
posts one anywhere is a tool nobody can use. Thesis text does leave, to
Anthropic's API with a key the user pastes, and the page says so in red.

From positions alone it produces a manager summary, the holdings grid, the
concentration ladder against a permutation floor, net against gross, pairwise
overlap against a chance floor, and an exit-exposure ranking. Where the names
resolve against a returns bundle it adds per-manager Sharpe, the raw-against-
residual scatter, and a priced unwind. After the text step: driver groups, the
two-axis quadrant, a pairs table and a catalyst calendar.

**Coverage floor 80%.** A manager needs 80% of their book to resolve against the
bundle and at least two managers must clear it, or the whole section is replaced
by a note stating the coverage and refusing. A residual correlation computed on
40% of a book looks identical to one computed on all of it, so it is refused
rather than shown with a caveat nobody reads.

### The demo pack

`docs/examples/` holds six managers built by `pods8` from real CRSP securities.
One crowded pair sharing four of six long names; one manager on the same bet
holding **not one name in common with either**; three independents. The three
sharing the driver carry long-leg betas of 1.52–1.73 and the independents
0.68–1.01, ranges disjoint by 0.51, which is what makes it a test rather than a
demonstration.

`docs/examples/README.md` states exactly what is planted, including what is
imperfect about it. `PLANTED.csv` sits at the repository root, read by nothing,
so anyone can check the tool found what was planted rather than taking it on
trust.

**The returns bundle is not published.** `demo_returns.json` carries daily
returns for the 68 names the demo managers hold, and those come from CRSP, which
is licensed to the university rather than to me. Redistributing security-level
daily return series publicly is not clearly within those terms, and "not
clearly outside" is not the standard a licence deserves. The file is generated
locally and gitignored; `docs/data/README.md` says which three analyses are
absent without it and why. The page degrades and names the missing half rather
than showing a number that looks complete and is not.

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

**Two definitions of one thing.** The unwind script ranked eras by the count of
names held by three or more pods; the web exporter ranked them by the share of
gross book in those names. Both are defensible and they select different
quarters, so the terminal output and the dashboard named different victims and
different multiples for what was described as the same test. Caught by reading
the deployed page against the terminal. One definition now lives in one
function and both import it.

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

## Errors in the text layer

**Coincidental naming cannot work.** The first thesis design asked each document,
in isolation, to invent a canonical slug that would happen to match the slug a
different isolated call invented for a different document. That is exact-string
agreement over an unbounded vocabulary, and it failed as it had to: five pods on
one datacenter bet produced five different labels, and the label was stable
across repeat runs on 8 of 20 pods. Meanwhile `direction`, a closed choice
between two values, was stable on 20 of 20. Replaced with a pass that
*compares* rather than names.

**Prompting alone did not stop fabricated flags.** The mandate-drift prompt
explicitly forbids speculative drift. The model produced three anyway, hedged
with "implied" and "cannot be confirmed". Filtered in code now, and the dropped
flags are printed as dropped rather than silently discarded.

**Weights hid pod size.** Aggregating own-NAV weights treats a $500m book and a
$50m book as equal. Exposure is carried in currency wherever a NAV or a nominal
is supplied, with a red warning when it is not.

**A manager can be in two groups, and the code assumed otherwise.** Nothing in
the clustering prompt forbids overlapping membership and a manager genuinely can
be running two bets, but `assignment()` wrote one label per pod, so a second
group containing a pod silently overwrote the first. The group table rendered
the model's output directly while the pair logic read the collapsed dict — one
artifact answering the same question two ways, which is the era-definition bug
again. Membership is a set now, and two pods share a bet when their sets
intersect.

**Keyed by index, which breaks the moment anything is dropped.** Fixing the
above surfaced a second problem underneath it. The model sees only the pods with
a usable extraction, so its index 3 is the fourth *survivor*, not the fourth
pod. An index-keyed assignment would attribute a group to the wrong manager,
silently, and only when an extraction had failed. Keyed by name now.

**A failed extraction was still allowed to vote.** Pods with no usable claim
were sent to the clustering pass anyway, rendering as `driver: ? | mechanism: ?
| direction: ?`. Two failures become two identical rows, and a model asked to
group by shared cause has every reason to put them together — so a pair of parse
errors manufactures a crowding alert. A false positive produced by an error path
is the worst kind, because it arrives looking like a finding. Failed pods are
dropped and named, and the partition is reported as partial.

**Truncation was reported as malformed JSON.** `call_anthropic` discarded
`stop_reason`, so a reply that hit the token cap arrived cut off mid-object,
failed to parse, and was reported as the model being unable to follow a format
it had followed perfectly. The two need opposite responses — raise the cap, or
fix the prompt — and the message could not distinguish them. Same error the
Analog Engine recorded as its seventh.

---

## Errors found by running the same thing twice

These are the ones worth dwelling on. Every error above was caught by running
the pipeline once and reading what came out. The following four were invisible
to that, and only appeared when two runs of an *unchanged* input were compared
against each other — which is a harder discipline than it sounds, because
nothing about a single run looks wrong.

**A p-value that moved between page loads.** The concentration null drew fresh
random numbers on every render. Two consecutive loads of identical books
returned p = 0.030 and p = 0.060, one either side of the conventional threshold.
A dashboard that changes its verdict on reload is not reporting a measurement,
and a reader has no way to tell an unstable estimate from a real one. Seeded
now, and the draw count raised from 200 to 2,000 — the seed only makes the
number repeatable, and the draws are what shrink the error.

**A calendar that resolved to different years.** A memo saying "December"
carries its year on the letterhead, not in the sentence, so an extraction with
no reference date has to guess — and guessed differently across runs, once
placing catalysts in a January that fell before the books existed. The pairs and
the grouping were stable across exactly those runs; only the calendar drifted,
which is why it survived several reproducibility checks. Anchored to an as-of
date in both the page and `pods7`.

**The same defect again, three fixes later.** Having written the paragraph above
about verdicts that move, the fallback for "no returns data loaded" was set to
the system clock — so on the published site, where the returns bundle is
deliberately absent, the catalyst calendar moved forward every day the page sat
unopened. Identical documents, a different answer each morning. It was invisible
locally because the bundle was always present and the fallback never ran. Fixed
to a stated constant, and the page now prints which date it used and where that
date came from.

**Two prompts that had quietly diverged.** The extraction prompt in the browser
was documented as a verbatim copy of the one in `pods7`. It was a trimmed
version, missing the clause forbidding the model from importing knowledge about
real companies. Since the demo theses name real tickers, a grouping could have
come from the model recognising SQ or MGM rather than from reading the
manager's argument — which would make the headline result ticker recognition
wearing the costume of semantic inference. Caught by diffing the two files
rather than by running either.

---

## Errors in the page itself

**A threshold standing in for a test.** Residual correlations in the upload
panel were flagged at a fixed `|r| > 0.1` with no null at all — the p-value floor
error in a different costume, since a fixed cut cannot know how large a residual
correlation this many days of data throws off by chance. On the demo pack it
called two unrelated pairs crowded at +0.121 and +0.113 beside a genuine one at
+0.191. Replaced with the circular block bootstrap and Benjamini-Hochberg the
Python pipeline already used, pooled across pairs for the same reason.

**A negative correlation labelled as shared risk.** The null is two-sided, which
is right — a correlation far from zero in either direction is a real
relationship. The label is not symmetric. Two books that offset each other once
factors are stripped are each other's hedge, and telling a risk desk they share
risk points it at the opposite of what is happening. A pair at −0.127 was
reported as shared risk until the demo pack surfaced it.

**A chart that dropped whoever ran the smallest book.** The holdings grid took
the top fifteen names by dollar exposure, and exposure is a currency figure, so
the smallest fund contributes the smallest per-name figure and its names fall
off the bottom. On the demo pack the two smallest managers had no name in the
top fifteen and rendered as entirely blank rows — which reads as "holds nothing"
rather than "holds nothing large enough for this list".

**And the fix for it, undone eleven lines later.** The corrected list guaranteed
every manager a column; a second `.slice(0,16)` further down then trimmed
whichever manager had been appended last, restoring one blank row. Both pieces
of code did exactly what they said. They disagreed, and the page did what the
second one said. Caught by looking at the picture and asking why a row was
empty.

**Rank exclusion is not separation.** In the demo builder the independent
managers were drawn from "everything outside the top 60 and bottom 40 by beta",
so the first name in the independent pool sat one rank below the last name in
the driver pool and carried essentially the same beta. The demo planted factor
overlap it did not intend, and the dashboard then reported that overlap as
residual correlation between managers labelled independent. The label was wrong,
not the detector. `--beta-gap` now requires a stated distance in beta units and
refuses rather than falling back.

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
```

That is the whole quantitative chain and it contains no AI — regression,
bootstrap and Benjamini-Hochberg. The text layer is separate and needs an API
key in `ANTHROPIC_API_KEY`:

```
python scripts\pods6_theses.py
python scripts\pods7_thesis_risk.py
python scripts\pods5_export_web.py
```

`pods5` runs twice because the second pass picks up the thesis layer. To rebuild
the demo pack and its returns bundle:

```
python scripts\pods8_export_demo.py --n-days 2000 --beta-gap 0.25
```

Then serve the dashboard:

```
python -m http.server 8000 --directory docs
```

A universe of roughly 500 names is needed: five factor pods at 20 names per leg
require 100 distinct high-beta names and the same on the low-beta side. The
simulator refuses and prints the arithmetic if the universe is too small.

**CRSP is licensed.** `data/` and `outputs/` are gitignored and must never be
committed. `docs/data/` holds aggregate statistics only, and the one file that
would not have — the demo returns bundle — is gitignored for that reason and
explained in `docs/data/README.md`.

**Never commit an API key.** It has happened once here, in a `.docx`, caught
before the push and resolved by resetting the commit and rotating the key.
`.gitignore` now excludes `*.docx`, `*.key`, `.env` and `secrets.*`.

**A note on two numbers.** The table at the top of this file reports factor
pairs residualising to +0.053, which is with all five components removed. The
deployed dashboard shows +0.101, which is with the market and the beta spread
alone. Both are correct and they answer slightly different questions, but a
reader comparing the two without this sentence would reasonably conclude one of
them is wrong.
