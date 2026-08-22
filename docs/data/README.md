# docs/data

The dashboard reads its numbers from this folder. Everything here is computed by
the scripts in `scripts/` before the page loads; the browser draws and does not
calculate.

## What is here

`orchestrator.json` — aggregate statistics for the twenty simulated pods: pair
correlations, concentration ladders, unwind costs, the thesis layer. Summary
figures only. No security-level return series.

## What is deliberately missing

`demo_returns.json` is **not published in this repository.**

The upload panel can run three analyses that a position file alone cannot
support — a Sharpe ratio, a residual correlation with factors regressed out, and
a priced unwind — and all three need daily returns for every name in the book.
For the demo pack in `docs/examples/`, those returns would come from CRSP, which
is licensed to the university rather than to me. Redistributing security-level
daily return series on a public repository is not obviously within those terms,
and "not obviously outside" is not the standard a licence deserves.

So the file is generated locally and gitignored. This is a licensing decision,
not a technical one: the code that reads the bundle is present and working, and
the analyses run the moment the file exists.

## What the page does without it

It degrades and says so. The upload panel still runs, from positions alone:
manager summary, holdings grid, the concentration ladder against a permutation
floor, net-versus-gross, pairwise overlap against a chance floor, and the exit
exposure ranking. The thesis layer is unaffected, because it reads text rather
than returns.

The three return-dependent analyses do not appear, and the panel states which
ones are missing and why rather than showing a number that looks complete and is
not. Naming the absent half is more useful than quietly presenting the rest.

## Regenerating it

With WRDS access and `panel_daily.csv` built:

```
python scripts\pods8_export_demo.py --n-days 2000 --beta-gap 0.25
```

That writes `docs/data/demo_returns.json` here, along with the six position
files and their answer key. The dashboard picks the bundle up on the next load
with no other change.

Anyone without CRSP access can still use the upload panel on their own book —
the position analyses and the thesis layer need nothing from this folder.
