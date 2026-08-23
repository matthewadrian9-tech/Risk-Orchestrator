"""
pods9_var.py
------------
Value at Risk and Expected Shortfall for the fund, decomposed by pod type.

WHY A RISK TOOL THAT ALREADY HAS A CROWDING TEST NEEDS THIS

  VaR is the number a risk committee actually reads. Every fund produces one
  daily. So the question worth asking is not "what is the fund's VaR" - which
  is arithmetic anyone can do - but whether VaR, computed correctly and
  decomposed honestly, points at the same pods the crowding test points at.

  The prediction, stated before running it: it does not, and it fails in the
  same direction the correlation matrix fails.

  VaR is driven by return covariance. The factor pods correlate at +0.764, so
  they move the fund's aggregate return more than any other group and should
  dominate the VaR decomposition. The crowded pods correlate at only +0.138 -
  they contribute little to the variance of the sum, and should look harmless.

  But the crowded pods are the expensive ones to lose: forcing one out costs
  the rest of the fund 4.6x-13.8x what forcing out an independent one costs,
  because five books holding one basket all sell into the same bid.

  If that prediction holds, this project shows the same blind spot twice, in
  two different instruments, one of which is the industry standard. If it does
  NOT hold - if VaR does flag the crowded pods - that is a genuine limit on the
  argument and belongs in the write-up rather than out of it.

WHAT IS COMPUTED

  Historical VaR and ES, not parametric. A normal distribution understates the
  tail of equity returns, and the whole point of using real CRSP returns rather
  than simulated ones is to keep the tail that actually happened. VaR at 95% is
  the 5th percentile of the daily fund return; ES is the mean of everything at
  or below it.

  Decomposition uses component VaR, the standard identity:

      component_i = w_i * cov(r_i, r_fund) / var(r_fund) * VaR_fund

  These sum to the fund VaR exactly, which is the property that makes them
  attributable. Note what drives it: covariance with the fund, not the pod's
  own volatility. A pod can be individually volatile and contribute almost
  nothing if it moves independently.

  Both are reported on the RESIDUAL series too - after factor exposure is
  regressed out - because that separates "this pod contributes to VaR because
  the whole fund is long the market" from "this pod contributes something of
  its own".

THE NULL, AND WHY EVEN VAR NEEDS ONE

  A pod type's share of VaR has to be compared against its share of the book.
  Five pods out of twenty carrying 25% of VaR is not a finding, it is counting.
  What matters is the ratio of VaR share to capital share.

Run:
  python scripts\\pods9_var.py
  python scripts\\pods9_var.py --alpha 0.99 --k 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pods2_crowding import stock_factors, residualise  # noqa: E402


def hist_var_es(x, alpha):
    """Historical VaR and ES at confidence alpha, both reported positive."""
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x <= q]
    return -float(q), -float(tail.mean()) if len(tail) else float("nan")


def component_var(R, var_fund, weights):
    """Component VaR per pod. Sums to var_fund by construction."""
    r_f = R.to_numpy() @ weights
    # ddof MUST MATCH. np.cov defaults to ddof=1 and np.var to ddof=0, so
    # mixing them scales every component by n/(n-1) and the parts no longer sum
    # to the whole - which is the one property that makes a decomposition
    # attributable at all. It is a 0.03% error on 3,000 days and would have
    # passed any eyeball check of the printed table.
    v = float(np.var(r_f, ddof=1))
    if v <= 0:
        return pd.Series(0.0, index=R.columns)
    X = R.to_numpy() - R.to_numpy().mean(axis=0)
    y = r_f - r_f.mean()
    cov = X.T @ y / (len(y) - 1)
    return pd.Series(weights * cov / v * var_fund, index=R.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=5, help="factors to residualise on")
    ap.add_argument("--nav", type=float, default=1000e6,
                    help="fund NAV, for stating VaR in currency")
    a = ap.parse_args()

    R = pd.read_csv(os.path.join(a.indir, "pods_returns.csv"),
                    index_col=0, parse_dates=True)
    truth = pd.read_csv(os.path.join(a.indir, "pods_truth.csv"))
    ty = dict(zip(truth["pod"], truth["type"]))
    pods = list(R.columns)
    n = len(pods)
    w = np.repeat(1.0 / n, n)          # equal capital, the allocator's default

    print(f"pods   : {n}")
    print(f"days   : {len(R)}  ({R.index[0].date()} to {R.index[-1].date()})")
    print(f"alpha  : {a.alpha:.0%}   NAV {a.nav/1e6:,.0f}m\n")

    r_fund = R.to_numpy() @ w
    var_f, es_f = hist_var_es(r_fund, a.alpha)

    print("=" * 74)
    print("FUND LEVEL, HISTORICAL")
    print("=" * 74)
    print(f"  {'VaR':<22}{var_f:>9.3%}{var_f*a.nav/1e6:>12,.1f}m")
    print(f"  {'Expected shortfall':<22}{es_f:>9.3%}{es_f*a.nav/1e6:>12,.1f}m")
    print(f"  {'worst single day':<22}{-r_fund.min():>9.3%}")
    print(f"  {'ES / VaR':<22}{es_f/var_f:>9.2f}")
    print("\n  ES exceeds VaR by construction. The gap is the tail's shape: a")
    print("  normal distribution would put it near 1.25, and further above that")
    print("  means the losses beyond the threshold are worse than a parametric")
    print("  VaR would tell you. This is why the returns are real.")

    # ---------------------------------------------------------- decomposition
    comp = component_var(R, var_f, w)
    resid = residualise(R, stock_factors(a.indir, R.index, a.k)[0], a.k)
    rf_res = resid.to_numpy() @ w
    var_res, _ = hist_var_es(rf_res, a.alpha)
    comp_res = component_var(resid, var_res, w)

    rows = []
    for t in ["crowded", "factor", "independent"]:
        members = [p for p in pods if ty.get(p) == t]
        if not members:
            continue
        rows.append({
            "type": t,
            "pods": len(members),
            "cap_share": len(members) / n,
            "var_share": comp[members].sum() / var_f,
            "var_res_share": comp_res[members].sum() / var_res,
            "own_vol": float(R[members].std().mean() * np.sqrt(252)),
        })
    D = pd.DataFrame(rows)
    D["var_per_cap"] = D["var_share"] / D["cap_share"]
    D["res_per_cap"] = D["var_res_share"] / D["cap_share"]

    print("\n" + "=" * 74)
    print("WHERE THE VaR ACTUALLY COMES FROM")
    print("=" * 74)
    print(f"  {'type':<14}{'pods':>5}{'capital':>9}"
          f"{'VaR':>9}{'per cap':>10}{'resid VaR':>11}{'per cap':>10}")
    for _, r in D.iterrows():
        print(f"  {r['type']:<14}{int(r['pods']):>5}{r['cap_share']:>9.0%}"
              f"{r['var_share']:>9.1%}{r['var_per_cap']:>9.2f}x"
              f"{r['var_res_share']:>11.1%}{r['res_per_cap']:>9.2f}x")

    print("\n  'per cap' is share of VaR divided by share of capital. A group")
    print("  holding a quarter of the book and carrying a quarter of the risk")
    print("  sits at 1.00 and is telling you nothing. Above 1 it contributes")
    print("  more than its size; below 1, less.")

    # ------------------------------------------------- the two columns differ
    # THIS IS THE RESULT, AND IT NEEDS THE ARITHMETIC DONE FOR THE READER.
    # The raw column and the residual column can rank the pod types
    # differently, and when they do it is the same mechanism as the headline
    # correlation finding: what the factor pods share IS a factor, so it leaves
    # when factors are removed, while what the crowded pods share does not.
    if len(D) >= 2:
        raw_rank = list(D.sort_values("var_per_cap", ascending=False)["type"])
        res_rank = list(D.sort_values("res_per_cap", ascending=False)["type"])
        print("\n  ranked by raw VaR      : " + " > ".join(raw_rank))
        print("  ranked by residual VaR : " + " > ".join(res_rank))
        if raw_rank != res_rank:
            print("\n  THE TWO RANKINGS DISAGREE, and the disagreement is the point.")
            print("  Removing common factor exposure does not shuffle these groups")
            print("  randomly: it removes exactly what the factor pods share and")
            print("  leaves exactly what the crowded pods share, because the second")
            print("  was never a factor. The same reversal appears in the pair")
            print("  correlations - factor pairs fall from +0.764 to +0.053 while")
            print("  crowded pairs sit unmoved at +0.138.")
            print()
            print("  A desk ranking on raw VaR and a desk ranking on residual VaR")
            print("  would call in different managers from the same fund on the")
            print("  same day.")
        else:
            print("\n  Both rankings agree. Removing factors does not change which")
            print("  group carries the risk here, so VaR and residual VaR are not")
            print("  telling different stories - report that rather than the")
            print("  reversal the pair correlations show.")

    # ---------------------------------------------------------- the reading
    if len(D) >= 2:
        top = D.loc[D["var_per_cap"].idxmax(), "type"]
        bot = D.loc[D["var_per_cap"].idxmin(), "type"]
        print("\n" + "=" * 74)
        print("WHAT A RISK COMMITTEE WOULD DO WITH THIS")
        print("=" * 74)
        print(f"  Largest contributor per unit of capital : {top}")
        print(f"  Smallest                                : {bot}")
        if top == "factor":
            print("\n  VaR points at the FACTOR pods - the five that hold no name in")
            print("  common with each other. They dominate because they correlate")
            print("  at +0.76, so they move the fund's aggregate return, and VaR")
            print("  is a statement about the variance of that aggregate.")
            print("\n  They are also the cheapest pods in the fund to lose. Forcing")
            print("  one out costs the others less than forcing out an independent")
            print("  pod, because nobody else holds their names.")
            print("\n  The crowded pods, which cost 4.6x-13.8x more to unwind, sit")
            print("  lower in this table. A committee cutting the largest VaR")
            print("  contributor would cut the wrong five - the same error the")
            print("  correlation matrix makes, in the number funds actually report.")
        elif top == "crowded":
            print("\n  VaR points at the CROWDED pods. That is the opposite of the")
            print("  prediction, and it weakens the argument rather than doubling")
            print("  it: the standard measure gets this right, and the crowding")
            print("  test is confirming rather than correcting it.")
            print("\n  Report it that way. A tool whose value rests on the standard")
            print("  measure being wrong has to survive the standard measure being")
            print("  right.")
        else:
            print("\n  Neither planted group dominates. VaR is close to uninformative")
            print("  about crowding here, which is a weaker claim than the intended")
            print("  one but still a claim: it does not find what the crowding test")
            print("  finds, so it is not a substitute for it.")

    print("\n" + "=" * 74)
    print("WHAT VaR CANNOT SEE, STATED PLAINLY")
    print("=" * 74)
    print("  VaR asks how much the fund loses if prices move against it. It is")
    print("  computed from the covariance of returns and it is silent about who")
    print("  holds what.")
    print()
    print("  Six managers holding six disjoint baskets look diversified to it,")
    print("  and stay looking diversified right up to the morning they all sell")
    print("  for the same reason. The loss then is not a price move - it is the")
    print("  cost of the exit, and it depends on the overlap of the books rather")
    print("  than on the covariance of the returns.")
    print()
    print("  That is why this project reports both, and why neither replaces the")
    print("  other.")

    os.makedirs("outputs", exist_ok=True)
    D.to_csv("outputs/var_by_type.csv", index=False)
    pd.DataFrame({"pod": pods, "type": [ty.get(p, "") for p in pods],
                  "component_var": comp.values,
                  "component_var_resid": comp_res.values}).to_csv(
        "outputs/var_by_pod.csv", index=False)
    print("\nwrote outputs/var_by_type.csv and outputs/var_by_pod.csv")


if __name__ == "__main__":
    main()
