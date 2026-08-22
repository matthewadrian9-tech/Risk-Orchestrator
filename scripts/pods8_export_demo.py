"""
pods8_export_demo.py
--------------------
Builds a demo fund out of real CRSP names and ships the return history those
names need, so the upload panel can run the SAME analyses as the precomputed
half of the dashboard rather than a subset of them.

THE PROBLEM THIS SOLVES

  The upload panel takes positions and text. From those it can aggregate
  exposure, measure overlap against a chance floor, and read the theses. What
  it cannot do is anything that needs to watch the books MOVE:

    Sharpe                  needs each manager's return history
    residual correlation    needs returns, to strip factors out of
    a dollar unwind         needs each name's volatility

  So the panel was structurally poorer than the section above it, and the page
  read as though the tool works better on the author's own data. That is the
  wrong impression to leave, and it is not true - the gap was never the method,
  it was that the browser had no returns.

  A position file will never carry them. But a demo does not have to be built
  from names with no history: build it from CRSP names and the history already
  exists. This script writes the daily returns for exactly the names the demo
  managers hold, plus the factor series, and the browser does the rest.

WHAT IS AND IS NOT SHIPPED

  Only the ~40 permnos the six demo managers hold, over the sample window.
  That is a few megabytes, not the 200 MB panel, and it is aggregate return
  data for a handful of large caps rather than a redistribution of the CRSP
  file. If that still reads as too close to the licence, drop --n-days and ship
  a shorter window; the analyses degrade gracefully.

  For anyone uploading their own book, the extra analyses simply do not appear,
  and the panel says why. That is the honest behaviour: the tool is not better
  on the author's data, it is better where returns are available, and it says
  which is which.

THE PLANTED STRUCTURE, WHICH THE TOOL NEVER SEES

  Two managers share a basket and a driver          -> crowded
  One manager shares the driver and NO name         -> latent, the whole point
  Three managers on drivers of their own            -> independent

  Names are picked by beta so the latent manager genuinely loads on the same
  factor as the crowded pair while holding nothing in common with them. That
  is what makes it a test rather than a demonstration.

Run:
  python scripts\\pods8_export_demo.py
  python scripts\\pods8_export_demo.py --n-days 2500 --outdir docs/data
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pods2_crowding import stock_factors  # noqa: E402

TICKER_PATH = "data/processed/tickers.csv"


def load_tickers():
    if not os.path.exists(TICKER_PATH):
        return {}
    t = pd.read_csv(TICKER_PATH, dtype=str).dropna()
    c = {x.lower(): x for x in t.columns}
    p, k = c.get("permno"), c.get("ticker")
    return dict(zip(t[p].astype(str), t[k])) if p and k else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/panel_daily.csv")
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--outdir", default="docs/data")
    ap.add_argument("--demodir", default="demo")
    ap.add_argument("--n-days", type=int, default=2000,
                    help="most recent N trading days to ship")
    ap.add_argument("--book", type=int, default=6, help="names per leg")
    ap.add_argument("--beta-gap", type=float, default=0.25,
                    help="how far below the high-beta pocket the independent "
                         "managers must sit, in beta units. Rank-based "
                         "exclusion is not separation: the name one rank "
                         "outside the pocket has the same beta as the name "
                         "one rank inside it.")
    ap.add_argument("--seed", type=int, default=20260823)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    os.makedirs(a.demodir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    tick = load_tickers()

    print(f"loading {a.panel} ...")
    panel = pd.read_csv(a.panel, parse_dates=["date"])
    panel = panel.dropna(subset=["ret_adj"])
    R = panel.pivot_table(index="date", columns="permno", values="ret_adj")
    R = R.sort_index().iloc[-a.n_days:]
    R = R.loc[:, R.notna().mean() > 0.98]
    cap = panel.groupby("permno")["mktcap"].median()
    print(f"  {R.shape[1]} names with full history over the last "
          f"{len(R)} trading days")

    if R.shape[1] < 60:
        raise SystemExit("too few names with complete history - lower --n-days")

    # ---- market and beta, so the latent manager can be built on a real factor
    mkt = R.mean(axis=1)
    m = mkt.to_numpy()
    beta = {}
    for c in R.columns:
        r = R[c].to_numpy()
        beta[c] = float(np.cov(r, m)[0, 1] / np.var(m))
    beta = pd.Series(beta).sort_values(ascending=False)

    hi = list(beta.index[:60])          # high beta, the shared driver
    lo = list(beta.index[-40:])         # low beta, the funding side

    # THE INDEPENDENTS WERE NEVER ACTUALLY SEPARATED, only excluded by rank.
    # `mid` used to be everything outside the top 60 and bottom 40, so the
    # first name in it sat one rank below the last name in `hi` and carried
    # essentially the same beta - the demo planted factor overlap it did not
    # intend, and then the dashboard reported that overlap as residual
    # correlation between managers labelled independent. The label was wrong,
    # not the detector.
    #
    # Independents are now required to sit a stated distance BELOW the pocket
    # rather than merely outside it. --beta-gap is that distance in beta units.
    hi_floor = float(beta.loc[hi].min())
    eligible = beta.index[(beta < hi_floor - a.beta_gap)]
    mid = [p for p in eligible if p not in set(lo)]
    rng.shuffle(mid)
    if len(mid) < 3 * a.book:
        raise SystemExit(
            f"only {len(mid)} names sit more than {a.beta_gap} in beta below "
            f"the high-beta pocket (floor {hi_floor:.2f}), and three "
            f"independent managers need {3 * a.book}.\n"
            f"  Lower --beta-gap, or widen the universe with --n-days.\n"
            f"  Refusing rather than falling back: drawing them closer in "
            f"would plant factor overlap in books labelled independent, which "
            f"is what this gap exists to prevent.")
    print(f"  high-beta pocket floor {hi_floor:.2f}; independents drawn from "
          f"{len(mid)} names below {hi_floor - a.beta_gap:.2f} "
          f"(gap {a.beta_gap})")

    B = a.book
    # crowded pair: overlapping longs out of the same high-beta pocket
    core = hi[:B - 2]
    schen_l = core + hi[B - 2:B]
    dkim_l = core + hi[B:B + 2]
    # latent: same high-beta driver, drawn from a DISJOINT slice of the tail
    far_l = hi[B + 2:B + 2 + B]
    assert not (set(schen_l) & set(far_l)), "latent manager must share no name"
    assert not (set(dkim_l) & set(far_l)), "latent manager must share no name"

    shorts = [lo[i * B:(i + 1) * B] for i in range(6)]
    inds = [mid[i * B:(i + 1) * B] for i in range(3)]

    managers = [
        ("schen", schen_l, shorts[0], "crowded"),
        ("dkim", dkim_l, shorts[1], "crowded"),
        ("mfarrell", far_l, shorts[2], "latent"),
        ("awong", inds[0], shorts[3], "independent"),
        ("rpatel", inds[1], shorts[4], "independent"),
        ("lmoreau", inds[2], shorts[5], "independent"),
    ]
    navs = {"schen": 400e6, "dkim": 250e6, "mfarrell": 180e6,
            "awong": 300e6, "rpatel": 220e6, "lmoreau": 150e6}

    used = sorted({p for _, L, S, _ in managers for p in L + S})
    label = {p: (tick.get(str(p)) or str(p)) for p in used}

    # ---- position files
    for name, L, S, _ in managers:
        rows = []
        for p in L:
            rows.append({"ticker": label[p], "weight": round(1.0 / len(L), 4),
                         "nav": int(navs[name]), "side": "long"})
        for p in S:
            rows.append({"ticker": label[p], "weight": round(1.0 / len(S), 4),
                         "nav": int(navs[name]), "side": "short"})
        pd.DataFrame(rows).to_csv(os.path.join(a.demodir, f"{name}.csv"),
                                  index=False)

    # ---- the returns bundle the browser needs
    F, share, n_stk = stock_factors(a.indir, R.index, 5)
    sub = R[used].fillna(0.0)
    bundle = {
        "dates": [str(d.date()) for d in R.index],
        "names": {label[p]: [round(float(v), 5) for v in sub[p]] for p in used},
        "vol": {label[p]: round(float(sub[p].std() * np.sqrt(252)), 4) for p in used},
        "cap": {label[p]: float(cap.get(p, np.nan)) for p in used},
        "factors": {c: [round(float(v), 6) for v in F[c]] for c in F.columns},
        "factor_names": list(F.columns),
        "n_stocks_factor": int(n_stk),
        "note": "Daily returns for the names in the demo pack only, so the "
                "upload panel can run the same analyses as the precomputed "
                "section. Not a redistribution of CRSP.",
    }
    p_out = os.path.join(a.outdir, "demo_returns.json")
    with open(p_out, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    mb = os.path.getsize(p_out) / 1e6

    truth = pd.DataFrame([{"manager": n, "planted": t,
                           "nav": navs[n], "longs": len(L), "shorts": len(S)}
                          for n, L, S, t in managers])
    truth.to_csv(os.path.join(a.demodir, "PLANTED.csv"), index=False)

    # ---------------------------------------------------------- report
    print("\n" + "=" * 72)
    print("THE DEMO FUND")
    print("=" * 72)
    for n, L, S, t in managers:
        print(f"  {n:<10} {t:<12} nav {navs[n]/1e6:>4.0f}m   "
              f"longs {', '.join(label[p] for p in L[:4])}...")

    print("\n  beta by manager, so the separation can be checked rather than assumed:")
    for n, L, S, t in managers:
        bl = float(beta.loc[L].mean())
        bmin, bmax = float(beta.loc[L].min()), float(beta.loc[L].max())
        print(f"    {n:<10} {t:<12} long-leg beta {bl:5.2f}  "
              f"(range {bmin:.2f}-{bmax:.2f})")
    drv_min = min(float(beta.loc[x].min()) for x in (schen_l, dkim_l, far_l))
    ind_max = max(float(beta.loc[x].max()) for x in inds)
    print(f"\n    lowest beta among the three sharing the driver : {drv_min:.2f}")
    print(f"    highest beta among the three independents      : {ind_max:.2f}")
    if ind_max >= drv_min:
        print("    THESE RANGES OVERLAP. An independent manager holds a name")
        print("    with higher beta than someone in the driver group, so any")
        print("    residual correlation between them is planted factor overlap")
        print("    wearing an 'independent' label. Raise --beta-gap.")
    else:
        print(f"    ranges are disjoint by {drv_min - ind_max:.2f} in beta.")

    sl, dl, fl = set(schen_l), set(dkim_l), set(far_l)
    print("\n  schen n dkim     :", len(sl & dl), "shared names",
          f"({len(sl & dl)/len(sl | dl):.0%} of the union)")
    print("  schen n mfarrell :", len(sl & fl), "shared names  <- the latent case")
    print("  dkim  n mfarrell :", len(dl & fl), "shared names")

    def podret(L, S):
        return (sub[L].mean(axis=1) - sub[S].mean(axis=1)).to_numpy()
    pr = {n: podret(L, S) for n, L, S, _ in managers}
    print("\n  return correlation between the three sharing a driver:")
    for i, x in enumerate(["schen", "dkim", "mfarrell"]):
        for y in ["schen", "dkim", "mfarrell"][i + 1:]:
            print(f"    {x:<10} {y:<10} {np.corrcoef(pr[x], pr[y])[0,1]:+.3f}")

    print(f"\nwrote {p_out}  ({mb:.1f} MB)")
    print(f"wrote six position files and PLANTED.csv to {a.demodir}/")
    print("\n  PLANTED.csv is the answer key. The upload panel never reads it -")
    print("  it is there so anyone can check the tool found what was planted")
    print("  rather than assuming it got lucky.")
    print("\n  Theses are not regenerated here. Rewrite demo/*.md to match the")
    print("  new tickers, or run pods6_theses.py against these books.")


if __name__ == "__main__":
    main()
