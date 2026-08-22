"""
pods1_simulate.py
-----------------
Builds a multi-manager fund out of real CRSP returns: roughly twenty pods,
each running its own long/short book, with crowding planted in a subset ON
PURPOSE so the detector built next has something to be scored against.

WHY SIMULATE THE PODS AND NOT THE RETURNS

  The returns are real. Only the allocation rules are invented.

  That split matters. If you simulate returns too, you have to invent a
  covariance structure, and whatever you invent will be wrong in ways that
  quietly decide the answer - too clean, too stationary, no fat tails, no
  volatility clustering. The correlation between two pods holding real stocks
  over real days is whatever it actually was. Nobody has to be trusted about it.

  What is invented is who held what, which is exactly the thing no outsider
  can observe about a real fund anyway.

THE THREE POD TYPES, AND WHY THE MIDDLE ONE IS THE POINT

  independent   draws its book at random from the eligible universe. No shared
                names by design, beyond what chance gives.

  factor        tilts long high-beta and short low-beta. Its holdings are
                drawn from a partition, so NO factor pod shares a name with
                another factor pod. Not one.

  crowded       draws a stated share of its long book from a common basket
                that every crowded pod also draws from. These pods genuinely
                hold the same idiosyncratic risk.

  The factor pods are the trap, and they are the reason this project exists.
  They will show high pairwise return correlation while holding nothing in
  common. A risk desk that flags them as crowded has misdiagnosed the problem:
  the remedy for shared factor exposure is to hedge the factor, and the remedy
  for genuine crowding is to make somebody cut. A tool that cannot tell them
  apart will recommend the wrong one, confidently.

  So the detector in the next script is not scored on "did it find correlated
  pods". It is scored on whether it separates these two populations, and the
  ground-truth labels written here are how that gets checked.

WHAT IS DELIBERATELY NOT CLAIMED

  These are not strategies and no part of this is a backtest. Nobody is
  claiming a pod would have made money. The pods exist to generate a realistic
  JOINT structure across books - which names overlap, how returns co-move,
  how that changes era to era - because that structure is what the risk tool
  has to work on. Reading a Sharpe ratio out of this file would be reading it
  wrong.

  Betas are estimated on the 252 days BEFORE each era and held fixed through
  it. Estimating them on the era itself would be look-ahead, and although a
  simulation could arguably get away with it, a habit of allowing it in places
  that "don't count" is how it ends up somewhere that does.

Run:
  python scripts\\pods1_simulate.py
  python scripts\\pods1_simulate.py --n-pods 20 --crowd-share 0.6
  python scripts\\pods1_simulate.py --era 63 --target-vol 0.10
"""

import argparse
import os

import numpy as np
import pandas as pd


def load_panel(path, market_path, min_days):
    panel = pd.read_csv(path, parse_dates=["date"])
    panel = panel.dropna(subset=["ret_adj"])
    counts = panel.groupby("permno").size()
    keep = counts[counts >= min_days].index
    panel = panel[panel["permno"].isin(keep)]

    R = panel.pivot_table(index="date", columns="permno", values="ret_adj")
    R = R.sort_index()

    if os.path.exists(market_path):
        mkt = pd.read_csv(market_path, parse_dates=["date"]).set_index("date")
        m = mkt["mkt_ret"].reindex(R.index)
    else:
        m = R.mean(axis=1)
    return R, m


def trailing_beta(R_win, m_win):
    """Beta of every stock against the market over one trailing window."""
    m = m_win.to_numpy()
    mv = np.nanvar(m)
    if mv <= 0:
        return pd.Series(1.0, index=R_win.columns)
    out = {}
    for c in R_win.columns:
        r = R_win[c].to_numpy()
        ok = np.isfinite(r) & np.isfinite(m)
        if ok.sum() < 60:
            out[c] = np.nan
            continue
        out[c] = float(np.cov(r[ok], m[ok])[0, 1] / np.var(m[ok]))
    return pd.Series(out)


def book_weights(longs, shorts, cols):
    """Dollar-neutral, equal weight within each leg."""
    w = pd.Series(0.0, index=cols)
    if len(longs):
        w[longs] += 0.5 / len(longs)
    if len(shorts):
        w[shorts] -= 0.5 / len(shorts)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/panel_daily.csv")
    ap.add_argument("--market", default="data/processed/market_daily.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--n-pods", type=int, default=20)
    ap.add_argument("--n-crowded", type=int, default=5,
                    help="pods drawing from a shared basket")
    ap.add_argument("--n-factor", type=int, default=5,
                    help="pods tilted on beta, holdings disjoint from each other")
    ap.add_argument("--book-size", type=int, default=20,
                    help="names per leg")
    ap.add_argument("--crowd-share", type=float, default=0.6,
                    help="fraction of a crowded pod's long leg from the shared basket")
    ap.add_argument("--basket-size", type=int, default=25)
    ap.add_argument("--era", type=int, default=63,
                    help="trading days between rebalances")
    ap.add_argument("--warmup", type=int, default=252)
    ap.add_argument("--target-vol", type=float, default=0.10,
                    help="annualised volatility each pod is scaled to")
    ap.add_argument("--min-days", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    if not os.path.exists(a.panel):
        raise SystemExit(
            f"{a.panel} not found.\n"
            "Re-run step 1 of the Analog Engine with --save-panel:\n"
            "  python scripts\\step1_load_crsp_v2.py --input data/raw/YOURFILE.csv "
            "--start-year 1992 --top-n 100 --save-panel"
        )

    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"loading {a.panel} ...")
    R, mkt = load_panel(a.panel, a.market, a.min_days)
    print(f"  {R.shape[1]:,} stocks x {R.shape[0]:,} trading days "
          f"({R.index[0].date()} to {R.index[-1].date()})")

    n_ind = a.n_pods - a.n_crowded - a.n_factor
    if n_ind < 0:
        raise SystemExit("--n-crowded plus --n-factor exceeds --n-pods")

    types = (["crowded"] * a.n_crowded + ["factor"] * a.n_factor
             + ["independent"] * n_ind)
    names = [f"pod_{i:02d}" for i in range(a.n_pods)]
    print(f"  {a.n_crowded} crowded, {a.n_factor} factor, {n_ind} independent")
    print(f"  book {a.book_size} names per leg, rebalanced every {a.era} days\n")

    starts = list(range(a.warmup, len(R) - 1, a.era))
    pod_ret = pd.DataFrame(0.0, index=R.index, columns=names)
    hold_rows = []

    for s in starts:
        e = min(s + a.era, len(R))
        win = R.iloc[s - a.warmup:s]
        eligible = win.columns[win.notna().mean() > 0.9]
        if len(eligible) < a.book_size * 4:
            continue

        beta = trailing_beta(win[eligible], mkt.iloc[s - a.warmup:s]).dropna()
        eligible = beta.index
        hi = beta.sort_values(ascending=False).index
        lo = beta.sort_values().index

        # the shared basket every crowded pod draws from this era
        basket = rng.choice(eligible, size=min(a.basket_size, len(eligible)),
                            replace=False)

        # factor pods get a partition of the high/low beta tails, so they
        # cannot share a name even by accident
        n_take = a.book_size * max(a.n_factor, 1)
        hi_pool = list(hi[:min(n_take, len(hi))])
        lo_pool = list(lo[:min(n_take, len(lo))])
        rng.shuffle(hi_pool)
        rng.shuffle(lo_pool)

        f_seen = 0
        for p, (nm, ty) in enumerate(zip(names, types)):
            if ty == "crowded":
                k = int(round(a.book_size * a.crowd_share))
                core = rng.choice(basket, size=min(k, len(basket)), replace=False)
                rest_pool = np.setdiff1d(eligible, core)
                rest = rng.choice(rest_pool, size=a.book_size - len(core),
                                  replace=False)
                longs = np.concatenate([core, rest])
                shorts = rng.choice(np.setdiff1d(rest_pool, rest),
                                    size=a.book_size, replace=False)
            elif ty == "factor":
                i0 = f_seen * a.book_size
                longs = np.array(hi_pool[i0:i0 + a.book_size])
                shorts = np.array(lo_pool[i0:i0 + a.book_size])
                f_seen += 1
                # The first version of this script quietly drew random names
                # when the beta tails ran out, which turned a factor pod into
                # an independent one while the report went on asserting the
                # factor pods shared no holdings "by construction". It printed
                # that sentence next to a measured overlap of 5.9%. A fallback
                # that changes what the data MEANS has to fail loudly, because
                # the output stays plausible either way and nothing downstream
                # can tell.
                if len(longs) < a.book_size or len(shorts) < a.book_size:
                    raise SystemExit(
                        f"\nSTOP: the universe cannot support this configuration.\n"
                        f"  {a.n_factor} factor pods x {a.book_size} names per leg "
                        f"needs {a.n_factor * a.book_size} distinct high-beta names\n"
                        f"  and the same again on the low-beta side, but only "
                        f"{len(eligible)} stocks are eligible in this era.\n\n"
                        f"  Fix it one of two ways:\n"
                        f"    smaller books   --book-size {max(5, len(eligible)//(4*max(a.n_factor,1)))} "
                        f"--n-factor {a.n_factor}\n"
                        f"    wider universe  re-run step1_load_crsp_v2.py with a larger --top-n\n\n"
                        f"  Substituting random names here would silently turn a "
                        f"factor pod into an independent one\n"
                        f"  and every number downstream would still look reasonable."
                    )
            else:
                longs = rng.choice(eligible, size=a.book_size, replace=False)
                shorts = rng.choice(np.setdiff1d(eligible, longs),
                                    size=a.book_size, replace=False)

            w = book_weights(longs, shorts, eligible)

            # scale to a common target volatility using the TRAILING window,
            # so every pod looks similarly sized to an allocator
            hist = win[eligible].fillna(0.0).to_numpy() @ w.to_numpy()
            sd = hist.std() * np.sqrt(252)
            scale = (a.target_vol / sd) if sd > 1e-8 else 1.0
            scale = float(np.clip(scale, 0.1, 10.0))

            seg = R.iloc[s:e][eligible].fillna(0.0).to_numpy() @ w.to_numpy()
            pod_ret.iloc[s:e, pod_ret.columns.get_loc(nm)] = seg * scale

            for t_ in longs:
                hold_rows.append({"era": R.index[s].date(), "pod": nm,
                                  "permno": t_, "side": "long"})
            for t_ in shorts:
                hold_rows.append({"era": R.index[s].date(), "pod": nm,
                                  "permno": t_, "side": "short"})

    pod_ret = pod_ret.iloc[a.warmup:]
    pod_ret = pod_ret.loc[(pod_ret != 0).any(axis=1)]

    truth = pd.DataFrame({"pod": names, "type": types})
    holds = pd.DataFrame(hold_rows)

    pod_ret.to_csv(os.path.join(a.outdir, "pods_returns.csv"))
    holds.to_csv(os.path.join(a.outdir, "pods_holdings.csv"), index=False)
    truth.to_csv(os.path.join(a.outdir, "pods_truth.csv"), index=False)

    # ------------------------------------------------------------- report
    ann = pod_ret.mean() * 252
    vol = pod_ret.std() * np.sqrt(252)
    shp = ann / vol

    print("=" * 72)
    print("EACH POD ON ITS OWN")
    print("=" * 72)
    print(f"  {'pod':<10} {'type':<13} {'ann ret':>9} {'ann vol':>9} {'sharpe':>8}")
    for nm, ty in zip(names, types):
        print(f"  {nm:<10} {ty:<13} {ann[nm]:>8.1%} {vol[nm]:>9.1%} {shp[nm]:>8.2f}")

    C = pod_ret.corr().to_numpy()
    iu = np.triu_indices(len(names), 1)
    tmap = {nm: ty for nm, ty in zip(names, types)}

    def pair_mean(t1, t2):
        vals = [C[i, j] for i, j in zip(*iu)
                if {tmap[names[i]], tmap[names[j]]} == {t1, t2}]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 72)
    print("WHAT THE CORRELATION MATRIX ALONE WOULD TELL YOU")
    print("=" * 72)
    print(f"  crowded pair, mean correlation      : {pair_mean('crowded','crowded'):+.3f}")
    print(f"  factor pair, mean correlation       : {pair_mean('factor','factor'):+.3f}")
    print(f"  independent pair, mean correlation  : {pair_mean('independent','independent'):+.3f}")

    # how much do books actually overlap
    last = holds["era"].max()
    h = holds[holds["era"] == last]
    sets = {nm: set(h[(h["pod"] == nm) & (h["side"] == "long")]["permno"])
            for nm in names}

    def overlap_mean(t1, t2):
        vals = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if {tmap[names[i]], tmap[names[j]]} != {t1, t2}:
                    continue
                A, B = sets[names[i]], sets[names[j]]
                if A and B:
                    vals.append(len(A & B) / len(A | B))
        return float(np.mean(vals)) if vals else float("nan")

    print("\n  long-book overlap in the final era (Jaccard):")
    print(f"    crowded pair                      : {overlap_mean('crowded','crowded'):.3f}")
    print(f"    factor pair                       : {overlap_mean('factor','factor'):.3f}")
    print(f"    independent pair                  : {overlap_mean('independent','independent'):.3f}")

    print("\n" + "=" * 72)
    print("THE PROBLEM, IN TWO NUMBERS")
    print("=" * 72)
    fo, co = overlap_mean("factor", "factor"), overlap_mean("crowded", "crowded")
    io = overlap_mean("independent", "independent")
    print(f"  factor pods    correlate {pair_mean('factor','factor'):+.3f}   share {fo:.1%} of long books")
    print(f"  crowded pods   correlate {pair_mean('crowded','crowded'):+.3f}   share {co:.1%}")
    print(f"  independent    correlate {pair_mean('independent','independent'):+.3f}   share {io:.1%}")
    print()
    # Chance overlap is not zero and pretending otherwise would set the
    # detector an easier problem than the real one. With B names per leg drawn
    # from U eligible, two books share about B^2/U names by luck alone.
    print(f"  With {a.book_size} names per leg drawn from {len(eligible)} eligible, two")
    print(f"  unrelated books share about {a.book_size**2/max(len(eligible),1):.1f} names by chance -")
    print(f"  a Jaccard of roughly {a.book_size**2/max(len(eligible),1)/(2*a.book_size):.1%}. Any crowding")
    print("  measure has to clear that floor before it means anything.")
    print()
    print("  A risk report built on the correlation matrix cannot tell these")
    print("  apart, and would prescribe the same remedy for both. One of them")
    print("  needs a factor hedge; the other needs somebody to cut. Separating")
    print("  them is what the next script has to do, and the labels in")
    print("  pods_truth.csv are how it gets scored.")

    print(f"\nwrote pods_returns.csv, pods_holdings.csv, pods_truth.csv to {a.outdir}")


if __name__ == "__main__":
    main()
