"""
pods4_unwind.py
---------------
Forces one pod to liquidate and measures what it costs everyone else.

WHY THIS SCRIPT EXISTS

  pods3 says a few percent of the fund's gross book sits in names that four or
  more pods hold. That is a true sentence and it is hard to act on, because
  nobody allocates capital against a percentage. The question an investment
  committee actually asks is: if one of these managers is stopped out on
  Monday, what does it cost the other nineteen?

  Crowding is only latent risk. It becomes realised risk at the moment
  somebody sells, and the damage lands on people who did nothing.

THE HONEST PROBLEM WITH ANY UNWIND MODEL

  Market impact - how far a price moves when you sell into it - depends on
  volume, and CRSP daily returns do not carry volume. Every number in this
  script therefore rests on an assumption about liquidity, and there is no
  way to estimate that assumption from the data available.

  Pretending otherwise would be the worst thing this project could do. So the
  impact coefficient is not estimated, it is SWEPT, and the output is read
  accordingly:

    Absolute losses are a SCENARIO. They say "if liquidity is this bad, the
    damage is this large", and they are only as good as that if.

    The RATIO of damage across pod types is close to invariant to the
    coefficient, because it scales nearly everything equally. That ratio is
    the finding. It says who gets hurt, which does not depend on knowing how
    much.

  Read the sweep and you can see this directly: the dollar figures move by a
  factor of ten down the table while the ratio barely moves.

THE MODEL

  Square-root impact, the standard workhorse:

      price move = -coefficient * volatility * sqrt(quantity / capacity)

  Quantity is the liquidating pod's position in a name. Capacity is a proxy
  for daily traded volume, taken as --turnover times market capitalisation,
  because CRSP has market cap and not volume. Volatility is the name's own
  realised volatility over the era.

  Every other pod holding that name is then marked at the moved price. A pod
  that is SHORT a name being liquidated makes money, which is why the report
  separates gainers from losers rather than reporting a net figure.

WHO GETS LIQUIDATED

  By default the pod with the largest overlap into names others also hold -
  chosen from HOLDINGS ONLY, never from pods_truth.csv. A risk desk does not
  know which of its managers is the crowded one; that is what it is trying to
  work out. The truth labels are read at the end, to check whether the pod the
  tool picked was in fact one of the planted crowded ones.

  --pod names a specific pod instead. --random-pod liquidates a randomly
  chosen one, which is the comparison that matters: if liquidating a random
  pod hurts as much as liquidating the crowded one, the crowding is not doing
  any work.

Run:
  python scripts\\pods4_unwind.py
  python scripts\\pods4_unwind.py --coefficients 0.1 0.3 1.0 3.0
  python scripts\\pods4_unwind.py --pod pod_07 --turnover 0.002
"""

import argparse
import os

import numpy as np
import pandas as pd


def era_books(holds, pods, era):
    h = holds[holds["era"] == era]
    out = {}
    for p in pods:
        hp = h[h["pod"] == p]
        longs = hp[hp["side"] == "long"]["permno"].to_numpy()
        shorts = hp[hp["side"] == "short"]["permno"].to_numpy()
        w = {}
        for n in longs:
            w[n] = w.get(n, 0.0) + 0.5 / max(len(longs), 1)
        for n in shorts:
            w[n] = w.get(n, 0.0) - 0.5 / max(len(shorts), 1)
        out[p] = w
    return out


def pick_victim(books, pods):
    """The pod whose long book overlaps most with everyone else's.

    Computed from holdings alone. This is the pod a risk desk would worry
    about if it could see the books, which is the situation being modelled.
    """
    best, best_score = None, -1.0
    for p in pods:
        mine = {n for n, w in books[p].items() if w > 0}
        if not mine:
            continue
        score = 0.0
        for q in pods:
            if q == p:
                continue
            theirs = {n for n, w in books[q].items() if w > 0}
            if theirs:
                score += len(mine & theirs) / len(mine | theirs)
        if score > best_score:
            best, best_score = p, score
    return best, best_score


def name_stats(panel, era_start, era_end):
    """Volatility and a capacity proxy for every name traded in the era."""
    seg = panel[(panel["date"] >= era_start) & (panel["date"] < era_end)]
    g = seg.groupby("permno")
    vol = g["ret_adj"].std()
    cap = g["mktcap"].median()
    return vol, cap


def unwind(books, victim, pods, vol, cap, coef, turnover, nav):
    """One-round liquidation. Returns per-pod mark-to-market P&L."""
    pnl = {p: 0.0 for p in pods}
    moves = {}
    for n, w in books[victim].items():
        if w <= 0:                      # only model the long side being sold
            continue
        v = float(vol.get(n, np.nan))
        c = float(cap.get(n, np.nan))
        if not np.isfinite(v) or not np.isfinite(c) or c <= 0:
            continue
        capacity = turnover * c          # dollars tradeable in a day, proxy
        qty = w * nav
        if capacity <= 0:
            continue
        move = -coef * v * np.sqrt(qty / capacity)
        moves[n] = move
        for p in pods:
            wp = books[p].get(n, 0.0)
            if wp == 0.0:
                continue
            pnl[p] += wp * move * nav
    return pnl, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--era", default=None,
                    help="era start date; default is the most crowded era")
    ap.add_argument("--pod", default=None, help="force a specific pod to liquidate")
    ap.add_argument("--random-pod", action="store_true")
    ap.add_argument("--coefficients", type=float, nargs="+",
                    default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--turnover", type=float, default=0.005,
                    help="fraction of market cap tradeable per day, the liquidity proxy")
    ap.add_argument("--nav", type=float, default=100e6,
                    help="capital per pod, in dollars")
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    pods = sorted(holds["pod"].unique())
    eras = sorted(holds["era"].unique())
    rng = np.random.default_rng(a.seed)

    # choose the era: the one with the most names held by 3+ pods
    if a.era:
        era = a.era
    else:
        best, era = -1, eras[0]
        for e in eras:
            h = holds[(holds["era"] == e) & (holds["side"] == "long")]
            cnt = h.groupby("permno")["pod"].nunique()
            score = int((cnt >= 3).sum())
            if score > best:
                best, era = score, e
    i = eras.index(era)
    era_start = pd.Timestamp(era)
    era_end = pd.Timestamp(eras[i + 1]) if i + 1 < len(eras) else era_start + pd.Timedelta(days=90)

    panel = pd.read_csv(os.path.join(a.indir, "panel_daily.csv"),
                        parse_dates=["date"],
                        usecols=["permno", "date", "ret_adj", "mktcap"])
    vol, cap = name_stats(panel, era_start, era_end)

    books = era_books(holds, pods, era)
    if a.pod:
        victim, score = a.pod, float("nan")
    elif a.random_pod:
        victim, score = str(rng.choice(pods)), float("nan")
    else:
        victim, score = pick_victim(books, pods)

    print(f"era        : {era} to {era_end.date()}")
    print(f"pods       : {len(pods)}, {a.nav/1e6:.0f}m each, "
          f"{len(pods)*a.nav/1e9:.1f}bn fund")
    print(f"liquidating: {victim}" + ("" if np.isnan(score)
          else f"  (largest long-book overlap, score {score:.2f})"))
    print(f"liquidity  : {a.turnover:.1%} of market cap tradeable per day\n")

    truth_path = os.path.join(a.indir, "pods_truth.csv")
    ty = {}
    if os.path.exists(truth_path):
        t = pd.read_csv(truth_path)
        ty = dict(zip(t["pod"], t["type"]))

    print("=" * 76)
    print("DAMAGE TO THE OTHER PODS, ACROSS THE LIQUIDITY ASSUMPTION")
    print("=" * 76)
    print(f"  {'coef':>6}{'victim':>12}{'others total':>15}"
          f"{'worst other':>13}{'n hurt':>8}{'crowded - indep':>17}")

    rows = []
    for coef in a.coefficients:
        pnl, moves = unwind(books, victim, pods, vol, cap, coef, a.turnover, a.nav)
        others = {p: v for p, v in pnl.items() if p != victim}
        total = sum(others.values())
        worst = min(others.values()) if others else 0.0
        n_hurt = sum(1 for v in others.values() if v < 0)

        # A RATIO WAS WRONG HERE. The first version divided crowded mean P&L by
        # independent mean P&L, and independent pods are close to flat in an
        # unwind - the denominator sits near zero, so the ratio blew up to
        # -71.7x on real data and its sign carried no information. A difference
        # in mean P&L has none of that pathology, and dividing it by the
        # coefficient makes it comparable down the table: the impact model is
        # linear in coef, so the normalised gap should be constant, and if it
        # is not, something other than impact is moving.
        if ty:
            cr = [v for p, v in others.items() if ty.get(p) == "crowded"]
            ind = [v for p, v in others.items() if ty.get(p) == "independent"]
            gap = (np.mean(cr) - np.mean(ind)) if (cr and ind) else np.nan
        else:
            gap = np.nan

        gs = "" if np.isnan(gap) else f"{gap/1e6:>15.3f}m"
        print(f"  {coef:>6.1f}{pnl[victim]/1e6:>11.2f}m{total/1e6:>14.2f}m"
              f"{worst/1e6:>12.2f}m{n_hurt:>8}{gs}")

        for p, v in pnl.items():
            rows.append({"coef": coef, "pod": p, "type": ty.get(p, ""),
                         "pnl_usd": round(v, 2), "is_victim": p == victim,
                         "victim": victim})

    res = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    res.to_csv("outputs/unwind.csv", index=False)

    # ---------------------------------------------------- who gets hurt
    mid = a.coefficients[len(a.coefficients) // 2]
    sub = res[(res["coef"] == mid) & (~res["is_victim"])]
    print("\n" + "=" * 76)
    print(f"AT COEFFICIENT {mid}, WHO ABSORBS IT  (truth read only here)")
    print("=" * 76)
    if ty:
        for t in ["crowded", "factor", "independent"]:
            s_ = sub[sub["type"] == t]
            if len(s_):
                print(f"  {t:<14}{len(s_):>3} pods   mean {s_['pnl_usd'].mean()/1e6:>8.3f}m"
                      f"   worst {s_['pnl_usd'].min()/1e6:>8.3f}m")
        print(f"\n  the liquidated pod was type: {ty.get(victim, 'unknown')}")

    # ------------------------------------------------------- the control
    # A SINGLE RANDOM POD IS NOT A CONTROL. With five of twenty pods planted
    # crowded, one random draw lands on a crowded pod a quarter of the time -
    # and on the first real run it did exactly that, so the "control" was a
    # second copy of the treatment. Every pod is liquidated in turn instead,
    # and the results are grouped by the type of pod that was liquidated.
    if ty and not a.pod and not a.random_pod:
        print("\n" + "=" * 76)
        print(f"CONTROL: LIQUIDATE EVERY POD IN TURN, coefficient {mid}")
        print("=" * 76)
        by_type = {}
        for v in pods:
            pnl_v, _ = unwind(books, v, pods, vol, cap, mid, a.turnover, a.nav)
            others_v = {p: x for p, x in pnl_v.items() if p != v}
            by_type.setdefault(ty.get(v, "?"), []).append({
                "total": sum(others_v.values()),
                "worst": min(others_v.values()) if others_v else 0.0,
                "n_hurt": sum(1 for x in others_v.values() if x < 0),
            })
        print(f"  {'liquidated pod':<16}{'n':>4}{'mean damage':>15}"
              f"{'worst single':>15}{'pods hurt':>12}")
        for t in ["crowded", "factor", "independent"]:
            v = by_type.get(t)
            if not v:
                continue
            print(f"  {t:<16}{len(v):>4}"
                  f"{np.mean([x['total'] for x in v])/1e6:>14.2f}m"
                  f"{np.mean([x['worst'] for x in v])/1e6:>14.2f}m"
                  f"{np.mean([x['n_hurt'] for x in v]):>12.1f}")
        cr = [x["total"] for x in by_type.get("crowded", [])]
        ind = [x["total"] for x in by_type.get("independent", [])]
        if cr and ind:
            print(f"\n  Liquidating a crowded pod costs the rest of the fund")
            print(f"  {abs(np.mean(cr))/max(abs(np.mean(ind)), 1e-9):.1f}x what liquidating an independent one costs.")
            print("  Averaged over every pod of each type, so no single unlucky")
            print("  draw decides it.")

    print("\n" + "=" * 76)
    print("READING THIS")
    print("=" * 76)
    print("  The dollar column moves by roughly the ratio of the coefficients")
    print("  down the table, because it is proportional to an assumption that")
    print("  cannot be estimated from CRSP daily data. Do not quote it as a")
    print("  loss estimate. It is a scenario.")
    print()
    print("  The crowded-minus-independent gap scales with the coefficient")
    print("  because the impact model is linear in it. What does NOT depend on")
    print("  the assumption is which group absorbs the loss, and the control")
    print("  table above is the test of that: if liquidating an independent pod")
    print("  did the same damage, crowding would not be the mechanism and this")
    print("  tool would be measuring nothing.")
    print("\nwrote outputs/unwind.csv")


if __name__ == "__main__":
    main()
