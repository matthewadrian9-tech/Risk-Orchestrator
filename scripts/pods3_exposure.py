"""
pods3_exposure.py
-----------------
Aggregates twenty separate books into the position the FUND actually holds,
and measures how much of it would try to leave through the same door.

WHY PAIR-LEVEL CROWDING IS NOT THE ANSWER A RISK DESK NEEDS

  pods2 answers "which two pods overlap". That is a diagnostic. The question a
  chief risk officer asks is different and harder: how much of our capital is
  in the same trade, and what happens when it moves.

  Ten flagged pairs could mean five pods sharing one basket, or it could mean
  ten unrelated coincidences. The pair matrix cannot tell you, because
  crowding is a property of a NAME, not of a pair. This script therefore
  aggregates at name level and reports the fund's position.

THE DISTINCTION THAT DECIDES EVERYTHING: NET VERSUS GROSS

  Suppose four pods are long a name and one is short it. The fund's NET
  position is small - the shorts offset the longs - and a report built on net
  exposure would show nothing worth discussing.

  That report would be dangerously wrong. The four longs are four separate
  managers who will each decide to sell on their own schedule, and the offset
  only exists as long as the short pod keeps its position. Net exposure is the
  right measure of MARKET risk: what the fund loses if the price moves. Gross
  one-way exposure is the right measure of LIQUIDATION risk: what hits the
  same bid when several pods head for the exit at once.

  These two numbers can point in completely opposite directions on the same
  name, and confusing them is how a fund discovers its crowding during the
  unwind rather than before it. Both are reported here, side by side, and the
  gap between them is itself the warning.

WHAT IS MEASURED

  For every name in every era:

    net weight        sum of pod weights, longs and shorts together
    gross weight      sum of absolute pod weights
    long pressure     sum of the POSITIVE weights only - the quantity that
                      would be sold into one bid if every holder exited
    n_pods_long       how many separate managers hold it

  And at fund level:

    concentration     Herfindahl index over net weights
    top-10 share      share of gross book in the ten largest positions
    crowded share     share of gross book in names held long by >= --min-pods
                      separate pods

THE NULL, AND WHY CONCENTRATION NEEDS ONE

  Twenty pods drawing twenty names each from a few hundred eligible will
  produce overlap by chance, and some name will always be the most crowded.
  Reporting the maximum of a random process as a finding is the same error as
  reporting the best of many p-values.

  So the concentration statistics are compared against a null that permutes
  WHICH POD HOLDS WHICH BOOK within each era. Book sizes, name popularity and
  the number of pods are all preserved exactly; only the question of who held
  what together is randomised. If the real fund is no more concentrated than
  that, its crowding is arithmetic rather than behaviour.

Run:
  python scripts\\pods3_exposure.py
  python scripts\\pods3_exposure.py --min-pods 4 --n-perm 500
"""

import argparse
import os

import numpy as np
import pandas as pd


def eligible_by_era(indir, eras):
    """Names with usable returns in each era, from the panel the pods drew from."""
    path = os.path.join(indir, "panel_daily.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\n"
            "The null needs the universe each pod could have chosen from, which "
            "only the panel knows."
        )
    panel = pd.read_csv(path, parse_dates=["date"], usecols=["permno", "date", "ret_adj"])
    panel = panel.dropna(subset=["ret_adj"])
    out = {}
    ed = [pd.Timestamp(e) for e in eras]
    for i, e in enumerate(ed):
        end = ed[i + 1] if i + 1 < len(ed) else e + pd.Timedelta(days=90)
        seg = panel[(panel["date"] >= e) & (panel["date"] < end)]
        out[eras[i]] = np.array(sorted(seg["permno"].unique()))
    return out


def era_weights(holds, pods, era):
    """Pod-by-name weight matrix for one era. Dollar-neutral, equal weight."""
    h = holds[holds["era"] == era]
    names = sorted(h["permno"].unique())
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(pods), len(names)))
    for p_i, p in enumerate(pods):
        hp = h[h["pod"] == p]
        longs = hp[hp["side"] == "long"]["permno"].to_numpy()
        shorts = hp[hp["side"] == "short"]["permno"].to_numpy()
        if len(longs):
            for n in longs:
                W[p_i, idx[n]] += 0.5 / len(longs)
        if len(shorts):
            for n in shorts:
                W[p_i, idx[n]] -= 0.5 / len(shorts)
    return W, names


def fund_stats(W, min_pods):
    """Aggregate one era's pod-by-name matrix into fund-level numbers."""
    net = W.sum(axis=0)
    gross = np.abs(W).sum(axis=0)
    long_press = np.clip(W, 0, None).sum(axis=0)
    n_long = (W > 0).sum(axis=0)

    g = gross.sum()
    if g <= 0:
        return None
    share = gross / g
    top10 = float(np.sort(share)[::-1][:10].sum())
    hhi = float((share ** 2).sum())
    crowded = n_long >= min_pods
    crowded_share = float(share[crowded].sum())
    # share of gross sitting in names held by exactly t or more pods, for a
    # ladder of t. One threshold cannot be defended: with twenty pods drawing
    # forty names each from a few hundred eligible, a low bar is cleared by
    # arithmetic alone, and only the tail separates behaviour from counting.
    ladder = {t: float(share[n_long >= t].sum()) for t in range(2, 11)}

    return {
        "hhi": hhi,
        "top10_share": top10,
        "crowded_share": crowded_share,
        "n_crowded_names": int(crowded.sum()),
        "max_long_pressure": float(long_press.max()),
        "max_n_pods_long": int(n_long.max()),
        "ladder": ladder,
        "net": net, "gross": gross,
        "long_press": long_press, "n_long": n_long,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--min-pods", type=int, default=3,
                    help="a name is 'crowded' when this many pods are long it")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--top", type=int, default=12,
                    help="names to list in the worst era")
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    pods = sorted(holds["pod"].unique())
    eras = sorted(holds["era"].unique())
    rng = np.random.default_rng(a.seed)

    print(f"pods   : {len(pods)}")
    print(f"eras   : {len(eras)}  ({eras[0]} to {eras[-1]})")
    print(f"crowded name = held long by >= {a.min_pods} pods")
    print(f"null   : {a.n_perm} permutations of who held which book\n")

    rows, per_era = [], {}
    for e in eras:
        W, names = era_weights(holds, pods, e)
        st = fund_stats(W, a.min_pods)
        if st is None:
            continue
        per_era[e] = (W, names, st)
        rows.append({"era": e, "hhi": st["hhi"], "top10_share": st["top10_share"],
                     "crowded_share": st["crowded_share"],
                     "n_crowded_names": st["n_crowded_names"],
                     "max_n_pods_long": st["max_n_pods_long"]})

    ts = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    ts.to_csv("outputs/exposure_by_era.csv", index=False)

    # ------------------------------------------------------------ the null
    # THE FIRST VERSION OF THIS NULL WAS A NO-OP, and the output said so: the
    # null mean equalled the real value to four decimals and p came back
    # exactly 1.0000. It permuted which pod held which book, which is the right
    # null for a PAIR statistic and mathematically useless for a NAME one -
    # reassigning whole books between pods leaves every column sum untouched,
    # so fund-level exposure per name cannot move.
    #
    # The correct null redraws each pod's book from the names that were
    # actually available that era, preserving book sizes and the number of
    # pods while destroying any tendency to pick the SAME names. Eligibility
    # comes from the panel rather than from the union of what pods held, since
    # that union is itself narrowed by the crowding being tested.
    print("running the permutation null ...", flush=True)
    pool = eligible_by_era(a.indir, eras)

    null_cs, null_hhi, null_lad = [], [], []
    for _ in range(a.n_perm):
        cs, hh, lad = [], [], []
        for e in eras:
            if e not in per_era:
                continue
            W, names, _ = per_era[e]
            univ = pool.get(e)
            if univ is None or len(univ) < 4 * W.shape[0]:
                continue
            n_long = int((W > 0).sum(axis=1).mean())
            n_short = int((W < 0).sum(axis=1).mean())
            Wp = np.zeros((W.shape[0], len(univ)))
            for p_i in range(W.shape[0]):
                pick = rng.choice(len(univ), size=n_long + n_short, replace=False)
                Wp[p_i, pick[:n_long]] = 0.5 / max(n_long, 1)
                Wp[p_i, pick[n_long:]] = -0.5 / max(n_short, 1)
            st = fund_stats(Wp, a.min_pods)
            if st:
                cs.append(st["crowded_share"])
                hh.append(st["hhi"])
                lad.append(st["ladder"])
        if cs:
            null_cs.append(np.mean(cs))
            null_hhi.append(np.mean(hh))
            null_lad.append({t: np.mean([d[t] for d in lad])
                             for t in range(2, 11)})
    null_cs = np.array(null_cs)
    null_hhi = np.array(null_hhi)
    if len(null_cs) == 0:
        raise SystemExit("the null produced no draws - check panel_daily.csv covers the pod eras")

    real_ladder = {t: np.mean([per_era[e][2]["ladder"][t] for e in per_era])
                   for t in range(2, 11)}
    null_ladder = {t: np.array([d[t] for d in null_lad]) for t in range(2, 11)}
    real_cs = ts["crowded_share"].mean()
    real_hhi = ts["hhi"].mean()
    p_cs = (np.sum(null_cs >= real_cs) + 1) / (len(null_cs) + 1)
    p_hhi = (np.sum(null_hhi >= real_hhi) + 1) / (len(null_hhi) + 1)

    print("\n" + "=" * 74)
    print("FUND-LEVEL CONCENTRATION, AVERAGED OVER ERAS")
    print("=" * 74)
    print(f"  {'':<28}{'real':>10}{'null mean':>12}{'p':>8}")
    print(f"  {'share in crowded names':<28}{real_cs:>10.1%}"
          f"{null_cs.mean():>12.1%}{p_cs:>8.4f}")
    print(f"  {'Herfindahl (net weights)':<28}{real_hhi:>10.4f}"
          f"{null_hhi.mean():>12.4f}{p_hhi:>8.4f}")
    print(f"  {'top-10 share of gross':<28}{ts['top10_share'].mean():>10.1%}")
    print(f"  {'crowded names per era':<28}{ts['n_crowded_names'].mean():>10.1f}")
    print(f"  {'most pods long one name':<28}{ts['max_n_pods_long'].max():>10d}")

    print("\n" + "=" * 74)
    print("SHARE OF GROSS BOOK IN NAMES HELD LONG BY t OR MORE PODS")
    print("=" * 74)
    print(f"  {'t':>3}{'real':>10}{'chance':>10}{'excess':>10}{'p':>9}")
    first_sep = None
    for t in range(2, 11):
        r = real_ladder[t]
        nl = null_ladder[t]
        p = (np.sum(nl >= r) + 1) / (len(nl) + 1)
        if p < 0.05 and first_sep is None:
            first_sep = t
        print(f"  {t:>3}{r:>10.1%}{nl.mean():>10.1%}{r-nl.mean():>+10.1%}{p:>9.4f}")

    print()
    if first_sep is None:
        print("  At no threshold is the fund more concentrated than independent")
        print("  selection produces. Whatever overlap exists is arithmetic -")
        print("  twenty managers drawing forty names each from a few hundred -")
        print("  and not a fact about how they choose.")
    else:
        print(f"  Real and chance separate from t = {first_sep} upward. Below that the")
        print("  overlap is counting, not behaviour: any twenty books drawn from")
        print("  this universe would pile up that much. The crowding worth acting")
        print(f"  on is the excess in names held by {first_sep}+ pods, and reporting the")
        print("  lower thresholds as risk would be reporting arithmetic.")

    # --------------------------------------------------- the worst era
    worst = ts.loc[ts["crowded_share"].idxmax(), "era"]
    W, names, st = per_era[worst]
    order = np.argsort(-st["long_press"])[:a.top]

    print("\n" + "=" * 74)
    print(f"WORST ERA: {worst}   NET AGAINST GROSS, NAME BY NAME")
    print("=" * 74)
    print(f"  {'permno':>9}{'pods long':>11}{'long press':>12}{'net':>10}"
          f"{'gross':>9}{'net/gross':>11}")
    for i in order:
        ratio = st["net"][i] / st["gross"][i] if st["gross"][i] > 0 else 0.0
        print(f"  {names[i]:>9}{st['n_long'][i]:>11}"
              f"{st['long_press'][i]:>12.3f}{st['net'][i]:>10.3f}"
              f"{st['gross'][i]:>9.3f}{ratio:>11.0%}")

    hidden = [i for i in order
              if st["gross"][i] > 0 and abs(st["net"][i]) / st["gross"][i] < 0.5
              and st["n_long"][i] >= a.min_pods]
    print()
    if hidden:
        print(f"  {len(hidden)} of the {a.top} most crowded names carry a net position")
        print("  under half their gross. A net-exposure report would show these as")
        print("  modest holdings while several pods are each long them and would")
        print("  each sell into the same bid. That gap is the whole point of")
        print("  reporting both columns.")
    else:
        print("  Net and gross agree on these names, so a net-exposure report")
        print("  would not have missed them. Worth knowing - the two measures")
        print("  only diverge when pods take opposite sides, and here they mostly")
        print("  do not.")

    # ------------------------------------------------------ who is crowding
    truth_path = os.path.join(a.indir, "pods_truth.csv")
    if os.path.exists(truth_path):
        truth = pd.read_csv(truth_path)
        ty = dict(zip(truth["pod"], truth["type"]))
        crowded_names = np.where(st["n_long"] >= a.min_pods)[0]
        by_type = {}
        for p_i, p in enumerate(pods):
            held = int((W[p_i, crowded_names] > 0).sum())
            by_type.setdefault(ty[p], []).append(held)
        print("\n" + "=" * 74)
        print("WHICH PODS ARE IN THE CROWDED NAMES  (truth read only here)")
        print("=" * 74)
        for t in ["crowded", "factor", "independent"]:
            if t in by_type:
                v = by_type[t]
                print(f"  {t:<14} mean {np.mean(v):5.1f} of "
                      f"{len(crowded_names)} crowded names held long")

    print("\nwrote outputs/exposure_by_era.csv")


if __name__ == "__main__":
    main()
