"""
pods2_crowding.py
-----------------
Finds pods holding the same idiosyncratic risk, and refuses to confuse them
with pods that merely share a factor.

THE PROBLEM, IN THE NUMBERS THE SIMULATOR PRINTS

    factor pods    correlate +0.764   share  0.0% of long books
    crowded pods   correlate +0.138   share 18.1%
    independent    correlate -0.003   share  2.6%

  Rank pod pairs by return correlation and every pair at the top is a factor
  pair holding nothing in common. The pods that genuinely share positions sit
  near the bottom, indistinguishable from noise.

  This is not a quirk of the simulation. Shared beta dominates return
  correlation because beta is the largest common component of any equity
  book, and it swamps a partial overlap in names. So the obvious risk report -
  a heatmap of pod correlations - points confidently at the wrong pods.

  The two failures also need opposite remedies. Shared factor exposure is
  hedged centrally, at the fund level, without anyone changing a position.
  Genuine crowding means several pods will try to sell the same names into the
  same bid, and the only fix is for someone to cut before that happens. A tool
  that reports one number for both will prescribe the wrong one.

WHAT THIS SCRIPT DOES

  1. RESIDUALISE. Regress each pod's daily return on a set of common factors
     and keep the residual. What is left is the part of the pod's return that
     the factors do not explain. Two pods sharing only beta have uncorrelated
     residuals; two pods holding the same names do not.

     The factors are built from the pods themselves - the leading principal
     components of the pod return matrix - rather than imported from
     elsewhere. That is deliberate. A central desk sees its own pods' returns;
     it may not have a licensed factor model, and it certainly cannot assume
     the factors that matter to its book are the published ones. The number of
     components is a parameter and the script sweeps it, because that choice
     is a forking path: remove too few and factor pairs survive as false
     crowding, remove too many and real crowding is absorbed into "factors".

  2. TEST EVERY PAIR AGAINST A NULL. With 20 pods there are 190 pairs. At
     p<0.05, about 9 will look significant on pure noise. A crowding report
     listing nine pairs with no correction is manufacturing findings.

     The null is a circular block bootstrap of the residual series: it keeps
     each pod's own volatility clustering and autocorrelation intact while
     destroying the alignment BETWEEN pods. That answers the right question -
     could this much co-movement arise from two series with these dynamics
     that have nothing to do with each other?

     Benjamini-Hochberg is then applied across all 190 pairs.

  3. MEASURE OVERLAP AGAINST ITS CHANCE FLOOR. Two books of B names drawn
     from U eligible share about B^2/U names by luck. Reporting raw overlap
     without that floor makes every pair look somewhat crowded. The script
     reports excess over the floor and its own p-value from a permutation of
     the holdings.

  4. COMBINE THE TWO. A pair is flagged as crowded only if BOTH the residual
     correlation and the holdings overlap survive correction. Either alone is
     weak evidence: correlated residuals can come from a factor the components
     missed, and shared names can be a coincidence of two managers liking the
     same sector. Together they are the signature.

HOW IT IS SCORED

  pods_truth.csv is never read by the detection code - only by the scoring
  block at the end, after every decision has been made. The score is not
  "how many crowded pairs did it find". It is a confusion matrix against all
  three populations, because the failure that matters is flagging a factor
  pair, and a detector that flags everything would score perfectly on recall
  while being useless.

Run:
  python scripts\\pods2_crowding.py
  python scripts\\pods2_crowding.py --factors 1 2 3 5
  python scripts\\pods2_crowding.py --n-boot 2000 --alpha 0.05
"""

import argparse
import os

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ stats

def benjamini_hochberg(p, alpha):
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    passed = ranked <= alpha * np.arange(1, n + 1) / n
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cut = np.max(np.where(passed)[0])
        keep[order[:cut + 1]] = True
    return keep


def stock_factors(indir, dates, k_max, warmup=252, era=63):
    """Factor time series built from the stock universe.

    TWO EARLIER VERSIONS OF THIS FUNCTION WERE WRONG, IN DIFFERENT WAYS.

    The first took principal components of the POD returns. That cannot work:
    with twenty pods and five crowded into a shared basket, the crowded cluster
    IS a principal component, so removing the leading components removes
    exactly what is being detected. The scorecard showed it - crowded pairs
    kept +0.13 residual correlation at k=1 and inverted to -0.24 at k=2, as the
    second component absorbed the cluster.

    The second took principal components of the STOCK returns, which fixed that
    but introduced a subtler failure. On real CRSP data the factor pods went
    from 0.76 raw correlation to only 0.44 after five components were removed -
    barely stripped at all. Two reasons, both worth keeping:

      - The factor pods are dollar-neutral, long high-beta and short low-beta.
        That exposure is a SPREAD. The leading principal component of a stock
        universe is approximately the market, and a dollar-neutral book is
        largely insulated from the market by construction, so PC1 removes
        almost nothing from it. The exposure that matters was never in the
        components being removed.
      - Requiring 80% non-missing coverage over 1993-2024 left 193 of 645
        stocks. Those are the names that persisted for thirty years, which is a
        survivorship-selected subset, and estimating factors from it is exactly
        the bias step0_diagnose.py exists to catch.

    So this version CONSTRUCTS the factors that are actually there rather than
    hoping a decomposition finds them:

      MKT   the value-weighted market return
      HML_B a high-minus-low beta spread, formed the same way pods1 forms its
            books: rank on trailing beta, long the top decile, short the
            bottom, rebalanced on the same era grid
      PC1..  principal components of the residual stock returns, for whatever
            pervasive co-movement the first two do not capture

    Betas are estimated on the `warmup` days before each era and held fixed
    through it, matching the simulator and avoiding look-ahead.
    """
    path = os.path.join(indir, "panel_daily.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\n"
            "The factors are estimated from the stock universe, so the panel "
            "the pods were built from has to be present."
        )
    panel = pd.read_csv(path, parse_dates=["date"])
    panel = panel.dropna(subset=["ret_adj"])
    S = panel.pivot_table(index="date", columns="permno", values="ret_adj")
    S = S.sort_index()

    mpath = os.path.join(indir, "market_daily.csv")
    if os.path.exists(mpath):
        mkt = pd.read_csv(mpath, parse_dates=["date"]).set_index("date")["mkt_ret"]
        mkt = mkt.reindex(S.index)
    else:
        mkt = S.mean(axis=1)

    # ---- the beta spread, rebuilt era by era on trailing information only
    spread = pd.Series(0.0, index=S.index)
    m_np = mkt.to_numpy()
    for st in range(warmup, len(S), era):
        en = min(st + era, len(S))
        win = S.iloc[st - warmup:st]
        cols = win.columns[win.notna().mean() > 0.9]
        if len(cols) < 40:
            continue
        mw = m_np[st - warmup:st]
        mv = np.nanvar(mw)
        if not (mv > 0):
            continue
        W = win[cols].to_numpy()
        ok = np.isfinite(W).all(axis=0)
        cols = cols[ok]
        if len(cols) < 40:
            continue
        W = win[cols].to_numpy()
        beta = ((W - W.mean(0)) * (mw - mw.mean())[:, None]).mean(0) / mv

        n_take = max(5, len(cols) // 10)
        order = np.argsort(beta)
        lo = cols[order[:n_take]]
        hi = cols[order[-n_take:]]
        seg = S.iloc[st:en]
        spread.iloc[st:en] = (seg[hi].mean(axis=1).fillna(0.0)
                              - seg[lo].mean(axis=1).fillna(0.0))

    built = pd.DataFrame({"MKT": mkt.fillna(0.0), "HML_B": spread}).reindex(dates)
    built = built.fillna(0.0)

    # ---- principal components of what those two do not explain.
    # Coverage is required WITHIN the pod window only, not across all of
    # history, so the component estimate is not restricted to thirty-year
    # survivors.
    Sw = S.reindex(dates)
    Sw = Sw.loc[:, Sw.notna().mean() > 0.8].fillna(0.0)
    n_stk = Sw.shape[1]
    k_pc = max(0, k_max - built.shape[1])
    share = np.array([])
    if k_pc > 0 and n_stk > k_pc:
        X = Sw.to_numpy()
        A = np.column_stack([np.ones(len(built)), built.to_numpy()])
        coef, *_ = np.linalg.lstsq(A, X, rcond=None)
        Xr = X - A @ coef
        sd = Xr.std(axis=0)
        sd[sd == 0] = 1.0
        Z = (Xr - Xr.mean(axis=0)) / sd
        U, Sv, _ = np.linalg.svd(Z, full_matrices=False)
        share = (Sv[:k_pc] ** 2).cumsum() / (Sv ** 2).sum()
        pcs = pd.DataFrame(U[:, :k_pc], index=dates,
                           columns=[f"PC{i+1}" for i in range(k_pc)])
        F = pd.concat([built, pcs], axis=1)
    else:
        F = built

    return F, share, n_stk


def residualise(R, F, k, win=63):
    """Regress every pod on the first k factors and keep the residual, fitting
    the regression SEPARATELY IN EACH BLOCK OF `win` DAYS.

    One full-sample regression was not enough, and the reason is structural.
    The pods redraw holdings and recompute their volatility scaling every era,
    so a factor pod's loading is a step function - constant within an era,
    different across them. A single coefficient fits the average loading and
    leaves the variation in the residual. That leftover is COMMON to all the
    factor pods, because they all rescale on the same grid, so it reappears as
    exactly the residual correlation the test is trying to remove. On real
    CRSP data a full-sample fit left factor pairs at 0.41 after removing the
    market and a beta spread; the exposure was there, the regression just
    could not follow it.

    Fitting inside each block tracks the step function. It is also what a risk
    desk does in practice: exposures are estimated on recent data, because a
    loading averaged over thirty years describes no position anyone currently
    holds.

    The block costs k+1 degrees of freedom out of `win` observations, which is
    why the residual is standardised per block afterwards - otherwise a short
    block would look artificially quiet next to a long one.
    """
    X = R.to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    if k <= 0:
        return pd.DataFrame(Z, index=R.index, columns=R.columns)

    Fv = F.to_numpy()[:, :k]
    E = np.zeros_like(Z)
    n = len(Z)
    for s0 in range(0, n, win):
        s1 = min(s0 + win, n)
        if s1 - s0 < k + 5:                 # too short to fit; leave as is
            E[s0:s1] = Z[s0:s1]
            continue
        A = np.column_stack([np.ones(s1 - s0), Fv[s0:s1]])
        beta, *_ = np.linalg.lstsq(A, Z[s0:s1], rcond=None)
        r = Z[s0:s1] - A @ beta
        rs = r.std(axis=0)
        rs[rs == 0] = 1.0
        E[s0:s1] = r / rs                   # comparable across blocks
    return pd.DataFrame(E, index=R.index, columns=R.columns)


def block_shift(x, rng, block):
    """Circular block bootstrap: same dynamics, different alignment."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
    return x[idx]


def pair_pvalues(E, n_boot, block, rng):
    """Two-sided p for every pair's residual correlation.

    The null values from every pair and every draw are POOLED into one
    reference distribution rather than each pair being compared only against
    its own draws. Under the null the pairs are exchangeable, so this is
    legitimate - and it is necessary, not merely convenient. With 190 pairs,
    Benjamini-Hochberg's tightest threshold is 0.05/190 = 0.00026, while a
    per-pair p-value from 500 draws cannot go below 1/501 = 0.002. Nothing
    could ever survive correction; the first run of this script flagged zero
    crowded pairs for exactly that reason, which is the same p-value floor
    error the Analog Engine hit at n_boot=500 and recorded as error 10.
    Pooling 500 draws across 190 pairs gives 95,000 reference values and a
    floor near 1e-5, comfortably below the threshold.
    """
    X = E.to_numpy()
    n, m = X.shape
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    real = (Xs.T @ Xs) / n
    iu = np.triu_indices(m, 1)

    pool = []
    for _ in range(n_boot):
        Yb = np.column_stack([block_shift(Xs[:, j], rng, block) for j in range(m)])
        Yb = (Yb - Yb.mean(axis=0)) / np.where(Yb.std(axis=0) == 0, 1, Yb.std(axis=0))
        Cb = (Yb.T @ Yb) / n
        pool.append(np.abs(Cb[iu]))
    pool = np.sort(np.concatenate(pool))

    obs = np.abs(real[iu])
    ge = len(pool) - np.searchsorted(pool, obs, side="left")
    return real, iu, (ge + 1) / (len(pool) + 1)


# --------------------------------------------------------------- holdings

def overlap_stats(holds, pods, rng, n_perm):
    """Mean long-book Jaccard per pair, its chance floor, and a p-value from
    permuting which pod holds which book within each era."""
    eras = sorted(holds["era"].unique())
    m = len(pods)
    idx = {p: i for i, p in enumerate(pods)}
    obs = np.zeros((m, m))
    cnt = np.zeros((m, m))

    era_books = []
    for e in eras:
        h = holds[(holds["era"] == e) & (holds["side"] == "long")]
        books = {p: set(h[h["pod"] == p]["permno"]) for p in pods}
        era_books.append(books)
        for i in range(m):
            for j in range(i + 1, m):
                A, B = books[pods[i]], books[pods[j]]
                if not A or not B:
                    continue
                obs[i, j] += len(A & B) / len(A | B)
                cnt[i, j] += 1

    obs = np.divide(obs, np.maximum(cnt, 1))

    # null: shuffle which book belongs to which pod, era by era. Book sizes and
    # the popularity of individual names are preserved exactly; only the
    # question of who held them together is randomised.
    iu = np.triu_indices(m, 1)
    # Pooled across pairs, for the same reason as the residual test: a per-pair
    # p-value from n_perm draws is floored at 1/(n_perm+1), which sits far above
    # BH's threshold once there are 190 pairs.
    pool = []
    for _ in range(n_perm):
        acc = np.zeros((m, m))
        c2 = np.zeros((m, m))
        for books in era_books:
            vals = list(books.values())
            perm = rng.permutation(len(vals))
            sh = [vals[k] for k in perm]
            for i in range(m):
                for j in range(i + 1, m):
                    A, B = sh[i], sh[j]
                    if not A or not B:
                        continue
                    acc[i, j] += len(A & B) / len(A | B)
                    c2[i, j] += 1
        acc = np.divide(acc, np.maximum(c2, 1))
        pool.append(acc[iu])
    pool = np.sort(np.concatenate(pool))

    ge = len(pool) - np.searchsorted(pool, obs[iu], side="left")
    return obs, iu, (ge + 1) / (len(pool) + 1)


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 5],
                    help="numbers of principal components to sweep")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--resid-window", type=int, default=63,
                    help="days per factor-regression block; match the pods' rebalance era")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    R = pd.read_csv(os.path.join(a.indir, "pods_returns.csv"),
                    index_col=0, parse_dates=True)
    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    pods = list(R.columns)
    m = len(pods)
    n_pairs = m * (m - 1) // 2

    print(f"pods            : {m}   pairs: {n_pairs}")
    print(f"days            : {len(R):,}  ({R.index[0].date()} to {R.index[-1].date()})")
    print(f"null            : circular block bootstrap, block {a.block}, "
          f"{a.n_boot} draws")
    print(f"holdings null   : {a.n_perm} permutations of who held which book")
    print(f"correction      : Benjamini-Hochberg at alpha {a.alpha}")
    print(f"factor fit      : re-estimated every {a.resid_window} days")
    print(f"expected by luck: {a.alpha*n_pairs:.1f} pairs at p<{a.alpha} "
          f"before correction\n")

    rng = np.random.default_rng(a.seed)

    # ---------------------------------------------------- holdings overlap
    print("measuring long-book overlap ...", flush=True)
    ov, iu_o, p_ov = overlap_stats(holds, pods, rng, a.n_perm)

    B = int(holds[holds["side"] == "long"].groupby(["era", "pod"]).size().mean())
    U = holds["permno"].nunique()
    floor = B * B / U / (2 * B)
    print(f"  books average {B} names, {U} distinct names held across the sample")
    print(f"  chance Jaccard floor ~ {floor:.1%}\n")

    # ---------------------------------------------- sweep component counts
    print("building factors from the stock universe ...", flush=True)
    F, share_cum, n_stk = stock_factors(a.indir, R.index, max(a.factors))
    print(f"  {n_stk} stocks -> factors: " + ", ".join(F.columns))
    if len(share_cum):
        print("  extra components explain "
              + ", ".join(f"{v:.0%}" for v in share_cum)
              + " of what MKT and HML_B leave")
    print()

    floor_res = 1.0 / (a.n_boot * n_pairs + 1)
    floor_ov = 1.0 / (a.n_perm * n_pairs + 1)
    tight = a.alpha / n_pairs
    print(f"  BH tightest threshold      : {tight:.6f}")
    print(f"  pooled p floor, residual   : {floor_res:.6f}"
          + ("  OK" if floor_res < tight else "  TOO COARSE - raise --n-boot"))
    print(f"  pooled p floor, overlap    : {floor_ov:.6f}"
          + ("  OK" if floor_ov < tight else "  TOO COARSE - raise --n-perm"))
    print()

    rows = []
    for k in a.factors:
        E = residualise(R, F, k, a.resid_window)
        C, iu, p_res = pair_pvalues(E, a.n_boot, a.block, rng)

        keep_res = benjamini_hochberg(p_res, a.alpha)
        keep_ov = benjamini_hochberg(p_ov, a.alpha)
        both = keep_res & keep_ov

        raw = int((p_res < a.alpha).sum())
        tag = "k=%d [%s]" % (k, ",".join(F.columns[:k]))
        print(f"  {tag:<32}"
              f"raw p<{a.alpha}: {raw:<3} BH residual: {int(keep_res.sum()):<3} "
              f"BH overlap: {int(keep_ov.sum()):<3} BOTH: {int(both.sum())}",
              flush=True)

        for t in range(len(iu[0])):
            i, j = iu[0][t], iu[1][t]
            rows.append({
                "k": k, "pod_a": pods[i], "pod_b": pods[j],
                "raw_corr": round(float(R.corr().to_numpy()[i, j]), 4),
                "resid_corr": round(float(C[i, j]), 4),
                "p_resid": round(float(p_res[t]), 4),
                "overlap": round(float(ov[i, j]), 4),
                "p_overlap": round(float(p_ov[t]), 4),
                "bh_resid": bool(keep_res[t]),
                "bh_overlap": bool(keep_ov[t]),
                "flagged": bool(both[t]),
            })

    res = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    res.to_csv("outputs/crowding_pairs.csv", index=False)

    # ------------------------------------------------------------- scoring
    truth_path = os.path.join(a.indir, "pods_truth.csv")
    if not os.path.exists(truth_path):
        print("\nno pods_truth.csv - skipping the scorecard")
        return
    truth = pd.read_csv(truth_path)
    ty = dict(zip(truth["pod"], truth["type"]))

    def label(r):
        s = {ty[r["pod_a"]], ty[r["pod_b"]]}
        if s == {"crowded"}:
            return "crowded"
        if s == {"factor"}:
            return "factor"
        if s == {"independent"}:
            return "independent"
        return "mixed"

    res["true_type"] = res.apply(label, axis=1)

    print("\n" + "=" * 74)
    print("SCORECARD  (pods_truth.csv was not read until this line)")
    print("=" * 74)
    for k in a.factors:
        sub = res[res["k"] == k]
        print(f"\n  k = {k} components removed")
        print(f"    {'true type':<14}{'pairs':>7}{'mean raw':>10}{'mean resid':>12}"
              f"{'flagged':>9}")
        for t in ["crowded", "factor", "independent", "mixed"]:
            s = sub[sub["true_type"] == t]
            if not len(s):
                continue
            print(f"    {t:<14}{len(s):>7}{s['raw_corr'].mean():>10.3f}"
                  f"{s['resid_corr'].mean():>12.3f}{int(s['flagged'].sum()):>9}")

        tp = int(sub[(sub["true_type"] == "crowded") & sub["flagged"]].shape[0])
        fp_f = int(sub[(sub["true_type"] == "factor") & sub["flagged"]].shape[0])
        fp_i = int(sub[(sub["true_type"] == "independent") & sub["flagged"]].shape[0])
        n_cr = int((sub["true_type"] == "crowded").sum())
        print(f"    -> caught {tp}/{n_cr} crowded pairs, "
              f"{fp_f} factor pairs wrongly flagged, "
              f"{fp_i} independent pairs wrongly flagged")

    print("\n" + "=" * 74)
    print("READING THIS")
    print("=" * 74)
    print("  The column that matters is 'factor pairs wrongly flagged'. A")
    print("  detector that flags every correlated pair scores perfectly on")
    print("  crowded pairs and is useless, because it sends the fund to cut")
    print("  positions when the actual exposure is a factor that should have")
    print("  been hedged centrally.")
    print()
    print("  Compare the mean raw and mean residual correlation for the factor")
    print("  row. If residualising worked, raw is high and residual is near")
    print("  zero. If the crowded row keeps its residual correlation while the")
    print("  factor row loses it, the separation is real.")
    print()
    print("  The number of components is a choice, which is why it is swept.")
    print("  A pair flagged at one k and not its neighbours is an artifact of")
    print("  that choice, not a finding about the fund.")
    print("\nwrote outputs/crowding_pairs.csv")


if __name__ == "__main__":
    main()
