"""
pods5_export_web.py
-------------------
Runs the whole chain once and writes the numbers to JSON, so the dashboard is
a renderer and not a calculator.

WHY EVERYTHING IS PRECOMPUTED

  Every figure the page displays comes out of this script. The browser draws
  and nothing else - it does not resample, it does not fit a regression, it
  does not decide what is significant. Two reasons, and the second matters
  more than the first.

  The practical one: the permutation nulls are thousands of resamples over a
  thirty-year panel. That is a server job, not a main-thread job.

  The real one: a chart that computes its own statistics can quietly disagree
  with the scripts that produced them, and then there are two answers with no
  way to tell which is the project's. Here there is exactly one path from data
  to number, it runs in Python, and the page can only show what that path
  produced.

Run:
  python scripts\\pods5_export_web.py
  python scripts\\pods5_export_web.py --n-boot 500 --n-perm 300
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pods2_crowding import (  # noqa: E402
    benjamini_hochberg, overlap_stats, pair_pvalues, residualise, stock_factors,
)
from pods3_exposure import era_weights, eligible_by_era, fund_stats  # noqa: E402
from pods4_unwind import (  # noqa: E402
    era_books, name_stats, pick_victim, rank_eras, unwind,
)


def jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--outdir", default="docs/data")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--resid-window", type=int, default=63)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-pods", type=int, default=3)
    ap.add_argument("--n-eras", type=int, default=5)
    ap.add_argument("--turnover", type=float, default=0.005)
    ap.add_argument("--nav", type=float, default=100e6)
    ap.add_argument("--coefficients", type=float, nargs="+",
                    default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    R = pd.read_csv(os.path.join(a.indir, "pods_returns.csv"),
                    index_col=0, parse_dates=True)
    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    truth = pd.read_csv(os.path.join(a.indir, "pods_truth.csv"))
    ty = dict(zip(truth["pod"], truth["type"]))
    pods = list(R.columns)
    eras = sorted(holds["era"].unique())
    m = len(pods)
    n_pairs = m * (m - 1) // 2

    out = {
        "meta": {
            "n_pods": m, "n_pairs": n_pairs, "n_eras": len(eras),
            "n_days": len(R),
            "start": str(R.index[0].date()), "end": str(R.index[-1].date()),
            "nav_per_pod": a.nav, "alpha": a.alpha,
            "n_boot": a.n_boot, "n_perm": a.n_perm,
            "min_pods": a.min_pods, "turnover": a.turnover,
            "resid_window": a.resid_window,
        }
    }

    # ------------------------------------------------------- pod summary
    ann = R.mean() * 252
    vol = R.std() * np.sqrt(252)
    out["pods"] = [{
        "id": p, "type": ty.get(p, ""),
        "ann_ret": jsonable(ann[p]), "ann_vol": jsonable(vol[p]),
        "sharpe": jsonable(ann[p] / vol[p] if vol[p] else np.nan),
    } for p in pods]

    # ------------------------------------------------------------- pairs
    print("[1/4] factors and residual correlations ...", flush=True)
    F, share_cum, n_stk = stock_factors(a.indir, R.index, max(a.factors))
    out["meta"]["factor_names"] = list(F.columns)
    out["meta"]["n_stocks_factor"] = int(n_stk)

    ov, iu_o, p_ov = overlap_stats(holds, pods, rng, a.n_perm)
    keep_ov = benjamini_hochberg(p_ov, a.alpha)
    raw_corr = R.corr().to_numpy()

    # ------------------------------------------------------------- VaR
    # THE SAME REVERSAL, IN THE NUMBER FUNDS ACTUALLY REPORT.
    #
    # Everything else on this page is a crowding statistic, and a risk
    # committee could reasonably ask why it should care about a measure it does
    # not currently run. So the fund's Value at Risk is computed here and
    # decomposed by pod type, twice: on raw returns and on returns with common
    # factor exposure removed.
    #
    # Component VaR uses the standard identity
    #     component_i = w_i * cov(r_i, r_fund) / var(r_fund) * VaR_fund
    # which sums to the fund VaR exactly - the property that makes it
    # attributable. Note that ddof must match between the covariance and the
    # variance or the parts stop summing to the whole; numpy defaults differ
    # (cov is ddof=1, var is ddof=0) and mixing them scales every component by
    # n/(n-1).
    #
    # This is arithmetic and there is no AI in it. It is here because it tests
    # the project's claim against the industry's own instrument rather than
    # only against a correlation matrix.
    w_eq = np.repeat(1.0 / m, m)

    def _var_es(x, alpha):
        q = np.quantile(x, 1.0 - alpha)
        tail = x[x <= q]
        return -float(q), (-float(tail.mean()) if len(tail) else float("nan"))

    def _component(M, var_fund):
        rf = M.to_numpy() @ w_eq
        v = float(np.var(rf, ddof=1))
        if v <= 0:
            return pd.Series(0.0, index=M.columns)
        X = M.to_numpy() - M.to_numpy().mean(axis=0)
        y = rf - rf.mean()
        return pd.Series(w_eq * (X.T @ y / (len(y) - 1)) / v * var_fund,
                         index=M.columns)

    r_fund = R.to_numpy() @ w_eq
    var_f, es_f = _var_es(r_fund, 0.95)
    comp_raw = _component(R, var_f)
    E_var = residualise(R, F, max(a.factors), a.resid_window)
    var_r, _ = _var_es(E_var.to_numpy() @ w_eq, 0.95)
    comp_res = _component(E_var, var_r)

    var_types = []
    for t in ["crowded", "factor", "independent"]:
        mem = [p_ for p_ in pods if ty.get(p_) == t]
        if not mem:
            continue
        cap = len(mem) / m
        vs = float(comp_raw[mem].sum() / var_f)
        rs = float(comp_res[mem].sum() / var_r)
        var_types.append({"type": t, "n": len(mem), "cap_share": cap,
                          "var_share": vs, "var_per_cap": vs / cap,
                          "res_share": rs, "res_per_cap": rs / cap})

    rank_raw = [d["type"] for d in sorted(var_types,
                key=lambda d: -d["var_per_cap"])]
    rank_res = [d["type"] for d in sorted(var_types,
                key=lambda d: -d["res_per_cap"])]

    out["var"] = {
        "alpha": 0.95,
        "var": jsonable(var_f), "es": jsonable(es_f),
        "es_over_var": jsonable(es_f / var_f if var_f else np.nan),
        "worst_day": jsonable(-float(r_fund.min())),
        "nav": a.nav * m,
        "by_type": var_types,
        "rank_raw": rank_raw, "rank_res": rank_res,
        "rankings_disagree": rank_raw != rank_res,
        "k_factors": int(max(a.factors)),
    }

    out["pairs"] = {}
    out["separation"] = []
    for k in a.factors:
        E = residualise(R, F, k, a.resid_window)
        C, iu, p_res = pair_pvalues(E, a.n_boot, a.block, rng)
        keep_res = benjamini_hochberg(p_res, a.alpha)
        both = keep_res & keep_ov

        rows = []
        for t in range(len(iu[0])):
            i, j = iu[0][t], iu[1][t]
            s = {ty.get(pods[i]), ty.get(pods[j])}
            rows.append({
                "a": pods[i], "b": pods[j],
                "raw": jsonable(raw_corr[i, j]),
                "resid": jsonable(C[i, j]),
                "overlap": jsonable(ov[i, j]),
                "flagged": bool(both[t]),
                "true": (list(s)[0] if len(s) == 1 else "mixed"),
            })
        out["pairs"][str(k)] = rows

        by = {}
        for t_ in ["crowded", "factor", "independent", "mixed"]:
            sel = [r for r in rows if r["true"] == t_]
            if sel:
                by[t_] = {
                    "n": len(sel),
                    "raw": jsonable(np.mean([r["raw"] for r in sel])),
                    "resid": jsonable(np.mean([r["resid"] for r in sel])),
                    "flagged": int(sum(r["flagged"] for r in sel)),
                }
        out["separation"].append({"k": k,
                                  "factors": list(F.columns[:k]), "by_type": by})
        print(f"      k={k} done", flush=True)

    # -------------------------------------------------- concentration
    print("[2/4] fund-level concentration ...", flush=True)
    per_era, ts_rows = {}, []
    for e in eras:
        W, names = era_weights(holds, pods, e)
        st = fund_stats(W, a.min_pods)
        if st is None:
            continue
        per_era[e] = (W, names, st)
        ts_rows.append({"era": str(e), "hhi": jsonable(st["hhi"]),
                        "crowded_share": jsonable(st["crowded_share"]),
                        "top10": jsonable(st["top10_share"]),
                        "n_crowded": int(st["n_crowded_names"])})
    out["era_series"] = ts_rows

    pool = eligible_by_era(a.indir, eras)
    null_lad = []
    for _ in range(a.n_perm):
        lad = []
        for e in eras:
            if e not in per_era:
                continue
            W, names, _ = per_era[e]
            univ = pool.get(e)
            if univ is None or len(univ) < 4 * W.shape[0]:
                continue
            nl = int((W > 0).sum(axis=1).mean())
            ns = int((W < 0).sum(axis=1).mean())
            Wp = np.zeros((W.shape[0], len(univ)))
            for p_i in range(W.shape[0]):
                pick = rng.choice(len(univ), size=nl + ns, replace=False)
                Wp[p_i, pick[:nl]] = 0.5 / max(nl, 1)
                Wp[p_i, pick[nl:]] = -0.5 / max(ns, 1)
            st = fund_stats(Wp, a.min_pods)
            if st:
                lad.append(st["ladder"])
        if lad:
            null_lad.append({t: np.mean([d[t] for d in lad]) for t in range(2, 11)})

    out["ladder"] = []
    for t in range(2, 11):
        real = float(np.mean([per_era[e][2]["ladder"][t] for e in per_era]))
        nl = np.array([d[t] for d in null_lad])
        p = float((np.sum(nl >= real) + 1) / (len(nl) + 1))
        out["ladder"].append({"t": t, "real": jsonable(real),
                              "chance": jsonable(nl.mean()), "p": jsonable(p)})

    # --------------------------------------------------------- worst era
    # The SAME ranking pods4 uses. These two scripts briefly disagreed about
    # which era was "most crowded" - one counted names, the other weighted by
    # gross book - so the terminal and the dashboard named different victims
    # and different multiples for what was described as one test.
    ranked = [e for e in rank_eras(holds, eras, a.min_pods) if e in per_era]
    worst = ranked[0]
    W, names, st = per_era[worst]
    order = np.argsort(-st["long_press"])[:14]
    # who holds each crowded name, so the overlap is visible as a grid rather
    # than only as a summary statistic
    crowded_idx = [i for i in range(len(names)) if st["n_long"][i] >= a.min_pods]
    crowded_idx.sort(key=lambda i: -st["long_press"][i])
    crowded_idx = crowded_idx[:20]
    out["holdings_grid"] = {
        "era": str(worst),
        "names": [int(names[i]) for i in crowded_idx],
        "n_long": [int(st["n_long"][i]) for i in crowded_idx],
        "pods": pods,
        "types": [ty.get(p, "") for p in pods],
        # -1 short, 0 flat, 1 long
        "cells": [[int(np.sign(W[p_i, i])) for i in crowded_idx]
                  for p_i in range(len(pods))],
    }

    out["worst_era"] = {
        "era": str(worst),
        "names": [{"permno": int(names[i]), "n_long": int(st["n_long"][i]),
                   "long_press": jsonable(st["long_press"][i]),
                   "net": jsonable(st["net"][i]),
                   "gross": jsonable(st["gross"][i])} for i in order]
    }

    # ------------------------------------------------------------ unwind
    print("[3/4] unwind stress ...", flush=True)
    i_e = eras.index(worst)
    e0 = pd.Timestamp(worst)
    e1 = pd.Timestamp(eras[i_e + 1]) if i_e + 1 < len(eras) else e0 + pd.Timedelta(days=90)
    panel = pd.read_csv(os.path.join(a.indir, "panel_daily.csv"),
                        parse_dates=["date"],
                        usecols=["permno", "date", "ret_adj", "mktcap"])
    nvol, ncap = name_stats(panel, e0, e1)
    books = era_books(holds, pods, worst)
    victim, score = pick_victim(books, pods)

    sweep = []
    for coef in a.coefficients:
        pnl, _ = unwind(books, victim, pods, nvol, ncap, coef, a.turnover, a.nav)
        others = {p: v for p, v in pnl.items() if p != victim}
        cr = [v for p, v in others.items() if ty.get(p) == "crowded"]
        ind = [v for p, v in others.items() if ty.get(p) == "independent"]
        sweep.append({
            "coef": coef, "victim_pnl": jsonable(pnl[victim]),
            "others_total": jsonable(sum(others.values())),
            "worst_other": jsonable(min(others.values())),
            "n_hurt": int(sum(1 for v in others.values() if v < 0)),
            "gap": jsonable(np.mean(cr) - np.mean(ind) if cr and ind else np.nan),
        })

    mid = a.coefficients[len(a.coefficients) // 2]
    control = {}
    for v in pods:
        pnl_v, _ = unwind(books, v, pods, nvol, ncap, mid, a.turnover, a.nav)
        others_v = {p: x for p, x in pnl_v.items() if p != v}
        control.setdefault(ty.get(v, "?"), []).append({
            "total": sum(others_v.values()),
            "worst": min(others_v.values()),
            "n_hurt": sum(1 for x in others_v.values() if x < 0)})
    # the multiple across several eras, because one era's figure is not stable
    across = []
    for e in ranked[:a.n_eras]:
        i2 = eras.index(e)
        s0 = pd.Timestamp(e)
        s1 = pd.Timestamp(eras[i2 + 1]) if i2 + 1 < len(eras) else s0 + pd.Timedelta(days=90)
        v2, c2 = name_stats(panel, s0, s1)
        bk = era_books(holds, pods, e)
        agg = {}
        for v in pods:
            pl, _ = unwind(bk, v, pods, v2, c2, mid, a.turnover, a.nav)
            agg.setdefault(ty.get(v, "?"), []).append(
                sum(x for p, x in pl.items() if p != v))
        cr2 = float(np.mean(agg.get("crowded", [np.nan])))
        in2 = float(np.mean(agg.get("independent", [np.nan])))
        across.append({"era": str(e), "crowded": jsonable(cr2),
                       "independent": jsonable(in2),
                       "multiple": jsonable(abs(cr2 / in2) if in2 else np.nan)})

    out["unwind"] = {
        "across_eras": across,
        "era": str(worst), "victim": victim, "overlap_score": jsonable(score),
        "victim_true_type": ty.get(victim, ""),
        "coefficient_sweep": sweep, "control_coef": mid,
        "control": [{"type": t, "n": len(v),
                     "mean_damage": jsonable(np.mean([x["total"] for x in v])),
                     "worst": jsonable(np.mean([x["worst"] for x in v])),
                     "n_hurt": jsonable(np.mean([x["n_hurt"] for x in v]))}
                    for t, v in control.items()],
    }

    # -------------------------------------------------- the thesis layer
    # Optional: present only when pods6/pods7 have been run. The position
    # analysis above stands on its own without it.
    tp = "outputs/thesis_pairs.csv"
    tx = "outputs/thesis_extract.json"
    tc = "outputs/thesis_clusters.json"
    if os.path.exists(tp) and os.path.exists(tx):
        tpairs = pd.read_csv(tp)
        ext = json.load(open(tx))
        first_ex = {k: (v[0] if v else None) for k, v in ext.items()}

        groups = []
        if os.path.exists(tc):
            cl = json.load(open(tc))
            runs_c = cl.get("runs", [])
            if runs_c and runs_c[0]:
                order = sorted(runs_c[0].get("groups", []),
                               key=lambda g: -len(g.get("members", [])))
                names_c = sorted(ext.keys())
                for g in order:
                    mem = [names_c[i] for i in g.get("members", [])
                           if i < len(names_c)]
                    if not mem:
                        continue
                    groups.append({
                        "plain": g.get("plain", ""),
                        "members": mem,
                        "types": [ty.get(m, "") for m in mem],
                        "opposed": [names_c[i] for i in (g.get("opposed") or [])
                                    if i < len(names_c)],
                        "confidence": g.get("confidence", ""),
                    })

        cal_p = "outputs/thesis_calendar.csv"
        weeks = []
        if os.path.exists(cal_p):
            cal = pd.read_csv(cal_p, parse_dates=["date"])
            cal["week"] = cal["date"].dt.to_period("W").astype(str)
            wk = (cal.groupby("week")
                     .agg(pods=("pod", "nunique"), decisive=("decisive", "sum"),
                          n=("what", "size"))
                     .reset_index().sort_values("week"))
            weeks = [{"week": r["week"], "pods": int(r["pods"]),
                      "decisive": int(r["decisive"]), "n": int(r["n"])}
                     for _, r in wk.iterrows()]

        # the two axes, with the residual correlation each pair carries
        kk = res_pairs = None
        cr_path = "outputs/crowding_pairs.csv"
        merged = []
        if os.path.exists(cr_path):
            cr = pd.read_csv(cr_path)
            kk = int(cr["k"].max())
            cr = cr[cr["k"] == kk][["pod_a", "pod_b", "raw_corr", "resid_corr"]]
            mm = tpairs.merge(cr, left_on=["a", "b"],
                              right_on=["pod_a", "pod_b"], how="left")
        else:
            mm = tpairs.assign(raw_corr=np.nan, resid_corr=np.nan)
        jac_hi = max(float(tpairs["book_overlap"].quantile(0.9)), 0.05)
        for _, r in mm.iterrows():
            t = float(r["thesis_overlap"]) >= 0.5
            b = float(r["book_overlap"]) >= jac_hi
            merged.append({
                "a": r["a"], "b": r["b"],
                "thesis": jsonable(r["thesis_overlap"]),
                "book": jsonable(r["book_overlap"]),
                "raw": jsonable(r.get("raw_corr")),
                "resid": jsonable(r.get("resid_corr")),
                "true": r["type_a"] if r["type_a"] == r["type_b"] else "mixed",
                "cell": ("crowded" if (t and b) else "latent" if t
                         else "coincidental" if b else "independent"),
            })

        man_p = os.path.join(a.indir, "theses_manifest.csv")
        gen_by = ""
        if os.path.exists(man_p):
            gen_by = str(pd.read_csv(man_p)["generated_by"].iloc[0])

        out["thesis"] = {
            "groups": groups,
            "weeks": weeks,
            "pairs": merged,
            "k_factors": kk,
            "generated_by": gen_by,
            "n_runs": max((len(v) for v in ext.values()), default=1),
            "drivers": {p: (first_ex[p] or {}).get("driver_plain", "")
                        for p in first_ex},
        }
        print(f"      thesis layer: {len(groups)} groups, {len(weeks)} weeks, "
              f"{len(merged)} pairs", flush=True)

    # -------------------------------------------------------------- write
    print("[4/4] writing ...", flush=True)
    path = os.path.join(a.outdir, "orchestrator.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    mb = os.path.getsize(path) / 1e6
    print(f"\nwrote {path}  ({mb:.2f} MB)")
    print(f"  {m} pods, {n_pairs} pairs x {len(a.factors)} factor settings")
    print(f"  {len(ts_rows)} eras, ladder t=2..10, unwind over "
          f"{len(a.coefficients)} coefficients")
    print("\npreview locally:  python -m http.server 8000 --directory docs")


if __name__ == "__main__":
    main()
