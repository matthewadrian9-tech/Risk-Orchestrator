"""
pods7_thesis_risk.py
--------------------
Reads every pod's thesis, extracts a structured account of what each one is
actually betting on, and turns that into risk the position files cannot show.

THE GAP THIS FILLS

  pods2 finds pods holding the same names. pods3 aggregates that to the fund.
  pods4 prices what it costs when one is forced out. All three start from
  positions, and positions are where convergence ENDS.

  A discretionary pod writes before it trades. If five managers independently
  conclude the same thing about datacenter power supply and express it through
  five disjoint baskets, every tool upstream of this one is blind: no shared
  ticker, no overlap to measure, and the return correlation stays modest until
  the shared driver actually moves. The bet was one bet the whole time.

  Text is the only place that is visible early, which is why a language model
  is load-bearing here rather than decorative. There is no arithmetic on
  positions that recovers it.

WHAT IT PRODUCES

  1. DRIVER OVERLAP. Each thesis is reduced to a structured claim - the
     underlying driver, the mechanism, the direction, what would falsify it.
     Overlap is then computed on THOSE FIELDS, not on raw text similarity.
     Deliberate: two pods writing about semiconductors in similar prose may be
     making opposite bets, and a text-embedding cosine would score them as
     twins. Comparing extracted claims can distinguish them; comparing
     paragraphs cannot.

  2. THE TWO-AXIS MATRIX. Position overlap and thesis overlap are independent,
     and crossing them is the output a risk committee can act on:

                          books overlap        books do not
       theses overlap     crowded, someone     shared driver, latent -
                          has to cut           watch, do not force a cut
       theses do not      coincidence,         independent
                          usually fine

     The top-right cell is the one nothing else in this project can reach.

  3. EVENT CALENDAR. Catalysts and their dates, pulled from the text and
     stacked. Three pods waiting on the same week is concentration in TIME,
     which no exposure report shows because it is not an exposure.

  4. MANDATE DRIFT. Every thesis states its own mandate. Where the thesis or
     the expression contradicts the mandate the pod wrote for itself, that is
     a flag - and it is checkable, because both halves are in the same
     document.

WHAT KEEPS IT HONEST

  The extraction is a language model and a language model is a sampler. Run it
  twice and it can say different things, so --runs repeats the extraction and
  reports how much of the structured output is STABLE across runs. Anything
  that flips between runs is reported as run-dependent, not as a finding. This
  is the same discipline the Analog Engine applied to its red-team pass, and
  the same reason: an unstable audit is a coin flip with good vocabulary.

  The model never sees pods_truth.csv, theses_manifest.csv, or any label. It
  sees twenty documents.

  And the whole thing is validated against something external to itself: pods
  scoring high on thesis overlap should show elevated RESIDUAL return
  correlation - the correlation left after factor exposure is stripped out,
  which pods2 already computes. If they do, the text found something real. If
  they do not, the layer adds nothing and that is the finding to report.

Run:
  python scripts\\pods7_thesis_risk.py --dry-run
  python scripts\\pods7_thesis_risk.py --runs 3
  python scripts\\pods7_thesis_risk.py --offline    # reuse a saved extraction
"""

import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

MODEL = "claude-sonnet-4-6"

SYSTEM = """\
You are a risk analyst at a multi-manager hedge fund. You are given one \
portfolio manager's internal investment thesis. Reduce it to a structured \
claim so that theses from different managers can be compared with each other.

You are comparing BETS, not prose. Two managers can write about the same \
sector while betting in opposite directions, and two managers can describe one \
bet in completely different vocabulary. Extract what is being bet on and which \
way, not what the document is about.

Hard constraints:
- Use only what the document states. Do not infer a driver the manager did not \
describe, and do not import knowledge about any real company or market.
- Describe the driver in plain words. Do NOT try to invent a canonical label \
or slug: grouping happens in a later pass that can see every thesis at once, \
so your job here is an accurate description of THIS one, not a guess at what \
vocabulary another document might have used.
- mandate_consistent is false ONLY when the document contradicts itself in \
terms you can quote: the mandate states a limit and the thesis or expression \
states something that plainly breaks it. Sizing you cannot see is not a \
contradiction. A limit that "risks" being breached, or "implies" a breach, or \
that you "cannot confirm", is NOT a contradiction - return true. If your \
explanation needs the words implied, likely, risks, or cannot be confirmed, \
the answer is true. A fabricated flag is worse than a missed one, because a \
risk desk that receives three speculative flags stops reading the fourth.
- Dates: convert relative timing to an approximate ISO date where the document \
gives enough to do so, otherwise null. Do not guess a precise date from vague \
language.

Respond with JSON only, no prose outside it, no markdown fences:

{
  "driver_plain": "<one sentence: the underlying cause being bet on>",
  "direction": "long_the_driver" | "short_the_driver",
  "mechanism": "<one sentence: how the driver reaches the P&L>",
  "horizon_quarters": <number>,
  "conviction_stated": "high" | "medium" | "low",
  "catalysts": [
    {"what": "<short>", "approx_date": "<YYYY-MM-DD or null>", "decisive": true|false}
  ],
  "falsifier": "<what the manager says would prove the thesis wrong>",
  "mandate_summary": "<one sentence on the stated constraints>",
  "mandate_consistent": true|false,
  "mandate_quote_limit": "<if false, the exact mandate clause breached; else empty>",
  "mandate_quote_breach": "<if false, the exact clause that breaks it; else empty>",
  "mandate_note": "<if false, one sentence naming the contradiction; else empty>"
}"""


def call_anthropic(system, user_text, temperature=None, max_tokens=1500):
    import urllib.error
    import urllib.request
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "No ANTHROPIC_API_KEY found.\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "Or --dry-run to print the prompt, or --offline to reuse a saved extraction."
        )
    body = {"model": MODEL, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": user_text}]}
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API error {e.code}: {e.read().decode()[:400]}")

    text = "".join(b.get("text", "") for b in data.get("content", []))

    # A RESPONSE THAT HIT THE CAP IS NOT A MALFORMED RESPONSE, and reporting
    # them the same way hides which one happened. A truncated reply arrives as
    # a JSON object cut off mid-object; parse_json returns None; the run then
    # prints "unparseable JSON" and the reader concludes the model cannot
    # follow the format. It could, and it did - it ran out of room.
    #
    # The two need opposite responses: raise max_tokens, or fix the prompt.
    # This is the same error the Analog Engine recorded as #7, where a 2,000
    # token cap silently truncated red-team output into invalid JSON and the
    # determinism harness was what caught it.
    if data.get("stop_reason") == "max_tokens":
        raise SystemExit(
            f"The model hit the {max_tokens}-token cap and the reply is cut "
            f"off mid-sentence.\n"
            f"  This is NOT a formatting failure - the JSON is truncated, not "
            f"malformed.\n"
            f"  Raise max_tokens for this call, or shorten the input.\n"
            f"  Last 120 characters received: ...{text[-120:]}")
    return text


def parse_json(raw):
    c = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(c)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", c, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None



CLUSTER_SYSTEM = """\
You are a risk analyst at a multi-manager hedge fund. You are given a numbered \
list of the bets each portfolio manager is making, one line per manager, each \
already reduced from their written thesis.

Group the managers who are betting on THE SAME UNDERLYING CAUSE. Two managers \
belong together when the same event or condition in the world would make both \
of them right, or both of them wrong - regardless of how differently they \
describe it, which sector they express it in, or which names they hold.

Judgements you must get right:
- Different vocabulary, same cause -> SAME group. One manager writing about \
transformer lead times and another about grid interconnection queues may be \
making one bet.
- Same sector, different cause -> DIFFERENT groups. Two managers in \
semiconductors, one betting on a supply shortage and one on an inventory \
correction, are not the same bet.
- Opposite sides of one cause -> SAME group. Note them as opposed; they are \
each other's hedge, and a risk desk needs to see both.
- A manager whose cause is genuinely their own -> a group of one. Do not force \
everyone into a group. Over-grouping produces false crowding alerts, which is \
the failure mode that gets a tool ignored.

Respond with JSON only, no prose outside it, no markdown fences:

{
  "groups": [
    {
      "group_id": "<short lowercase slug>",
      "plain": "<one sentence naming the shared cause>",
      "members": [<the numbers of the managers in this group>],
      "opposed": [<numbers of members betting AGAINST the cause, subset of members>],
      "confidence": "high" | "medium" | "low"
    }
  ]
}"""


def cluster_drivers(first, pods, temperature, sleep, runs):
    """Group pods by shared driver, comparing all theses in one pass.

    THE FIRST VERSION OF THIS DID NOT EXIST, and that was the design error.
    Each thesis was extracted in isolation and asked to emit a canonical slug
    that would coincidentally match the slug another isolated call invented for
    a different document. That is exact-string agreement over an unbounded
    vocabulary, and it failed exactly as it had to: five pods describing one
    datacenter bet produced five different slugs, and the label was stable
    across repeat runs on only 8 of 20 pods. Meanwhile `direction`, a closed
    choice between two values, was stable on 20 of 20.

    The task is comparison, not naming. Shown all twenty descriptions at once,
    the model is deciding whether two stated causes are the same cause - which
    is a question with a defensible answer - instead of guessing a shared
    vocabulary it has no way to coordinate on.
    """
    # A FAILED EXTRACTION MUST NOT VOTE. The earlier version wrote
    # `d = first[p] or {}` and sent the pod on regardless, which rendered as
    # "driver: ? | mechanism: ? | direction: ? | horizon: ? quarters". Two
    # failures become two IDENTICAL rows, and a model asked to group managers
    # by shared cause has every reason to put them together - so a pair of
    # parse errors manufactures a crowding alert. A false positive produced by
    # an error path is the worst kind, because it arrives looking like a
    # finding.
    #
    # Failed pods are dropped and named. The indices the model sees are indices
    # into the SURVIVING list, so `kept` is returned to map them back.
    kept = [p for p in pods if first.get(p)]
    dropped = [p for p in pods if not first.get(p)]
    if dropped:
        print(f"  dropping {len(dropped)} pod(s) with no usable extraction "
              f"from the clustering input: {', '.join(dropped)}")
        print("  (a pod with no claim cannot be grouped by claim; leaving it in")
        print("   would let two failures group with each other)")
    if len(kept) < 2:
        print("  fewer than two usable extractions - clustering skipped")
        return [], ""

    lines = []
    for i, p in enumerate(kept):
        d = first[p]
        lines.append(
            f"{i}. driver: {d.get('driver_plain','?')} | mechanism: "
            f"{d.get('mechanism','?')} | direction: {d.get('direction','?')} | "
            f"horizon: {d.get('horizon_quarters','?')} quarters")
    user = ("The managers:\n\n" + "\n".join(lines)
            + "\n\nGroup them by shared underlying cause.")

    out = []
    for r in range(runs):
        obj = parse_json(call_anthropic(CLUSTER_SYSTEM, user, temperature, 2500))
        out.append(obj)
        if r < runs - 1:
            time.sleep(sleep)
    return out, user, kept


def assignment(obj, kept):
    """-> {pod_name: set(group_ids)}, or None.

    Keyed by NAME rather than index: the model is shown only the pods with a
    usable extraction, so its index 3 is the fourth SURVIVOR, not the fourth
    pod. `kept` maps back. An index-keyed dict would misalign silently the
    moment one extraction failed.

    ONE LABEL PER POD WAS WRONG AND SILENTLY SO. Nothing in the clustering
    prompt forbids a manager from appearing in two groups, and a manager
    genuinely can be running two bets. The earlier version wrote
    `m[i] = gid`, so a second group containing pod i overwrote the first -
    and the group table renders the model's groups directly while the pair
    logic read this dict, so one artifact answered the same question two
    ways. That is the same failure as pods4 and pods5 ranking eras by
    different definitions.

    Membership is a set. Two pods share a bet when their sets intersect.
    """
    if not obj:
        return None
    m = {}
    for g in obj.get("groups", []):
        gid = str(g.get("group_id", ""))
        for i in g.get("members", []):
            try:
                i = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(kept):
                m.setdefault(kept[i], set()).add(gid)
    return m


def partition_agreement(assigns, names):
    """Share of PAIRS placed together-or-apart identically across runs.

    Comparing group LABELS across runs would repeat the original mistake -
    run 2 may call a group something else and be right. What is comparable is
    the partition: for each of the n(n-1)/2 pairs, same group or not. That is
    a closed binary decision per pair, so it can be measured.
    """
    good = [a for a in assigns if a]
    if len(good) < 2:
        return None, 0
    same = 0
    tot = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p, q = names[i], names[j]
            # "together" now means the two pods share at least one group.
            # Still a closed binary decision per pair, so still comparable.
            vals = [bool(a.get(p) and a.get(q) and (a[p] & a[q])) for a in good]
            tot += 1
            same += len(set(vals)) == 1
    return same / tot, tot


# --------------------------------------------------------------- overlap

def thesis_overlap(a, b, ga=None, gb=None):
    """Similarity between two extracted claims, on fields not prose.

    Same driver in the same direction is the whole signal; everything else is
    a small adjustment. Same driver in OPPOSITE directions scores near zero on
    purpose - those two pods are each other's hedge, not each other's crowd,
    and a text-similarity score would call them identical.
    """
    if not a or not b:
        return 0.0
    # ga and gb are SETS of group ids. Two pods share a driver when the sets
    # intersect, not when a single label matches.
    same_driver = bool(ga and gb and (set(ga) & set(gb)))
    same_dir = a.get("direction") == b.get("direction")
    if not same_driver:
        return 0.0
    if not same_dir:
        return 0.0
    s = 0.75
    ha, hb = a.get("horizon_quarters"), b.get("horizon_quarters")
    if isinstance(ha, (int, float)) and isinstance(hb, (int, float)):
        s += 0.10 * max(0.0, 1.0 - abs(ha - hb) / 4.0)
    ca = {c.get("what", "").lower()[:18] for c in (a.get("catalysts") or [])}
    cb = {c.get("what", "").lower()[:18] for c in (b.get("catalysts") or [])}
    if ca and cb:
        s += 0.15 * len(ca & cb) / len(ca | cb)
    return round(min(s, 1.0), 3)


def stability(runs_by_pod, field):
    """Share of pods where `field` is identical across every run."""
    ok = 0
    tot = 0
    for pod, runs in runs_by_pod.items():
        vals = [r.get(field) for r in runs if r]
        if len(vals) < 2:
            continue
        tot += 1
        ok += len(set(json.dumps(v, sort_keys=True) for v in vals)) == 1
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--thesis-dir", default="data/processed/theses")
    ap.add_argument("--runs", type=int, default=2,
                    help="extractions per thesis; >1 measures stability")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="reuse outputs/thesis_extract.json")
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--as-of", default=None,
                    help="date the books are as of, YYYY-MM-DD. Relative timing "
                         "in a thesis resolves forward from it. Defaults to the "
                         "last era in pods_holdings.csv. Without an anchor the "
                         "extraction guesses the year and the catalyst calendar "
                         "stops being reproducible.")
    a = ap.parse_args()

    truth = pd.read_csv(os.path.join(a.indir, "pods_truth.csv"))
    ty = dict(zip(truth["pod"], truth["type"]))
    files = sorted(f for f in os.listdir(a.thesis_dir) if f.endswith(".md"))
    pods = [f[:-3] for f in files]
    texts = {p: open(os.path.join(a.thesis_dir, p + ".md"), encoding="utf-8").read()
             for p in pods}
    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------- the timing anchor
    # A MEMO CARRIES ITS YEAR ON THE LETTERHEAD, NOT IN THE SENTENCE. A thesis
    # saying "December" or "late January" is unambiguous to whoever wrote it
    # and ambiguous to anything reading it afterwards, so with no reference
    # date the extraction has to guess the year - and guesses differently on
    # different runs. Two runs over identical theses in the browser produced
    # different calendars, one placing catalysts BEFORE the books existed,
    # while the pairs and the grouping stayed stable across those same runs.
    # Only the calendar drifted, which is why it survived several
    # reproducibility checks.
    #
    # The anchor is the fix; a firmer instruction is not. The prompt already
    # forbids guessing a date from vague language, and "December" does not
    # read as vague.
    as_of = a.as_of
    if as_of is None:
        _h = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"),
                         usecols=["era"])
        as_of = str(sorted(_h["era"].unique())[-1])[:10]
    date_note = (
        f"These positions are as of {as_of}. Relative timing in the document "
        f'("December", "next quarter", "late January") is relative to that '
        f"date and resolves forward from it. If a month is named with no year, "
        f"take the next occurrence of it after {as_of}. If the timing is "
        f"genuinely vague, return null rather than choosing a date.\n\n---\n\n")
    print(f"as of      : {as_of}  (catalyst dates resolve forward from this)")

    if a.dry_run:
        print("=" * 72)
        print("SYSTEM")
        print("=" * 72)
        print(SYSTEM)
        print("\n" + "=" * 72)
        print(f"USER  ({pods[0]})")
        print("=" * 72)
        print(date_note + texts[pods[0]])
        return

    # ------------------------------------------------------- extraction
    store = "outputs/thesis_extract.json"
    if a.offline:
        if not os.path.exists(store):
            raise SystemExit(f"{store} not found - run once without --offline.")
        runs_by_pod = json.load(open(store))
        n_runs = max(len(v) for v in runs_by_pod.values())
        print(f"reusing {store}  ({len(runs_by_pod)} pods, {n_runs} runs)\n")
    else:
        print(f"model      : {MODEL}")
        print(f"runs       : {a.runs} per thesis  ({len(pods)*a.runs} calls)")
        print(f"temperature: {'pipeline default' if a.temperature is None else a.temperature}\n")
        runs_by_pod, bad = {}, 0
        for p in pods:
            runs_by_pod[p] = []
            for r in range(a.runs):
                raw = call_anthropic(SYSTEM, date_note + texts[p], a.temperature)
                obj = parse_json(raw)
                if obj is None:
                    bad += 1
                runs_by_pod[p].append(obj)
                time.sleep(a.sleep)
            d = runs_by_pod[p][0]
            ok = d is not None
            print(f"  {p:<10} {'ok  ' if ok else 'PARSE FAIL'} "
                  f"{(d or {}).get('direction',''):<18} "
                  f"{(d or {}).get('driver_plain','')[:44]}", flush=True)
        json.dump(runs_by_pod, open(store, "w"), indent=1)
        if bad:
            print(f"\n  {bad} extraction(s) returned unparseable JSON - itself a "
                  f"reproducibility finding, reported rather than retried.")

    first = {p: (v[0] if v else None) for p, v in runs_by_pod.items()}

    # -------------------------------------------------------- stability
    if max(len(v) for v in runs_by_pod.values()) > 1:
        print("\n" + "=" * 74)
        print("IS THE EXTRACTION REPRODUCIBLE?")
        print("=" * 74)
        for f in ["direction", "mandate_consistent", "conviction_stated"]:
            ok, tot = stability(runs_by_pod, f)
            print(f"  {f:<22} identical across runs on {ok}/{tot} pods"
                  + ("" if ok == tot else "   <- unstable, report as run-dependent"))

    # ------------------------------------------------------- clustering
    cstore = "outputs/thesis_clusters.json"
    if a.offline and os.path.exists(cstore):
        cl = json.load(open(cstore))
        cruns, cuser = cl["runs"], cl["prompt"]
        # older caches predate `kept`; fall back to every pod, which is what
        # they were written under
        ckept = cl.get("kept") or list(pods)
    else:
        print("\n  grouping by shared driver ...", flush=True)
        cruns, cuser, ckept = cluster_drivers(first, pods, a.temperature,
                                              a.sleep, max(2, a.runs))
        json.dump({"runs": cruns, "prompt": cuser, "kept": ckept},
                  open(cstore, "w"), indent=1)

    assigns = [assignment(o, ckept) for o in cruns]
    agree, npair = partition_agreement(assigns, ckept)
    grp = assigns[0] or {}

    multi = sorted(p for p, g in grp.items() if len(g) > 1)
    if multi:
        print(f"\n  {len(multi)} manager(s) appear in more than one group: "
              f"{', '.join(multi)}")
        print("  Nothing forbids it and a manager can be running two bets, so")
        print("  membership is a set and two pods share a bet when their sets")
        print("  intersect. An earlier version kept one label per pod, which")
        print("  silently dropped every group but the last.")
    if len(ckept) < len(pods):
        print(f"\n  the grouping ran on {len(ckept)} of {len(pods)} pods - "
              f"read the partition as partial, not complete")

    print("\n" + "=" * 74)
    print("IS THE GROUPING REPRODUCIBLE?")
    print("=" * 74)
    if agree is None:
        print("  fewer than two valid clustering runs - cannot measure")
    else:
        print(f"  {agree:.1%} of {npair} pairs placed together-or-apart identically")
        print(f"  across {len([x for x in assigns if x])} runs.")
        print()
        print("  Note what is measured: the PARTITION, not the labels. Comparing")
        print("  group names across runs would repeat the error this pass was")
        print("  built to fix - run two may name a group differently and still")
        print("  be right. Whether two pods sit together is a closed question")
        print("  and can be scored; what the group is called cannot.")

    print("\n" + "=" * 74)
    print("WHAT THE FUND IS ACTUALLY BETTING ON")
    print("=" * 74)
    g0 = (cruns[0] or {}).get("groups", [])
    for g in sorted(g0, key=lambda x: -len(x.get("members", []))):
        mem = [ckept[i] for i in g.get("members", []) if i < len(ckept)]
        if not mem:
            continue
        tag = Counter(ty.get(p, "?") for p in mem)
        opp = [ckept[i] for i in (g.get("opposed") or []) if i < len(ckept)]
        print(f"  {g.get('plain','?')[:58]:<60} {len(mem):>2} pods  "
              f"{', '.join(f'{k}:{v}' for k, v in tag.items())}"
              + (f"   opposed: {', '.join(opp)}" if opp else ""))

    # ------------------------------------------------ the two-axis matrix
    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    era = sorted(holds["era"].unique())[-1]
    h = holds[(holds["era"] == era) & (holds["side"] == "long")]
    books = {p: set(h[h["pod"] == p]["permno"]) for p in pods}

    rows = []
    for i in range(len(pods)):
        for j in range(i + 1, len(pods)):
            p, q = pods[i], pods[j]
            ov = thesis_overlap(first[p], first[q], grp.get(p), grp.get(q))
            A, B = books.get(p, set()), books.get(q, set())
            jac = len(A & B) / len(A | B) if A and B else 0.0
            rows.append({"a": p, "b": q, "thesis_overlap": ov,
                         "book_overlap": round(jac, 4),
                         "type_a": ty.get(p, ""), "type_b": ty.get(q, "")})
    pairs = pd.DataFrame(rows)
    pairs.to_csv("outputs/thesis_pairs.csv", index=False)

    jac_hi = pairs["book_overlap"].quantile(0.9)
    def cell(r):
        t = r["thesis_overlap"] >= a.min_overlap
        b = r["book_overlap"] >= max(jac_hi, 0.05)
        return ("crowded" if (t and b) else "latent" if (t and not b)
                else "coincidental" if (b and not t) else "independent")
    pairs["cell"] = pairs.apply(cell, axis=1)

    print("\n" + "=" * 74)
    print("THE TWO AXES  (thesis overlap against book overlap)")
    print("=" * 74)
    counts = pairs["cell"].value_counts().to_dict()
    print(f"  {'':<26}{'books overlap':>16}{'books do not':>16}")
    print(f"  {'theses overlap':<26}{counts.get('crowded',0):>16}{counts.get('latent',0):>16}")
    print(f"  {'theses do not':<26}{counts.get('coincidental',0):>16}{counts.get('independent',0):>16}")

    lat = pairs[pairs["cell"] == "latent"]
    print(f"\n  {len(lat)} pair(s) share a thesis with no meaningful book overlap.")
    if len(lat):
        same = int((lat["type_a"] == lat["type_b"]).sum())
        print(f"  {same} of them are same-type pairs. Ground truth says the factor")
        print("  pods were briefed on one driver and hold NO name in common, so")
        print("  those pairs are invisible to every position-based tool here.")
        print("\n" + lat.head(12)[["a", "b", "thesis_overlap",
                                   "book_overlap", "type_a"]].to_string(index=False))

    # ------------------------------------------------------- event risk
    cal = []
    for p in pods:
        for c in ((first[p] or {}).get("catalysts") or []):
            d = c.get("approx_date")
            if not d:
                continue
            try:
                d = pd.Timestamp(d)
            except Exception:
                continue
            cal.append({"pod": p, "type": ty.get(p, ""), "date": d,
                        "what": c.get("what", ""),
                        "decisive": bool(c.get("decisive"))})
    cal = pd.DataFrame(cal)
    print("\n" + "=" * 74)
    print("CONCENTRATION IN TIME")
    print("=" * 74)
    if cal.empty:
        print("  No dated catalysts extracted.")
    else:
        cal.to_csv("outputs/thesis_calendar.csv", index=False)
        cal["week"] = cal["date"].dt.to_period("W").astype(str)
        wk = cal.groupby("week").agg(pods=("pod", "nunique"),
                                     decisive=("decisive", "sum")).reset_index()
        wk = wk.sort_values("pods", ascending=False)
        print(f"  {len(cal)} dated catalysts across {cal['pod'].nunique()} pods")
        print(f"\n  {'week':<26}{'pods waiting':>14}{'decisive':>10}")
        for _, r in wk.head(6).iterrows():
            print(f"  {r['week']:<26}{r['pods']:>14}{int(r['decisive']):>10}")
        top = wk.iloc[0]
        if top["pods"] >= 3:
            print(f"\n  {int(top['pods'])} pods are waiting on the same week. That is")
            print("  concentration no exposure report shows, because it is not an")
            print("  exposure - the books can be perfectly diversified and still")
            print("  resolve together on one date.")

    # ----------------------------------------------------- mandate drift
    # A flag whose own note concedes the evidence is missing is not a flag.
    # The prompt forbids speculative drift and the model raised it anyway, so
    # the rule is enforced here as well: a contradiction has to quote both the
    # limit and the clause that breaks it, and must not hedge in the note.
    HEDGE = ("implies", "implied", "likely", "risks", "cannot be confirmed",
             "does not provide", "may exceed", "press against", "rests solely",
             "not mention", "unaddressed", "plausible")
    drift, dropped = [], []
    for p in pods:
        d = first[p]
        if not d or d.get("mandate_consistent", True):
            continue
        note = str(d.get("mandate_note", ""))
        lim = str(d.get("mandate_quote_limit", "")).strip()
        brk = str(d.get("mandate_quote_breach", "")).strip()
        if not lim or not brk or any(h in note.lower() for h in HEDGE):
            dropped.append((p, note))
        else:
            drift.append((p, lim, brk, note))
    print("\n" + "=" * 74)
    print("MANDATE DRIFT")
    print("=" * 74)
    for p, lim, brk, note in drift:
        print(f"  {p:<10} limit:  {lim[:90]}")
        print(f"  {'':<10} breach: {brk[:90]}")
        print(f"  {'':<10} {note[:90]}\n")
    if not drift:
        print("  No pod's thesis contradicts a limit it stated for itself, in")
        print("  terms that can be quoted from both sides.")
    if dropped:
        print(f"\n  {len(dropped)} flag(s) raised by the model and DROPPED here, because")
        print("  the justification hedged or one of the two quotes was missing:")
        for p, note in dropped:
            print(f"    {p:<10} {note[:96]}")
        print()
        print("  Worth keeping in the output rather than hiding. The prompt")
        print("  forbids speculative drift explicitly and the model produced it")
        print("  anyway, which is a fact about what prompting alone can enforce.")
        print("  A risk desk sent three speculative flags stops reading the")
        print("  fourth, so the filter is applied in code as well as in words.")

    # -------------------------------------------------------- validation
    print("\n" + "=" * 74)
    print("DOES THESIS OVERLAP PREDICT ANYTHING?  <- the test that matters")
    print("=" * 74)
    cp = "outputs/crowding_pairs.csv"
    if not os.path.exists(cp):
        print(f"  {cp} not found - run pods2_crowding.py first.")
    else:
        cr = pd.read_csv(cp)
        k = cr["k"].max()
        cr = cr[cr["k"] == k][["pod_a", "pod_b", "resid_corr", "raw_corr"]]
        m = pairs.merge(cr, left_on=["a", "b"], right_on=["pod_a", "pod_b"])
        if m.empty:
            print("  no overlapping pairs - check pod naming between the two files")
        else:
            hi = m[m["thesis_overlap"] >= a.min_overlap]
            lo = m[m["thesis_overlap"] < a.min_overlap]
            print(f"  factor model: k={k} factors removed, {len(m)} pairs matched\n")
            print(f"  {'':<34}{'pairs':>7}{'raw corr':>11}{'residual':>11}")
            print(f"  {'thesis overlap >= '+str(a.min_overlap):<34}{len(hi):>7}"
                  f"{hi['raw_corr'].mean():>11.3f}{hi['resid_corr'].mean():>11.3f}")
            print(f"  {'thesis overlap below':<34}{len(lo):>7}"
                  f"{lo['raw_corr'].mean():>11.3f}{lo['resid_corr'].mean():>11.3f}")

            lat_m = m[(m["thesis_overlap"] >= a.min_overlap)
                      & (m["book_overlap"] < max(jac_hi, 0.05))]
            if len(lat_m):
                print(f"\n  Of those, {len(lat_m)} share a thesis with NO book overlap:")
                print(f"    mean raw correlation      {lat_m['raw_corr'].mean():+.3f}")
                print(f"    mean residual correlation {lat_m['resid_corr'].mean():+.3f}")
                print("\n  Read this carefully. High RAW correlation with a shared")
                print("  thesis and no shared book is what a common factor looks")
                print("  like - the text found the exposure, which is useful and")
                print("  is NOT crowding. Elevated RESIDUAL correlation would be")
                print("  the stronger claim: a shared bet that survives stripping")
                print("  out the factors, held through different names.")
            if len(hi) and len(lo):
                d = hi["resid_corr"].mean() - lo["resid_corr"].mean()
                print(f"\n  residual gap, high vs low thesis overlap: {d:+.3f}")
                if abs(d) < 0.02:
                    print("  Essentially nothing. On this corpus, thesis overlap does")
                    print("  not predict residual co-movement, and the honest report")
                    print("  is that the text layer surfaces shared drivers but adds")
                    print("  no forecasting power over the position tools.")
    pairs.to_csv("outputs/thesis_pairs.csv", index=False)
    print("\nwrote outputs/thesis_pairs.csv, thesis_extract.json"
          + (", thesis_calendar.csv" if not cal.empty else ""))


if __name__ == "__main__":
    main()
