"""
pods6_theses.py
---------------
Writes an investment thesis for every pod, in the register a discretionary PM
actually writes in, grounded in that pod's real holdings.

WHY THIS FILE EXISTS

  Everything upstream reasons about positions. Positions are the LAST thing to
  converge. A pod shop's pods write before they trade: a mandate defining what
  they may hold, a thesis for each position, a monthly letter explaining what
  moved. Polymer runs discretionary fundamental long/short teams, so those
  documents exist by construction - a discretionary manager who cannot write
  down why they are long has nothing to take to a risk committee.

  The failure this makes visible: five pods can hold five disjoint baskets and
  be making one bet. One is long the equipment maker, one the memory supplier,
  one the power infrastructure name. No shared ticker, no position overlap, and
  a correlation matrix sees very little until the shared driver actually moves.
  The convergence happened in the reasoning, months before it reached the book.

THE HONEST PROBLEM WITH GENERATING THESE

  Real PM theses are confidential and no student has them. So they are
  generated - and that creates a trap worth naming loudly, because it would
  quietly invalidate everything downstream:

      If a template writes the theses and a model reads them back, the
      detector is recovering the template, not detecting anything.

  Two defences are built in here.

  First, generation goes through a language model at high temperature with only
  a driver and a holdings list - never the phrasing another pod used. Pods
  sharing a driver receive the same brief and independently choose their own
  words. That is the actual task: recognising one bet described five ways.

  Second, --fake produces deterministic template text for plumbing tests. It is
  named so that no result from it can be mistaken for a finding. Any number
  reported from a --fake corpus is a test of the code, not of the method.

WHAT GROUND TRUTH LOOKS LIKE HERE

  pods_truth.csv already says which pods were built crowded. This adds the
  driver assignment: which pods were briefed on the same underlying bet. The
  two are deliberately not the same partition.

    crowded pods     one shared basket AND one shared driver
                     -> overlap in both positions and theses
    factor pods      one shared driver, disjoint holdings by construction
                     -> overlap in theses ONLY. This is the cell the position
                        tools cannot reach, and the reason for the text layer.
    independent      a driver of their own each

  So the scorecard is not "did it find similar theses". It is whether the
  method separates shared-driver-with-shared-book from shared-driver-with-no-
  shared-book, because those need opposite remedies.

Run:
  python scripts\\pods6_theses.py --dry-run          # print a prompt, no API call
  python scripts\\pods6_theses.py --fake             # deterministic, for plumbing
  python scripts\\pods6_theses.py                    # needs ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

MODEL = "claude-sonnet-4-6"

# Drivers a discretionary Asia-focused long/short desk might actually be
# running. Wording here is the BRIEF, not the thesis - the model writes the
# thesis, so pods sharing a driver do not share phrasing.
DRIVERS = {
    "datacenter_buildout": "accelerating capital expenditure on datacenter and "
        "power infrastructure, with the position expressed through suppliers "
        "rather than the obvious end names",
    "china_stimulus": "a policy-driven recovery in Chinese domestic demand "
        "following expected fiscal support, expressed in cyclicals",
    "japan_governance": "corporate governance reform in Japan forcing balance "
        "sheet efficiency, cross-shareholding unwinds and buybacks",
    "rate_sensitivity": "a turn in the rate cycle favouring duration-sensitive "
        "and highly cyclical equities over defensives",
    "supply_normalisation": "post-shortage normalisation in component supply "
        "chains compressing margins for intermediaries",
    "consumer_downtrade": "sustained consumer trading-down behaviour favouring "
        "value retail formats over premium",
    "energy_transition": "grid investment and transmission constraints as the "
        "binding limit on renewable deployment",
    "healthcare_pricing": "reimbursement pressure compressing margins across "
        "branded pharmaceutical distribution",
    "logistics_capacity": "freight capacity coming back online faster than "
        "demand recovers",
    "financials_credit": "credit normalisation in regional banks after a "
        "period of unusually low provisioning",
}

SYSTEM = """\
You are a portfolio manager at a multi-manager hedge fund writing the internal \
investment thesis for your book. You are writing for your own risk committee, \
not for clients.

Write the way a discretionary equity long/short PM actually writes: specific, \
compressed, unsentimental, willing to state what would prove you wrong. No \
marketing language, no hedging every sentence, no bullet-point deck prose.

Cover, in continuous prose under the headings given:

MANDATE - one sentence on what this book is allowed to do and how it is \
constrained.
THESIS - the core bet in three or four sentences. What has to be true.
EXPRESSION - why these particular names express it, and why not the obvious \
ones.
CATALYSTS - the specific events you are waiting for, each with an approximate \
timing. Invent plausible dates within the next two quarters.
RISKS - what breaks this, stated as something checkable.

Hard constraints:
- 220 to 300 words total. This is an internal memo, not an essay.
- Refer to holdings by the identifiers given. Do not invent company names, \
tickers, or real firms.
- Do not use the exact phrasing of the driver brief you were given. Write it \
as you would say it. Two managers briefed on the same macro view should sound \
like two different people, because they are.
- No preamble, no sign-off. Start at MANDATE."""


def fake_thesis(pod, driver, longs, shorts, rng):
    """Deterministic template text. NOT for measurement - see the docstring."""
    d = DRIVERS[driver].split(",")[0]
    return (
        f"MANDATE\nThis book runs {len(longs)} long and {len(shorts)} short "
        f"positions, market neutral, sized to a common volatility target.\n\n"
        f"THESIS\nThe book is positioned for {d}. The market is discounting "
        f"the persistence of this and we think the adjustment happens over two "
        f"to three quarters rather than at once.\n\n"
        f"EXPRESSION\nLongs are {', '.join(str(x) for x in longs[:6])} and "
        f"others. Shorts are {', '.join(str(x) for x in shorts[:4])}. The "
        f"expression avoids the crowded end of the trade.\n\n"
        f"CATALYSTS\nQuarterly results in the next reporting season; a policy "
        f"decision expected mid-quarter; a supply datapoint in the following "
        f"month.\n\nRISKS\nIf the driver does not appear in reported numbers "
        f"within two quarters the position is wrong and should be cut."
    )


def call_anthropic(system, user_text, temperature, max_tokens=1200):
    import urllib.error
    import urllib.request

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "No ANTHROPIC_API_KEY found.\n"
            "  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "Or use --dry-run to print the prompts, or --fake for plumbing tests."
        )
    body = {
        "model": MODEL, "max_tokens": max_tokens, "system": system,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_text}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API error {e.code}: {e.read().decode()[:400]}")
    return "".join(b.get("text", "") for b in data.get("content", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/processed")
    ap.add_argument("--outdir", default="data/processed/theses")
    ap.add_argument("--era", default=None, help="default: the last era")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="high on purpose: pods sharing a driver must not share wording")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", action="store_true",
                    help="deterministic template text, for plumbing only")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    holds = pd.read_csv(os.path.join(a.indir, "pods_holdings.csv"))
    truth = pd.read_csv(os.path.join(a.indir, "pods_truth.csv"))
    ty = dict(zip(truth["pod"], truth["type"]))
    pods = sorted(holds["pod"].unique())
    era = a.era or sorted(holds["era"].unique())[-1]
    rng = np.random.default_rng(a.seed)

    # ---- assign drivers so the two axes come apart on purpose
    names = list(DRIVERS)
    crowded = [p for p in pods if ty.get(p) == "crowded"]
    factor = [p for p in pods if ty.get(p) == "factor"]
    indep = [p for p in pods if ty.get(p) == "independent"]

    shared_crowd = "datacenter_buildout"
    shared_factor = "rate_sensitivity"
    rest = [n for n in names if n not in (shared_crowd, shared_factor)]
    rng.shuffle(rest)

    driver = {}
    for p in crowded:
        driver[p] = shared_crowd     # same basket AND same story
    for p in factor:
        driver[p] = shared_factor    # same story, provably disjoint books
    for i, p in enumerate(indep):
        driver[p] = rest[i % len(rest)]

    os.makedirs(a.outdir, exist_ok=True)
    h = holds[holds["era"] == era]

    print(f"era        : {era}")
    print(f"pods       : {len(pods)}")
    print(f"drivers    : {shared_crowd} x{len(crowded)} (crowded), "
          f"{shared_factor} x{len(factor)} (factor), "
          f"{len(set(driver[p] for p in indep))} distinct (independent)")
    print(f"mode       : "
          + ("dry run" if a.dry_run else "TEMPLATE (not for measurement)"
             if a.fake else f"{MODEL} at temperature {a.temperature}") + "\n")

    rows = []
    for p in pods:
        hp = h[h["pod"] == p]
        longs = list(hp[hp["side"] == "long"]["permno"])
        shorts = list(hp[hp["side"] == "short"]["permno"])
        brief = (
            f"Your book, as of {era}.\n\n"
            f"Long positions (identifiers): {', '.join(str(x) for x in longs)}\n"
            f"Short positions (identifiers): {', '.join(str(x) for x in shorts)}\n\n"
            f"The view you are expressing: {DRIVERS[driver[p]]}\n\n"
            f"Write the internal thesis for this book."
        )

        if a.dry_run:
            print("=" * 72)
            print(f"{p}  driver={driver[p]}  type={ty.get(p)}")
            print("=" * 72)
            print("--- SYSTEM ---")
            print(SYSTEM)
            print("\n--- USER ---")
            print(brief)
            break

        text = (fake_thesis(p, driver[p], longs, shorts, rng) if a.fake
                else call_anthropic(SYSTEM, brief, a.temperature))
        path = os.path.join(a.outdir, f"{p}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        rows.append({"pod": p, "type": ty.get(p, ""), "driver": driver[p],
                     "era": str(era), "words": len(text.split()),
                     "path": path})
        print(f"  {p:<10} {ty.get(p,''):<12} {driver[p]:<22} "
              f"{len(text.split()):>4} words", flush=True)
        if not a.fake:
            time.sleep(a.sleep)

    if a.dry_run:
        return

    man = pd.DataFrame(rows)
    man["generated_by"] = "template" if a.fake else MODEL
    man.to_csv(os.path.join(a.indir, "theses_manifest.csv"), index=False)

    print("\n" + "=" * 72)
    print("GROUND TRUTH, WHICH THE DETECTOR NEVER SEES")
    print("=" * 72)
    print(f"  {len(crowded)} pods briefed on '{shared_crowd}' and holding one shared basket")
    print(f"  {len(factor)} pods briefed on '{shared_factor}' with NO name in common")
    print(f"  {len(indep)} pods on drivers of their own")
    print()
    print("  The second group is the test. Their books do not overlap at all,")
    print("  so every position-based tool in this project is blind to them.")
    print("  If thesis similarity finds them, it found something the positions")
    print("  had not yet revealed. If it does not, the text layer adds nothing")
    print("  and that is the result to report.")
    if a.fake:
        print("\n  WARNING: --fake was used. This corpus is template text and")
        print("  any downstream number is a test of the plumbing, not a finding.")
    print(f"\nwrote {len(rows)} theses to {a.outdir} and theses_manifest.csv")


if __name__ == "__main__":
    main()
