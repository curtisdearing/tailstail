# `decision-card/1` — the private page, and what stays off the public site

## Why there are two artifacts

A weekly run produces two different things and only one of them may be
published.

**Public.** Tailstail's own projections and the aggregate model grading. They
name no league and no person, and publishing them is the point of the site:
`fantasy.html` → `_site/index.html`, and the allow-listed
`data/fantasy_public.json` → `_site/fantasy_latest.json`.

**Private.** The ESPN league snapshot and everything derived from it — rosters,
team names, manager identity, the league id — plus the personalised decision
card. These are written to `private/`, which is gitignored, and never leave the
machine.

This split is not hypothetical tidying. Before it existed, `fantasy.html`
embedded the eight-section `my_team` contract, the workflow copied that file to
`_site/index.html`, and `data/fantasy_latest.json` — `my_team` and all — was
copied next to it. Every step was locally reasonable, and the result published a
private league's rosters every week. So the guard is a **positive allow-list**
(`nflvalue/fantasy/private_boundary.PUBLIC_PAYLOAD_KEYS`) plus an assertion that
runs in the pipeline, not only in tests: a section somebody adds next year is
private until it is deliberately named.

The ESPN external-challenger comparison is split the same way. Its per-player
rows were fetched under terms that grant no redistribution right — the snapshots
themselves record that they are "retained for audit, not republication" — so the
public payload carries the week-by-week aggregate grading and says, in the file,
that the rows were withheld and why.

Where those rows live between runs took two attempts. They came out of the
`fantasy-model-state` release asset, because a release on a public repository is
published material — and went into an `actions/cache`, described at the time as
private. **That was wrong.** GitHub documents that a workflow triggered by a
pull request can restore caches created on the default branch, and says in terms
not to store sensitive information in one
([dependency caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)).
An Actions cache is not a security boundary.

The split that replaced it does not make the public history depend on the
private rows at all:

* **`data/espn_comparison_history.json`** (`espn-comparison-history/1`) is the
  public, durable aggregate grading history — one immutable entry per graded
  week, counts, MAEs, closer/tie tallies, by-position aggregates and the
  ledger's own `projections_sha256` as non-reversible audit linkage. It carries
  nothing that could be a player. It is loaded and validated independently of
  the raw ledger, and it rides in the checksummed public state release, so a run
  that never reaches the private store still republishes every week already
  graded. Weeks are immutable: a conflicting rewrite raises rather than
  overwriting.
* **The row-level ledger and the raw captures** live in a separate private
  repository (`curtisdearing/tailstail-state`), reached with a write-scoped
  deploy key that only a trusted production run is given.
  `nflvalue/fantasy/private_state.py` is the only thing that copies between it
  and the worktree: two allow-listed paths, symlinks and path traversal refused
  outright, files carrying league identity, a personalised contract, roster or
  member data or anything credential-shaped refused outright, the restored
  ledger's own row hashes verified, and never a file's contents in a log line.
  An unavailable store is a `::warning::` and a run without raw state — never a
  cache, a release, an artifact or Pages.

A private-store miss can therefore cost this week's raw rows. It cannot reset a
single previously graded aggregate week, and it never touches the projections.

## What the card is

`nflvalue/fantasy/decision_card.py` reads one `my_team/1.0.0` contract and
states four things:

1. the current legal lineup, each seat marked already-set or not;
2. only the seats that differ from what is set;
3. at most three actionable decisions;
4. freshness and unresolved-identity alerts.

`nflvalue/fantasy/decision_page.py` renders it: one self-contained HTML file,
no script, no external reference, everything escaped. The full eight-section
contract is available below the card, collapsed, as the record the decisions
were drawn from rather than as the decisions themselves.

### Every actionable decision carries

| Field | Note |
|---|---|
| `status` | `start` · `sit` · `add` · `drop` · `hold` · `shadow` · `no_current_pick` |
| `subject` / `alternative` | model id, ESPN id, name, position, NFL club, fantasy team, slot |
| `mean_delta` / `median_delta` | read from the contract; never recomputed here |
| `interval` | p10–p90 over **paired** simulated weeks, or a stated absence |
| `model_relative_probability` | labelled model-relative, and not a calibrated confidence |
| `drivers` | at most two cited notes, each with a source and an as-of timestamp |
| `risk` | exactly one visible counter-case |
| `invalidation_trigger` | one observable thing that would make it wrong |
| `provenance` | model version, scoring hash, snapshot hash, freshness state |

### The rules the validator enforces

`decision_card.validate()` runs on every card before it is returned, including
one an LLM has touched, and refuses rather than repairs.

* **No model-internal vocabulary.** Composite scores, learner names, and the
  unqualified word "confidence" are refused anywhere in the card's prose. The
  only permitted use of the word is inside the phrase *not a calibrated
  confidence*. Upstream reasons are not forwarded: the card composes its own
  sentences from a reason code, so jargon has no door to arrive through — not
  even inside a message explaining why something was rejected.
* **Provenance is all-or-nothing.** A run that cannot say which model, scoring
  rules and capture produced it emits one `no_current_pick` and no decisions.
* **An action needs a range.** A swap recommended on its mean alone is refused;
  it renders as `NO CURRENT PICK` with the reason. The two exceptions are a
  `hold` and a *forced* seat — a bye, an out designation, an empty slot — where
  there is nothing to measure, and which therefore carry no delta at all rather
  than the arithmetic difference against a player who cannot play.
* **Three actions.** Anything beyond the budget is held back, counted, and named
  in an alert. Shadow seats and refusals never consume it.
* **K and D/ST stay visible and unpromoted** until their own season-forward gate
  passes; waiver rows appear only once the planner's gate has passed; trades are
  absent.
* **Stale, future-dated or partial means one reason and no card content.** A
  capture dated ahead of the clock is *not* fresh — `age < FRESH_HOURS` is true
  for a negative age, which is the one answer that makes the problem invisible,
  so `my_team.freshness` names that state `future`. Both artifacts are rewritten
  every run, so a run that can say nothing replaces the page instead of leaving
  last week's answer on disk looking current.

### Cited context

Team and injury news is passed to `build(..., context=[...])` as items carrying
`text`, `source` and `as_of`. An item without both provenance fields is dropped
and counted. Context is consulted only after ranking is finished, so it can add
a driver, replace the risk with cited counter-evidence, and nothing else;
`tests/test_decision_card.py` asserts the card is identical with and without it
apart from those fields.

### The optional rewrite

The default path uses no language model. `apply_prose_rewrite(card, rewriter)`
will let one restate `headline` and `invalidation_trigger` after the numbers and
the order are fixed, and accepts a candidate only if it carries the same
numerals, still names every player, slot and position the original named,
introduces no forbidden wording, and does not double in length. Any failure
keeps the original sentence and is recorded, and the whole card is re-validated.

### Recommendation only

Nothing in this layer touches the network. ESPN is read once, upstream, for
display; no lineup is set, no claim is placed, no trade is proposed.

## Running it

The weekly pipeline writes the card automatically. By hand:

```python
from nflvalue.fantasy import decision_card, decision_page, my_team

contract = my_team.build(snapshot, now=now, contract=league_contract,
                         crosswalk=crosswalk, projections=projections,
                         byes=byes, samples=paired_samples)
card = decision_card.build(contract, now=now, model_version=git_sha)
decision_page.write(card, json_path="private/my_team_latest.json",
                    html_path="private/my_team.html", my_team=contract)
```

`samples` is the simulation's own draw matrix — one column per player, one row
per simulated week. Without it every swap is unmeasured, and the card will say
so instead of recommending on a mean.
