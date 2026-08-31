#!/usr/bin/env bash
# Decide whether this workflow run is TRUSTED PRODUCTION, and validate its inputs.
#
# Trust is a property of the ref, not of the inputs. The previous version of
# this logic asked only whether the optional dispatch inputs were blank, so a
# `workflow_dispatch` from any branch with nothing filled in classified as
# production -- and would then have been handed the private-state deploy key,
# permission to publish Pages, and the release asset. A branch anybody with
# write access can create is not a trust boundary; the default branch is.
#
# Every value arrives through the environment. `${{ }}` expression text spliced
# into a `run:` block is substituted by the runner BEFORE bash parses the
# script, so a dispatch input of `"; curl … | sh; #` executes with the job's
# token in scope. Nothing here is interpolated.
#
# Inputs (environment):
#   GH_REF     github.ref
#   GH_EVENT   github.event_name
#   IN_SEASON  optional season override      (makes the run diagnostic)
#   IN_WEEK    optional week override        (makes the run diagnostic)
#   IN_FAST    reduced-learner flag          (makes the run diagnostic)
# Output:
#   production=true|false  ->  $GITHUB_OUTPUT
set -euo pipefail

DEFAULT_BRANCH_REF="refs/heads/main"
TRUSTED_EVENTS="schedule workflow_dispatch"

season="${IN_SEASON:-}"
week="${IN_WEEK:-}"
fast="${IN_FAST:-false}"
ref="${GH_REF:-}"
event="${GH_EVENT:-}"

if [[ -n "$season" ]]; then
  [[ "$season" =~ ^[0-9]{4}$ ]] || { echo "::error::season must be a 4-digit year"; exit 1; }
  (( season >= 1999 && season <= 2099 )) || { echo "::error::season out of range 1999-2099"; exit 1; }
fi
if [[ -n "$week" ]]; then
  [[ "$week" =~ ^[0-9]{1,2}$ ]] || { echo "::error::week must be 1-2 digits"; exit 1; }
  (( week >= 1 && week <= 22 )) || { echo "::error::week out of range 1-22"; exit 1; }
fi

production=true

# 1. Trust: the default branch, reached by a trigger a person or the schedule
#    owns. An exact string compare -- `refs/heads/mainline` and
#    `refs/heads/Main` are different branches.
if [[ "$ref" != "$DEFAULT_BRANCH_REF" ]]; then
  production=false
  echo "not production: ref '$ref' is not $DEFAULT_BRANCH_REF"
fi
if [[ " $TRUSTED_EVENTS " != *" $event "* ]]; then
  production=false
  echo "not production: event '$event' is not one of $TRUSTED_EVENTS"
fi

# 2. Unmodified pipeline. A reduced-learner or back-dated run must not publish
#    state, a release or Pages: a --fast model writing the season ledger
#    silently corrupts every later grading comparison.
if [[ "$fast" == "true" || -n "$season" || -n "$week" ]]; then
  production=false
  echo "not production: this run overrides the pipeline (fast=$fast season='$season' week='$week')"
fi

echo "production=$production" >> "${GITHUB_OUTPUT:-/dev/stdout}"
echo "run mode: production=$production ref=$ref event=$event"
