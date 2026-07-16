# Fantasy factor catalog

Every scored factor must be known before the projection timestamp, work for
changing rosters, expose sample size, and survive an untouched season. “Context”
means the data may be displayed or researched but cannot move points yet.

| Factor family | Current treatment | Status / next evidence needed |
|---|---|---|
| 2/4/8-game PPR and expected points | Position model inputs | Implemented |
| Expected TD/yards/receptions/first downs by phase | Lagged opportunity-quality inputs | Implemented |
| Attempts, carries, targets, shares, snaps, WOPR | Player + team role state | Implemented |
| Reconciled team volume → player shares | Active-roster constrained attempts/targets/carries | Implemented |
| Coach pace/pass rate/position allocation | Prior-only 8-game coach tendencies | Implemented; missing history is explicit |
| Vacated RB/WR/TE opportunity | Current availability × prior role | Implemented |
| QB/player chemistry | Shrunk prior player × starting-QB performance and sample | Implemented; route-level separation is next |
| Trade/team or QB change | Change-point flags; history retained but role can move | Implemented |
| Spread, total, implied points, rest, home/division | Near-kickoff game state | Implemented |
| Opponent points allowed by position | Prior-only defense/position EWM | Implemented; coverage/scheme is richer |
| Stadium, referee, surface, primetime | Individual shrunk uplift plus current condition | Implemented conservatively |
| Weather | Wind/temp/roof plus missingness | Implemented; historical observed weather is not a Wednesday forecast |
| Injury/practice | Out/doubtful/questionable/DNP/limited hurdle and features | Implemented |
| Age, experience, draft capital | Player prior and cold-start support | Implemented |
| Route participation and routes per dropback | Better target denominator than snaps | Not in base cache; add participation feed |
| Slot/wide/inline alignment | Player-specific coverage and formation response | Research; requires charting/participation joins |
| 11/12/21 personnel, trips, motion, play action | Conditional role/efficiency scenarios | FTN/participation extension; enforce pregame aggregation |
| Man/zone, shell, pressure/blitz | Receiver/QB matchup interactions | Research; minimum multi-season OOF support |
| Offensive-line starters/injuries | QB pressure and rush efficiency scenarios | Needs position-group availability and projected starters |
| Offensive play-caller/scheme change | Head-coach allocation priors plus team change point | Head coach implemented; play-caller identity needs a maintained table |
| Referee penalty style | Team pace/first-down extension, not player folklore | Officials exist; derive prior crew rates from PBP |
| Birthday, revenge, contract, personal narrative | Context only | No score until predeclared, multiple-testing-adjusted evidence |
| Travel distance, time-zone/body-clock | Team/player random slope | Research; stadium coordinates and kickoff local time needed |
| Altitude and dome-to-weather transition | Efficiency/uncertainty interaction | Research; must beat generic stadium/surface features |
| Backup QB × wind × turf conjunction | Bayesian interaction with shrinkage | Candidate; never promote from one hand-picked split |
| TE1 out → RB2 TD or other role chains | Generated through availability/Dirichlet/TD simulation | Implemented structurally; calibrate each chain OOS |
| Fantasy ADP/expert consensus | External benchmark only | Optional prior only with verified pre-kickoff snapshot lineage |

For any new interaction, report raw and effective sample size, seasons, players,
teams, posterior effect, credible interval, and season-forward change in MAE,
RMSE, ranking, and calibration. Player-specific effects remain in the model at
small samples but shrink toward zero; they are not deleted behind an arbitrary
`n` cutoff or presented as proven.
