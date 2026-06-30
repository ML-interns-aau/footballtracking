# Event-Detection Validation (vs StatsBomb)

A harness that scores the pipeline's detected events (`events.json`) against
**StatsBomb open-data** ground truth and reports precision / recall / F1 per
event type. It turns "improve event-detection accuracy" into a measurable loop:
run → score → inspect false positives / negatives → fix → re-score.

> StatsBomb open-data ships **event data only, no video**. The harness can
> therefore only score a clip that *is* a StatsBomb-covered match, for which you
> know the **match id** and the **kickoff offset** (the match-clock time at the
> clip's first frame).

## What it scores

Headline metrics cover the four event types that map cleanly between our
detector and StatsBomb:

| Our event | StatsBomb source |
|-----------|------------------|
| `pass` | `Pass`, completed (no `pass.outcome`), not a cross |
| `cross` | `Pass` with `pass.cross == true` |
| `interception` | `Interception` |
| `recovery` | `Ball Recovery` |

`switch_of_play`, `skill_move`, and the zone-entry events are emitted by the
detector but only map to *derived* StatsBomb ground truth, so they are kept out
of the headline numbers (see `DIAGNOSTIC_TYPES` in
[`src/validation/events/model.py`](../src/validation/events/model.py)).

## Matching model

Events are matched per type within a time tolerance (default ±3 s), minimising
total time error (optimal assignment via SciPy, greedy fallback otherwise):

- **TP** — our event paired with a StatsBomb event,
- **FP** — our event with no match (a likely false detection),
- **FN** — a StatsBomb event we missed.

Team-aware matching is **off by default** because our team-0/1 labels come from
jersey clustering and are arbitrary; turn it on with a `team_map` once you have
a reliable side mapping.

Coordinates are reconciled by rescaling StatsBomb's 120×80 grid onto our
105×68 metric pitch, so `mean Δxy` in the report is in metres. Spatial error is
diagnostic only — it does **not** gate matching (homography calibration is not
yet reliable enough to trust position alone).

## Usage

### 1. Get the StatsBomb events file

Download the match's events JSON from the StatsBomb open-data repo
(`data/events/<match_id>.json`) and place it somewhere local, e.g.
`data/statsbomb/events/<match_id>.json`.

### 2. Single run (flags)

```bash
python tools/validate_against_statsbomb.py \
  --run-dir data/insights/<your_run> \
  --statsbomb-events data/statsbomb/events/<match_id>.json \
  --clip-start-s 0 \         # match-clock seconds at the clip's first frame
  --period 1 \               # optional: restrict to one half
  --tolerance-s 3.0 \
  --name my_clip
```

### 3. Batch (config)

Fill in [`configs/validation_matches.json`](../configs/validation_matches.json)
(one entry per run) and:

```bash
python tools/validate_against_statsbomb.py --config configs/validation_matches.json
```

### Outputs

Written into each run directory:

- `event_validation.json` — machine-readable metrics + FP/FN lists,
- `event_validation.md` — the same, plus per-type **review lists** with match
  clocks so you can jump to each false positive / false negative in the clip.

A summary table is also printed to stdout.

## The iteration loop (Yosef's task)

1. Run the pipeline on a StatsBomb-covered clip.
2. Run the harness; read `event_validation.md`.
3. For each **false positive**, open the annotated video at that clock and
   confirm whether the detection is spurious; for each **false negative**, see
   what the detector missed.
4. Adjust detection logic / thresholds in
   [`src/analytics/events.py`](../src/analytics/events.py).
5. Re-run and confirm precision/recall improved without regressing other types.

## Tests

```bash
python test_event_validation.py   # harness: mapping, matcher, metrics
python test_zone_events.py        # detector: zone events + predicted-ball guard
```

## Programmatic API

```python
from src.validation.events import RunSpec, validate_run

result = validate_run(RunSpec(
    name="my_clip",
    run_dir="data/insights/my_run",
    statsbomb_events="data/statsbomb/events/3795506.json",
    clip_start_s=0.0,
    period=1,
    tolerance_s=3.0,
))
print(result.overall_precision, result.overall_recall, result.overall_f1)
```
