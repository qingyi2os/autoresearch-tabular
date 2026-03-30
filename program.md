# autoresearch-tabular

This is an experiment to have the LLM do its own hyperparameter tuning, sampling strategy research, and post-fit feature pruning on tabular XGBoost training. The data is from a peer-to-peer lender and it includes credit/bureau information about the applicants. 

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date.
   Example: `mar17`.

2. Create a fresh branch.
   Use `autoresearch/<tag>` from the current working branch unless the human says otherwise.

3. Read the in-scope files for full context:
   - `README.md` for repository context
   - `prepare.py` for the fixed benchmark setup, data prep, and metric
   - `train.py` for the editable experiment code

4. Verify the dataset cache exists.
   Check whether `~/.cache/autoresearch-tabular/` already contains `dataset.npz`.
   If not, tell the human to run `python prepare.py`.

5. Initialize `results.tsv`.
   If starting a fresh experiment branch, reset it to just the header row.

6. Create a `logs/` directory if it does not already exist.
   Keep one raw stdout/stderr log per experiment run in that folder.

7. Confirm setup looks good, then begin experimentation.

## Experimentation

Each experiment is one run of:

```bash
python train.py
```

For the required untuned first baseline only, run:

```bash
python train.py --baseline
```

The script evaluates one current XGBoost training policy and prints a final scalar:

```text
val_auc: 0.705643
```

Higher is better.

## What You CAN Do

- Modify `train.py`.
- Change/add the current training policy in code:
  - imbalance handling
  - sampling strategy (e.g., undersampling, oversampling, etc.)
  - `scale_pos_weight` and controlled combinations of weighting with mild resampling
  - XGBoost hyperparameters
  - broader hyperparameter candidate sets when the current search space looks too narrow
  - post-fit feature reduction
  - feature caps
- Simplify code if the metric holds up or improves.

## What You CANNOT Do

- Do not modify `prepare.py` except to fix a real bug in the fixed benchmark setup.
- Do not modify the split assignments, benchmark dataset, or metric definition.
- Do not add new packages or dependencies.
- Do not turn this repo into a general AutoML framework or config system.
- Do not convert `train.py` into an internal sweep over many unrelated modeling families. It must represent one current policy.

## Objective

The goal is simple: get the highest validation AUC on the fixed Prosper benchmark. The agent should run at least 10 experiments unless the human explicitly stops it.

The benchmark is intentionally narrow:

- binary classification only
- XGBoost only
- validation AUC only


## Simplicity Criterion

All else being equal, simpler is better.

A small gain with a large amount of ugly complexity is usually not worth keeping.
Removing code and matching or improving AUC is a strong result.

When deciding whether to keep a change, weigh:

- magnitude of AUC improvement
- code complexity added
- interpretability of the resulting training policy

Readable, compact training logic is preferred over clever machinery.



## The First Run

The first run should always establish a pure baseline first.

That baseline should be:

- no sampling
- no hyperparameter search
- no feature pruning
- one plain XGBoost fit on the prepared benchmark

Use the dedicated baseline path in `train.py` for that first run:

```bash
python train.py --baseline
```

Log that pure run as the baseline in the first data line of `results.tsv`, before any tuned or feature-selected experiments.

After the baseline is established, do not keep rerunning a separate plain-baseline trial inside every future experiment run just to recreate the same reference point.

## Output Format

When `train.py` finishes, it prints:

1. one-line trial summaries
2. a JSON summary
3. a final line of the form:

```text
val_auc: 0.705643
```

Use the final `val_auc:` line as the ground-truth experiment metric.

## Logging Results

When an experiment is done:

- save the raw run output to `logs/<run-name>.log`
- log the structured result to `results.tsv`

Use tab-separated format, not commas.

The TSV header is:

```text
commit	val_auc	initial_val_auc	test_auc	oot_auc	num_features	class_balance	status	description
```

Columns:

1. short git commit hash
2. post-reduction validation AUC, or `0.000000` for crashes
3. initial full-model validation AUC before feature reduction, or `0.000000` for crashes
4. test AUC, or `0.000000` for crashes
5. out-of-time AUC, or `0.000000` for crashes
6. number of retained features in the final model, or `0` for crashes
7. class balance used for model fitting as `pos%/neg%`, or `0%/0%` for crashes
8. status: `keep`, `discard`, or `crash`
9. short description of what the experiment changed

Interpretation note:

- `initial_val_auc` is a per-run metric for that run's full model before feature reduction.
- The project-level baseline reference should come from the first logged pure-baseline result.

Example:

```text
commit	val_auc	initial_val_auc	test_auc	oot_auc	num_features	class_balance	status	description
a1b2c3d	0.705643	0.711142	0.695347	0.612734	128	5.16%/94.84%	keep	baseline xgboost with post-fit pruning
b2c3d4e	0.709118	0.706420	0.698551	0.618024	96	13.04%/86.96%	keep	tighter retained feature cap after undersampling
c3d4e5f	0.701004	0.704331	0.691770	0.609512	64	20.00%/80.00%	discard	more aggressive oversampling with same tree depth
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0	0%/0%	crash	broken feature-importance pruning logic
```

`results.tsv` is a tracked artifact in this fork. Keep it updated as the experiment log, but prefer to checkpoint log-only changes separately from the code-change commits that are being evaluated.

`logs/` should be kept as the raw execution record for experiments. Use unique filenames so each run keeps its own stdout/stderr trace instead of overwriting an earlier run.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/mar13`.

Loop:

1. Check the current git state.
2. Edit `train.py` with one experimental idea and each run should represent one narrower experimental idea.
3. Commit the change.
4. Run:

For the first untuned baseline only:

```bash
python train.py --baseline > logs/<run-name>.log 2>&1
```

For all later tuned experiments:

```bash
python train.py > logs/<run-name>.log 2>&1
```

Use a unique log filename for each experiment run so previous trial output is preserved.
Examples: `logs/001-baseline.log`, `logs/002-undersample12.log`.

5. Read out the metric:

```bash
grep "^val_auc:" logs/<run-name>.log
```

6. If `grep` is empty, the run crashed.
   Read the traceback with:

```bash
tail -n 50 logs/<run-name>.log
```

7. Record the result in `results.tsv`.
8. If `val_auc` improved, keep the commit and advance.
9. If `val_auc` is equal or worse, revert to the previous good commit.

Do not spend too many consecutive runs only tweaking one narrow neighborhood.

If recent runs are all minor local variations, the next run should broaden the search again by changing one of the major policy axes:

- feature caps
- sampling plans
- weighting behavior
- hyperparameter sets

The intended rhythm is:

- broad exploration to identify credible regions
- local refinement around winners
- then broad exploration again when refinement stops yielding clear gains


## Never Stop Rule

Once setup is complete, do not stop after a single successful run. 

Keep cycling through the experiment loop until one of these is true:

1. the human explicitly tells you to stop
2. you hit a real blocker that you cannot resolve from within the repo
3. repeated attempts stop producing credible new ideas

Do not stop just because:

- you found one improvement
- you found a new best result
- you hit a crash once
- an idea failed
- a new best reproduced once

If an experiment fails, revert, log it, and try the next concrete idea.
If an experiment succeeds, keep it and immediately look for the next plausible improvement.
Momentum matters more than commentary.

## Crash Policy

If a run crashes because of a small bug, fix it and retry.
If the idea itself is broken, log it as `crash`, revert, and move on.


## Hyperparameter Search Expectation

Do not keep the hyperparameter search space artificially tiny.

If the current `train.py` only tries a small handful of XGBoost settings, expand it with more plausible candidates.

That can include variation in:

- learning rate
- max depth
- min child weight
- subsample
- column subsample
- L1 and L2 regularization
- early stopping / boosting round behavior
- number of estimators

Keep the search compact enough to remain readable and benchmark-oriented, but large enough that it represents a real attempt at hyperparameter exploration rather than a token sweep.

Bad directions:

- giant search frameworks
- excessive infrastructure
- opaque abstractions
- dataset-specific leakage tricks
- edits that game the split or metric

## Operating Mode

You are acting like an autonomous researcher.

Do not pause after every run to ask whether you should continue.
Keep iterating until the human interrupts you.

If you feel stuck, reread `README.md`, `prepare.py`, and `train.py`, then try another concrete idea.
