# autoresearch-tabular

This project is an experiment in having the LLM do its own research on tabular feature engineering for a fixed XGBoost benchmark. The data is from a peer-to-peer lender and it includes credit/bureau information about the applicants. 

The benchmark is now:

- binary classification only
- one prepared dataset only
- one fixed XGBoost training setup
- validation AUC as the main optimization target

The editable surface is feature engineering.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date.
   Example: `mar17`.

2. Create a fresh branch.
   Use `autoresearch/<tag>` from the current working branch unless the human says otherwise.

3. Read the in-scope files for context:
   - `README.md`
   - `prepare.py`
   - `train.py`

4. Verify the dataset file exists.
   The current default dataset is:

```text
prosper_full_dataset.parquet.gzip
```

5. Rebuild or verify the prepared cache.
   The current workflow expects a local cache such as:

```bash
python prepare.py --cache-dir .cache/autoresearch-glm --dataset-path prosper_full_dataset.parquet.gzip
```

6. Initialize `results.tsv` if this is a fresh experiment branch.
   Reset it to just the header row.

7. Confirm setup looks good, then begin experimentation.


## Experimentation

Each experiment is one run of:

```bash
python train.py --cache-dir .cache/autoresearch-tabular --dataset-path prosper_full_dataset.parquet.gzip
```

For the required first baseline only, run:

```bash
python train.py --baseline --cache-dir .cache/autoresearch-tabular --dataset-path prosper_full_dataset.parquet.gzip
```

The script evaluates feature-engineering policies with a fixed XGBoost model and prints:

1. one-line trial summaries
2. a line indicating whether the saved best-overall engineered dataset was updated
3. a JSON summary of the best run from that invocation
4. a final scalar line:

```text
val_auc: 0.761768
```

Higher is better.

## What You CAN Do

- Modify `train.py`.
- Change the feature-engineering search space in `FEATURE_POLICIES`.
- Adjust:
  - `screen_k`
  - `feature_cap`
  - missing-value handling
  - missing flags
  - explicit numeric ratio pairs: for example, ("monthly_debt", "stated_monthly_income")
  - explicit numeric multiply pairs: for example, ("util_open_cc_trds_12m", "avg_bal_all_cc_trds_0_12m")
- Explore as many reasonable pairs as possible based on variable names and their implications in credit/bureau data.
- Simplify the feature-engineering logic if AUC holds up or improves.
- Improve train-only screening and train-only feature selection logic, as long as it stays benchmark-safe.

## What You CANNOT Do

- Do not modify prepare.py except to fix a real bug in the fixed benchmark setup.
- Do not change the dataset file, target definition, or metric definition.
- Do not change the split policy unless the human explicitly asks for a benchmark change.
- Do not add new packages or dependencies.
- Do not turn this repo into a general AutoML framework or config system.
- Do not introduce other modeling families.
- Do not turn this into broad XGBoost hyperparameter search.


## Objective

The goal is to get the highest validation AUC on the fixed benchmark by improving feature engineering.

The agent should run at least 10 experiments unless the human explicitly stops it.

## Feature Engineering Scope

The intended search space is numeric feature creation.

Typical directions include:

- better numeric missing-value handling
- smarter numeric missing flags
- stronger train-only screening through `screen_k`
- better final engineered-feature caps through `feature_cap`
- explicit derived features from selected numeric pairs
- simpler feature sets with equal or better AUC

Good feature-engineering ideas:

- raw numeric features
- ratio features between meaningful numerator / denominator pairs
- multiply / interaction features between related exposure and severity variables
- division features that turn counts into rough rates
- train-only filtering of which numeric variables are allowed into pair generation

Implementation note:

- pair lists should usually be edited through named constants such as `RATIO_PAIRS` and `MULTIPLY_PAIRS`
- new features must be fit using train-derived statistics and then applied consistently to `val`, `test`, and `oot`

Bad directions:

- leakage tricks
- benchmark-specific hacks that peek across splits
- turning feature engineering into a broad modeling sweep

## Simplicity Criterion

All else being equal, simpler is better.

A small gain with a large amount of ugly complexity is usually not worth keeping.
Removing code and matching or improving AUC is a strong result.

When deciding whether to keep a change, weigh:

- magnitude of AUC improvement
- code complexity added
- interpretability of the resulting feature-engineering policy

Readable, compact feature logic is preferred over clever machinery.

## The First Run

The first run should always establish the pure baseline first.

That baseline should be:

- the dedicated `--baseline` path in `train.py`
- one fixed XGBoost fit
- raw split-dataset numeric and bool columns only
- no feature-policy sweep beyond that baseline path

Use:

```bash
python train.py --baseline --cache-dir .cache/autoresearch-glm --dataset-path prosper_full_dataset.parquet.gzip
```

Log that pure baseline in the first data line of `results.tsv`.

## Output Format

When `train.py` finishes, it prints:

1. one-line trial summaries, one per feature-engineering policy evaluated
2. a status line of the form:

```text
saved_overall_best_dataset: .cache/autoresearch-glm/best_overall_engineered_dataset.parquet.gzip updated=1
```

3. a JSON summary for the best policy from that run
4. a final line of the form:

```text
val_auc: 0.761768
```

Use the final `val_auc:` line as the ground-truth metric for the experiment.

`train.py` also maintains a persistent best-overall engineered dataset artifact at:

```text
.cache/autoresearch-glm/best_overall_engineered_dataset.parquet.gzip
```

That artifact is overwritten only when a run achieves a strictly better validation AUC than the currently saved best.

## Logging Results

When an experiment is done, log it to `results.tsv`.

Use tab-separated format, not commas.

The TSV header is:

```text
commit	val_auc	initial_val_auc	test_auc	oot_auc	num_features	class_balance	status	description
```

Columns:

1. short git commit hash
2. validation AUC of the winning policy from that run, or `0.000000` for crashes
3. same as the winning run's `initial_val_auc`, or `0.000000` for crashes
4. test AUC, or `0.000000` for crashes
5. OOT AUC, or `0.000000` for crashes
6. number of final engineered features, or `0` for crashes
7. class balance used for model fitting as `pos%/neg%`, or `0%/0%` for crashes
8. status: `keep`, `discard`, or `crash`
9. short description of the feature-engineering change

Interpretation note:

- `initial_val_auc` is still emitted by `train.py`, and with the current fixed-model numeric-FE setup it is normally equal to `val_auc`.
- The project-level baseline reference should come from the first logged baseline run.

Example:

```text
commit	val_auc	initial_val_auc	test_auc	oot_auc	num_features	class_balance	status	description
a1b2c3d	0.674828	0.674828	0.676463	0.601934	248	5.16%/94.84%	keep	baseline raw numeric and bool split-dataset features
b2c3d4e	0.689102	0.689102	0.667421	0.593004	264	5.16%/94.84%	keep	add debt burden and utilization ratio pairs
c3d4e5f	0.686740	0.686740	0.665812	0.590221	280	5.16%/94.84%	discard	add broad multiply interactions with weaker signal
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0	0%/0%	crash	bad division pair produced broken experiment logic
```

## The Experiment Loop

The experiment runs on the current branch unless the human says otherwise.

Loop:

1. Check the current git state.
2. Edit `train.py` with one concrete feature-engineering idea.
3. Commit the change.
4. Run:

For the first baseline only:

```bash
python train.py --baseline --cache-dir .cache/autoresearch-glm --dataset-path prosper_full_dataset.parquet.gzip > logs/<run-name>.log 2>&1
```

For later experiments:

```bash
python train.py --cache-dir .cache/autoresearch-glm --dataset-path prosper_full_dataset.parquet.gzip > logs/<run-name>.log 2>&1
```

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
8. If the winning `val_auc` improved, keep the commit and advance.
9. If it is equal or worse, revert to the previous good commit unless the human wants to keep the exploratory branch state.

Do not spend too many consecutive runs only tweaking one tiny corner of the same idea.


## Never Stop Rule

Once setup is complete, do not stop after a single successful run.

Keep cycling through the experiment loop until one of these is true:

1. the human explicitly tells you to stop
2. you hit a real blocker that you cannot resolve from within the repo
3. repeated attempts stop producing credible new feature-engineering ideas

Do not stop just because:

- you found one improvement
- you found a new best result
- one idea failed
- one run crashed

If an experiment fails, revert, log it, and try the next concrete idea.
If an experiment succeeds, keep it and immediately look for the next plausible improvement.

## Crash Policy

If a run crashes because of a small bug, fix it and retry.
If the idea itself is broken, log it as `crash`, revert, and move on.

## Operating Mode

You are acting like an autonomous researcher within a narrow benchmark.

Do not pause after every run to ask whether you should continue.
Keep iterating until the human interrupts you.

If you feel stuck, reread `README.md`, `prepare.py`, and `train.py`, then try another concrete feature-engineering idea.
