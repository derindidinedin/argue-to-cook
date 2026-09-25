# Exploratory material

These analyses support the exploratory results in the thesis and report summaries for each seed and, where stated, bootstrap intervals.

Performance is summarised as the mean of the final 50 logged episode outcomes. These are not necessarily consecutive episodes because only the last episode to finish in each rollout is logged.

## Analysis scripts

* `exploratory_analysis.py` reads the cached runs and reports five exploratory analyses:

  * changes in the three scoring weights during training
  * one run per configuration comparing two profile selection strategies and three potential component ablations
  * performance on the Coordination Ring layout
  * the shaping strength comparison against the PPO baseline and ArgRL-Full
  * convergence during training

  Convergence is measured using normalised area under the training curve and the first step at which a run reaches 80% of its own final performance.

* `bandit_analysis.py` compares the exploratory intention bias experiment with ArgRL-Full and PPO baseline. Its mean performance falls between the two, and both 95% bootstrap intervals for the mean differences include zero.

* `shaping_only_analysis.py` reports the ArgRL-Shape condition, which uses reward shaping but hides the intention profile from the policy observation, so the shaping channel is active and the observation channel is disabled.

All three use the same loading, summary and bootstrap functions as the confirmatory analysis in `../analysis/`. The intention bias mechanism is implemented in `../af/bandit.py` and trained through the main `train.py` script.

The scripted executor comparison is implemented in `../af/scripted_executor.py` because it generates new episodes rather than reading cached training runs. It evaluates six runs for each intention source. Only the random source changes with the seed. The framework and fixed role sources are deterministic, so their repeated runs check reproducibility rather than variation across independent seeds.

## Running the analyses

With the dependencies installed and the cached runs available in `results/`, run:

```bash
python exploratory/exploratory_analysis.py
python exploratory/bandit_analysis.py
python exploratory/shaping_only_analysis.py
python -m af.scripted_executor
```

Each script prints plain text tables to standard output. `bandit_analysis.py`
takes `--verbose`, which adds the run directory and logged point count behind
each seed. Those are diagnostics rather than results, so they are hidden unless
a run is missing.

## Retraining the exploratory runs

The shaping strength comparison uses six seeds for each setting:

```bash
for seed in 0 1 2 3 4 5; do
  python analysis/train.py --condition argrl_full --n-envs 4 --scale 5 --seed $seed
  python analysis/train.py --condition argrl_full --n-envs 4 --scale 25 --seed $seed
done
```

The exploratory intention bias experiment and ArgRL-Shape condition also use six seeds:

```bash
for seed in 0 1 2 3 4 5; do
  python analysis/train.py --condition bandit --n-envs 4 --seed $seed
  python analysis/train.py --condition argrl_shape --n-envs 4 --seed $seed
done
```

The following comparisons use one run per configuration with the default seed:

```bash
python analysis/train.py --condition argrl_full --n-envs 4
python analysis/train.py --condition argrl_full --n-envs 4 --strategy conservative
python analysis/train.py --condition argrl_full --n-envs 4 --ablate coalition
python analysis/train.py --condition argrl_full --n-envs 4 --ablate load_ready
python analysis/train.py --condition argrl_full --n-envs 4 --ablate individual
```
