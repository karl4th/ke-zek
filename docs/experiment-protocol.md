# Experiment protocol

These rules apply to every experiment in this repository. An experiment exists to make a
decision, not to explore.

## 1. Every outcome maps to an action, in advance

- Each experiment answers at most two questions and ends in an action ("keep / drop /
  choose X"). An outcome of "inconclusive, needs more work" is not allowed.
- Each protocol contains an **outcome → action** table that covers every case.
- **A grey zone always goes to the simpler or cheaper option**, the one that adds no
  component, data or stage. The burden of proof is on the added complexity.
- No extending a run, adding seeds or changing the metric after launch. If that seems
  necessary, it is a new experiment with a new protocol. The decision on the current one is
  already made.

## 2. One primary metric per question

All other metrics are guardrails. A guardrail can block a win but cannot create one.

## 3. Seeds

Key decisions use 2 seeds. A clear win means the mean passes the threshold **and** each seed
on its own beats the baseline. Anything else is a grey zone, and rule 1 applies.

## 4. Fair comparison

- Same data, batch order, number of steps and seconds of audio. Parameters and FLOPs of the
  arms within ±5%.
- Never compare training losses between arms with different inputs or targets.
- The evaluation protocol, test sets and scripts are committed **before** training starts.
  That commit hash is recorded in the experiment.
- Test recordings and their speakers never appear in training, in any arm.
- KezekBench is a test set only. Hyper-parameters are tuned on a separate dev set.

## 5. Report

Each experiment closes with a table of all arms against the primary metric and guardrails
(mean, both seeds, 95% bootstrap CI over episodes), the rule applied and the action taken.
The table goes into the experiment folder (`experiments/<id>/RESULT.md`).

## 6. Experiment folder

```
experiments/<id>/
  README.md     protocol: questions, hypothesis, arms, data, metrics, outcome → action table
  RESULT.md     written after the run: table, rule applied, decision
configs/<id>.yaml
```
