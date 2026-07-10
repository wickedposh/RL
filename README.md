# EXP3 Recommender with User Satiation

An EXP3 (adversarial multi-armed bandit) recommender where the reward for a
genre *decays* when it's recommended repeatedly (user satiation), and recovers
when it's rested.

## Key finding
Sweeping the satiation-retention parameter (20 seeds each), I show it
**monotonically controls how much the policy concentrates on the favorite**:

| retention (weaker satiation →) | top-arm probability |
|---|---|
| 0.1 | 0.233 ± 0.013 |
| 0.5 | 0.292 ± 0.015 |
| 0.9 | 0.513 ± 0.065 |

Strong satiation forces diversification (top arm ~23%); weak satiation lets the
policy concentrate on the favorite (~51%). Monotonic across all 9 settings,
error bars non-overlapping.

## Run
`python exp3.py`
