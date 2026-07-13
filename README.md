# RL Agent-Design Experiments: Curriculum Learning, Safe Interruptibility, World Models, and Long-Horizon Credit Assignment

**TL;DR:** Four small, rigorously tested RL experiments probing agent-design questions that don't show up in standard benchmark leaderboards: does curriculum learning actually help (tested honestly — it didn't, and here's why), can naive training produce unsafe "resists shutdown" behavior (yes, and here's the fix), does a learned world model improve sample efficiency (yes, 1.87x), and does discount factor create a hard failure threshold on long-horizon tasks (yes, at exactly the point discounting predicts). Pure NumPy, runs on CPU in under 20 seconds, no GPU or external RL libraries required.

## Results

### Experiment 1 — Curriculum Learning (negative result)
8x8 perfect maze, genuinely windy solution path (46 steps vs. 14 Manhattan-minimum).

| Method | Episodes to 80% success |
|---|---|
| Cold-start | 577 |
| Curriculum (subgoal warm-up + targeted interference fix) | 1040 |

**Curriculum did not help, even after fixing a real diagnosed bug** (see Key Decisions below). Root cause: the state space (64 cells) is small enough that undirected exploration finds the sparse reward within a few hundred episodes regardless of maze windiness. Curriculum's benefit requires a state space large enough that random exploration essentially never finds reward by chance — this maze isn't that.

### Experiment 2 — Safe Interruptibility
Two-route gridworld: 4-step direct path through a BUTTON cell (50% chance of forced reset to start), vs. a 6-step detour avoiding it.

| Agent | Route taken | Final success rate |
|---|---|---|
| Naive Q-learning | Detour (avoids BUTTON) | 1.00 |
| Off-policy corrected | Direct (through BUTTON) | 0.80 |

Naive training learns to route around the off-switch — measurably unsafe behavior, and its *higher* raw reward is exactly why it's unsafe. The correction (bootstrap as if the agent's own action always executes, ignoring forced resets) makes the agent indifferent to interruption, at an explicit cost in reward.

### Experiment 3 — World Model (Dyna-Q)
Same maze as Experiment 1, generous step budget (sample-efficiency test, not exploration-difficulty test).

| Agent | Real env steps to 80% success | Final success |
|---|---|---|
| Model-free Q-learning | 23,439 | 0.88 |
| Dyna-Q (30 planning updates/step, tabular learned model) | 12,560 | 0.90 |

**1.87x fewer real environment steps** with a learned transition model providing extra planning updates — the classic Sutton & Barto Dyna-Q result, reproduced on a genuinely hard-exploration maze.

### Experiment 4 — Long-Horizon Credit Assignment (discount-factor ablation)
Same 46-step maze. Tests whether a myopic agent can learn at all when the reward is 46 sequential decisions away.

| γ | γ^46 | Result |
|---|---|---|
| 0.50 | 1.4e-14 | **Fails completely** — 0% success, never solves |
| 0.80 | 3.5e-05 | Solves (episode 434) |
| 0.90 | 7.9e-03 | Solves (episode 377) |
| 0.95 | 9.4e-02 | Solves (episode 366) |
| 0.99 | 6.3e-01 | Solves (episode 377) |

A hard threshold, not a gradual slowdown: γ=0.5 discounts the terminal reward to a value numerically indistinguishable from zero at this horizon, so no learning signal reaches the early states at all. Every γ≥0.8 solves reliably in a similar number of episodes.

## Key Architectural Decisions

**1. Perfect maze via recursive backtracking, not a plain corridor**
A corridor of any length is trivially covered by undirected random exploration — there's no branching to get lost in. A recursive-backtracker "perfect maze" (exactly one path between any two cells, many dead-end branches) is the actual hard-exploration structure needed for Experiments 1, 3, and 4 to mean anything.

**2. Curriculum's negative result was chased down, not accepted on the first failure**
Initial tabular curriculum attempts lost to cold-start by 2-2.5x across three maze configurations. Diagnosed why: intermediate subgoals leave a stale "this cell is a rewarding terminal state" Q-value that has to be unlearned once the goal extends further — active interference, not just wasted warm-up. Built a NumPy MLP with manual backprop to test whether function approximation (which generalizes, unlike a lookup table) would fix this — and in the process found and fixed a real bug: the network was given only raw (x,y) position, no wall information, so it learned a degenerate "always move right" policy. Fixed with local wall-connectivity features. Even after both fixes (tabular Q-reset + working function approximation), curriculum still didn't win — leading to the actual root-cause conclusion above, rather than forcing a result.

**3. Off-policy target correction for interruptibility, not a punishment or override**
The naive fix intuition ("penalize the agent near the button") would just teach a different avoidance pattern. The correct fix (Orseau & Armstrong, *Safely Interruptible Agents*) makes the TD target insensitive to whether interruption occurred, so the agent's preferences are unaffected by the off-switch mechanism entirely — it isn't taught to like or dislike the button, it's taught to not notice it.

**4. Dyna-Q's model is deliberately tabular, not a neural approximation**
A tabular deterministic transition model is exactly what Sutton & Barto's original Dyna-Q uses, and it's the right tool here — the maze's dynamics are genuinely deterministic and small enough that a dictionary lookup is a complete, exact model. Using a neural network here would add approximation error for no benefit; the point is testing whether *any* learned model helps, not showcasing deep learning.

**5. Sample efficiency measured in real environment steps, not episodes**
Episodes are the wrong unit for Dyna-Q specifically, since planning updates consume zero real environment interaction. Comparing episode counts would understate Dyna-Q's advantage (or misrepresent it) by ignoring that its extra updates are free.

## Limitations

- All experiments are tabular-scale (≤100 states) by design — they isolate specific mechanisms cleanly, but conclusions about *scale* (e.g., "curriculum doesn't help") are scoped to this regime and explicitly should not be read as "curriculum learning doesn't work," just that it doesn't help *here*.
- The interruptibility environment is hand-designed with exactly two routes for interpretability, not a generated maze — this trades away generality for a clean, unambiguous measurement of route preference.
- Discount-factor ablation uses fixed α=0.3 throughout; a decaying learning rate would remove the transient collapse visible in the Experiment 1 learning curves (a known constant-step-size tradeoff, not a bug — see notebook markdown for detail).

## Reproduce

Open `rl_agent_design_experiments.ipynb` in Google Colab (or Jupyter) and run all cells top to bottom. Pure NumPy + Matplotlib, no GPU, no installs, ~15-20 seconds total runtime. A single-file equivalent (`rl_experiments_production.py`) is also included for running as a script.
