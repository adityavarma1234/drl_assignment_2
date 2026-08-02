# Assignment II — Robust RL under Stochastic Action Failure

DRL course assignment (15 marks, due 2026-08-07). Solution lives in
[Assignment2_DQN_DDQN_Solution.ipynb](Assignment2_DQN_DDQN_Solution.ipynb), built to run top-to-bottom in Google Colab.
Full spec: [Assignment 2-v2.pdf](Assignment 2-v2.pdf).

## Task
Wrap `LunarLander-v3` with `gym.Wrapper` to simulate stochastic actuator failure, then train and
compare DQN vs DDQN on the original vs the modified environment (4 agents total).

## Wrapper spec (must match exactly — this is graded)
On every `step(action)`:
1. Store the agent's selected action `a` before modification.
2. If `a != 0`: draw `r ~ U[0,1)`; if `r < 0.15`, replace with `a_exec = 0` (misfire), else `a_exec = a`.
   The agent must never see that a swap happened (nothing added to `info`).
3. Execute `a_exec` on the base env.
4. Reward: `R = R_base − 0.3·1[a ∈ {1,2,3}] + bonus`. The fuel penalty is based on the
   **selected** action `a`, not `a_exec` — it applies even on a misfire.
5. `bonus = +50` only if `terminated and not truncated` and, at the final observation:
   both legs in contact (`obs[6]==1`, `obs[7]==1`) and `|obs[2]|<0.10`, `|obs[3]|<0.10`,
   `|obs[4]|<0.10` (horizontal vel, vertical vel, orientation angle). Otherwise `bonus = 0`.
6. Return `(observation, R, terminated, truncated, info)` unchanged otherwise — observation
   space, action space, and termination/truncation conditions stay identical to the base env.

Implemented as `FaultyLunarLander` in the notebook. It also tracks a private
`_last_decomposition` / `stats` dict for the task-(a) verification cells only — never leak
this into the `info` dict returned to the agent.

## Experiment constraints (also graded)
- DQN and DDQN must share network architecture, optimizer, replay buffer, ε-greedy schedule,
  seed, hyperparameters, and episode count — see the single `HYPERPARAMS` dict and shared
  `train_dqn(..., double_dqn=True/False)` function. The **only** allowed difference between
  DQN and DDQN is the target-Q line inside that function.
- All 4 runs (DQN-orig, DDQN-orig, DQN-mod, DDQN-mod) reuse `SEED` and the same fixed
  `VALIDATION_STATES` set (sampled once, before training) for the avg-Q-value tracking plot.
- Landing-success flag for plots/metrics uses `FaultyLunarLander.is_safe_landing(...)`
  applied to the final `obs` regardless of which environment is used, so it's comparable
  across original vs modified runs.

## Submission requirements (don't lose easy marks)
- Comment every function/operation (1 mark auto-deducted if missing) — already done in the
  notebook; keep it that way if you edit.
- Fill in the group contribution table (top of notebook) and the 5 discussion answers
  (task e, bottom of notebook) using the actual plots produced by your run — they're
  currently placeholders.
- Final submission must be **one PDF** (`Team# - Q_learning_DQN_DDQN`) containing all code,
  outputs, and every training iteration — export the executed notebook (`File > Print` or
  `File > Download > PDF` in Colab) after running all cells, don't submit the `.ipynb` itself.
- Needs virtual-lab execution with timestamped screenshots attached.
- First version due 2 days before the 2026-08-07 deadline (i.e. by 2026-08-05).

## Working in this repo
- No package.json/build system — this is a single self-contained Colab notebook. Bash/pip
  installs happen inside the first notebook cell (`box2d`, `gymnasium[box2d]`), not locally.
- If you regenerate the notebook programmatically (e.g. to fix a cell), build valid nbformat
  v4 JSON (list of cells, each with `cell_type`, `metadata`, `source` as a list of lines) —
  don't hand-edit the raw JSON for anything beyond trivial text tweaks.
