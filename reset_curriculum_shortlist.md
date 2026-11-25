# Reset curricula: ranked shortlist of methods, scores and tasks

Source: the 206 papers in `papers/text`. 121 have a full-read note; the other 85 got a light pass (abstract, method, main results, limitations). Citations are arXiv id plus line numbers in the `.txt` file, e.g. `2610.12465 L892-906`.

Three labels are used throughout:

- **Shown**: an experiment in the cited paper reports it.
- **Claimed**: the authors say it, with no experiment behind that statement.
- **Ours**: an inference or proposal from this review. No paper tests it in this form.

Unless a sentence carries a citation, treat it as Ours. In the tables, the "Before first reward" and "Worst case" columns are all Ours.

Terms: an **IC** is a start state. A **score** is a number per IC saying how much there is to learn from it. A **method** is the rule that builds and updates the set of ICs. **Task start** means the task's own start distribution, the one you evaluate on. **p_task** is the share of resets sent to the task start.

Most of the cited evidence is about choosing levels, goals, task parameters or demonstration frames, not start states for a plain PPO policy. Applying it to start states is a transfer, and is Ours every time.

## 1. Bottom line

1. **No reward-based score gives a signal before the first reward (Ours, by arithmetic).** With a pure success-indicator reward, success rate, p(1-p), learning progress and best-return-so-far are all identical across ICs while every return is zero, so sampling by them is uniform sampling. No paper in the folder measures a score across a zero-reward phase; the supporting evidence is indirect (finding A). The only signals left are reward-free ones such as visit counts, and no paper here shows that counts help choose start states for PPO.
2. **States the robots actually visited are the cheapest candidate set that meets your rules.** They need no labels and are reachable by construction. Evidence that restarting from them helps is mixed, and if too few resets go to the task start it can hurt (finding B).
3. **After the first reward, the best-supported idea is restarting from states on a trajectory that reached it.** Whether a backward ordering adds anything over drawing uniformly along that trajectory is unsettled: one paper shows a gain, one shows mainly a speed difference. Every paper gets its trajectories from demonstrations, planners or hand-built ladders. Using the agent's own first successes is Ours, is untested, and does nothing until some robot reaches reward (finding C).
4. **Plain baselines are often close to published curricula.** Not always: the curricula keep a lead on held-out terrain in one benchmark and on the hardest SGS task (finding D). Section 6 lists the controls that separate the two cases.
5. **Suggested build (Ours):** restarts from visited states, weighted by visit count, until reward appears; then restarts along the agent's own successful trajectories with a success-band score; a fixed share of resets always at the task start. Count-weighting needs some notion of two states being "the same", which is where your no-descriptor and distance constraints bite (method 3). Nothing in the papers shows this combination works.

## 2. What the papers show

### A. Reward-based scores before first reward (indirect evidence)

- SGS (2610.12465): rod-in-hole with the Near-Goal resets removed gives 0.0% success on the other two reset types (Shown, 3 seeds, `L892-906`). SGS does not generate states; it samples a fixed set built before training (`L189-193`, `L454-456`) from hand-defined reset strategies with task-specific assembly paths and grasp points (`L711-725`).
- OmniReset (2603.15789): four hand-defined reset categories sampled uniformly, no score (`L289-317`, `L330-339`). It needs a goal sampler (`L247-262`). Its reward includes two dense distance terms (`L346-355`).
- AlphaNIST (2610.13354): all success estimates start at 0 (`L313`). Ours: by its sampling formula (`L227-235`) that makes initial sampling uniform over a bank that already holds near-goal states (`L161-176`).
- PAIRED (2012.02096): with one adversary parameterisation the solved path length was about 2 instead of about 15. The authors hypothesise (Claimed) that the adversary mostly built impossible levels, so no agent learned and there was no regret signal (`L1201-1212`).
- Kinetix (2410.23208): levels the agent never solves get p(1-p) = 0 and are dropped (`L316-317`). PPO from scratch never reached the goal on the deceptive Car-Ramp task in 1B steps; a pretrained agent did solve it, one seed (`L496-506`).
- Ours: for value-based scores (mean |GAE|, value loss) the pre-reward reading is critic noise that decays toward zero. With any non-zero penalty terms in the reward, returns are not all equal and this argument weakens.

### B. Restarting from visited states: mixed evidence

- UFLP (2301.12579) keeps a buffer of visited states and restarts from it, picking the best of B candidates by an uncertainty score; B=1 is a uniform draw. Discrete-action agents (DQN variants and an approximate-policy-iteration agent), not PPO.
  - Hard Cartpole, one of the DQN agents: B = 1, 5, 25, 125 gave comparable results; the authors conclude the gain there came from starting mid-trajectory (Shown, `L1225-1230`).
  - Deep Sea: selecting over the whole buffer beat B=5 and B=1 (Shown, `L702-705`). On Montezuma, with a count-based score, B = 8 and 64 were significantly better than B=1 (`L891-894`). So selection by score mattered on the harder exploration tasks.
  - Never using the task start hurt on Cartpole and Montezuma (`L730-732`, `L894`); on Deep Sea the best task-start share was 0.0-0.3 (`L701`).
- Contrastive Initial State Buffer (2309.09752), quadruped, 5 seeds: a pre-filtered buffer of visited states, drawn uniformly and mixed 80/20 with the task start, scored 9.58 ± 0.36 against 9.21 ± 0.43 for no buffer (`L264-271`). The authors report an early speed-up and the same final result (`L285-290`). Weak evidence either way.
- AlphaZero search control (2602.20809), board games: uniform restarts from visited states led early and the lead faded (`L455-462`), but that variant still beat AlphaZero against external programs on all three games (`L466-469`). Mixed.
- DemoStart (2409.06613), plug insertion: 80% uniform resets over demonstration states plus 20% task start gave 0%; so did keeping any state with one success; keeping only states with mixed outcomes gave 97.2% (`L261-270`, `L290-294`). These are demonstration states, not visited states, and seeds are not stated.
- Theory (2504.05405, finite actions, not PPO): a lower bound where restart mass on states the task start cannot reach breaks two classical policy-search algorithms (`L442-451`). The same passage has an exponential-in-horizon lower bound for one of them even when resets are reachable (`L444-448`), so reachable resets are not a guarantee either.

### C. Restarting along a successful trajectory

- Montezuma from one demonstration (1812.03381): a frontier moves back along the demo when success reaches a 20% bar (`L303-305`). Single run. On Gravitar and Pitfall the authors found no hyperparameters that completed the curriculum (`L365-368`).
- Backplay (1807.06919), 100 mazes: standard PPO found the optimal path on 0% of mazes. Starting uniformly along the demo gave 27/51/49% optimal against 31/37/54% for the backward schedule (`L424-444`). The uniform row is the best seed (`L443-444`); the authors say the schedule converged faster (`L454`) and that uniform failed on their other domain (`L507`).
- 2610.24392, reset choices on 10 tasks, reference motions: backward curriculum over validated states 9/10; uniform random frame 5/10; first frame only 2/10; backward curriculum over raw poses with interpenetration 0/10 (`L378-382`, `L401-424`). Seeds not stated. This is the one result where ordering clearly beat uniform-along-trajectory.
- Per-context reverse curricula (2607.26417): one frontier per context, with advance and retreat thresholds (`L90`); ladders come from planners, solvers or hand design. On DoorKey a random-level baseline had the higher median score (0.363 vs 0.300, Table 1 near `L276`); the checker reports random was lower in the other five settings. On shelf-place the method reached 0.617 against 0.477 for random (`L758-771`). The authors state their sign tests do not survive multiple-comparison correction at six seeds (`L539-541`).
- 2610.12470 (resets uniform over hand-derived stages, not backward): removing the dense reward gave 0% throughout even with the resets (`L556-558`). Seeds not stated.

### D. How strong plain baselines are

- 2610.40364, five quadrupeds, 5 seeds, 8192 envs, velocity-tracking reward. "DR-easy" draws fresh levels from a fixed easy distribution with no buffer or score. In-distribution means, DR-easy vs ACCEL: 90.2 vs 90.5, 61.6 vs 53.7, 33.2 vs 27.8, 59.5 vs 54.7, and 14.1 vs 17.4 on Spot (`L279-294`). By the paper's own significance test only one of those pairs separates (`L306`). On held-out terrain ACCEL was ahead of DR-easy on all five robots. Plain uniform randomisation over the full range was in no top group (`L375-376`).
- JaxUED (2403.13091), 10 seeds: uniform random levels 0.69 ± 0.05, PLR 0.72 ± 0.08, ACCEL 0.72 ± 0.05 (Table 2, `L421`). With walls capped at 25 instead of 60: uniform 0.54 against 0.17, 0.47 and 0.46 for the three curricula run (`L341-342`). The paper fixes environment interactions across methods, though the checker could not confirm that for ACCEL (`L410-415`).
- Craftax (2402.16801): the authors report most curricula "roughly the same" as random levels; only plain PLR was significantly better (`L447-451`, `L541-543`).
- SGS (2610.12465): at 64K envs SGS was about equal to uniform on five insertion tasks (for example 96.09 vs 94.24). It was far ahead on nut-and-bolt at 256K envs (90.04 vs 48.83) (`L937-948`), and on rod-in-hole at 32K it succeeded where uniform and PLR were at zero (`L358-359`). Seeds for the table are not stated.
- 1704.03003 (supervised learning, not RL): the authors call uniform sampling a "surprisingly strong benchmark" (`L508-509`).

### E. Distance and feasibility

- Optimal-transport curricula over goals (2309.14091), maze, 20 seeds: one method reached 0.62 with a shortest-path distance (its best setting) and 0 with Euclidean distance at every setting; a second method got 0.60 and 0.01-0.03 (`L618-632`).
- Raw, interpenetrating reset poses did worse than always starting at the first frame (0/10 vs 2/10, 2610.24392 above). OmniReset's authors say (Claimed) that invalid ICs destabilise learning, which is why they validate offline (`2603.15789 L331-333`).
- Ours: a success-based score cannot separate "impossible" from "not solved yet"; both read zero. The feasibility paper (2602.15817) states the asymmetry for safety under deterministic dynamics: a safe rollout proves a parameter feasible, an unsafe one proves nothing (`L297-299`).
- Go-Explore's robotics result (2004.12919) used hand-designed cells: gripper voxel, contact flags, object-in-shelf flag, door positions (`L790-800`). Its automatic cell is defined for images (`L731-764`). As published it does not meet your no-descriptor rule.

### F. Parallel PPO facts that affect reset design

- Synchronized episode clocks hurt on-policy learning with short rollouts; staggering resets fixes it (2511.21011, `L22-34`, `L142-179`). The authors also say incidental desynchronization is slow and not enough (`L106-109`, `L312-315`). Ours: a curriculum changes episode timing, so give every arm random initial time offsets; how large this confound is for your setup is unknown.
- Success per finished env was 99% while per-object success was 80%, which the authors attribute to resets after success over-counting easy objects (`2407.02274 L364-367`). Ours: count success per IC, not per finished episode.
- Forgetting: a motion-tracking paper reports the policy forgets earlier clips while learning hard ones, and that training on failures eases this but has the same problem (`2305.06456 L234-238`). In the feasibility paper, removing the rehearsal buffer raised coverage loss from 0.01 to 0.10 on one task (`2602.15817 L708-720`).
- None of 8 plain-PPO seeds at 8192 envs reached the target on a hard reorientation task (`2305.12127 L547-548`). The paper does not vary env count.
- A simulator benchmark on one chaotic scene (2607.13059): envs started from identical states diverge within a run in Isaac Lab and ManiSkill (`L345-356`) but not in MJX (`L357-366`). The same table shows MuJoCo Playground varying from run to run and MuJoCo Warp varying both ways (`L379-384`, `L417-418`). Before relying on stored-state restarts, restore one stored state in two envs with the same actions and confirm the trajectories match, in whichever backend you end up using.

## 3. Methods

Rows 1 and 8 are controls and are not ranked. Rows 2-7 are ranked by worst-case behaviour under your seven constraints, then cost, then evidence. "Existing" means already in the repo.

| # | Method | Status | Before first reward | Worst case | Extra cost |
|---|---|---|---|---|---|
| 1 | `uniform` (control) | existing | plain PPO | no learning on hard exploration | none |
| 2 | `visited`: ring buffer of visited states, uniform draw, mixed with task start | new | starts spread along what the policy already reaches; undirected | like `uniform` but with only p_task of the resets on the task start; worse than `uniform` if p_task is too low | one state copy every k steps |
| 3 | `visited` with count-weighted draw (your `tree` is in this family) | new / existing | directed toward rarely visited states | large frontier of dead ends; needs a notion of "same state" | counters plus a neighbour count or hash |
| 4 | Restart along the agent's own successful trajectories | new | inert | never starts; or sits on a fluke trajectory | a few stored trajectories |
| 5 | `sgs` over the `visited` buffer, with staleness and rehearsal | existing, extended | uniform over the buffer | uniform; impossible ICs keep their floor probability | counters per IC |
| 6 | `walk`: random-action walks from parent ICs | existing | expands outward from parents; undirected | drifts into dead states on unstable systems | walk steps, charged to budget |
| 7 | `genetic`, `ball` | existing | propose unvisited points | infeasible ICs; collapse onto few lineages | a validity rollout per child if screened |
| 8 | `point` (control) | existing | n/a | n/a | none |

**1. `uniform`.** Finding D is the reason to give it the same tuning effort as the methods it is compared against.

**2. `visited`.** The reference every buffer method has to beat; it is UFLP's B=1. Evidence: finding B. On the task-start share: 0 hurt on two of three UFLP tasks (above). In 2602.20809, 90% restarts from a regret-prioritised buffer was unstable late in training on Hex and marginal on Go; they used 50% (`L1033-1038`). In 2609.35715 (demonstration states, one robot, 3 seeds) any tested buffer share up to 0.9 gave 55-61 against 8 at zero (`L377-382`). Ours: fill the buffer from all time offsets, not only episode ends.

**3. Count-weighted `visited`.** One of two items here that are directed before reward (the other is a novelty network, score 7). Support: UFLP's score-based selection beat uniform-from-buffer on Deep Sea and Montezuma (finding B), with discrete actions. Go-Explore selects with weight 1/sqrt(count + 1) (`2004.12919 L1632-1633`); its Fetch result supports the whole method (restore, then random-action exploration, hand-designed cells), not that weight in isolation. Ways to count without a hand-designed descriptor, none tested for start states with PPO:
- Neighbour pseudocount in a buffer. 2609.08642 does this over standardised goal features with a fixed radius, for goals of an off-policy goal-conditioned agent (`L257-260`). Ours: on raw state vectors, neighbours across a wall get counted together, and in high dimensions every count tends to 1. A miscount costs efficiency, not feasibility, since every candidate is a visited state.
- A learned step-distance embedding (2407.08464): one small encoder trained on consecutive states. A side model, so it must earn its cost. Its headline comparison is at equal wall-clock, where three exploration baselines had 6-24 times fewer samples, PEG 47-81 times fewer, and METRA more (`L687-693`).
- Ours: lineage depth, the number of steps along the actual restart chain back to the task start. Needs no metric and is free. Biased toward chains that wander in place. Untested anywhere.

**4. Restart along own successes.** When any env first reaches reward, store that trajectory's states and restart from them. Two versions to compare: uniform along the stored trajectories, and a frontier that moves back when success from a window behind it passes a bar and forward when it collapses. Bars in the papers: advance at 20% (1812.03381); advance 0.8, retreat 0.15 (2607.26417). Evidence: finding C, none of it with self-generated trajectories. Risks: no hyperparameters completing the curriculum (1812.03381 on two games); the authors of Backplay argue (Claimed, a thought experiment) that a demo through a poor basin misleads the learner (`L341-344`); Ours: all envs concentrating on one path.

**5. `sgs` over `visited`.** Additions with some support:
- A staleness term in sampling. PLR found sampling with no staleness term worse than uniform (`2010.03934 L898-902`).
- Discounting old outcomes and requiring a minimum count before trusting a score (`2607.26417 L799-800`, `L554`).
- A rehearsal share for ICs that were solved and now fail (2602.15817, finding F).
- Ours, adapted from that paper's observed-safe set: an "ever succeeded" bit per IC, to separate solved-before from never-solved.
- Lanes: several envs share one IC so success per IC is measured within one iteration (`2610.40364 L205-208`).

**6. `walk`.** The reverse-curriculum paper (1707.05300) and the Voronoi-biased curriculum (2610.03395) grow the IC set by random-action walks, so children are reached states. In 2610.03395 the backward half starts from a known goal state; the forward-only ablation never reached 75% on two of eight tasks (`L500-506`). BaRC's related-work section says (Claimed) that for unstable systems random actions from a good start are likely to land far away and unlikely to give good starts (`1806.06161 L85-93`). Ours: an inverted pendulum is that kind of system, so test `walk` there first.

**7. `genetic`, `ball`.** They move points in a box, so children are not visited states (see finding E). If kept, screen each child with a short hold rollout before admitting it and charge the steps: 2609.35715 holds with gravity disabled and rejects on termination or excess speed (`L688-710`); 2609.13761 uses a 2 s rollout. Ours: a hold test still passes stable states from which the task cannot be solved. ACCEL reports concentration onto ten parent levels in a small-buffer ablation without the generator (`2203.01302 L1275-1278`). For `genetic`: an evolutionary-PPO paper (about populations of policies, not ICs) runs selection only when the fitness spread exceeds a threshold relative to the median (`2503.19037 L298-312`); the analogue would stop culling while all scores are near zero.

**8. `point`.** Keep as a control.

Lower priority: a learned generator of levels or goals (PAIRED, Goal GAN, 2410.19715) adds a network and outputs unvisited states. A second policy for regret adds a second learner; in one buffer test it scored lower than positive value loss, 0.50 ± 0.14 vs 0.61 ± 0.03 over 5 seeds, a gap inside the spread (2308.10797). The world-model method 2411.01396 lists 36 hours of runtime for 1e6 steps on two of its tasks (`L1130-1134`).

## 4. Scores

| # | Score | Status | Before first reward | Main weakness | Cost |
|---|---|---|---|---|---|
| 1 | Success band: `sgs`, p(1-p) | existing | flat | cannot tell coin-flip ICs from learnable ones | counters |
| 2 | Visit count, 1/sqrt(count + 1) | new | informative | needs a notion of "same state"; ignores reward | counters |
| 3 | `slope` (return slope) | existing | flat | noisy; can down-weight stuck-but-unsolved ICs | window per IC |
| 4 | Positive value loss, MaxMC, mean \|GAE\| per IC | new | critic noise | value loss and MaxMC track success rate; \|GAE\| has thin evidence | free from PPO |
| 5 | Critic value at the start state | new | critic noise | only as good as the critic on rarely seen states | one critic call |
| 6 | Steps-to-reward along a stored successful trajectory | new | undefined | defined only on those trajectories | free |
| 7 | Learned side models: success classifier, novelty network, step-distance embedding | new | classifier flat; novelty informative | cost; targets keep moving | a network each |

**1. Success band.**
- For: AlphaNIST, same bank, uniform vs success-band sampling at 256K envs: 84.3% vs 97.4% (`L432-436`; number of training runs not stated). SGS on its hardest tasks (finding D).
- Not evidence: in "No Regrets" (2408.15099) the p(1-p) buffer holds levels with mean learnability 0.22 against 0.01-0.02 for value-loss and MaxMC buffers (`L1539-1550`). That buffer is the top-K by p(1-p), so this is true by construction.
- Mixed: in 2610.40364, learnability inside ACCEL scored 83.8 against 91.2 for the matching value-loss variant on one robot, and had the higher mean on two others.
- Shape may not matter: SGS's sensitivity table (locomotion, 32K envs, 2 seeds per setting) gives 0.42-0.52 for target success rates from 0.33 to 0.90 (`L854-886`). With 2 seeds, noise is not ruled out.
- Weakness: DemoStart reports a "dice-roll" state (a cube dropped from height) that kept passing its mixed-outcome filter late into training (`2409.06613 L853-862`). 2402.12284 builds test levels where a regret score is high and cannot be reduced; that a success band gets stuck the same way is Ours. Ours: down-weight an IC whose success rate has not moved over K visits.

**2. Visit count.** See method 3. Ours: before the first reward use the count alone; afterwards use count × (learnability + ε). 2609.08642 multiplies a reachability estimate by novelty for goals and reports the additive form less stable (`L190-199`); a plain product with a reward-based score would be zero everywhere before reward, hence the ε.

**3. `slope`.** In 2610.40364 the learning-progress variant had the best pooled in-distribution score (0.94) and a weak pooled held-out score (0.61), with a seed spread of 12.7 on one robot (Table 10; `L290`, `L388`). A multi-task paper found a return-slope score down-weighted a stuck unsolved task and was slower than uniform there (`2605.14350 L458-461`). If kept, also test the absolute-value form (1707.00183) and fast-minus-slow averages of success (2106.14876, one seed).

**4. Value loss, MaxMC, |GAE|.** Free, which is the reason to try them. On one JaxNav policy, MaxMC and positive value loss correlated with success rate at r = 0.61 and 0.51 (`2408.15099 L251`); the paper describes value loss as weakly correlated with learnability. L1 value loss correlated at 0.07 with success rate and with return (its Fig. 13). In a driving benchmark, PLR with mean |GAE| beat random levels only "with a small margin" (`2608.22549 L436`), while the checker notes the same passage rates value loss among the better scores. Evidence here conflicts; treat as a cheap experiment, not a recommendation.

**5. Critic at the start state.** V(s0)·(Vmax - V(s0)) is the uniform-target ablation of 2405.02481; it costs one batched critic call and that paper tunes a temperature per environment (β = 90, 110, 130, `L851-853`). 2512.22824 uses the variance over time of mean Q per goal with an off-policy learner; using variance of V(s0) for PPO is Ours. Neither paper isolates the pre-reward phase.

**6. Steps-to-reward.** Ours: an ordering for method 4 that needs no state-space metric (Backplay orders by step index along the demo).

**7. Side models.** You said these must pay for themselves, so log wall-clock per iteration with and without.
- Success classifier with a band (2311.09195 uses 0.3-0.7). Ours: it generalises across states where per-IC counters cannot, and is flat before first success. In 2602.15817 a feasibility classifier beat two density models on one task (0.61 vs 0.40 and 0.27 coverage gain, `L734-746`).
- Novelty network (RND). In 2602.15817, RND as a reward bonus on top of uniform sampling gained 0.16 and 0.01 on two tasks where the paper's method gained 0.61 and 0.64 (`L1557-1588`). That tests RND as a bonus, not as an IC score.
- Step-distance embedding. The offline goal-reaching paper 2307.11949 reports that learned value gets noisier as distance grows (`L258-262`, `L592-596`). Ours: use such an embedding for near / not-near decisions, not for ranking far pairs.

Low priority: second-policy regret (cost; method list above). L1 value loss (the one correlation above). Per-IC update-size scores: 2605.01358 scores levels by the norm of the policy-parameter change and selected hyperparameters on test levels in one of its domains. Early-termination length (2507.04606); Ours: it is constant when nothing terminates early, as in your maze and pendulum. A learnability-only bandit in LLM fine-tuning was +0.3 over random sampling on one model and -0.7 on another (`2606.25178 L557-564`), which is weak evidence outside RL control.

## 5. Tasks

| # | Task | Status | What it tests |
|---|---|---|---|
| 1 | `maze`, plus a narrow-passage variant and a size sweep | existing + new | constraints 1 and 4 together |
| 2 | Sparse versions of `pendulum` and `arm`; sparse cartpole / acrobot swing-up | new variants | constraint 1 on unstable dynamics |
| 3 | Dead-region twin: an irreversible region plus a reversible copy | new | constraint 5 for visited-state methods |
| 4 | Infeasible-mass start region | new | constraint 5 for generated ICs |
| 5 | Coin-flip region | new | whether a score gets absorbed by noise |
| 6 | Deceptive task | new | where plain PPO fails outright |
| 7 | `claw`, dense `pendulum`, dense `arm` | existing | sanity; a curriculum should not hurt |

1. **Maze.** Keep it. 2610.03395 has an N-maze with narrow passages: vanilla PPO succeeded on a few seeds in the U-maze only (`L387-388`), a random curriculum plateaued near 40% and a reverse curriculum stayed below about 35% in the N-maze (`L389-392`). Ours: add it as the harder maze. Where a method uses a distance, run it with Euclidean and with a path-aware distance (finding E). Sweep maze size; 1803.10227 runs grid sizes 5 to 20 and reports the advantage growing (`L233-244`; discrete, DDQN).
2. **Sparse classics.** Reward = success indicator plus small penalties, as in SGS's manipulation reward (`2610.12465 L801-834`). MuJoCo Playground ships AcrobotSwingupSparse, CartpoleBalanceSparse and CartpoleSwingupSparse on GPU (`2502.08844 L593-598`).
3. **Dead-region twin.** From REVERSAL-BENCH (2609.17745): a parameter scales an irreversible region, and a twin with the same geometry makes the trap escapable (`L96-105`, `L117-145`). Ours: report the share of the IC pool from which no rollout has ever succeeded.
4. **Infeasible mass.** The feasibility paper's ToyLevels puts 90% of mass on easy, 0.01% on hard-but-feasible and 9.99% on infeasible parameters (`2602.15817 L2216-2217`). TeachMyAgent (2103.09815) uses mostly-infeasible and mostly-trivial variants of a task-parameter space the same way.
5. **Coin-flip region.** Ours: a block of ICs where success is random and outside the policy's control, added to the maze. Modelled on the irreducible-regret levels in 2402.12284, which test a regret-maximising sampler.
6. **Deceptive task.** Kinetix's Car-Ramp, where the agent must first move away from the goal (finding A). Ours: a reset method that restarts past the deceptive part may solve it; plain PPO from the task start did not.
7. **Existing dense tasks.** Finding D: a gain here may come from the easy start.

## 6. Controls every run needs

Except where a paper is cited for the design, these are Ours.

1. **Same candidate set, uniform draw.** Separates the score from the set (AlphaNIST's uniform-vs-band ablation; UFLP's B=1).
2. **Initial IC set, no adaptation.** Modelled on DR-easy in 2610.40364, which samples a fixed easy distribution with no buffer or score.
3. **Inverted score, and random score.** 2502.15662 runs an anti-curriculum (`L391-394`). If either does as well, the score is not doing the work.
4. **Equal environment steps, counting every scoring, screening and walk step.** JaxUED fixes environment interactions and charges replay and mutation cycles (`2403.13091 L316-325`), though it leaves out one method's level-editor steps (`L327-328`).
5. **Evaluate only from the task start, on fixed states** (already in the repo). Also log return on task-start envs next to return on curriculum envs during training; 2402.03479 defines a similar gap for levels (`L347-355`).
6. **Random episode-time offsets in every arm** (finding F).
7. **Success per IC, not per finished episode** (finding F), plus worst-group success, since a mean can hide a failed group (`2607.26417 L87-88`).
8. **Share of ICs that never succeeded, and buffer concentration,** over training.
9. **Seeds.** 2108.13264 recommends interquartile mean with bootstrap intervals (`L127-161`). 2304.01315 says "in almost all cases 5 runs is insufficient to make strong claims" (`L846-847`), suggests paired comparisons (`L1723-1724`), and warns that an interquartile mean on a single environment hides failed runs (`L743-758`). Report how many seeds end at zero.
10. **Sweep p_task including 0 and 1.**

## 7. Suggested order of runs (Ours)

Cheapest first. Fix in advance, per task, the number of seeds and the gap that counts as a difference; drop a method when its control is within that gap.

1. `maze` and sparse `pendulum`: `uniform`, `point`, `visited` at p_task in {0, 0.2, 0.5, 0.8}. Measure the share of seeds that reach any reward within the budget, and steps to first reward for those that do.
2. Add count-weighted `visited` with the two cheapest counts (raw-state neighbour count; lineage depth). In the maze, log how many counted neighbours lie across a wall; that is the measurement that decides step 6.
3. Once some run reaches reward: restart along stored successful trajectories, uniform and with a frontier.
4. `sgs` scoring over the buffer with staleness and rehearsal, against uniform over the same buffer.
5. Dead-region twin and infeasible-mass variants, to measure wasted resets for `visited`, `walk`, `ball`, `genetic`.
6. Only if step 2 shows many across-wall neighbours and no gain over uniform-from-buffer: try the learned step-distance embedding, with wall-clock logged.

## 8. Limits of this review

- 85 papers got a light pass, not a full read.
- Many figures did not survive text extraction. Where a result exists only as a curve, it is given as the authors' prose or left out.
- 34 of the 206 papers have arXiv ids of 2607 or later. Those could be checked only against their own text.
- The orderings in Sections 3-5 are judgments. No paper compares these options head to head under your constraints.
- Claim check: after drafting, five independent readers checked every cited claim against the paper text without access to the reading notes. Of 152 claims, 94 were supported as written, 54 were right in substance but needed a corrected number, scope or label, 2 were wrong, and 2 could not be checked. All 58 were corrected or cut in this version. The corrections themselves were not re-checked by a third reader, apart from three spot checks against the text (SGS at 32K envs, UFLP on Montezuma, the 2411.01396 runtime).
