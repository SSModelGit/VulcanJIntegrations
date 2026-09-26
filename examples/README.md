# VulcanJIntegrations examples

Run from this `examples/` directory with Julia 1.11 or later:

```sh
julia --project=. -e 'using Pkg: Pkg; Pkg.instantiate()'
julia --project=. MuKumari/ergodic_conditional_gp.jl
```

To run all 13 examples, from this `examples/` directory:

```sh
julia --project=. run_all_examples.jl
```

Equivalently, from the `VulcanJIntegrations` package directory, run
`julia --project=examples/ examples/run_all_examples.jl`.

The runner visits MuKumari, SCRIBE, then MuKumariSCRIBE, with filenames sorted
within each group. Each example runs in a separate process, one at a time, with
one Julia thread and one BLAS thread. A failed example stops the run. Figures
still save to `examples/res/<group>/<example>/ground_truth_and_learned.png`; paths are anchored to
the example files, independently of the working directory.

`Project.toml` selects the local integration package, VulcanJ, and sibling
MuKumari/SCRIBE checkouts through `[sources]`. Change those dependency locations
when using a different checkout layout. Example code is self-contained here;
it does not include core VulcanJ examples. There is no integration `test/` folder.

Each script explicitly imports and uses `VulcanJIntegrations` adapter functions.
The shared `problems/` files define demonstration problems and configuration,
not execution loops, plotting implementations, or replacement model/state adapters. In particular, MuKumari
state extraction, timestamps, history, and posterior binding come from the
package, as do SCRIBE prediction, conditioning, and information computations.

## MuKumari

| File | Demonstration |
| --- | --- |
| [ergodic_conditional_gp.jl](MuKumari/ergodic_conditional_gp.jl) | Ergodic conditional; GP |
| [riskless_conditional_gp.jl](MuKumari/riskless_conditional_gp.jl) | Riskless conditional; obstacles and GP |
| [performance_conditional_unobserved.jl](MuKumari/performance_conditional_unobserved.jl) | Performance-guided conditional; obstacles and thresholded unobserved GP |

## SCRIBE

| File | Demonstration |
| --- | --- |
| [ergodic_trajectory_absolute.jl](SCRIBE/ergodic_trajectory_absolute.jl) | Ergodic full trajectory; absolute posterior mean field |
| [ergodic_trajectory_information.jl](SCRIBE/ergodic_trajectory_information.jl) | Ergodic full trajectory; information density |
| [riskless_trajectory.jl](SCRIBE/riskless_trajectory.jl) | Riskless full trajectory |
| [riskless_conditional.jl](SCRIBE/riskless_conditional.jl) | Riskless conditional |
| [performance_conditional.jl](SCRIBE/performance_conditional.jl) | Performance-guided conditional |
| [performance_conditional_unobserved.jl](SCRIBE/performance_conditional_unobserved.jl) | Performance-guided conditional; thresholded unobserved SCRIBE |

## MuKumariSCRIBE

| File | Demonstration |
| --- | --- |
| [ergodic_trajectory_absolute.jl](MuKumariSCRIBE/ergodic_trajectory_absolute.jl) | Ergodic full trajectory; absolute SCRIBE posterior mean field |
| [ergodic_conditional_absolute.jl](MuKumariSCRIBE/ergodic_conditional_absolute.jl) | Ergodic conditional; absolute SCRIBE posterior mean field |
| [riskless_trajectory.jl](MuKumariSCRIBE/riskless_trajectory.jl) | Riskless full trajectory; obstacles |
| [performance_conditional_unobserved.jl](MuKumariSCRIBE/performance_conditional_unobserved.jl) | Performance-guided conditional; obstacles and thresholded unobserved SCRIBE |

## Execution and figures

Every example requests 60 actions and writes
`res/<group>/<example>/ground_truth_and_learned.png`, independently of the working
directory. The 1×2 figure shows actual ground truth on the left and the final
learned posterior mean with the executed exploration path on the right. Both
panels use the same signed field scale. Even when planning uses absolute field
magnitude or unobserved-information gain, the learned panel shows the environmental
field itself. Unobserved-phenomena wrappers delegate this prediction to their
underlying GP or SCRIBE model.

Conditional MCTS updates its posterior and replans after every acquired sample.
Ergodic planning holds its initial model and target density fixed for the entire
requested horizon. Conditional action calls track one reference trajectory;
collected observations update the model as one batch after execution. Full-path
examples execute the planned actions using POMDPTools `PlaybackPolicy` before
conditioning on the actual observations. The initial reading is already in the
starting posterior and is not counted again.

MuKumari execution uses its native POMDPTools `stepthrough` simulator. SCRIBE
prediction and batch conditioning use its public `posterior_model_moments` and
`condition_on_measurement` functions. Figures use VulcanJ's
`plot_environment_comparison`, which composes `plot_simulated_path` panels using
Plots. No example implements its own simulation or plotting loop.

MuKumari examples use collision-blocking `KAgentMDP` dynamics, a 0–10 world,
speed 1, agent width 0.1, and three-decimal rounding. Riskless examples explicitly
map risk to zero. Performance examples select obstacle-derived risk using
`score -> -expm1(k*score)`, with `k=1` for GP and `k=0.5` for SCRIBE. Their information
objectives remain independent of risk extraction. See the [adapter choices](../README.md).

MCTS uses a 12-step lookahead and a 0.3-second decision budget, or one second for
performance-guided SCRIBE examples. Ergodic references cover the full 60-step
execution horizon. Finite-risk examples use a mission budget of 0.6. Scripts
report executed length; search can stop before the requested horizon.

Objective integration uses 21×21 sites, unobserved-phenomena models retain their
6×6 cell partition and threshold 0.5, and figures evaluate fields on 101×101 points.
These display and integration grids do not discretize predictions or motion.
