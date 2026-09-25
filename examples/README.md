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
still save to `examples/res/<group>/<example>/path.png`; paths are anchored to
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

## Interpretation

Each script requests 60 actions and saves `res/<group>/<example>/path.png`.
For example, `MuKumari/ergodic_conditional_gp.jl` writes
`res/MuKumari/ergodic_conditional_gp/path.png`. Conditional
examples call `simulate_info_path` with their already-conditioned starting model,
execute through the original problem, update from acquired measurements, and replan. Full-trajectory examples return a predicted realization
without executing it. Figures identify paths as executed or predicted.

Absolute-field objectives use the magnitude of SCRIBE's posterior mean, obtained
through its public prediction API. Information objectives use the integration's
expected-information method. Most backgrounds show the initial planning objective; conditional planning
recomputes it as the posterior changes. The combined performance example saves
the absolute SCRIBE field/path as `path.png` and its separate unobserved-information
map as `acquisition.png`. These are distinct
spatial targets, rather than two labels for the same density.

MuKumari examples use the scale demonstrated in its examples: a 0–10 world,
speed 1.0, agent width 0.1, and three-decimal rounding. Model length scales,
obstacles, cellsites, and ergodic controls use the same spatial units.
Obstacles are actual polygons in its traversable world. These examples explicitly use collision-blocking `KAgentMDP` dynamics. The
riskless cases pass `risk_probability=score->0.0`. Performance-guided cases use
`extract_obstacle_risk` with an explicit conversion `score -> -expm1(k*score)`
of MuKumari's summed negative obstacle penalties, with `k=1.0` for GP and `k=0.5`
for SCRIBE. The information objective and default `performance_alpha` schedule
remain independent choices. The displayed
execution paths are nonfailure realizations; execution stops early if the search
finds no admissible continuation within its budget. This does not prove that no
feasible path exists.
Unobserved models wrap the GP or SCRIBE posterior with a presence cutoff of 0.5,
and receive existing history through the integration's accessor.

The SCRIBE problem outside MuKumari defines its own simple motion and history.
Its custom `:absolute_field` objective belongs to the demonstration. MuKumari
risk extraction is supplied by the integration package, not reimplemented here.

MCTS examples use a 12-step planning lookahead. The search budget per decision
is one second for performance-guided SCRIBE models and 0.3 seconds otherwise.
Conditional ergodic examples use a 30-step lookahead; both planner families
request 60 actions so their coverage is visually comparable. Finite-risk examples
use a mission budget of 0.6. Standalone SCRIBE examples retain their own
exposure field (scale 0.5, approximately 0.001–0.0085 per action). MuKumari
examples instead evaluate obstacle-derived risk at the propagated successor;
there is no added uniform exposure baseline or action-independent Gaussian field.
See the [adapter choices](../README.md) for reward-derived, obstacle-derived,
crossing, and explicit objective extraction.

Spatial resolutions serve separate purposes: objective evaluation and ergodic
target construction use 21×21 sites (normalized spacing 0.05); figures evaluate
the background independently on a 51×51 grid. The default ergodic density
bandwidth is 0.075 in normalized coordinates. Unobserved-phenomena examples
explicitly retain their 6×6 phenomenon-cell partition through `sites`, independent
of both grids. These resolutions do not quantize model predictions or motion.

MuKumari execution uses the integration's `simulate_info_path` specialization and
POMDPTools `stepthrough`, the simulator used by MuKumari itself. It records the
actual successor, including the final acquired observation, without regenerating
that transition. MuKumari policy dispatch also consumes the newest history record
when an external simulator presents the next state to `action`. Repeated queries
at the same timestamp do not condition the observation again. Initialize the
policy with `set_environment_model!` before using an external simulator; use
`update_environment_model!(policy, final_state, final_action)` to consume a final
successor when no subsequent action is requested. The supplied simulation helper
already handles this final update.

All figures call VulcanJ's `plot_simulated_path`, with explicit domain bounds,
background labels, and obstacle polygons. SCRIBE predictions and information
remain delegated to its public model APIs; objective maps retain their absolute
field or information meaning rather than being replaced with posterior-mean maps.

Thresholded examples explicitly use `ThresholdPresence(0.5)`, enabling predictive
Gaussian tail probabilities instead of threshold-indicator quadrature. The
combined performance acquisition map uses higher-order outcome quadrature and
retains true phenomenon-cell boundaries. It contains no obstacle penalty; the
obstacle polygon is a separate physical overlay. For a traversable POMDP risk
region, the plotting function supports `obstacle_alpha=0.2` and
`obstacle_label="Risk region"` instead of an opaque barrier.
