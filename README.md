# VulcanJIntegrations

Load alongside VulcanJ, MuKumari and SCRIBE. The package reexports its adapters;
none of the following extraction choices is installed automatically.

## MuKumari risk and objectives

```julia
using MuKumari: MuKumari
using VulcanJIntegrations: extract_reward_risk, extract_obstacle_risk,
    extract_objective, crossing_exposure

# The conversion is part of the user's problem definition.
probability = reward -> -expm1(0.5 * min(reward, 0.0))
reward_risk = extract_reward_risk(problem; probability)
obstacle_risk = extract_obstacle_risk(problem; probability)
objective = extract_objective(problem)
```

The risk objects are called as `risk(state, action)`. Bind the selected one to
the problem's existing `VulcanJ.get_failure_prob` specialization. The examples
store it in a `Dict` keyed by problem. Independently pass `objective` to `solve`
or keep VulcanJ's information objective.

- `extract_reward_risk` uses the reward component of `problem.obj`, including
  goal offsets, horizon penalties and MuKumari's rounding.
- `extract_obstacle_risk` sums the native penalties of **all** `:sobc` and `:robc`
  groups. It excludes goals and horizon terms. `:sobc` uses polygon distance;
  `:robc` uses centroid distance. Neither is inherently a failure probability.
- `extract_objective` selects the composed reward itself as the planning score,
  without automatically adding information gain or interpreting reward as risk.

Risk evaluation uses MuKumari's motion distribution and `physical_step`, with
Gauss–Hermite motion quadrature (`order=5`, one outcome for zero motion noise).
It converts each outcome's score to probability **before** averaging. A projected
successor advances history length for the built-in horizon reward, retains the
existing measurements, and never queries the sensor. This projection is only
for geometric/time preferences; it is not an acquired environmental observation.

Endpoint proximity and crossing a region are explicit, different choices:

```julia
crossing_risk = extract_obstacle_risk(problem; probability,
    exposure=crossing_exposure(problem))
```

Crossing exposure returns minus the sum of `:impact` values for obstacle polygons
touched by the propagated segment, including endpoints. It detects narrow regions
with both endpoints outside. This explicitly replaces the proximity interpretation;
it does not pretend that a centroid penalty is a polygon-intersection probability.

The adapter preserves MuKumari's current dynamics: `KAgentMDP` blocks obstacle
holes and outer walls; `KAgentPOMDP` blocks outer walls only. `:sobc`/`:robc` choose
penalties, not the physical collision mode, and neither currently terminates on
collision. MuKumari's world construction takes the first obstacle group; risk
extraction still includes every group. The adapter does not silently rebuild the
world or change the problem type.

For an ergodic target, supply an explicit spatial projection and nonnegative
conversion of signed reward:

```julia
objective = extract_objective(problem;
    projection=X -> MuKumari.pseudo_agent_placement(reference_state, X),
    density=reward -> exp(reward))
```

Here the caller chooses the reference time/history for a spatial snapshot. The
composed reward remains the realized score; `density` is only its spatial target
representation. No conversion is inferred from signs or solver settings.

## Threshold presence and figures

`VulcanJ.ThresholdPresence(c)` explicitly specifies the scalar event `z > c`.
VulcanJ uses predictive tail probabilities for scalar distributions with a CDF;
the SCRIBE adapter uses public predictive Gaussian moments for its scalar tail.
Other presence functions retain observation quadrature. This avoids integrating
a discontinuous threshold with a handful of nodes while preserving phenomenon
cells and the underlying model.

See [examples](examples/README.md) for execution commands and figure meanings.
