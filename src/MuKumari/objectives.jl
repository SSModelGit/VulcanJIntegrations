# Explicit interpretations of MuKumari's composed preferences. None is installed
# as the problem's risk or objective automatically.
export extract_reward_risk, extract_obstacle_risk, extract_objective,
    crossing_exposure

obstacle_groups(problem) = [(kind, descriptors) for (kind, descriptors) in
    problem.objl.objectives if kind in (:sobc, :robc)]

struct MuKumariObjective{F,P,D}
    reward::F
    projection::P
    density::D
end
(objective::MuKumariObjective)(state) = first(objective.reward(state))

"""
    extract_objective(problem; projection=nothing, density=nothing)

Select MuKumari's composed reward as the planning objective. For an ergodic
spatial target, explicitly supply `projection(location)` returning a MuKumari
state and `density(reward)` returning a nonnegative value. No sign conversion or
mixing with information gain is performed implicitly.
"""
extract_objective(problem::KProblem; projection=nothing, density=nothing) =
    MuKumariObjective(problem.obj, projection, density)

VulcanJ.information_gain(objective::MuKumariObjective, problem, prior, posterior,
    state, observation) = objective(state)
VulcanJ.expected_information_gain(objective::MuKumariObjective, problem, model,
    state::MuKumari.KAgentState, order) = objective(state)
VulcanJ.expected_information_gain(objective::MuKumariObjective, problem, model,
    location::AbstractMatrix, order) = objective.density(objective(objective.projection(location)))

endpoint_exposure(score, state, successor) = score(successor)

struct MuKumariRisk{P,S,F,E,M}
    problem::P
    score::S
    probability::F
    exposure::E
    motion::M
end

function (risk::MuKumariRisk)(state, action)
    return sum(risk.motion[action]) do outcome
        position = MuKumari.physical_step(risk.problem, MuKumari.state(state), outcome.observation)
        # This is a geometry/time projection, not an acquired observation. The
        # built-in composed reward uses position and history length, not new z.
        successor = MuKumari.pseudo_agent_placement(state, position)
        push!(successor.hist, MuKumari.state(state))
        outcome.weight * risk.probability(risk.exposure(risk.score,state,successor))
    end
end

function extracted_risk(problem, score, probability, exposure, order)
    motion = Dict(a => VulcanJ.observation_outcomes(MuKumari.motion_distribution(problem,a),order)
                  for a in POMDPs.actions(problem))
    return MuKumariRisk(problem,score,probability,exposure,motion)
end

"""
    extract_reward_risk(problem; probability, order=5)

Return `(state, action) -> risk`, applying the supplied reward-to-probability
mapping to each motion outcome before averaging. Reuses the composed reward,
including goal offsets, rounding and the successor's horizon term. Does not
query the environmental sensor or change MuKumari's physical propagation.
"""
extract_reward_risk(problem::KProblem; probability, order=5) =
    extracted_risk(problem, s->first(problem.obj(s)), probability, endpoint_exposure, order)

"""
    extract_obstacle_risk(problem; probability, order=5, exposure=endpoint_exposure)

Extract every `:sobc` and `:robc` group, excluding goals and horizon penalties.
Endpoint exposure preserves MuKumari's polygon-distance and centroid-distance
penalty functions. The supplied probability mapping receives their summed
negative reward. Motion uncertainty is integrated after that mapping.

Explicitly pass `exposure=crossing_exposure(problem)` for a different meaning:
penalize crossing/touching an obstacle using its `:impact`, rather than proximity.
"""
function extract_obstacle_risk(problem::KProblem; probability, order=5, exposure=endpoint_exposure)
    functions = [kind == :sobc ? s->first(MuKumari.safety_obj(s,descriptors)) :
                                s->first(MuKumari.risk_obj(s,descriptors))
                 for (kind,descriptors) in obstacle_groups(problem)]
    score = s -> sum(f(s) for f in functions; init=0.0)
    return extracted_risk(problem,score,probability,exposure,order)
end

"""
    crossing_exposure(problem)

An explicitly selected segment-exposure rule for `extract_obstacle_risk`.
Returns minus the sum of impacts of obstacles intersecting the propagated
movement segment (including endpoints). This catches thin regions even when
both endpoints lie outside. Blocking/truncation remains MuKumari's responsibility.
"""
function crossing_exposure(problem::KProblem)
    obstacles = [(MuKumari.GI.Polygon([o[:poly]]),o[:impact])
                 for (_,group) in obstacle_groups(problem) for o in group]
    return function (score,state,successor)
        start, stop = Tuple(MuKumari.state(state)), Tuple(MuKumari.state(successor))
        movement = MuKumari.GI.LineString([start,stop])
        crosses(polygon) = MuKumari.GO.intersects(movement,polygon)
        return -sum(impact for (polygon,impact) in obstacles if crosses(polygon); init=0.0)
    end
end
