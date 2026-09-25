module MuKumariIntegration

import VulcanJ, POMDPs
import VulcanJ: state_time, extract_location, observation_history,
    generative_problem

export state_time, extract_location, observation_history, generative_problem,
    action, simulate_info_path
using VulcanJ: AbstractInfoMCTS, AbstractErgodicSolver, RiskBoundedInfoPolicy,
    ErgodicPolicy, conditional_observation_distribution
using MuKumari: MuKumari
import POMDPs: action
using POMDPTools: Deterministic, stepthrough
import VulcanJ: simulate_info_path

const KProblem = Union{MuKumari.KAgentMDP,MuKumari.KAgentPOMDP}
include("objectives.jl")
VulcanJ.state_time(problem, state::MuKumari.KAgentState) = MuKumari.t(state)

const VulcanPlanner = Union{AbstractInfoMCTS,AbstractErgodicSolver}

VulcanJ.extract_location(state::MuKumari.KAgentState) = MuKumari.state(state)

function VulcanJ.observation_history(::VulcanPlanner, ::KProblem, state::MuKumari.KAgentState)
    n = MuKumari.t(state)
    return ((location = i <= n ? state.hist[i] : MuKumari.state(state),
             observation = i <= n ? state.z[i] : MuKumari.z(state)) for i in 1:n+1)
end

# Rebind only the environment callbacks. MuKumari's own generator owns motion,
# measurement history, rewards, and POMDP observation construction.
function VulcanJ.generative_problem(problem::KProblem, model, rng)
    reading = Ref{Any}()
    names = problem.menv.μ_order
    callbacks = Dict{Symbol,Function}()
    for (i, name) in enumerate(names)
        callbacks[name] = X -> begin
            if i == 1
                reading[] = vcat(rand(rng, conditional_observation_distribution(problem, model, X)))
            end
            reading[][i]
        end
    end
    environment = MuKumari.MuEnv(length(names), names, callbacks)
    return typeof(problem)(; (key => (key == :menv ? environment : getfield(problem, key))
                              for key in fieldnames(typeof(problem)))...)
end

# The simulator owns transitions; successive states carry the acquired readings.
previous_action(policy::RiskBoundedInfoPolicy) = policy.root.branches[policy.root.best].action
previous_action(policy::ErgodicPolicy) = policy.selected_action

for PolicyType in (RiskBoundedInfoPolicy, ErgodicPolicy)
    @eval function POMDPs.action(policy::$PolicyType{P}, state::MuKumari.KAgentState) where {P<:KProblem}
        if !isnothing(policy.model) && MuKumari.t(state) > MuKumari.t(policy.root_state)
            VulcanJ.update_environment_model!(policy,state,previous_action(policy))
        end
        return invoke(POMDPs.action,Tuple{$PolicyType,Any},policy,state)
    end
    @eval POMDPs.action(policy::$PolicyType{P}, belief::Deterministic{MuKumari.KAgentState}) where {P<:KProblem} =
        action(policy, belief.val)
end

# Use the same POMDPTools simulator as MuKumari.stepthrough_sim, retaining its
# actual successor instead of regenerating the final transition in that wrapper.
function VulcanJ.simulate_info_path(problem::MuKumari.KAgentMDP,
    policy::Union{RiskBoundedInfoPolicy,ErgodicPolicy}, n_steps::Integer;
    initial_state, model, rng=policy.solver.rng,
    observe_fn=(p,s)->MuKumari.z(s), update_model=true, mission_start_time=0)
    VulcanJ.initialize_simulation!(policy,initial_state,model,n_steps,mission_start_time)
    states, taken, observations = Any[initial_state], Any[], Any[]
    rewards, risks = Float64[], Float64[]
    steps = stepthrough(problem,policy,initial_state,"s,a,sp";rng,max_steps=n_steps)
    cursor = nothing
    for i in 1:n_steps
        POMDPs.isterminal(problem,policy.root_state) && break
        # Plan before advancing the simulator: an infeasible policy has no action.
        # Its subsequent action query reuses the cached plan.
        isnothing(action(policy,policy.root_state)) && break
        step, cursor = i == 1 ? iterate(steps) : iterate(steps,cursor)
        s,a,sp = step
        step = VulcanJ.update_environment_model!(policy,sp,a;
            observation=observe_fn(problem,sp),update_model)
        push!(states,sp); push!(taken,a); push!(observations,step.observation)
        push!(rewards,step.gain); push!(risks,step.risk)
    end
    return (;states, actions=taken, observations, information_rewards=rewards, risks,
             model=policy.model, policy)
end

end
