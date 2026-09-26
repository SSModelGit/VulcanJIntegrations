using POMDPTools: PlaybackPolicy
using VulcanJ: simulate_info_path
include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_scribe, initial_reading, spatial_bounds, obstacle_polygons, ground_truth, learned_field
using VulcanJ: RiskBoundedInfoMCTS, plan_trajectory, plot_environment_comparison
using VulcanJIntegrations: condition_environment_model
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_scribe(;risk_scale=0.0)
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:mutual_information)
plan=plan_trajectory(solver,problem,state,model,60;objective)
result = simulate_info_path(problem,PlaybackPolicy(plan.actions),length(plan.actions);
    initial_state=state,model,rng,observe_fn=(p,s)->initial_reading(s))
plot_environment_comparison(problem,result.states,result.observations;
    ground_truth_fn=X->ground_truth(problem,X),learned_fn=X->learned_field(result.model,X),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    save_path=joinpath(@__DIR__,"..","res","SCRIBE","riskless_trajectory","ground_truth_and_learned.png"),
    title="SCRIBE — riskless trajectory")
println("Executed actions: ",length(result.actions)," / 60")
