include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_gp, initial_reading, spatial_bounds, obstacle_polygons, ground_truth, learned_field
using VulcanJ: RiskBoundedInfoMCTS, plot_environment_comparison
using VulcanJIntegrations: condition_environment_model, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_gp(;obstacles=true,risk_probability=score->0.0)
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:mutual_information)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
plot_environment_comparison(problem,result.states,result.observations;
    ground_truth_fn=X->ground_truth(problem,X),learned_fn=X->learned_field(result.model,X),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    save_path=joinpath(@__DIR__,"..","res","MuKumari","riskless_conditional_gp","ground_truth_and_learned.png"),
    title="MuKumari — riskless conditional gp")
println("Executed actions: ",length(result.actions)," / 60")
