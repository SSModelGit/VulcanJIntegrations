include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_gp, initial_reading, spatial_bounds, obstacle_polygons
using VulcanJ: RiskBoundedInfoMCTS, plot_simulated_path
using VulcanJIntegrations: condition_environment_model, expected_information_gain, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_gp(;obstacles=true,risk_probability=score->0.0)
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:mutual_information)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
background=X->expected_information_gain(objective,problem,model,X,3)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","MuKumari","riskless_conditional_gp","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="MuKumari riskless conditional gp (executed)",observation_fn=background)
