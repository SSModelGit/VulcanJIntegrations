include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_scribe, initial_reading, spatial_bounds, obstacle_polygons
using VulcanJ: RiskBoundedInfoMCTS, plan_trajectory, plot_simulated_path
using VulcanJIntegrations: condition_environment_model, expected_information_gain
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_scribe(;risk_scale=0.0)
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:mutual_information)
result=plan_trajectory(solver,problem,state,model,60;objective)
background=X->expected_information_gain(objective,problem,model,X,3)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","SCRIBE","riskless_trajectory","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="SCRIBE riskless trajectory (predicted)",observation_fn=background)
