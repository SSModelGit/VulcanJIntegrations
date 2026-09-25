include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_gp, initial_reading, phenomenon_sites, spatial_bounds, obstacle_polygons
using VulcanJ: RiskBoundedInfoMCTS, UnobservedPhenomenaModel, ThresholdPresence, plot_simulated_path
using VulcanJIntegrations: condition_environment_model, observation_history, expected_information_gain, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_gp(;obstacles=true,risk_probability=score->-expm1(1.0*score))
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=0.6,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
# A presence threshold over the supplied environmental model, on spatial cells.
model=UnobservedPhenomenaModel(problem,model,ThresholdPresence(0.5);
    sites=phenomenon_sites(problem),history=observation_history(solver,problem,state),quadrature_order=5)
objective=Val(:mutual_information)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
background=X->expected_information_gain(objective,problem,model,X,3)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","MuKumari","performance_conditional_unobserved","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="MuKumari performance conditional unobserved (executed)",observation_fn=background)
println("Executed actions: ",length(result.actions)," / 60")
println("Spent risk / total information: ",(sum(result.risks),sum(result.information_rewards)))
println("Resolved cells: ",count(result.model.observed))
