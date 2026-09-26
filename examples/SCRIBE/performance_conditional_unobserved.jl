include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_scribe, initial_reading, phenomenon_sites, spatial_bounds, obstacle_polygons, ground_truth, learned_field
using VulcanJ: RiskBoundedInfoMCTS, UnobservedPhenomenaModel, ThresholdPresence, plot_environment_comparison, simulate_info_path
using VulcanJIntegrations: condition_environment_model, observation_history
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_scribe(;risk_scale=0.5)
solver=RiskBoundedInfoMCTS(;lookahead=12,time_budget=1.0,risk_budget=0.6,reference_reward=0.05,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
# A presence threshold over the supplied environmental model, on spatial cells.
model=UnobservedPhenomenaModel(problem,model,ThresholdPresence(0.5);
    sites=phenomenon_sites(problem),history=observation_history(solver,problem,state),quadrature_order=5)
objective=Val(:mutual_information)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
plot_environment_comparison(problem,result.states,result.observations;
    ground_truth_fn=X->ground_truth(problem,X),learned_fn=X->learned_field(result.model,X),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    save_path=joinpath(@__DIR__,"..","res","SCRIBE","performance_conditional_unobserved","ground_truth_and_learned.png"),
    title="SCRIBE — performance conditional unobserved")
println("Executed actions: ",length(result.actions)," / 60")
println("Resolved cells: ",count(result.model.observed))
