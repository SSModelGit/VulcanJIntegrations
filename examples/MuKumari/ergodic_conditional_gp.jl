include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_gp, initial_reading, spatial_bounds, obstacle_polygons
using VulcanJ: ErgodicSolver, plot_simulated_path
using VulcanJIntegrations: condition_environment_model, expected_information_gain, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_gp(;obstacles=false,risk_probability=score->0.0)
solver=ErgodicSolver(;lookahead=30,optimizer_iters=30,max_speed=1.0,quad_order=3,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:mutual_information)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
background=X->expected_information_gain(objective,problem,model,X,3)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","MuKumari","ergodic_conditional_gp","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="MuKumari ergodic conditional gp (executed)",observation_fn=background)
