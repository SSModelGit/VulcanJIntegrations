include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_scribe, initial_reading, absolute_field, spatial_bounds, obstacle_polygons
using VulcanJ: ErgodicSolver, plot_simulated_path
using VulcanJIntegrations: condition_environment_model, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_scribe(;obstacles=false,risk_probability=score->0.0)
solver=ErgodicSolver(;lookahead=30,optimizer_iters=30,max_speed=1.0,quad_order=3,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:absolute_field)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
background=X->absolute_field(model,X)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","MuKumariSCRIBE","ergodic_conditional_absolute","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="MuKumariSCRIBE ergodic conditional absolute (executed)",observation_fn=background)
