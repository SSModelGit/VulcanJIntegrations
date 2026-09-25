include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_scribe, initial_reading, absolute_field, spatial_bounds, obstacle_polygons
using VulcanJ: ErgodicSolver, plan_trajectory, plot_simulated_path
using VulcanJIntegrations: condition_environment_model
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_scribe(;risk_scale=0.0)
solver=ErgodicSolver(;lookahead=30,optimizer_iters=30,max_speed=0.16,quad_order=3,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:absolute_field)
result=plan_trajectory(solver,problem,state,model,60;objective)
background=X->absolute_field(model,X)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"..","res","SCRIBE","ergodic_trajectory_absolute","path.png"),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    heatmap_resolution=51,marker_size=2,colorbar_title="Initial planning objective",
    title="SCRIBE ergodic trajectory absolute (predicted)",observation_fn=background)
