include("../problems/IntegrationExamples.jl")
using .IntegrationExamples: setup_mukumari_scribe, initial_reading, spatial_bounds, obstacle_polygons, ground_truth, learned_field
using VulcanJ: ErgodicSolver, plot_environment_comparison
using VulcanJIntegrations: condition_environment_model, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng=MersenneTwister(17)
problem,state,prior=setup_mukumari_scribe(;obstacles=false,risk_probability=score->0.0)
solver=ErgodicSolver(;lookahead=60,optimizer_iters=30,max_speed=1.0,quad_order=3,rng)
model=condition_environment_model(problem,prior,state,initial_reading(state))
objective=Val(:absolute_field)
policy=solve(solver,problem;objective)
result=simulate_info_path(problem,policy,60;initial_state=state,model)
plot_environment_comparison(problem,result.states,result.observations;
    ground_truth_fn=X->ground_truth(problem,X),learned_fn=X->learned_field(result.model,X),
    bounds=spatial_bounds(problem),obstacles=obstacle_polygons(problem),
    save_path=joinpath(@__DIR__,"..","res","MuKumariSCRIBE","ergodic_conditional_absolute","ground_truth_and_learned.png"),
    title="MuKumariSCRIBE — ergodic conditional absolute")
println("Executed actions: ",length(result.actions)," / 60")
