# All SCRIBE prediction and conditioning adapters come from VulcanJIntegrations.
# This file defines a demonstration problem and its chosen spatial objective.
function scribe_models(;spatial_scale=1.)
    centers = spatial_scale * [0.25 0.75; 0.75 0.75; 0.55 0.25]
    prior_parameters = LGSFModelParameters(;μ=centers,σ=[0.25spatial_scale],τ=[1.],ϕ₀=[-0.6,0.8,0.3])
    truth_parameters = LGSFModelParameters(;μ=centers,σ=[0.25spatial_scale],τ=[1.],ϕ₀=[-1.,1.4,0.5])
    smodel = initialize_SCRIBEModel_from_parameters(prior_parameters)
    truth = initialize_SCRIBEModel_from_parameters(truth_parameters)
    information = init_agent_info(prior_parameters;prior_covariance=Matrix{Float64}(I,3,3))
    return SCRIBEModelState(smodel,information,0.01),truth
end

# Magnitude of the posterior mean field, using SCRIBE's public prediction API.
absolute_field(m::SCRIBEModelState,X) = abs(only(posterior_model_moments(m.smodel,m.information,X)[:μ]))
VulcanJ.expected_information_gain(::Val{:absolute_field},problem,m::SCRIBEModelState,s,order) =
    absolute_field(m,extract_location(s))
VulcanJ.information_gain(::Val{:absolute_field},problem,prior::SCRIBEModelState,posterior,s,observation) =
    absolute_field(prior,extract_location(s))

struct FieldState
    location::Matrix{Float64}
    history::Vector
end
struct FieldProblem <: MDP{FieldState,Symbol}
    sensor::Function
    truth::Function
    steps::Int
    risk_scale::Float64
end
const directions = Dict(:n=>(0.,1.),:ne=>(1.,1.),:e=>(1.,0.),:se=>(1.,-1.),
                        :s=>(0.,-1.),:sw=>(-1.,-1.),:w=>(-1.,0.),:nw=>(-1.,1.))
field_step(s,a) = clamp.(s.location+0.12reshape(collect(directions[a]),1,2),0.,1.)
VulcanJ.extract_location(s::FieldState) = s.location
VulcanJ.state_time(::FieldProblem,s) = length(s.history)-1
VulcanJ.observation_history(::Union{AbstractInfoMCTS,AbstractErgodicSolver},::FieldProblem,s) = s.history
# Numerical objective integration is independent of the phenomenon-cell partition.
spatial_bounds(::FieldProblem) = (xmin=0., xmax=1., ymin=0., ymax=1.)
VulcanJ.cellsites(::FieldProblem) = [[x y] for x in range(0.,1.;length=21) for y in range(0.,1.;length=21)]
phenomenon_sites(::FieldProblem) = [[x y] for x in 0.:0.2:1. for y in 0.:0.2:1.]
POMDPs.actions(::FieldProblem,s) = keys(directions)
POMDPs.isterminal(p::FieldProblem,s) = length(s.history)>p.steps
function POMDPs.gen(p::FieldProblem,s,a,rng)
    X=field_step(s,a)
    y=p.sensor(X,rng)
    return (sp=FieldState(X,[s.history;(location=X,observation=y)]),r=0.)
end
VulcanJ.generative_problem(p::FieldProblem,m,rng) =
    FieldProblem((X,r)->rand(r,conditional_observation_distribution(p,m,X)),p.truth,p.steps,p.risk_scale)
VulcanJ.get_failure_prob(p::FieldProblem,s,a) =
    p.risk_scale*(0.002+0.015exp(-sum(abs2,field_step(s,a)-[0.5 0.45])/0.04))
initial_reading(s::FieldState) = last(s.history).observation
function setup_scribe(;steps=60,risk_scale=0.,rng=MersenneTwister(7))
    prior,truth=scribe_models()
    truth_field=X->predict_SCRIBEModel(truth,X)
    problem=FieldProblem((X,r)->[truth_field(X)+0.1randn(r)],truth_field,steps,risk_scale)
    X=[0.15 0.2]
    state=FieldState(X,[(location=X,observation=problem.sensor(X,rng))])
    return problem,state,prior
end
obstacle_polygons(::FieldProblem) = []

ground_truth(p::FieldProblem,X) = p.truth(X)
