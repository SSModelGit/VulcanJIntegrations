# Only demonstration configuration is defined here. State extraction, history,
# timestamps, posterior binding, and action dispatch are package adapters.
const settings = Dict{KAgentMDP,NamedTuple}()
function mukumari_problem(sensor;obstacles=false,risk_probability)
    polygons = obstacles ? [[(4.,3.5),(5.5,3.5),(5.5,6.5),(4.,6.5),(4.,3.5)]] : []
    obcs=[Dict(:poly=>p,:risk=>2.0,:impact=>0.1) for p in polygons]
    goal=Dict(:target=>[9.5 9.5],:strength=>0.,:influence=>2.0,:size=>0.0)
    landscape=AgentObjectiveLandscape(;objectives=[(:goal,goal),(:robc,obcs),(:horz,0.)])
    # Use MuKumari's demonstrated scale: world 0–10, speed 1, width 0.1, digits 3.
    problem=init_standard_KAgentMDP(;name="adaptive_sampling",start=[1.5 2.0],
        dimensions=(0.,10.),objl=landscape,menv=MuEnv(1,[:signal],Dict(:signal=>sensor)),
        digits=3,agent_width=0.1,agent_speed=1.0,ag_mvt_noise=0.,obs_noise=0.)
    risk=extract_obstacle_risk(problem; probability=risk_probability)
    settings[problem]=(;risk,polygons)
    state=MuKumari.blindstart_KAgentState(problem,problem.start)
    return problem,state
end
# Numerical objective integration is independent of the phenomenon-cell partition.
spatial_bounds(p::KAgentMDP) = (xmin=p.dimensions[1], xmax=p.dimensions[2],
                                    ymin=p.dimensions[1], ymax=p.dimensions[2])
VulcanJ.cellsites(::KAgentMDP) = [[x y] for x in range(0.,10.;length=21) for y in range(0.,10.;length=21)]
phenomenon_sites(::KAgentMDP) = [[x y] for x in 0.:2.:10. for y in 0.:2.:10.]
# Explicitly chosen obstacle-only risk; positive calibration converts negative
# MuKumari obstacle rewards to probabilities. A zero calibration is riskless.
VulcanJ.get_failure_prob(p::KAgentMDP,s,a) = settings[p].risk(s,a)
VulcanJ.posterior_phenomenon_prob(p::KAgentMDP,m::GPE,X) =
    1-cdf(conditional_observation_distribution(p,m,X),0.5)
initial_reading(s::MuKumari.KAgentState) = MuKumari.z(s)
obstacle_polygons(p::KAgentMDP) = settings[p].polygons
function setup_mukumari_gp(;obstacles=false,risk_probability)
    sensor=X->1.4exp(-sum(abs2,X-[7.5 7.5])/9.0)-0.7exp(-sum(abs2,X-[2.5 8.0])/5.0)
    problem,state=mukumari_problem(sensor;obstacles,risk_probability)
    prior=GPE(zeros(2,0),Float64[],MeanZero(),SEIso(log(3.0),0.0))
    return problem,state,prior
end
function setup_mukumari_scribe(;obstacles=false,risk_probability)
    prior,truth=scribe_models(;spatial_scale=10.)
    problem,state=mukumari_problem(X->predict_SCRIBEModel(truth,X);obstacles,risk_probability)
    return problem,state,prior
end
