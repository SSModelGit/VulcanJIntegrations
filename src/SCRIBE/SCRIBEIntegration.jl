module SCRIBEIntegration

import VulcanJ
import VulcanJ: set_environment_model!, conditional_observation_distribution,
    observation_outcomes, condition_environment_model, information_gain,
    expected_information_gain

export set_environment_model!, conditional_observation_distribution,
    observation_outcomes, condition_environment_model, information_gain,
    expected_information_gain
using VulcanJ: RiskBoundedInfoPolicy, ErgodicPolicy, cellsites
using SCRIBE: SCRIBE, SCRIBEModel, SCRIBEModelState

VulcanJ.conditional_observation_distribution(problem, m::SCRIBEModelState, state) =
    SCRIBE.posterior_measurement_distribution(m.smodel, m.information,
        VulcanJ.extract_location(state), m.R)

VulcanJ.observation_outcomes(distribution::SCRIBE.Gaussian, order) =
    VulcanJ.gaussian_observation_outcomes(VulcanJ.mean(distribution),
                                        VulcanJ.cov(distribution), order)


# Use the public predictive distribution moments, not SCRIBE information internals.
function VulcanJ.presence_probability(p::VulcanJ.ThresholdPresence,
    distribution::SCRIBE.Gaussian, location, order)
    marginal = VulcanJ.Normal(only(VulcanJ.mean(distribution)), sqrt(only(VulcanJ.cov(distribution))))
    return VulcanJ.presence_probability(p,marginal,location,order)
end

function VulcanJ.condition_environment_model(problem, m::SCRIBEModelState, state, observation)
    info = SCRIBE.condition_on_measurement(m.smodel, m.information,
        VulcanJ.extract_location(state), observation, m.R)
    return SCRIBEModelState(m.smodel, info, m.R)
end

VulcanJ.information_gain(::Val{:mutual_information}, problem,
    prior::SCRIBEModelState, posterior::SCRIBEModelState, state, observation) =
    SCRIBE.D_KL(posterior.information, prior.information)

VulcanJ.expected_information_gain(::Val{:mutual_information}, problem,
    m::SCRIBEModelState, state, order::Integer) =
    SCRIBE.mutual_information(m.smodel, m.information, VulcanJ.extract_location(state), m.R)

VulcanJ.expected_information_gain(::Val{:variance_reduction}, problem,
    m::SCRIBEModelState, state, order::Integer) =
    SCRIBE.integrated_variance_reduction(m.smodel, m.information,
        VulcanJ.extract_location(state), reduce(vcat, cellsites(problem)), m.R)

function VulcanJ.information_gain(::Val{:variance_reduction}, problem,
    prior::SCRIBEModelState, posterior::SCRIBEModelState, state, observation)
    sites = reduce(vcat, cellsites(problem))
    return SCRIBE.predict_model_uncertainty(prior.smodel, prior.information, sites;
               metric=:mean_variance) -
           SCRIBE.predict_model_uncertainty(posterior.smodel, posterior.information, sites;
               metric=:mean_variance)
end

# Scalar uncertainty decreases and information increases are positive rewards.
# Expected rewards use VulcanJ's common outcome integration of these same metrics.
for (metric, direction) in ((:differential_entropy, -1), (:total_variance, -1),
                            (:mean_variance, -1), (:logdet_information, 1),
                            (:minimum_information_eigenvalue, 1))
    @eval VulcanJ.information_gain(::Val{$(QuoteNode(metric))}, problem,
        prior::SCRIBEModelState, posterior::SCRIBEModelState, state, observation) =
        $direction * (SCRIBE.evaluate_information_metric(posterior.information; metric=$(QuoteNode(metric))) -
                      SCRIBE.evaluate_information_metric(prior.information; metric=$(QuoteNode(metric))))
end

for PolicyType in (RiskBoundedInfoPolicy, ErgodicPolicy)
    @eval function VulcanJ.set_environment_model!(policy::$PolicyType, state,
        model::SCRIBEModel; information, R, kwargs...)
        return set_environment_model!(policy, state, SCRIBEModelState(model, information, R); kwargs...)
    end
end

end
