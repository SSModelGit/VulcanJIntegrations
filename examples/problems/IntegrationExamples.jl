module IntegrationExamples

import VulcanJ, POMDPs
using VulcanJ: AbstractInfoMCTS, AbstractErgodicSolver
using VulcanJIntegrations: conditional_observation_distribution,
    condition_environment_model, information_gain, expected_information_gain,
    extract_location, observation_history, state_time, extract_obstacle_risk
using POMDPs: MDP
using MuKumari: MuKumari, KAgentMDP, MuEnv, AgentObjectiveLandscape, init_standard_KAgentMDP
using SCRIBE: SCRIBE, SCRIBEModelState, LGSFModelParameters,
    initialize_SCRIBEModel_from_parameters, init_agent_info, predict_SCRIBEModel,
    posterior_model_moments
using GaussianProcesses: GPE, MeanZero, SEIso
using Distributions: cdf
using LinearAlgebra: I
using Random: MersenneTwister

export setup_mukumari_gp, setup_mukumari_scribe, setup_scribe,
    initial_reading, absolute_field, phenomenon_sites, spatial_bounds, obstacle_polygons

include("scribe.jl")
include("mukumari.jl")
end
