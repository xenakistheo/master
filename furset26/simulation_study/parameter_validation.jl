include("../parameters.jl")
include("../spectral_discretization.jl")
include("../temporal_discretization.jl")
include("../spatial_discretization.jl")
include("../GRF.jl")
include("../kalman.jl")

using LinearAlgebra
using Plots
using JLD2

# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations
Y_obs = y_LL[:, 1, :]


# Fixed parameters
D = Rectangle(1.0, 1.0) # Define spatial domain. 
m_order = 2
Δt = 1.0
N_init = 1000
Mx_inf, My_inf = 8, 8
M_inf = Mx_inf * My_inf # Spatial modes for inference

H_spatial = build_observation_matrix(spatial_locations, Mx_inf, My_inf, D)

loglikelihood(η) = loglikelihood(η; H_spatial=H_spatial, D=D, Y_obs=Y_obs, Δt=Δt, N_init=N_init, Mx_inf=Mx_inf, My_inf=My_inf, m_order=m_order)


function loglikelihood_beta(β_s)

    ν_s = 1.0
    ν_t = 1.0
    r_t = 10.0
    r_s = 1.0
    σ = 3.5
    σ_obs_LL = 0.35 
    
    θ = [
        ν_s,
        ν_t,
        r_t,
        r_s,
        σ,
        β_s,
        σ_obs_LL,
    ]

    η = θ_to_η(θ)
    return loglikelihood(η)
end

β_grid = range(0.05, 0.5, length=50)

ll = [
    loglikelihood_beta(β)
    for β in β_grid
]

plot(
    β_grid,
    ll,
    xlabel="βₛ",
    ylabel="log-likelihood",
    legend=false,
)

β_s_LL = 0.25
vline!([β_s_LL], linestyle=:dash)

