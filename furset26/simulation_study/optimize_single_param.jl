include("../likelihood.jl")
include("../GRF.jl")

using JLD2
using Optim

# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations
Y_obs = y_LL[:, 1, :]


# Fixed parameters
D = Rectangle(1.0, 1.0) # Define spatial domain. 
m_order = 2
Δt = 1.0
N_init = 1000
Mx_inf, My_inf = 8, 8


H_spatial = build_observation_matrix(spatial_locations, Mx_inf, My_inf, D)

loglikelihood(η) = loglikelihood_fast(η; H_spatial=H_spatial, D=D, Y_obs=Y_obs, Δt=Δt, N_init=N_init, Mx_inf=Mx_inf, My_inf=My_inf, m_order=m_order)

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


#####
objective_beta(β) = -loglikelihood_beta(β)

result = optimize(
    objective_beta,
    0.001,
    0.999,
    Brent()
)

β_hat = Optim.minimizer(result) # True is β_s_LL = 0.25

println("β̂_s = ", β_hat)
