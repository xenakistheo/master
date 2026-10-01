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

function loglikelihood(θ)
    ν_s, ν_t, r_t, r_s, σ, β_s, σ_obs = θ

    params_interp = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s, σ=σ)
    params = interpretable_to_matern(params_interp)

    # Build F. 
    F = build_F_blockmatrix(; Mx=Mx_inf, My=My_inf, D=D, m_params=params, Δt=Δt, m=m_order, p=[0.9974898057151861, -1.765795611691862, 0.769541794952949], q=[1.0, -1.2950912717347545, 0.326822511078717])

    σ_vec_initial = ones(M_inf)
    Σ0 = build_Σ_blockmatrix(M_inf; σ_vec=σ_vec_initial, m=m_order, γ=params.γ)
    S0 = 100 .* I

    # Perform initial steps to get the initial covariance matrix
    for i in 1:N_init
        S0 = F * S0 * F' + Σ0
    end

    # Compute updated covariance matrix for noise
    σ_k = compute_sigma_k_vec(S0; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params, D=D)
    Σ = build_Σ_blockmatrix(M_inf; σ_vec=σ_k, m=m_order, γ=params.γ)

    H_full = build_full_spatial_matrix(H_spatial; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params)


    LOGLIKELIHOOD = KalmanFilter(; 
    m_hat_0=zeros(size(F, 1)), # mean
    S_hat_0=S0, # covariance
    F=F, # state transition matrix
    Σ=Σ, # process noise covariance
    H=H_full, # spatial coefficients matrix
    Y=Y_obs, # observation matrix 
    σ_obs=σ_obs, #IS THIS CORRECT?
    )

    return LOGLIKELIHOOD

end


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
        r_s,
        r_t,
        β_s,
        σ,
        σ_obs_LL,
    ]

    return loglikelihood(θ)
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




##### CHOOSE p, q parameters. 
# γ = 1.5
# m = 2 : order of p and q 
# Σ_x (p(x)/q(x) - (1-x)^(γ - ⌊γ⌋) )^2 , x in 0:0.001:1
using Optim

γ = 1.5
m = 2

xgrid = 0.0:0.001:1.0
target(x) = (1 - x)^(γ - floor(γ))   # sqrt(1-x)

# θ = [a₀, ..., aₘ, b₁, ..., bₘ]
function objective(θ)
    a = θ[1:m+1]
    b = θ[m+2:end]

    loss = 0.0

    for x in xgrid
        p = sum(a[k+1] * x^k for k in 0:m)
        q = 1.0 + sum(b[k] * x^k for k in 1:m)

        loss += (p / q - target(x))^2
    end

    return loss
end

θ0 = zeros(2m + 1)
θ0[1] = 1.0   # p(x) ≈ 1 initially

result = optimize(objective, θ0, BFGS())

θ_opt = Optim.minimizer(result)

a_opt = θ_opt[1:m+1]
b_opt = vcat(1.0, θ_opt[m+2:end])

println("p coefficients = ", a_opt)
println("q coefficients = ", b_opt)
println("minimum loss = ", Optim.minimum(result))