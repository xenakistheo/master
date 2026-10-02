include("../parameters.jl")
include("../spectral_discretization.jl")
include("../temporal_discretization.jl")
include("../spatial_discretization.jl")
include("../GRF.jl")
include("../kalman.jl")

using LinearAlgebra
using Plots
using JLD2

#### experiment

### Define configuration parameters
D = Rectangle(1.0, 1.0) # Define spatial domain. 
σ_obs = 0.1 # Standard deviation of observation noise.


ν_s = 1.0
ν_t = 1.0
r_t = 10.0
r_s = 1.0
σ = 3.5

β_s_LL, σ_obs_LL = 0.25, 0.35 

params_LL_interp = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s_LL, σ=σ)
params_LL = interpretable_to_matern(params_LL_interp)

#########
m_order = 2
Δt = 1.0
N_init = 100 * r_t / Δt
Mx_inf, My_inf = 8, 8
M_inf = Mx_inf * My_inf # Spatial modes for inference

# Build F. 
F = build_F_blockmatrix(; Mx=Mx_inf, My=My_inf, D=D, m_params=params_LL, Δt=Δt, m=m_order, p=fill(1.0, M_inf), q=fill(1.0, M_inf))

σ_vec_initial = ones(M_inf)
Σ0 = build_Σ_blockmatrix(M_inf; σ_vec=σ_vec_initial, m=m_order, γ=params_LL.γ)
S0 = 100 .* I

# Perform initial steps to get the initial covariance matrix
for _ in 1:N_init
    S0 = F * S0 * F' + Σ0
end

# Compute updated covariance matrix for noise
σ_k = compute_sigma_k_vec(S0; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params_LL, D=D)
Σ = build_Σ_blockmatrix(M_inf; σ_vec=σ_k, m=m_order, γ=params_LL.γ)

S_init = 100 .* I
for _ in 1:N_init
    S_init = F * S_init * F' + Σ
end


# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations
Y_obs = y_LL[:, 1, :]


# Create observation matrices
H_spatial = build_observation_matrix(spatial_locations, Mx_inf, My_inf, D)
H_full = build_full_spatial_matrix(H_spatial; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params_LL)
d_block = 2*m_order + floor(Int, params_LL.γ) # state dimension per mode, as in build_full_spatial_matrix
idx = 1:d_block:size(F, 1) # index of c_k for each mode k

# Run through Kalman Filter
# Run through Kalman Filter
# ll_plain = KalmanFilter(; 
#     m_hat_0=zeros(size(F, 1)), # mean
#     S_hat_0=S_init, # covariance
#     F=F, # state transition matrix
#     Σ=Σ, # process noise covariance
#     H=H_full, # spatial coefficients matrix
#     Y=Y_obs, # observation matrix 
#     σ_obs=σ_obs_LL, 
#     )

ll_batch = KalmanFilter_fast(; 
    m_hat_0=zeros(size(F, 1)),
    S_hat_0=S_init,
    F=F,
    Σ=Σ,
    H_spatial=H_spatial,
    idx=idx,
    Y=Y_obs,
    σ_obs=σ_obs_LL, 
)

ll_batch2 = KalmanFilter_fast2(; 
    m_hat_0=zeros(size(F, 1)),
    S_hat_0=S_init,
    F=F,
    Σ=Σ,
    H_spatial=H_spatial,
    idx=idx,
    Y=Y_obs,
    σ_obs=σ_obs_LL, 
)

ll_batch3 = KalmanFilter_fast3(; 
    m_hat_0=zeros(size(F, 1)),
    S_hat_0=S_init,
    F=F,
    Σ=Σ,
    H_spatial=H_spatial,
    idx=idx,
    Y=Y_obs,
    σ_obs=σ_obs_LL, 
)

ll_seq = KalmanFilter_sequential(; 
    m_hat_0=zeros(size(F, 1)),
    S_hat_0=S_init,
    F=F,
    Σ=Σ,
    H_spatial=H_spatial,
    idx=idx,
    Y=Y_obs,
    σ_obs=σ_obs_LL, 
)


@show ll_plain
@show ll_batch
@show ll_seq

@show ll_batch - ll_seq
@show ll_batch - ll_plain

##### BENCHMARK TIME
# @btime KalmanFilter(; 
#     m_hat_0=zeros(size($F, 1)), # mean
#     S_hat_0=$S_init, # covariance
#     F=$F, # state transition matrix
#     Σ=$Σ, # process noise covariance
#     H=$H_full, # spatial coefficients matrix
#     Y=$Y_obs, # observation matrix 
#     σ_obs=$σ_obs_LL, 
#     )

@btime KalmanFilter_fast(; 
    m_hat_0=zeros(size($F, 1)),
    S_hat_0=$S_init,
    F=$F,
    Σ=$Σ,
    H_spatial=$H_spatial,
    idx=$idx,
    Y=$Y_obs,
    σ_obs=$σ_obs_LL, 
)

@btime KalmanFilter_fast2(; 
    m_hat_0=zeros(size($F, 1)),
    S_hat_0=$S_init,
    F=$F,
    Σ=$Σ,
    H_spatial=$H_spatial,
    idx=$idx,
    Y=$Y_obs,
    σ_obs=$σ_obs_LL, 
)

@btime KalmanFilter_fast3(; 
    m_hat_0=zeros(size($F, 1)),
    S_hat_0=$S_init,
    F=$F,
    Σ=$Σ,
    H_spatial=$H_spatial,
    idx=$idx,
    Y=$Y_obs,
    σ_obs=$σ_obs_LL, 
)

@btime KalmanFilter_fast4(; 
    m_hat_0=zeros(size($F, 1)),
    S_hat_0=$S_init,
    F=$F,
    Σ=$Σ,
    H_spatial=$H_spatial,
    idx=$idx,
    Y=$Y_obs,
    σ_obs=$σ_obs_LL, 
)

@btime KalmanFilter_sequential(; 
    m_hat_0=zeros(size($F, 1)),
    S_hat_0=$S_init,
    F=$F,
    Σ=$Σ,
    H_spatial=$H_spatial,
    idx=$idx,
    Y=$Y_obs,
    σ_obs=$σ_obs_LL, 
)



