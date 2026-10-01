include("parameters.jl")
include("spectral_discretization.jl")
include("temporal_discretization.jl")
include("spatial_discretization.jl")
include("GRF.jl")
include("kalman.jl")


function loglikelihood(η; H_spatial, D::Rectangle, Y_obs, Δt, N_init::Int=1000, Mx_inf::Int=8, My_inf::Int=8, m_order::Int=2)
    θ = η_to_θ(η)
    M_inf = Mx_inf * My_inf
    ν_s, ν_t, r_t, r_s, σ, β_s, σ_obs = θ

    params_interp = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s, σ=σ)
    params = interpretable_to_matern(params_interp)

    p, q = evaluate_pq(params.γ, m_order)

    # Build F. 
    F = build_F_blockmatrix(; Mx=Mx_inf, My=My_inf, D=D, m_params=params, Δt=Δt, m=m_order, p=p, q=q)

    σ_vec_initial = ones(M_inf)
    Σ0 = build_Σ_blockmatrix(M_inf; σ_vec=σ_vec_initial, m=m_order, γ=params.γ)
    S0 = 100 .* I

    # Perform initial steps to get the initial covariance matrix
    for _ in 1:N_init
        S0 = F * S0 * F' + Σ0
    end

    # Compute updated covariance matrix for noise
    σ_k = compute_sigma_k_vec(S0; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params, D=D)
    Σ = build_Σ_blockmatrix(M_inf; σ_vec=σ_k, m=m_order, γ=params.γ)


    S_init = 100 .* I
    for _ in 1:N_init
        S_init = F * S_init * F' + Σ
    end

    H_full = build_full_spatial_matrix(H_spatial; Mx=Mx_inf, My=My_inf, m=m_order, m_params=params)


    LOGLIKELIHOOD = KalmanFilter(; 
    m_hat_0=zeros(size(F, 1)), # mean
    S_hat_0=S_init, # covariance
    F=F, # state transition matrix
    Σ=Σ, # process noise covariance
    H=H_full, # spatial coefficients matrix
    Y=Y_obs, # observation matrix 
    σ_obs=σ_obs, 
    )

    return LOGLIKELIHOOD

end