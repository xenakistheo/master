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

    # Index of c_k (the observed component) in the state vector for each mode k
    w = 2*m_order + floor(Int, params.γ)
    idx = [1 + (k-1)*w for k in 1:M_inf]

    LOGLIKELIHOOD = KalmanFilter_fast4(; 
    m_hat_0=zeros(size(F, 1)), # mean
    S_hat_0=S_init, # covariance
    F=F, # state transition matrix
    Σ=Σ, # process noise covariance
    H_spatial=H_spatial, # eigenfunctions at the observation locations
    idx=idx, # observed state components
    Y=Y_obs, # observation matrix 
    σ_obs=σ_obs, 
    )

    return LOGLIKELIHOOD

end


"""
Same log-likelihood as loglikelihood, but faster:
  - evaluate_pq_fast: identical p, q, with the grid precomputed and the result cached per γ.
  - The two N_init warm-up recursions are done per w x w block of the block diagonal F by
    repeated doubling (propagate_block). Since Σ_k = σ_k Σ0_k, one pass gives both S0 and S_init.
    Equal to the iterated result up to floating point round-off.
  - S_init is assembled directly as a dense matrix for KalmanFilter_fast4.
"""
function loglikelihood_fast(η; H_spatial, D::Rectangle, Y_obs, Δt, N_init::Int=1000, Mx_inf::Int=8, My_inf::Int=8, m_order::Int=2)
    θ = η_to_θ(η)
    M_inf = Mx_inf * My_inf
    ν_s, ν_t, r_t, r_s, σ, β_s, σ_obs = θ

    params_interp = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s, σ=σ)
    params = interpretable_to_matern(params_interp)

    p, q = evaluate_pq_fast(params.γ, m_order)

    # Build F. 
    F = build_F_blockmatrix(; Mx=Mx_inf, My=My_inf, D=D, m_params=params, Δt=Δt, m=m_order, p=p, q=q)

    # Block size, and index of c_k (the observed component) in the state vector for each mode k
    w = 2*m_order + floor(Int, params.γ)
    idx = [1 + (k-1)*w for k in 1:M_inf]

    # Per block k: S_N = F_k^N (100 I) F_k^N' + σ W_k, with W_k the N-step sum for Σ0_k (σ = 1)
    Σ0k = Matrix(build_Σk_matrix(; σ=1.0, m=m_order, γ=params.γ))
    A_N = Vector{Matrix{Float64}}(undef, M_inf)
    W_N = Vector{Matrix{Float64}}(undef, M_inf)
    S0_diag = zeros(size(F, 1))
    for k in 1:M_inf
        r = idx[k]:idx[k]+w-1
        A_N[k], W_N[k] = propagate_block(Matrix(F[r, r]), Σ0k, N_init)
        S0_diag[r] .= diag(100 .* (A_N[k] * A_N[k]') .+ W_N[k])
    end

    # Compute updated covariance matrix for noise (only the diagonal of S0 is used)
    σ_k = compute_sigma_k_vec(sparse(Diagonal(S0_diag)); Mx=Mx_inf, My=My_inf, m=m_order, m_params=params, D=D)
    Σ = build_Σ_blockmatrix(M_inf; σ_vec=σ_k, m=m_order, γ=params.γ)

    S_init = zeros(size(F))
    for k in 1:M_inf
        r = idx[k]:idx[k]+w-1
        S_init[r, r] .= 100 .* (A_N[k] * A_N[k]') .+ σ_k[k] .* W_N[k]
    end

    LOGLIKELIHOOD = KalmanFilter_fast4(; 
    m_hat_0=zeros(size(F, 1)), # mean
    S_hat_0=S_init, # covariance
    F=F, # state transition matrix
    Σ=Σ, # process noise covariance
    H_spatial=H_spatial, # eigenfunctions at the observation locations
    idx=idx, # observed state components
    Y=Y_obs, # observation matrix 
    σ_obs=σ_obs, 
    )

    return LOGLIKELIHOOD

end
