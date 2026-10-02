using LinearAlgebra
using Random
using BenchmarkTools

include("parameters.jl")

function KalmanStep(; 
    m_hat, # mean
    S_hat, # covariance
    F, # state transition matrix
    Σ, # process noise covariance
    H, # spatial coefficients matrix
    G=nothing, # control input matrix
    β_coeffs=nothing, # regression coefficients
    y, # observations
    σ_obs, 
    )
    # mean and covariance matrix
    m_tilde = F * m_hat
    S_tilde = F * S_hat * F' + Σ

    # Compute the Kalman gain
    s = S_tilde * H'
    A = Symmetric(H * s + σ_obs^2 * I)
    cholA = cholesky(A)

    K = (cholA \ s')'

    # Ahead point forecast
    if G !== nothing && β_coeffs !== nothing
        y_tilde = G*β_coeffs + H*m_tilde
    else
        y_tilde = H*m_tilde
    end
    Δy = (y - y_tilde)
    # Final Step
    m_hat_new = m_tilde + K * Δy
    S_hat_new = (I - K * H) * S_tilde

    # Compute the log-likelihood of the parameters. 
    α = cholA \ Δy
    ll = -0.5 * (logdet(cholA) + dot(Δy, α) + length(y) * log(2π)) #log-likelihood


    return m_hat_new, S_hat_new, ll 
end 


function KalmanFilter(; 
    m_hat_0, # mean
    S_hat_0, # covariance
    F, # state transition matrix
    Σ, # process noise covariance
    H, # spatial coefficients matrix
    Y, # observation matrix 
    σ_obs,
    G=nothing, # control input matrix
    β_coeffs=nothing, # regression coefficients
    )

    Nsteps = size(Y, 1)
    m_hat = m_hat_0
    S_hat = S_hat_0

    LOG_SUM = 0.0
    for t in 1:Nsteps
        m_hat, S_hat, ll = KalmanStep(
            m_hat = m_hat,
            S_hat = S_hat,
            F = F,
            Σ = Σ,
            H = H,
            G = G,
            β_coeffs = β_coeffs,
            y = Y[t, :],
            σ_obs = σ_obs
        )
        LOG_SUM += ll
    end

    return LOG_SUM
end


"""
Faster log-likelihood for the model without covariates. Same result as KalmanFilter with
H = build_full_spatial_matrix(H_spatial; ...), but
  - S is kept dense (it becomes dense after the first update), F stays sparse, and
  - only the observed state components idx (c_k for each mode k) are multiplied by H_spatial.
"""
function KalmanFilter_fast(;
    m_hat_0, # mean
    S_hat_0, # covariance
    F, # state transition matrix (sparse, block diagonal)
    Σ, # process noise covariance
    H_spatial, # N_spatial x M matrix of eigenfunctions evaluated at the observation locations
    idx, # index of c_k in the state vector for each mode k
    Y, # observation matrix
    σ_obs,
    )

    m_hat = Vector{Float64}(m_hat_0)
    S_hat = Matrix{Float64}(S_hat_0)
    Σ_dense = Matrix{Float64}(Σ)

    LOG_SUM = 0.0
    for t in axes(Y, 1)
        m_tilde = F * m_hat
        S_tilde = (F * S_hat) * F' + Σ_dense

        s = S_tilde[:, idx] * H_spatial' # = S_tilde * H'
        cholA = cholesky(Symmetric(H_spatial * s[idx, :] + σ_obs^2 * I))

        Δy = Y[t, :] - H_spatial * m_tilde[idx]
        K = s / cholA

        m_hat = m_tilde + K * Δy
        S_hat = S_tilde - K * s' # = (I - K H) S_tilde
        S_hat = (S_hat + S_hat') / 2 # keep S symmetric to avoid accumulating round-off

        LOG_SUM += -0.5 * (logdet(cholA) + dot(Δy, cholA \ Δy) + length(Δy) * log(2π))
    end

    return LOG_SUM
end




"""
Faster log-likelihood for the model without covariates. Same result as KalmanFilter with
H = build_full_spatial_matrix(H_spatial; ...), but
  - S is kept dense (it becomes dense after the first update), F stays sparse, and
  - only the observed state components idx (c_k for each mode k) are multiplied by H_spatial.
"""

function KalmanFilter_fast2(;
    m_hat_0,
    S_hat_0,
    F,              # sparse block diagonal matrix
    Σ,
    H_spatial,
    idx,
    Y,
    σ_obs,
)

    # State
    m_hat = copy(m_hat_0)
    S_hat = Matrix{Float64}(S_hat_0)   # dense: S fills in after the first update

    Σ_dense = Matrix(Σ)

    d = length(m_hat)
    n_obs = size(H_spatial, 1)

    # ------------------------------------------------------------
    # Preallocate
    # ------------------------------------------------------------

    m_tilde = similar(m_hat)

    # Temporary for F*S
    FS = similar(S_hat)

    # Predicted covariance
    S_tilde = similar(S_hat)

    # S_tilde[:,idx] * H'
    s = zeros(Float64, d, n_obs)

    # innovation
    Δy = zeros(Float64, n_obs)

    # Kalman gain, and its transpose for the Cholesky solve
    K = zeros(Float64, d, n_obs)
    Kt = zeros(Float64, n_obs, d)

    # innovation covariance
    innovation_cov = zeros(Float64, n_obs, n_obs)

    Ht = Matrix(transpose(H_spatial))

    log2π = log(2π)

    LOG_SUM = 0.0


    for t in axes(Y,1)

        # ========================================================
        # Prediction
        # ========================================================

        # m^- = F*m
        mul!(m_tilde, F, m_hat)


        # S^- = F*S*F' + Q

        # FS = F*S
        mul!(FS, F, S_hat)

        # S_tilde = FS*F'
        mul!(S_tilde, FS, transpose(F))

        # + process noise
        S_tilde .+= Σ_dense



        # ========================================================
        # Observation update
        # ========================================================

        # s = P*H'
        #
        # Only c_k states are observed, hence idx
        mul!(s, view(S_tilde, :, idx), Ht)


        # Δy = y - H*m
        mul!(Δy, H_spatial, view(m_tilde, idx))
        Δy .= view(Y, t, :) .- Δy


        # Innovation covariance:
        #
        # A = H*S*H' + σ²I

        mul!(
            innovation_cov,
            H_spatial,
            view(s, idx, :)
        )

        for i in 1:n_obs
            innovation_cov[i,i] += σ_obs^2
        end


        cholA = cholesky!(
            Symmetric(innovation_cov)
        )


        # ========================================================
        # Kalman gain
        # ========================================================

        # K = s / A
        #
        # Solve A*K' = s'

        transpose!(Kt, s)

        ldiv!(cholA, Kt)

        transpose!(K, Kt)



        # ========================================================
        # Mean update
        # ========================================================

        mul!(m_hat, K, Δy)
        m_hat .+= m_tilde



        # ========================================================
        # Covariance update
        # ========================================================

        # S_hat = S_tilde - K*s'
        mul!(S_hat, K, transpose(s))

        S_hat .= S_tilde .- S_hat


        # Keep symmetric
        S_hat .= 0.5 .* (S_hat + transpose(S_hat))


        # ========================================================
        # Likelihood
        # ========================================================

        LOG_SUM += -0.5 * (
            logdet(cholA) +
            dot(Δy, cholA \ Δy) +
            n_obs * log2π
        )

    end

    return LOG_SUM
end



"""
Same log-likelihood as KalmanFilter_fast2, but the observation update is done in the
k = length(idx) dimensional space of the observed states instead of the n_obs dimensional
observation space. With R = σ²I and G = H'H (H = H_spatial, assumed full column rank),
Woodbury gives, for P = S_tilde, B = P[idx,idx] + σ² G⁻¹ and A = H P[idx,idx] H' + σ²I:
  - H' A⁻¹ H  = B⁻¹                         ⇒  S_hat = P - P[:,idx] B⁻¹ P[idx,:]
  - H' A⁻¹ Δy = B⁻¹ z,  z = G⁻¹H'y - c̃       ⇒  m_hat = m_tilde + P[:,idx] B⁻¹ z
  - logdet A  = (n_obs - k) log σ² + logdet G + logdet B
  - Δy'A⁻¹Δy  = (y'y - y'H G⁻¹H'y) / σ² + z'B⁻¹z
so no n_obs x n_obs matrix is ever formed. G⁻¹H'Y and the residual term are precomputed.
"""
function KalmanFilter_fast3(;
    m_hat_0,
    S_hat_0,
    F,              # sparse block diagonal matrix
    Σ,              # sparse process noise covariance
    H_spatial,
    idx,
    Y,
    σ_obs,
)

    d = length(m_hat_0)
    n_obs, k = size(H_spatial)
    σ2 = σ_obs^2

    # ------------------------------------------------------------
    # Precompute everything that does not depend on the state
    # ------------------------------------------------------------

    G = Symmetric(H_spatial' * H_spatial)
    cholG = cholesky(G)
    σ2_Ginv = σ2 .* inv(cholG)                   # σ² G⁻¹ (k x k)

    HtY = H_spatial' * transpose(Y)              # k x T
    Z_y = cholG \ HtY                            # G⁻¹H'y_t for every t

    # Part of the quadratic form orthogonal to range(H): y'y - y'H G⁻¹ H'y
    resid2 = [dot(view(Y, t, :), view(Y, t, :)) - dot(view(HtY, :, t), view(Z_y, :, t)) for t in axes(Y, 1)]

    const_ll = n_obs * log(2π) + (n_obs - k) * log(σ2) + logdet(cholG)

    # Nonzeros of Σ, added to S_tilde without forming a dense copy
    Σ_I, Σ_J, Σ_V = findnz(sparse(Σ))

    # ------------------------------------------------------------
    # Preallocate
    # ------------------------------------------------------------

    m_hat = Vector{Float64}(m_hat_0)
    S_hat = Matrix{Float64}(S_hat_0)

    m_tilde = similar(m_hat)
    FS = similar(S_hat)
    FSt = similar(S_hat)
    S_tilde = similar(S_hat)

    B = zeros(Float64, k, k)
    X = zeros(Float64, k, d)                     # U⁻ᵀ P[idx,:], with B = U'U
    z = zeros(Float64, k)

    LOG_SUM = 0.0

    for t in axes(Y, 1)

        # ========================================================
        # Prediction
        # ========================================================

        mul!(m_tilde, F, m_hat)

        # S_tilde = F*S*F' computed as F*(F*S)' so both products are sparse * dense
        mul!(FS, F, S_hat)
        transpose!(FSt, FS)
        mul!(S_tilde, F, FSt)

        @inbounds for n in eachindex(Σ_V)
            S_tilde[Σ_I[n], Σ_J[n]] += Σ_V[n]
        end

        # ========================================================
        # Observation update in the k observed states
        # ========================================================

        # B = P[idx,idx] + σ² G⁻¹
        B .= view(S_tilde, idx, idx) .+ σ2_Ginv
        cholB = cholesky!(Symmetric(B, :U))
        U = cholB.U

        # z = G⁻¹H'y - c̃
        z .= view(Z_y, :, t) .- view(m_tilde, idx)

        # X = U⁻ᵀ P[idx,:]  and  z ← U⁻ᵀ z
        X .= view(S_tilde, idx, :)
        ldiv!(U', X)
        ldiv!(U', z)

        # m_hat = m_tilde + P[:,idx] B⁻¹ z = m_tilde + X'(U⁻ᵀ z)
        copyto!(m_hat, m_tilde)
        mul!(m_hat, transpose(X), z, 1.0, 1.0)

        # S_hat = P - X'X (symmetric rank-k update, upper triangle, then mirrored)
        copyto!(S_hat, S_tilde)
        BLAS.syrk!('U', 'T', -1.0, X, 1.0, S_hat)
        LinearAlgebra.copytri!(S_hat, 'U')

        # ========================================================
        # Likelihood
        # ========================================================

        LOG_SUM += -0.5 * (const_ll + logdet(cholB) + resid2[t] / σ2 + dot(z, z))

    end

    return LOG_SUM
end



"""
C = A * F', given Ft = sparse(F') (column i of Ft holds row i of F).
Each column of C is a short linear combination of contiguous columns of A, which is much
faster than the generic sparse * dense mul! for the companion-form blocks of F.
"""
function _mul_Ft!(C::Matrix{Float64}, A::Matrix{Float64}, Ft::SparseMatrixCSC{Float64})
    n = size(A, 1)
    rv = rowvals(Ft)
    nzv = nonzeros(Ft)
    @inbounds for i in axes(C, 2)
        rng = nzrange(Ft, i)
        if isempty(rng)
            for r in 1:n
                C[r, i] = 0.0
            end
            continue
        end
        p = first(rng)
        j, v = rv[p], nzv[p]
        @simd for r in 1:n
            C[r, i] = v * A[r, j]
        end
        for p in first(rng)+1:last(rng)
            j, v = rv[p], nzv[p]
            @simd for r in 1:n
                C[r, i] = muladd(v, A[r, j], C[r, i])
            end
        end
    end
    return C
end


"""
Same algorithm as KalmanFilter_fast3, with a faster prediction step:
  - F*S*F' is computed as T = S*F', then F*S*F' = T'*F', both with the column kernel _mul_Ft!
    instead of the generic sparse * dense mul!.
  - Prediction and update are done in place in S_hat (no separate S_tilde copy).
"""
function KalmanFilter_fast4(;
    m_hat_0,
    S_hat_0,
    F,              # sparse block diagonal matrix
    Σ,              # sparse process noise covariance
    H_spatial,
    idx,
    Y,
    σ_obs,
)

    d = length(m_hat_0)
    n_obs, k = size(H_spatial)
    σ2 = σ_obs^2

    # ------------------------------------------------------------
    # Precompute everything that does not depend on the state
    # ------------------------------------------------------------

    G = Symmetric(H_spatial' * H_spatial)
    cholG = cholesky(G)
    σ2_Ginv = σ2 .* inv(cholG)                   # σ² G⁻¹ (k x k)

    HtY = H_spatial' * transpose(Y)              # k x T
    Z_y = cholG \ HtY                            # G⁻¹H'y_t for every t

    # Part of the quadratic form orthogonal to range(H): y'y - y'H G⁻¹ H'y
    resid2 = [dot(view(Y, t, :), view(Y, t, :)) - dot(view(HtY, :, t), view(Z_y, :, t)) for t in axes(Y, 1)]

    const_ll = n_obs * log(2π) + (n_obs - k) * log(σ2) + logdet(cholG)

    F_sp = SparseMatrixCSC{Float64}(sparse(F))
    Ft = SparseMatrixCSC{Float64}(sparse(F_sp'))

    # Nonzeros of Σ, added to S without forming a dense copy
    Σ_I, Σ_J, Σ_V = findnz(sparse(Σ))

    # ------------------------------------------------------------
    # Preallocate
    # ------------------------------------------------------------

    m_hat = Vector{Float64}(m_hat_0)
    S = Matrix{Float64}(S_hat_0)                 # holds S_hat, and S_tilde within a step

    m_tilde = similar(m_hat)
    T1 = similar(S)
    T2 = similar(S)

    B = zeros(Float64, k, k)
    X = zeros(Float64, k, d)                     # U⁻ᵀ P[idx,:], with B = U'U
    z = zeros(Float64, k)

    LOG_SUM = 0.0

    for t in axes(Y, 1)

        # ========================================================
        # Prediction
        # ========================================================

        mul!(m_tilde, F_sp, m_hat)

        _mul_Ft!(T1, S, Ft)                      # T1 = S F'
        transpose!(T2, T1)                       # T2 = F S
        _mul_Ft!(S, T2, Ft)                      # S  = F S F'

        @inbounds for n in eachindex(Σ_V)
            S[Σ_I[n], Σ_J[n]] += Σ_V[n]
        end

        # ========================================================
        # Observation update in the k observed states
        # ========================================================

        # B = P[idx,idx] + σ² G⁻¹
        B .= view(S, idx, idx) .+ σ2_Ginv
        cholB = cholesky!(Symmetric(B, :U))
        U = cholB.U

        # z = G⁻¹H'y - c̃
        z .= view(Z_y, :, t) .- view(m_tilde, idx)

        # X = U⁻ᵀ P[idx,:]  and  z ← U⁻ᵀ z
        X .= view(S, idx, :)
        ldiv!(U', X)
        ldiv!(U', z)

        # m_hat = m_tilde + X'(U⁻ᵀ z)
        copyto!(m_hat, m_tilde)
        mul!(m_hat, transpose(X), z, 1.0, 1.0)

        # S_hat = P - X'X (upper triangle, then mirrored)
        BLAS.syrk!('U', 'T', -1.0, X, 1.0, S)
        LinearAlgebra.copytri!(S, 'U')

        # ========================================================
        # Likelihood
        # ========================================================

        LOG_SUM += -0.5 * (const_ll + logdet(cholB) + resid2[t] / σ2 + dot(z, z))

    end

    return LOG_SUM
end



function KalmanFilter_sequential(;
    m_hat_0,
    S_hat_0,
    F,
    Σ,
    H_spatial,
    idx,
    Y,
    σ_obs,
)
    m_hat = Vector{Float64}(m_hat_0)
    S_hat = Matrix{Float64}(S_hat_0)
    Σ_dense = Matrix{Float64}(Σ)

    d = length(m_hat)
    M = length(idx)

    LOG_SUM = 0.0
    log2π = log(2π)

    # Work vector to avoid allocating for every observation
    a = zeros(Float64, d)

    for t in axes(Y, 1)

        # --------------------------------------------------
        # 1. Prediction
        # --------------------------------------------------
        m_hat = F * m_hat
        S_hat = (F * S_hat) * F' + Σ_dense

        # --------------------------------------------------
        # 2. Sequential observation updates
        # --------------------------------------------------
        for j in axes(H_spatial, 1)

            yj = Y[t, j]

            # optionally skip missing observations
            if ismissing(yj) || isnan(yj)
                continue
            end

            h = @view H_spatial[j, :]

            # Observation:
            #
            # y_j = h' * m[idx] + ε
            #
            # Innovation
            dy = yj - dot(h, @view m_hat[idx])

            # a = P * H_j'
            #
            # H_j only acts on state entries idx, so
            # instead of P * full_H_j' we use:
            #
            #     P[:, idx] * h
            #
            mul!(a, @view(S_hat[:, idx]), h)

            # scalar innovation variance:
            #
            # s_j = H_j P H_j' + σ²
            #
            sj = σ_obs^2 + dot(h, @view(a[idx]))

            # log likelihood contribution
            LOG_SUM += -0.5 * (
                log2π +
                log(sj) +
                dy^2 / sj
            )

            # --------------------------------------------------
            # posterior mean
            #
            # m <- m + a/sj * dy
            # --------------------------------------------------
            α = dy / sj
            @. m_hat += α * a

            # --------------------------------------------------
            # posterior covariance
            #
            # P <- P - a*a'/sj
            # --------------------------------------------------
            LinearAlgebra.BLAS.ger!(-1 / sj, a, a, S_hat)
        end

        # optional numerical cleanup
        S_hat .= 0.5 .* (S_hat .+ S_hat')
    end

    return LOG_SUM
end

#=
Paper Notation          Code notation

S_hat                   Σ
S_tilde                 S

m_hat                   x

m_tilde                 xp
H
F                       A 
Σ                       Q
K

y

y_tilde 

H

=#



