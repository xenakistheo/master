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



