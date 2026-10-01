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



