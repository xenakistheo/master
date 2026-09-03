using LinearAlgebra
using Random
using BenchmarkTools

struct KalmanModel
    A::Matrix{Float64}
    H::Matrix{Float64}
    Q::Matrix{Float64}
end 




# Standard version. 
function KalmanFW(x::Vector{Float64}, Σ::Matrix{Float64}, z::Vector{Float64}, R::Matrix{Float64}, km::KalmanModel)

    # Predict system state and covariance
    xp = km.A * x
    Σp = km.A * Σ * km.A' + km.Q

    # Compute Kalman gain
    S = km.H * Σp * km.H' + R
    K = Σp * km.H' * inv(S)    

    # Update system state and covariance using observations
    ν = z - km.H * xp  # innovation
    x_new = xp + K * ν
    Σ_new = (I - K * km.H) * Σp

    ll = -0.5 * (logdet(S) + ν' * inv(S) * ν + length(z) * log(2π)) #log-likelihood
    
    return x_new, Σ_new, ll
end 

#Uses Woodbury identity. Useful when number of observations is much larger than number of states. (m >> n)
function KalmanFW(x::Vector{Float64}, Σ::Matrix{Float64}, z::Vector{Float64}, R_diag::Vector{Float64}, km::KalmanModel)
    
    # Predict system state and covariance
    xp = km.A * x
    Σp = km.A * Σ * km.A' + km.Q


    # Innovation covariance, and inverses
    S = km.H * Σp * km.H' + Diagonal(R_diag)
    Rinv = Diagonal(1.0 ./ R_diag)  # Inverse of diagonal observation noise covariance
    Sinv = Rinv - Rinv * km.H *inv(inv(Σp) + km.H' * Rinv * km.H) * km.H' * Rinv    # Inverse of innovation covariance using Woodbury identity

    # Compute Kalman gain
    K = Σp * km.H' * Sinv

    # Update system state and covariance using observations
    ν = z - km.H * xp  # innovation
    x_new = xp + K * ν
    Σ_new = (I - K * km.H) * Σp

    ll = -0.5 * (logdet(S) + ν' * Sinv * ν + length(z) * log(2π)) #log-likelihood
    
    return x_new, Σ_new, ll
end



function test_KalmanFW()

    n = 10 # state dimension  
    m = 100 # observation dimension

    T = 10 # TIME STEPS
    # Example usage of KalmanFW
    A = rand(n, n)  # State transition matrix
    H = rand(m, n)  # Observation matrix
    Q = Matrix(0.1 * I(n))  # Process noise covariance

    R = Matrix(0.5 * I(m))  # Observation noise covariance
    R_diag = 0.5 * ones(m)  # Observation noise covariance as diagonal matrix

    model = KalmanModel(A, H, Q)

    for t in 1:T
        x = rand(n)  # Initial state estimate
        Σ = Matrix(0.1 * I(n))  # Initial covariance estimate
        z = rand(m)  # New observations

        x, Σ, ll = KalmanFW(x, Σ, z, R_diag, model)
    end
end

@btime test_KalmanFW()
# test_sequential_kf()


function sequential_kf_fast(x::Vector{Float64}, Σ::Matrix{Float64}, z::Vector{Float64}, R_diag::Vector{Float64}, km::KalmanModel; 
        NA_ind::Vector{Int} = Int[])
    n = length(x) # number of states
    m = length(z) # number of observations

    ll = 0
    log2pi = log(2 * π)

    for j in 1:m
        if j in NA_ind
            continue  # Skip this observation if it's marked as NA
        end

        # Innovation
        dz_j = z[j]
        for k in 1:n
            dz_j -= km.H[j, k] * x[k]
        end

        a = zeros(n)
        for r in 1:n
            sum = 0.0
            for c in 1:n
                sum += km.H[j, c] * Σ[r, c]
            end
        a[r] = sum
        end
    

        sj = R_diag[j] 
        for k in 1:n
            sj += km.H[j, k] * a[k]
        end

        ll -= 0.5 * (log(sj) + dz_j^2 / sj + log2pi)

        # Kalman gain
        for k in 1:n
            kj = a[k] / sj
            x[k] += kj * dz_j
            for r in 1:n
                Σ[k, r] -= (a[k] * a[r]) / sj
            end 
        end 

    end     

    return x, Σ, ll
end 


