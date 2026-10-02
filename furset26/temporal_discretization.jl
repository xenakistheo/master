using Optim
using SparseArrays
using Distributions
using SpecialFunctions
using DSP

include("spectral_discretization.jl")

function C_ck0(; D::Rectangle, Mx::Int, My::Int, m_params::maternParams)
    M = Mx * My

    ξ = zeros(M)
    λ = zeros(M)
    μ = zeros(M)

    C0 = zeros(M)

    for k in 1:M
        ξ[k], μ[k], λ[k] = eigenvalues(D, k; Mx=Mx, m=m_params)
    end 

    for k in 1:M
        C0[k] = λ[k] * gamma(2*m_params.γ - 1) / ((2*μ[k])^(2*m_params.γ - 1) * gamma(m_params.γ)^2)
    end

    λ ./= sum(C0)
    C0 ./= sum(C0)

    return (; C0, ξ, μ, λ)
end 

# Define helper functions
ζ_func(i; m, γ) = Int(min(max(m, floor(γ)), i, m + floor(γ) -  i))

θ_func(i, μ, qi, q0) = exp(-μ*i)*qi/q0


# Double check that indexing (0,1) is correctly translated to julia. 
  function ϕ_func(k, i; μ, Δt, γ, p, m)
      r = floor(Int, γ)
      P = conv(p, [(-1)^j * binomial(r, j) for j in 0:r])   # coefficients of p(z)(1-z)^⌊γ⌋
      return -exp(-μ * i * Δt) * P[i+1] / P[1]
  end 


function build_Fk_matrix(k; m_params::maternParams, μ::Float64, Δt::Float64, m::Int, p::Vector{Float64}, q::Vector{Float64})
    lγ = Int(floor(m_params.γ))
    F = spzeros(2*m + lγ, 2*m + lγ)

    F[1, 1:m+lγ] = [ϕ_func(k, i; μ = μ, Δt = Δt, γ = m_params.γ, p = p, m = m) for i in 1:m+lγ]
    F[1, m+lγ+1:2*m+lγ] = [θ_func(i, μ, q[i+1], q[1]) for i in 1:(m)]
    
    for i in 1:(m+lγ - 1)
        F[i+1, i] = 1
    end 

    for i in 1:(m-1)
        F[i+ lγ + 3, i + m +lγ] = 1
    end 
    
    return F
end 

function build_F_blockmatrix(M; m_params::maternParams, μ::Vector{Float64}, Δt::Float64, m::Int, p::Vector{Float64}, q::Vector{Float64})
    return blockdiag([build_Fk_matrix(k; m_params = m_params, μ = μ[k], Δt = Δt, m = m, p = p, q = q) for k in 1:M]...)
end

function build_F_blockmatrix(; Mx::Int, My::Int, D::Rectangle, m_params::maternParams, Δt::Float64, m::Int, p::Vector{Float64}, q::Vector{Float64})
    M = Mx * My
    μ = [eigenvalue_μ(k, m_params; Mx=Mx, D=D) for k in 1:M]
    return blockdiag([build_Fk_matrix(k; m_params = m_params, μ = μ[k], Δt = Δt, m = m, p = p, q = q) for k in 1:M]...)
end



function build_Σk_matrix(; σ::Float64, m::Int, γ::Float64)
    lγ = Int(floor(γ))
    Σ = spzeros(2*m + lγ, 2*m + lγ)
    Σ[1, 1] = σ
    Σ[1, m + lγ + 1] = σ
    Σ[m + lγ + 1, 1] = σ
    Σ[m + lγ + 1, m + lγ + 1] = σ

    return Σ
end

function build_Σ_blockmatrix(M; σ_vec::Vector{Float64}, m::Int, γ::Float64)
    return blockdiag([build_Σk_matrix(; σ = σ_vec[k], m = m, γ = γ) for k in 1:M]...)
end

function simulate_v(; σ::Float64, m::Int, γ::Float64)
    lγ = Int(floor(γ))
    v = spzeros(2*m + lγ)
    ϵ = rand(Normal(0, σ))
    v[1] = ϵ
    v[m + lγ + 1] = ϵ
    return v
end 

function simulate_v_long(M; σ_vec::Vector{Float64}, m::Int, γ::Float64)
    return [simulate_v(; σ = σ_vec[k], m = m, γ = γ) for k in 1:M]
end



function compute_sigma_k_vec(Σ::SparseMatrixCSC; Mx::Int, My::Int, m::Int, m_params::maternParams, D::Rectangle)
    M = Mx * My
    σ_k_vec = zeros(M)
    w = 2*m + floor(Int, m_params.γ) # Width/height of each block
    C0 = C_ck0(; D=D, Mx=Mx, My=My, m_params=m_params).C0
    c_star = sum(C0[k] * eigenfunction_f(D, 0.5, 0.5, k; Mx=Mx)^2 for k in 1:M)

    # for k in 1:M
    #     idx = 1 + (k-1) * w
    #     σ_k_vec[k] = C0[k] / Σ[idx, idx]
    # end 
    return [m_params.σ^2 * C0[k] / c_star / Σ[1 + (k-1)*w, 1 + (k-1)*w] for k in 1:M]
    # return σ_k_vec
end 




function evaluate_pq(γ, m)

    function objective(pq)
        a = pq[1:m+1]
        b = pq[m+2:end]

        loss = 0.0

        xgrid = 0.0:0.001:1.0
        target(x) = (1 - x)^(γ - floor(γ))

        for x in xgrid
            p = sum(a[k+1] * x^k for k in 0:m)
            q = 1.0 + sum(b[k] * x^k for k in 1:m)

            loss += (p / q - target(x))^2
        end

        return loss
    end

    pq = zeros(2m + 1)
    pq[1] = 1.0   # p(x) ≈ 1 initially

    result = optimize(objective, pq, BFGS())

    θ_opt = Optim.minimizer(result)

    p_opt = θ_opt[1:m+1]
    q_opt = vcat(1.0, θ_opt[m+2:end])

    return p_opt, q_opt
end


# Cache for evaluate_pq_fast, keyed on (γ, m). Locked since the gradient evaluates the likelihood on several threads.
const PQ_CACHE = Dict{Tuple{Float64, Int}, Tuple{Vector{Float64}, Vector{Float64}}}()
const PQ_CACHE_LOCK = ReentrantLock()

"""
Same result as evaluate_pq (bit for bit: same BFGS, same floating point operations in the objective),
but x^k and the target (1 - x)^(γ - ⌊γ⌋) are precomputed on the grid instead of recomputed in every
objective call, and the result is cached per (γ, m).
"""
function evaluate_pq_fast(γ, m)
    key = (Float64(γ), m)
    cached = lock(() -> get(PQ_CACHE, key, nothing), PQ_CACHE_LOCK)
    cached === nothing || return cached

    xgrid = collect(0.0:0.001:1.0)
    target = [(1 - x)^(γ - floor(γ)) for x in xgrid]
    xpow = [x^k for k in 0:m, x in xgrid] # (m+1) x length(xgrid)

    function objective(pq)
        loss = 0.0
        @inbounds for i in eachindex(xgrid)
            # p = a_0 + a_1 x + ... + a_m x^m, summed left to right as in evaluate_pq
            p = pq[1] * xpow[1, i]
            for k in 1:m
                p += pq[k+1] * xpow[k+1, i]
            end

            # q = 1 + b_1 x + ... + b_m x^m
            s = pq[m+2] * xpow[2, i]
            for k in 2:m
                s += pq[m+1+k] * xpow[k+1, i]
            end
            q = 1.0 + s

            loss += (p / q - target[i])^2
        end
        return loss
    end

    pq = zeros(2m + 1)
    pq[1] = 1.0   # p(x) ≈ 1 initially

    result = optimize(objective, pq, BFGS())

    θ_opt = Optim.minimizer(result)

    p_opt = θ_opt[1:m+1]
    q_opt = vcat(1.0, θ_opt[m+2:end])

    lock(() -> (PQ_CACHE[key] = (p_opt, q_opt)), PQ_CACHE_LOCK)
    return p_opt, q_opt
end


"""
N steps of S ← F S F' + Σ from S = s0*I, for one w x w block of the block diagonal F and Σ,
by repeated doubling (O(log N) small products instead of N):
    S_N = F^N (s0 I) F^N' + W_N,    W_N = Σ_{i<N} F^i Σ F^i'.
Returns (F^N, W_N) so that the same powers can be reused for any multiple of Σ.
"""
function propagate_block(Fk::Matrix{Float64}, Σk::Matrix{Float64}, N::Int)
    w = size(Fk, 1)
    A_res, W_res = Matrix{Float64}(I, w, w), zeros(w, w)   # 0 steps
    A_pow, W_pow = copy(Fk), copy(Σk)                      # 2^j steps
    n = N
    while n > 0
        if isodd(n)
            # (res steps) followed by (pow steps)
            W_res = W_pow + A_pow * W_res * A_pow'
            A_res = A_pow * A_res
        end
        n >>= 1
        n > 0 || break
        W_pow = W_pow + A_pow * W_pow * A_pow'
        A_pow = A_pow * A_pow
    end
    return A_res, W_res
end 
