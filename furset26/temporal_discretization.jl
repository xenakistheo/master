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
