include("parameters.jl")
include("spectral_discretization.jl")
using SpecialFunctions
using Distributions
using ToeplitzMatrices

function build_covariance_matrix(N, λ, μ, γ)
    γ > 0.5 || throw(ArgumentError("γ must be > 1/2"))

    ν = γ - 0.5
    prefactor = λ / (sqrt(pi) * gamma(γ))

    # Unique covariance values:
    # c[h+1] = C(h), h = 0,...,N-1
    c = Vector{Float64}(undef, N)

    c[1] = λ * gamma(ν) /
           (2sqrt(pi) * gamma(γ) * μ^(2ν))

    @inbounds for h in 1:N-1
        c[h+1] =
            prefactor *
            (h / (2μ))^ν *
            besselk(ν, μ*h)
    end

    # Fill Toeplitz matrix
    # Σ = Matrix{Float64}(undef, N, N)

    # @inbounds for j in 1:N
    #     for i in 1:N
    #         Σ[i,j] = c[abs(i-j) + 1]
    #     end
    # end
    Σ = SymmetricToeplitz(c)

    return Σ
end


"""
Create observation matrix H with entries
    H_{ik} = f_k(s_i)

Dimension: N_spatial x (Mx * My)
"""
function build_observation_matrix(spatial_locations::Vector{Tuple{Float64, Float64}}, Mx::Int, My::Int, D::Rectangle)
    
    N_spatial = length(spatial_locations)
    H = Matrix{Float64}(undef, N_spatial, Mx * My)

    for (i, si) in enumerate(spatial_locations)
        x, y = si
        for k in 1:(Mx * My)
            H[i, k] = eigenfunction_f(D, x, y, k; Mx=Mx)
        end
    end
    return H
end 


function stationary_variance(λ_tilde, μ, γ)
    return λ_tilde *
           gamma(2γ - 1) /
           (2^(2γ - 1) * gamma(γ)^2) *
           μ^(1 - 2γ)
end


function sample_GRF(m::maternParams,  H::Matrix{Float64}, D::Rectangle, Mx_sim::Int, My_sim::Int, N_spatial::Int, N_temporal::Int, σ_obs::Float64, R::Int=1)

    M_sim = Mx_sim * My_sim
    C = Array{Float64}(undef, N_temporal, R, M_sim)
    ϵ = σ_obs * randn(N_temporal, R, N_spatial)
    y = Array{Float64}(undef, N_temporal, R, N_spatial)

    ξ = Vector{Float64}(undef, M_sim)
    λ = Vector{Float64}(undef, M_sim)
    μ = Vector{Float64}(undef, M_sim)
    NORM = Vector{Float64}(undef, M_sim)

    for k in 1:M_sim
        ξ[k], μ[k], λ[k] = eigenvalues(D, k, Mx=Mx_sim, m=m)  
        NORM[k] = stationary_variance(λ[k], μ[k], m.γ)
    end 

    # Normalize lambda 
    C_norm = sum(NORM)
    NORM ./= C_norm
    λ ./= C_norm

    # Verify that NORM sums to 1
    @assert isapprox(sum(NORM), 1.0; atol=1e-10)

    x0, y0 = 0.5, 0.5
    c_star = 0.0

    for k in 1:M_sim
        fk_center = eigenfunction_f(D, x0, y0, k; Mx=Mx_sim)
        c_star += NORM[k] * fk_center^2
    end

    # Evaluate all eigenfunctions at (0.5, 0.5) and 


    # Compute covariance matrices for each mode and sample coefficients 
    for k in 1:M_sim
        Σk_toeplitz = build_covariance_matrix(N_temporal, λ[k], μ[k], m.γ)
        Σk = Matrix(Σk_toeplitz)
        Σk .*= m.σ^2 / c_star
        for r in 1:R
            C[:, r, k] = rand(MvNormal(Σk)) # C_kr
        end 
    end 

    # Combine sampled coefficients with observation and noise to 
    # form the observed data
    for n in 1:N_temporal, r in 1:R
        y[n, r, :] = H*C[n, r, :] + ϵ[n, r, :]
    end 

    return (; y, C, ϵ)
end 

