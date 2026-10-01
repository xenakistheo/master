include("parameters.jl")


# Indexing functions for the spatial grid
ij_to_k(i, j, Mx) = i + (j - 1) * Mx

function k_to_ij(k, Mx)
    i = mod(k - 1, Mx) + 1
    j = div(k - 1, Mx) + 1
    return i, j
end

##### EIGENVALUE ξ
function eigenvalue_ξ(i::Int, j::Int)
    return (i^2 + j^2)*pi^2
end 

function eigenvalue_ξ(D::Rectangle, i::Int, j::Int)
    return ((i^2 / D.A1^2) + (j^2 / D.A2^2))*pi^2
end 

function eigenvalue_ξ(D::Rectangle, k::Int; Mx::Int)
    i, j = k_to_ij(k, Mx)
    return eigenvalue_ξ(D, i, j)
end 

##### EIGENFUNCTION f
function eigenfunction_f(D::Rectangle, x::Float64, y::Float64, i::Int, j::Int)
    δ(u, v) = u == v ? 1 : 0
    normalization_factor = 2^(1 - 0.5*δ(i,0) - 0.5*δ(j,0))/sqrt(D.A1*D.A2)
    return normalization_factor*cos(i*pi*x/D.A1)*cos(j*pi*y/D.A2)
end

function eigenfunction_f(D::Rectangle, x::Float64, y::Float64, k::Int; Mx::Int)
    i, j = k_to_ij(k, Mx)
    return eigenfunction_f(D, x, y, i, j)
end

##### EIGENVALUE μ
function eigenvalue_μ(ξ; r::Float64=1.0, κ::Float64=1.0, α::Float64=1.0)
    return (κ^2 + ξ)^(α)/float(r)
end

function eigenvalue_μ(ξ::Float64, m::maternParams)
    return eigenvalue_μ(ξ; r=m.r, κ=m.κ, α=m.α)
end
function eigenvalue_μ(k::Int, m::maternParams; Mx::Int, D::Rectangle)
    ξ = eigenvalue_ξ(D, k, Mx)
    return eigenvalue_μ(ξ; r=m.r, κ=m.κ, α=m.α)
end


##### EIGENVALUE λ
function eigenvalue_λ_unnormalized(ξ; r::Float64=1.0, κ::Float64=1.0, β::Float64=1.0, γ::Float64=1.0, σ::Float64=1.0)
    return σ^2 * float(r)^(-2γ) * (κ^2 + ξ)^(-β)
end

function eigenvalue_λ_unnormalized(ξ::Float64, m::maternParams)
    return eigenvalue_λ_unnormalized(ξ; r=m.r, κ=m.κ, β=m.β, γ=m.γ, σ=m.σ)
end



##### EIGENVALUES
function eigenvalues(D::Rectangle, k::Int; Mx::Int, m::maternParams)
    ξ = eigenvalue_ξ(D, k, Mx)
    μ = eigenvalue_μ(ξ, m)
    λ_tilde = eigenvalue_λ_unnormalized(ξ, m)
    return (; ξ, μ, λ_tilde)
end 





