struct maternParams
    d::Int # Spatial dimension
    r::Float64
    κ::Float64
    α::Float64
    β::Float64
    γ::Float64
    σ::Float64
end

struct maternParams_interpretable
    d::Int # Spatial dimension
    ν_s::Float64
    ν_t::Float64
    r_s::Float64
    r_t::Float64
    β_s::Float64
    σ::Float64
end

function maternParams(;
    d::Int,
    r::Real,
    κ::Real,
    α::Real,
    β::Real,
    γ::Real,
    σ::Real,
)
    return maternParams(d, r, κ, α, β, γ, σ)
end

function maternParams_interpretable(;
    d::Int,
    ν_s::Real,
    ν_t::Real,
    r_s::Real,
    r_t::Real,
    β_s::Real,
    σ::Real,
)
    return maternParams_interpretable(d, ν_s, ν_t, r_s, r_t, β_s, σ)
end

function interpretable_to_matern(m::maternParams_interpretable)
    d, ν_s, ν_t, r_s, r_t, β_s, σ = m.d, m.ν_s, m.ν_t, m.r_s, m.r_t, m.β_s, m.σ

    β_star = ν_s /( ν_s + d/2)

    γ = ν_t * max(1, β_s/β_star) + 0.5
    α = ν_s / (2 * ν_t) * min(1, β_s/β_star)
    β = (1 - β_s)/β_star * ν_s
    κ = sqrt(8*ν_s)/r_s
    r = r_t * κ^(2*α) / sqrt(8* (γ - 0.5))

    return maternParams(d, r, κ, α, β, γ, σ)
end

function matern_to_interpretable(m::maternParams)
    d, r, κ, α, β, γ, σ = m.d, m.r, m.κ, m.α, m.β, m.γ, m.σ 

    ν_s = β + (2*γ - 1)*α - 0.5 
    ν_t = γ - 0.5 + (1/α)*min(β - 0.5*d, 0)
    β_s = (2*γ - 1)*α / (β + (2*γ - 1)*α) 
    r_s = sqrt(8 * ν_s) / κ
    r_t = r * sqrt(8 * (γ - 0.5)) / κ^(2 * α)

    return maternParams_interpretable(d, ν_s, ν_t, r_s, r_t, β_s, σ)
end




# Define types for different Domains.
abstract type Domain end
abstract type SpatialDomain <: Domain end

"""Rectangle is defined as D = Rectangle(a, b) which represents ([0,a] × [0,b])"""
struct Rectangle <: SpatialDomain
    A1::Float64
    A2::Float64
end 





#### Transform parameters to make them satisfy certain numerical bounds. 
function θ_to_η(η)

    ν_s = exp(η[1]) + 0.25 
    ν_t = 7.5 * exp(η[2]) / (3 + 2.5*exp(η[2])) + 0.25
    r_t = exp(η[3]) + 0.005 
    r_s = exp(η[4]) + 0.005 
    σ = exp(η[5]) + 0.005 
    β_s = exp(3*η[6]) / (2 + exp(3*η[6]))
    σ_obs = exp(η[7])

    θ = [
        ν_s,
        ν_t,
        r_t,
        r_s,
        σ,
        β_s,
        σ_obs,
    ]
    return θ
end 

function η_to_θ(θ)
    ν_s, ν_t, r_t, r_s, σ, β_s, σ_obs = θ
    return [
        log(ν_s - 0.25),
        log((ν_t - 0.25)/2.5) - log(1 - (ν_t - 0.25)/3),
        log(r_t - 0.005),
        log(r_s - 0.005),
        log(σ - 0.005),
        log(2 * β_s / (1 - β_s)) / 3,
        log(σ_obs),
    ]
end
