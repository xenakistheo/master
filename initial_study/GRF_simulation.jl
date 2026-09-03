using Distributions
using Plots



function eigenvalue_xi(i, j)
    return (i^2 + j^2)*pi^2
end 


function eigenfunction_f(x, y, i, j)
    normalization_factor = 2^(1 - 0.5*kron(i,0) - 0.5*kron(j,0))
    return normalization_factor*cos(i*pi*x)*sin(j*pi*y)
end

function eigenvalue_mu(ξ; r=1, κ=1, α=1)
    return float(r)^(-1)*(κ^2 + ξ)^(α)
end

function eigenvalue_lambda(ξ; r=1, κ=1, β=1, γ=1, C=1, σ=1)
    return C*σ^2 * float(r)^(-2γ) * (κ^2 + ξ)^(-β)
end

function simulate_ck(μ, λ, Δt, N)

    z = randn(N) # Generate N standard normal random variables
    c_array = zeros(N) # Initialize an array to store ck values
    c_array[1] = rand(Normal(0, sqrt(λ/(2*μ)))) # Initialize ck with the first value

    for k in 2:N
        c_array[k] = exp(-μ*Δt)*c_array[k-1] + sqrt(λ/(2*μ) * (1 - exp(-2*μ*Δt))) * z[k] 
    end 
    return c_array
end 

function simulate_GRF(Δx, Δy, xy_M, Δt, N; r=1, κ=1, α=1, β=1, γ=1, C=1, σ=1)
    X = collect(0:Δx:1) # Discretized x-coordinates
    Y = collect(0:Δy:1) # Discretized y-coordinates

    # Precompute eigenvalues and eigenfunctions
    ξ = [eigenvalue_xi(i, j) for i in 0:xy_M[1], j in 0:xy_M[2]]
    μ = [eigenvalue_mu(ξ_ij; r=r, κ=κ, α=α) for ξ_ij in ξ]
    λ = [eigenvalue_lambda(ξ_ij; r=r, κ=κ, β=β, γ=γ, C=C, σ=σ) for ξ_ij in ξ]


    field = zeros(length(X), length(Y), N) # Initialize the field array


    for i in 0:xy_M[1]
        for j in 0:xy_M[2]
            μ_ij = μ[i+1, j+1] # Adjust index for 1-based indexing
            λ_ij = λ[i+1, j+1] # Adjust index for 1-based indexing
            c_array = simulate_ck(μ_ij, λ_ij, Δt, N) # Simulate ck values

            for t in 1:N
                field[:, :, t] .+= c_array[t] * eigenfunction_f.(X', Y, i, j) # Update the field
            end
        end
    end
    return field
end 




xy_range = ((0, 1), (0, 1)) # Spatial domain
Δx = 0.01 # Spatial step size in x-direction
Δy = 0.01 # Spatial step size in y-direction

xy_M = (10, 10) # Number of eigenmodes in each direction
Δt = 0.01 # Temporal step size
N = 1000 # Number of time steps

r = 3.0
κ = 2.0
C = 2.4
σ = 3.0

α = 1
β = 1
γ = 1


FIELD = simulate_GRF(Δx, Δy, xy_M, Δt, N; r, κ, α, β, γ, C, σ)



# Make Video Animation



# Keep the same color scale across all frames
clims = extrema(FIELD)

anim = @animate for t in axes(FIELD, 3)
    heatmap(
        FIELD[:, :, t],
        clims = clims,
        aspect_ratio = :equal,
        color = :viridis,
        title = "t = $t",
        xlabel = "x",
        ylabel = "y",
    )
end

gif(anim, "field.gif", fps=10)