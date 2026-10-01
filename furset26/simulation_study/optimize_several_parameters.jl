include("../likelihood.jl")
include("../GRF.jl")

using LineSearches
using JLD2
using Optim

# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations
Y_obs = y_LL[:, 1, :]


# Fixed parameters
D = Rectangle(1.0, 1.0) # Define spatial domain. 
m_order = 2
Δt = 1.0
N_init = 1000
Mx_inf, My_inf = 8, 8

# Build the observation matrix
H_spatial = build_observation_matrix(spatial_locations, Mx_inf, My_inf, D)

# Define the objective function, scale by number of observations for numerical stability
objective(η) = -loglikelihood(η; H_spatial=H_spatial, D=D, Y_obs=Y_obs, Δt=Δt, N_init=N_init, Mx_inf=Mx_inf, My_inf=My_inf, m_order=m_order) / length(Y_obs)

η0 = θ_to_η([1.0, 1.0, 5.0, 0.5, 2.0, 0.3, 0.5]) # Initial guess for the parameters

function safe_objective(η)
    v = try objective(η) catch e
        e isa PosDefException || e isa DomainError || rethrow()
        return 1e10
    end
    return isfinite(v) ? v : 1e10
end

@time result = optimize(safe_objective, η0, BFGS(alphaguess = InitialStatic(scaled=true)), Optim.Options(show_trace=true))

η_opt = Optim.minimizer(result) # Extract the optimal parameters
θ_opt = η_to_θ(η_opt) # Convert the optimal parameters to the interpretable form

begin 
    println("ν_s = ", θ_opt[1])
    println("ν_t = ", θ_opt[2])
    println("r_t = ", θ_opt[3])
    println("r_s = ", θ_opt[4])
    println("σ = ", θ_opt[5])
    println("β_obs = ", θ_opt[6])
    println("σ_obs = ", θ_opt[7])
end 

# TRUE VALUES 
# ν_s = 1.0
# ν_t = 1.0
# r_t = 10.0
# r_s = 1.0
# σ = 3.5
# β_obs_LL = 0.25
# σ_obs_LL = 0.35 




