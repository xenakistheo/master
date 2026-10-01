include("../likelihood.jl")
include("../GRF.jl")

using LineSearches
using JLD2
using Optim
using LinearAlgebra

# The gradient is parallelized over Julia threads, so keep BLAS single-threaded to avoid oversubscription
Threads.nthreads() > 1 && BLAS.set_num_threads(1)

# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations

# Replication index r: from the SLURM array task id, else the first command-line argument, else 1
r = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", isempty(ARGS) ? "1" : ARGS[1]))
Y_obs = y_LL[:, r, :]
println("Estimating parameters for replication r = ", r)


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


# Central finite-difference gradient, with the 2 evaluations per coordinate run in parallel over threads.
# Same step size as FiniteDiff.jl's default for central differences.
function fd_grad!(g, η)
    Threads.@threads for i in eachindex(η)
        h = cbrt(eps(Float64)) * max(1.0, abs(η[i]))
        e = zeros(length(η))
        e[i] = h
        g[i] = (safe_objective(η .+ e) - safe_objective(η .- e)) / (2h)
    end
    return g
end


# Set stopping critera 
opts = Optim.Options(
    f_reltol   = 1e-8,    # relative change in objective
    x_abstol   = 1e-4,    # change in η (log-scale for most parameters)
    g_abstol   = 1e-5,    # realistic for finite-difference gradients
    iterations = 200,
    time_limit = 4 * 3600,
    show_trace = true,
)

@time result = optimize(safe_objective, fd_grad!, η0, BFGS(alphaguess = InitialStatic(scaled=true)), opts)

η_opt = Optim.minimizer(result) # Extract the optimal parameters
θ_opt = η_to_θ(η_opt) # Convert the optimal parameters to the interpretable form

println(result)
begin 
    println("ν_s = ", θ_opt[1])
    println("ν_t = ", θ_opt[2])
    println("r_t = ", θ_opt[3])
    println("r_s = ", θ_opt[4])
    println("σ = ", θ_opt[5])
    println("β_obs = ", θ_opt[6])
    println("σ_obs = ", θ_opt[7])
end 

# Save the estimates for this replication
results_dir = "furset26/simulation_study/results/LL"
mkpath(results_dir)
converged = Optim.converged(result)
iterations = Optim.iterations(result)
minimum_value = Optim.minimum(result)
@save joinpath(results_dir, "estimate_r$(lpad(r, 2, '0')).jld2") r θ_opt η_opt converged iterations minimum_value

# TRUE VALUES 
# ν_s = 1.0
# ν_t = 1.0
# r_t = 10.0
# r_s = 1.0
# σ = 3.5
# β_obs_LL = 0.25
# σ_obs_LL = 0.35 




