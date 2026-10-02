# Benchmark of loglikelihood (and loglikelihood_fast) at the starting point of optimize_several_parameters.jl.
# Run from the repository root:  julia --project=. furset26/benchmarking/benchmark_loglikelihood.jl

include("../likelihood.jl")

using BenchmarkTools
using JLD2
using LinearAlgebra

# Same setting as the optimizer: gradient threads, so single-threaded BLAS
BLAS.set_num_threads(1)

# Load simulated data and spatial locations
@load "furset26/simulation_study/furset26_simulation_data.jld2" y_LL spatial_locations
Y_obs = y_LL[:, 1, :]

# Fixed parameters (as in optimize_several_parameters.jl)
D = Rectangle(1.0, 1.0)
m_order = 2
Δt = 1.0
N_init = 1000
Mx_inf, My_inf = 8, 8

H_spatial = build_observation_matrix(spatial_locations, Mx_inf, My_inf, D)

η0 = θ_to_η([1.0, 1.0, 5.0, 0.5, 2.0, 0.3, 0.5]) # Initial guess for the parameters

kw = (; H_spatial=H_spatial, D=D, Y_obs=Y_obs, Δt=Δt, N_init=N_init, Mx_inf=Mx_inf, My_inf=My_inf, m_order=m_order)


println("loglikelihood:")
ll = loglikelihood(η0; kw...)
@show ll
@btime loglikelihood($η0; $kw...)


println("\nloglikelihood_fast:")
ll_fast = loglikelihood_fast(η0; kw...)
@show ll_fast ll_fast - ll

# First call for a new γ: p, q are fitted (cache emptied before every evaluation)
println("cold p, q cache:")
@btime loglikelihood_fast($η0; $kw...) setup=(empty!(PQ_CACHE)) evals=1

# Repeated γ (most evaluations within a finite-difference gradient)
println("warm p, q cache:")
@btime loglikelihood_fast($η0; $kw...)
