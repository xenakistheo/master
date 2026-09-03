include("src/kalman.jl")
using DelimitedFiles

# --- Fixed test data, identical to test_sequential_kf.R for direct comparison ---
x0 = [1.0, -2.0, 0.5]                     # prior state mean

Σ0 = [2.0   0.3   0.1;
      0.3   1.5  -0.2;
      0.1  -0.2   1.0]                    # prior state covariance (symmetric, PD)

H = [1.0  0.0   2.0;
     0.0  1.0  -1.0;
     1.0  1.0   0.0]                      # observation matrix (3 stations x 3 states)

z = [3.2, -1.5, 0.7]                      # observations
R_diag = [0.5, 0.8, 0.3]                  # observation noise variances

A = Matrix(1.0I, 3, 3)                    # unused by sequential_kf_fast, but KalmanModel needs a value
Q = Matrix(0.1I, 3, 3)                    # unused by sequential_kf_fast, but KalmanModel needs a value
km = KalmanModel(A, H, Q)

println("=== Case A: no missing stations ===")
xA, ΣA, llA = sequential_kf_fast(copy(x0), copy(Σ0), z, R_diag, km)
println("m_hat:")
println(xA)
println("s_hat:")
println(ΣA)
println("log_lik: ", llA)

println("\n=== Case B: station 2 missing ===")
xB, ΣB, llB = sequential_kf_fast(copy(x0), copy(Σ0), z, R_diag, km; NA_ind=[2])
println("m_hat:")
println(xB)
println("s_hat:")
println(ΣB)
println("log_lik: ", llB)

println("\n=== Speed (Case A, no missing stations) ===")
@btime sequential_kf_fast(copy($x0), copy($Σ0), $z, $R_diag, $km)

# --- Large-scale case: m=1000 states, n=250 stations, shared with test_sequential_kf.R ---
m_hat_L = vec(readdlm("kf_bench_data/m_hat.csv", ','))
H_L = readdlm("kf_bench_data/Hm.csv", ',')
R_diag_L = vec(readdlm("kf_bench_data/R_diag.csv", ','))
y_L = vec(readdlm("kf_bench_data/y.csv", ','))

m = length(m_hat_L)
Σ_L = Matrix(1.0I, m, m)

A_L = Matrix(1.0I, m, m)
Q_L = Matrix(1.0I, m, m)
km_L = KalmanModel(A_L, H_L, Q_L)

println("\n=== Large case: m=1000 states, n=250 stations ===")
xL, ΣL, llL = sequential_kf_fast(copy(m_hat_L), copy(Σ_L), y_L, R_diag_L, km_L)
println("log_lik: ", llL)

println("\n=== Speed (large case) ===")
@btime sequential_kf_fast(copy($m_hat_L), copy($Σ_L), $y_L, $R_diag_L, $km_L)
