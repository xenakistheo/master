using Random
using DelimitedFiles

# Generates one shared set of random matrices at the "real" problem scale
# (m states, n stations) and writes them to plain CSV so both the Julia
# and the R benchmark scripts load byte-identical data.

Random.seed!(42)

m = 1000   # state dimension
n = 250    # number of stations

m_hat = randn(m)
Hm = randn(n, m) .* 0.05        # scaled down so innovation variances stay reasonable
R_diag = 0.1 .+ 0.4 .* rand(n)  # noise variances in [0.1, 0.5]
y = Hm * m_hat .+ sqrt.(R_diag) .* randn(n)   # observations consistent with m_hat + noise

mkpath("kf_bench_data")
writedlm("kf_bench_data/m_hat.csv", m_hat, ',')
writedlm("kf_bench_data/Hm.csv", Hm, ',')
writedlm("kf_bench_data/R_diag.csv", R_diag, ',')
writedlm("kf_bench_data/y.csv", y, ',')

println("Wrote m=$m, n=$n test data to kf_bench_data/")
