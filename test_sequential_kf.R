library(Rcpp)
library(microbenchmark)
sourceCpp("sequential_kf.cpp")

options(digits = 10)

# --- Fixed test data, identical to test_sequential_kf.jl for direct comparison ---
# NOTE: Rcpp's NumericVector/NumericMatrix wrap R's memory directly, so passing
# the same object into two calls can let the first call's in-place mutation leak
# into the second. We rebuild the literals fresh for each case to avoid that.

new_m_hat <- function() c(1.0, -2.0, 0.5)

new_s_hat <- function() matrix(c(2.0,  0.3,  0.1,
                                  0.3,  1.5, -0.2,
                                  0.1, -0.2,  1.0), nrow = 3, byrow = TRUE)

Hm <- matrix(c(1.0, 0.0,  2.0,
               0.0, 1.0, -1.0,
               1.0, 1.0,  0.0), nrow = 3, byrow = TRUE)

y <- c(3.2, -1.5, 0.7)
R_diag <- c(0.5, 0.8, 0.3)

cat("=== Case A: no missing stations ===\n")
resA <- sequential_kf_fast(new_m_hat(), new_s_hat(), y, Hm, R_diag, integer(0))
cat("m_hat:\n"); print(resA$m_hat)
cat("s_hat:\n"); print(resA$s_hat)
cat("log_lik:", resA$log_lik, "\n")

cat("\n=== Case B: station 2 missing ===\n")
resB <- sequential_kf_fast(new_m_hat(), new_s_hat(), y, Hm, R_diag, as.integer(2))
cat("m_hat:\n"); print(resB$m_hat)
cat("s_hat:\n"); print(resB$s_hat)
cat("log_lik:", resB$log_lik, "\n")

cat("\n=== Speed (Case A, no missing stations) ===\n")
bm <- microbenchmark(
  sequential_kf_fast(new_m_hat(), new_s_hat(), y, Hm, R_diag, integer(0)),
  times = 10000
)
print(bm)

# --- Large-scale case: m=1000 states, n=250 stations, shared with test_sequential_kf.jl ---
m_hat_L <- as.vector(as.matrix(read.csv("kf_bench_data/m_hat.csv", header = FALSE)))
Hm_L <- as.matrix(read.csv("kf_bench_data/Hm.csv", header = FALSE))
R_diag_L <- as.vector(as.matrix(read.csv("kf_bench_data/R_diag.csv", header = FALSE)))
y_L <- as.vector(as.matrix(read.csv("kf_bench_data/y.csv", header = FALSE)))

m_L <- length(m_hat_L)
new_m_hat_L <- function() as.numeric(m_hat_L)  # explicit copy: Rcpp mutates its vector args in place
new_s_hat_L <- function() diag(1, m_L)

cat("\n=== Large case: m=1000 states, n=250 stations ===\n")
resL <- sequential_kf_fast(new_m_hat_L(), new_s_hat_L(), y_L, Hm_L, R_diag_L, integer(0))
cat("log_lik:", resL$log_lik, "\n")

cat("\n=== Speed (large case) ===\n")
bmL <- microbenchmark(
  sequential_kf_fast(new_m_hat_L(), new_s_hat_L(), y_L, Hm_L, R_diag_L, integer(0)),
  times = 50
)
print(bmL)
