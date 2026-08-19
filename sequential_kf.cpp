#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;

// [[Rcpp::export]]
List sequential_kf_fast(NumericVector m_hat, 
                        NumericMatrix s_hat, 
                        NumericVector y, 
                        NumericMatrix Hm, 
                        NumericVector R_diag, 
                        IntegerVector NA_ind) {
  
  int m = m_hat.size();
  int n_stations = y.size();
  double log_lik = 0.0;
  const double log2pi = std::log(2.0 * M_PI);
  
  // Track NAs for O(1) skipping
  std::vector<bool> is_na(n_stations, false);
  for(int k = 0; k < NA_ind.size(); ++k) {
    if(NA_ind[k] > 0 && NA_ind[k] <= n_stations) is_na[NA_ind[k] - 1] = true;
  }

  for(int j = 0; j < n_stations; ++j) {
    if(is_na[j]) continue;

    // 1. Innovation: dy = y[j] - H[j,] * m_hat
    double dy_j = y[j];
    for(int k = 0; k < m; ++k) {
      dy_j -= Hm(j, k) * m_hat[k];
    }

    // 2. Vector a = P * H'  (Dimension: m x 1)
    // This is essentially multiplying the matrix s_hat by the j-th row of Hm
    NumericVector a(m);
    for(int r = 0; r < m; ++r) {
      double sum = 0;
      for(int c = 0; c < m; ++c) {
        sum += s_hat(r, c) * Hm(j, c);
      }
      a[r] = sum;
    }

    // 3. Scalar Innovation Variance: sj = H * a + R[j]
    double sj = R_diag[j];
    for(int k = 0; k < m; ++k) {
      sj += Hm(j, k) * a[k];
    }

    // 4. Update Log-Likelihood
    log_lik -= 0.5 * (log2pi + std::log(sj) + (dy_j * dy_j) / sj);

    // 5. Kalman Gain: K = a / sj (Vector m x 1)
    // 6. Update State: m_hat = m_hat + K * dy_j
    for(int k = 0; k < m; ++k) {
      double kj = a[k] / sj;
      m_hat[k] += kj * dy_j;
      
      // 7. Update Covariance: P = P - K * a' (Rank-1 update)
      // Since K = a/sj, this is: P = P - (a * a') / sj
      for(int l = 0; l < m; ++l) {
        s_hat(k, l) -= (a[k] * a[l]) / sj;
      }
    }
  }

  return List::create(
    _["m_hat"]   = m_hat,
    _["s_hat"]   = s_hat,
    _["log_lik"] = log_lik
  );
}