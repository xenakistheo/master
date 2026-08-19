library(pracma)
library(vioplot)
library(Matrix)
library(optimParallel)
library(MASS)
library(verification)
library(latex2exp)
library(Rcpp)

sourceCpp(paste0(getwd(),"/sequential_kf.cpp"))
Rcpp.package.skeleton("KFParallel", cpp_files = "sequential_kf.cpp")
install.packages("KFParallel", repos = NULL, type = "source")

#change to basis = sin want sin basis
basis = cos
basis_is_cos = (basis(0) == 1)

#returns a list containing:
#eigen$LA: vector of lambda_k
#eigen$MU: vector of mu_k
#eigen$XI: vector of eigenvalues of laplacian 
#eigen$NORM: 
#eigen$keys: cosine basis is e_k(x) = C*sin(i*pi*j). 
#            eigen$keys is a matrix that maps k -> (i,j) such that keys[,k] = c(i,j)
#            the mapping is chosen so that the k largest eigenvalues are selected
EIGEN = function(ga, al, be, ka, r, sigma, M, a = 1, b = 1) {
  
  LA = c()
  MU = c()
  NORM = c()
  keys = matrix(data = NA, nrow = 3, ncol = 4*M^2)
  
  k = 1
  if(basis_is_cos) {I = 0:(2*M - 1)}
  else {I = 1:(2*M)}
  for(i in I) {
    for(j in I) {
      keys[,k] = c(i, j, (i^2 + j^2) * pi^2)
      k = k + 1
    }
  }
  keys = keys[, order(keys[3,])]
  keys = keys[,1:M^2]
  
  XI = keys[3,]
  
  k = 1
  for(k in 1:M^2) {
    i = keys[1,k]
    j = keys[2,k]
    LA = c(LA, r^(-2*ga) * (ka^2 + XI[k])^(-be))
    MU = c(MU, (ka^2 + XI[k])^al / r)
    NORM = c(NORM, LA[k] * MU[k]^(1 - 2 * ga))
    k = k + 1
  }
  
  #for(i in 0:(M - 1)) {
  #  for(j in 0:(M - 1)) {
  #    keys[i + 1, j + 1] = k
  #    XI = c(XI, (i^2 + j^2) * pi^2)
  #    LA = c(LA, r^(-2*ga) * (ka^2 + XI[k])^(-be))
  #    MU = c(MU, (ka^2 + XI[k])^al / r)
  #    NORM = c(NORM, LA[k] * MU[k]^(1 - 2 * ga))
  #    k = k + 1
  #  }
  #}
  
  NORM = gamma(2 * ga - 1) / gamma(ga)^2 * 2^(1 - 2*ga) * NORM
  C = sum(NORM)
  NORM = NORM / C
  LA = LA / C
  #NORM = sigma^2 * NORM / C
  #LA = sigma^2 * LA / C
  
  return(list("LAMBDA" = LA, "MU" = MU, "XI" = XI, "NORM" = NORM, "keys" = keys))
  
}

###The next three functions deal with the rational approximation
###There is no rational approximation used when gamma = 1, so you can ignore all of it except the alpha==0 case in arma_coef

#r_approx_func computes a rational approximation of (1-x)^(alpha)
#returns a list containing:
#p: vector of polynomial coefficients in numerator, ordered from highest to lowest power
#q: vector of polynomial coefficients in denominator, ordered from highest to lowest power
r_approx_func = function(alpha, deg) {
  xx = seq(0, 1, 0.001)
  yy = (1 - xx)^(alpha)
  v = rationalfit(xx, yy, d1 = deg, d2 = deg)
  return(list("p" = v$p1, "q" = v$p2))
}

#sup_norm_error returns the sup norm error of a rational approximation on the complex unit disc
#can also plot the error
sup_norm_error = function(p, q, alpha, plot = F) {
  pn = length(p)
  qn = length(q)
  
  r = 0.9
  
  theta = seq(0, 2*pi, 2*pi/10001)
  
  m = -Inf
  
  err = function(th) {
    z = r*exp(1i * th)
    z_p = z^((pn - 1):0)
    z_q = z^((qn - 1):0)
    a = p %*% z_p
    b = q %*% z_q
    e = (1 - z)^alpha - a/b
    return(e)
  }
  
  e = sapply(theta, err)
  
  if(plot == T) {
    plot(e, type = "l")
    points(0,0,col = "red")
  }
  
  return(max(abs(e)))
  
}

#arma_coef computes the coeffecients of the ARMA approximation on a basis function
#returns a list containing:
#phi: vector of AR coefficients, ordered from lowest to highest lag (if i recall correctly)
#theta: vector of MA coefficients, ordered from lowest to highest lag
#epsilon: sup norm error of the rational approximation
arma_coef = function(mu, ga, h, n, approx_func, override = F) {
  
  alpha = ga - floor(ga)
  
  if(alpha == 0) {
    
    ar_coef = rev(choose(ga, ga:1) * (-1)^((ga:1) + 1) * exp(-mu*h*(ga:1)))
    ma_coef = numeric(0)
    epsilon = NaN
    
  } else {
    
    rat_approx = approx_func(alpha, n)
    epsilon = sup_norm_error(rat_approx$p, rat_approx$q, alpha, plot = F)
    delta = 1 - exp(-mu*h)
    
    #this is a numerical stability condition that appears in my theoretical results
    #if override is TRUE, the code will stop if the stab.cond. is violated
    #disabled by default
    if(!override) {stopifnot(epsilon * delta^(-alpha)  < 1)}
    
    m = floor(ga)
    ar = choose(m, m:0) * (-1)^(m:0)
    
    ar_outer = outer(rat_approx$p, ar)
    ar_coef = as.vector(tapply(ar_outer, row(ar_outer) + col(ar_outer), sum)) * exp(-mu*h*((m + n):0))
    ar_coef = rev(-ar_coef[1:(m + n)] / ar_coef[m + n + 1])
    
    ma_coef = rat_approx$q * exp(-mu*h*(n:0))
    ma_coef = rev(-ma_coef[1:n] / ma_coef[n + 1])
    
  }
  
  return(list("phi" = ar_coef, "theta" = ma_coef, "epsilon" = epsilon))
  
}

#computes the exact covariance function for a coefficient process
#returns a function R+ -> R
covariance_function_exact = function(mu, ga, la) {
  C_ga = la / (2*mu)^(2*ga - 1) / gamma(ga)^2
  r = function(h) {
    f = function(u) {(u + 2*mu*h)^(ga - 1) * u^(ga - 1) * exp(-u)}
    s = integrate(f,0,Inf)$value
    return(C_ga * s * exp(- mu * h))
  }
  return(r)
}

#computes the observation matrix for a [b1,b2]x[b1,b2] grid with (Nf + 1)^2 equispaced nodes
full_structure_matrix = function(eigen, Nf = 100, b1 = 0, b2 = 1, p = 1, q = 0) {
  
  xobs = rep(seq(b1,b2, (b2-b1)/Nf), Nf + 1)
  yobs = sort(xobs)
  
  M = sqrt(length(eigen$MU)) 
  
  Hm = matrix(0, ncol = (p + q) * M^2, nrow = length(xobs))
  #for(l in 1:N_obs) {
  #  ind = (1:N)[-NA_ind][l]
  #  Hm[l, (p + q) * M^2 + 1:param_n] = c(1, altitude[ind], cos(pi / 12  * iter))
  #}
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    for(l in 1:length(xobs)) {
      n_ = (k - 1) * (p + q) +  1
      #ind = (1:N)[-NA_ind][l] ### needed for potential missing data example
      #print(c(N_obs, length(x), length(y), ind))
      Hm[l, n_] = C * basis(pi * xobs[l] * i) * basis(pi * yobs[l] * j)
    }
  }
  
  return(Hm)
  
}

#These two functions are commented out since they are not used in this implementation.
#They compute an inverse using a version of the Woodbury identity that can handle non-negative definite matrices.

#semi_cholesky = function(A) {
#  a = norMmix::ldl(A)
#  chol_ = a$L %*% diag(sqrt(pmax(0,a$D)))
#  return(chol_)
#}
#pseudoWoodburyInverse = function(Hm, tau, S) {
#  
#  A = semi_cholesky(S)
#  HA = Hm %*% A
#  
#  iden_n = diag(rep(1, dim(Hm)[1]))
#  iden_m = diag(rep(1, dim(A)[1]))
#  
#  Ff = iden_m + t(HA) %*% HA / tau^2
#  Ee = iden_n - HA %*% solve(Ff) %*% t(HA) / tau^2
#  inv = Ee / tau^2
#
#  return(inv)
#  
#}

#This is the simulation code.
#returns a list containing: 
#Y: an NxT matrix containg the data
#X_eff: an NxT matrix containing the data without measurement error
#X_raw: an NxT matrix containing the data without measurement error or covariates (currently X_eff = X_raw)
#spec_procs: an M^2 x T matrix containing the spectral processes on each coefficients
#x: a list of x values of the spatial locations
#y: a list of y values of the spatial locations
#eigen: the EIGEN-list associated with the simulation
#Hm: the observation matrix associated with the simulation
exact_simulate_data = function(v_t, v_s, k_, r_s, r_t, sigma, tau, M, h, Ti, N_obs = 100, x = c(0), y = c(0)) {
  
  #spatial locations are sampled uniformly unless spatial locations are specified
  if(sum(x) == 0) {
    x = runif(N_obs, 0.2, 0.8)
    y = runif(N_obs, 0.2, 0.8)
    altitude = floor(rexp(N_obs, 0.005)) #currently not being used for anything
  }
  
  #reparametrise
  bb = v_s / (v_s + 1)
  ga = v_t * max(1, k_/bb) + 0.5
  al = 0.5 * v_s/v_t * min(1, k_ / bb)
  be = (1 - k_)/bb * v_s
  ka = sqrt(8 * v_s) / r_s
  r = r_t * ka^(2*al) / sqrt(8 * (ga - 0.5))
  
  #compute eigenvalues
  eigen = EIGEN(ga, al, be, ka, r, sigma, M)
  
  #compute rescaling constant for the variance
  c_ = 0
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    c_ = c_ + eigen$NORM[k] * (C * basis(pi * 0.5 * i) * basis(pi * 0.5 * j))^2
  }
  
  #This loop computes the exact covariance matrix for each basis function and then samples
  X_field = matrix(NA, ncol = Ti, nrow = M^2)
  for(k in 1:M^2) {
    r = covariance_function_exact(eigen$MU[k], ga, eigen$LAMBDA[k])
    I = (0:(Ti - 1))*h
    rr = numeric(Ti) 
    for(l in 1:Ti) {
      rr[l] = r(I[l])
    }
    Z = toeplitz(rr) * sigma^2 / c_
    X_field[k,] = mvrnorm(1, rep(0,Ti), Z)
  }
  
  #This loop constructs the observation matrix for the (x,y)-locations
  Hm = matrix(0, ncol = M^2, nrow = N_obs)
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    for(l in 1:N_obs) {
      Hm[l, k] = C * basis(pi * x[l] * i) * basis(pi * y[l] * j)
    }
  }
  
  #simulate observation error
  error = matrix(rnorm(N_obs * Ti, 0, tau), ncol = Ti, nrow = N_obs)
  
  #compute Y from X_field and error
  X_raw = Hm %*% X_field
  X_eff = X_raw 
  Y = X_eff + error
  
  return(list("Y" = Y, "X_eff" = X_eff, "X_raw" = X_raw, "spec_procs" = X_field, "x" = x, "y" = y, "eigen" = eigen, "Hm" = Hm))
}

#This is the implementation of the (sequential) Kalman filter
#Is is called .._clean because it only returns the value of the (log)-likelihood-function, no predictions
KF_clean_seq = function(Y, x, y, v_t, v_s, k_, r_s, r_t, sigma, tau, M, ratdegree, h, Ti) {
  
  #This version never prints any debug information
  silent = T
  
  #reparametrise to spde-parameters
  bb = v_s / (v_s + 1)
  ga = v_t * max(1, k_/bb) + 0.5
  al = 0.5 * v_s/v_t * min(1, k_ / bb)
  be = (1 - k_)/bb * v_s
  ka = sqrt(8 * v_s) / r_s
  r = r_t * ka^(2*al) / sqrt(8 * (ga - 0.5))
  
  n_stations = dim(Y)[1]
  
  #Compute eigenvalues
  M_max = M
  eigen = EIGEN(ga, al, be, ka, r, sigma, max(M,M_max))
  
  #compute rescaling constant for variance (i.e. the variance at the centre of the domain)
  c_ = 0
  for(k in 1:max(M^2,M_max^2)) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    c_ = c_ + eigen$NORM[k] * (C * basis(pi * 0.5 * i) * basis(pi * 0.5 * j))^2
  }
  
  #Computation of rational coefficients
  coef = arma_coef(1, ga, h, ratdegree, r_approx_func, T)
  ar_coef = coef$phi
  ma_coef = coef$theta
  if(silent == F) {
    print(paste0("Rat.approx error: ", coef$epsilon))
    cond = (1 - exp(-min(eigen$MU)*h))^(-(ga - floor(ga))) * coef$epsilon
    print(paste0("Stab. cond. ", cond, " < 1 is ", cond < 1))
  }
  p = length(ar_coef)
  q = length(ma_coef)
  
  #Deprecated variable. Should always be zero.
  param_n = 0
  
  ### innovation matrix  
  Fm = Matrix(0, nrow = (p + q)*M^2 + param_n, ncol = (p + q) * M^2 + param_n, sparse = T)
  #Fm[(p + q)*M^2 + 1:param_n, (p + q)*M^2 + 1:param_n] = diag(1, param_n)
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    Fe = matrix(0, nrow = p + q, ncol = p + q)
    Fe[1, 1:p] = ar_coef * exp(-(eigen$MU[k]-1)*h*(1:p))
    if(q > 0) {Fe[1, (p + 1):(p + q)] = - ma_coef * exp(-(eigen$MU[k]-1)*h*(1:q))}
    if(p > 1) {for(l in 1:(p - 1)) {Fe[l + 1, l] = 1}}
    if(q > 1) {for(l in 1:(q - 1)) {Fe[l + p + 1, l + p] = 1}}
    n_ = (k - 1) * (p + q) +  1:(p + q)
    Fm[n_, n_] = Fe
  }
  
  ### innovation covariance (not scaled properly)
  Qm = Matrix(0, nrow = (p + q) * M^2 + param_n, ncol = (p + q) *  M^2 + param_n, sparse = T)
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    n_ = (k - 1) * (p + q) +  1:(p + q)
    rs = rep(1, p+q)
    
    Q = matrix(0, nrow = p + q, ncol = p + q)
    
    Q[1,1] = 1
    if(q > 0) {
      Q[p + 1, p + 1] = 1
      Q[p + 1, 1] = 1
      Q[1, p + 1] = 1
    }
    
    Qm[n_, n_] = Q
  }
  
  ### A priori mean and variances
  m_tilde = c(rep(0, M^2 * (p + q) + param_n))
  s_tilde = diag(100, M^2 * (p + q) + param_n)
  Ind = (0:(M^2 - 1))*(p + q) + 1
  
  ### Computation of the stationary covariance matrix
  for(i in 1:min(500,ceiling(r_t/h*5))) {
    s_tilde = Fm %*% s_tilde %*% t(Fm) + Qm
  }
  
  ### rescaling of the innovation covariance and the a priori variance
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    ck = eigen$LAMBDA[k] / (2 * eigen$MU[k])^(2 * ga - 1) * gamma(2*ga - 1) / gamma(ga)^2
    ck = ck / s_tilde[(k - 1)*(p + q) + 1, (k - 1)*(p + q) + 1]
    n_ = (k - 1) * (p + q) +  1:(p + q)
    s_tilde[n_, n_] = ck * s_tilde[n_, n_]
    Qm[n_, n_] = ck * Qm[n_, n_]
  }
  Qm = sigma^2 / c_ * Qm
  s_tilde[1:(M^2*(p + q)), 1:(M^2*(p + q))] = sigma^2 / c_ * s_tilde[1:(M^2*(p + q)), 1:(M^2*(p + q))]
  
  #Adjustment of measurement error
  meas_error = tau^2
  
  #A penalty is added for large temporal ranges
  loglik = 0 
  
  ### Observation matrix, defined inside for loop since the temperature data sometimes have missing data
  Hm = matrix(0, ncol = (p + q) * M^2 + param_n, nrow = n_stations)    
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    for(ind in 1:n_stations) {
      n_ = (k - 1) * (p + q) +  1
      Hm[ind, n_] = C * basis(pi * x[ind] * i) * basis(pi * y[ind] * j)
    }
  }
  
  for(iter in 1:Ti) {
  
    ### Compute index set of non-missing data (if any) (there isnt currently any)
    NA_ind = (1:n_stations)*is.na(Y[, iter])
    if(sum(NA_ind) == 0) {
      NA_ind = c(n_stations + 2)
    }
    
    ### measurement error matrix
    big_number = 1e6
    R = diag(meas_error, n_stations) + diag(is.na(Y[,iter])*big_number)
    
    ### Update step
    y_hat = Hm %*% m_tilde
    y = Y[,iter]
    y[is.na(y)] = 0
    
    result <- sequential_kf_fast(
      m_hat = as.vector(m_tilde), 
      s_hat = as.matrix(s_tilde), 
      y = as.vector(y), 
      Hm = as.matrix(Hm), 
      R_diag = as.vector(diag(R)), 
      NA_ind = as.vector(NA_ind)
    )
    m_hat = result$m_hat
    s_hat = result$s_hat
    
    ### Compute likelihood for step
    loglik = loglik + result$log_lik
    
    #state forecast
    m_tilde = Fm %*% m_hat
    s_tilde = Fm %*% s_hat %*% t(Fm) + Qm
    
    s_tilde = (s_tilde + t(s_tilde)) / 2 #symmetrisation is for stability, runs fine without as well for below example
    
  }
  
  return(loglik)
}

#This is another implementation of the (sequential) Kalman filter
#It returns filtered and one-step predicted means and covariance matrices as well.
#backpass = T also runs a Rauch–Tung–Striebel smoother, but this is not used in the paper
KF_seq = function(Y, x, y, v_t, v_s, k_, r_s, r_t, sigma, tau, M, ratdegree, h, Ti, backpass = F, verbose = T) {
  
  silent = T
  
  #reparametrise to spde-parameters
  bb = v_s / (v_s + 1)
  ga = v_t * max(1, k_/bb) + 0.5
  al = 0.5 * v_s/v_t * min(1, k_ / bb)
  be = (1 - k_)/bb * v_s
  ka = sqrt(8 * v_s) / r_s
  r = r_t * ka^(2*al) / sqrt(8 * (ga - 0.5))
  
  #number of obs/timestep
  n_stations = dim(Y)[1]
  
  #compute eigenvalues
  M_max = M #eigenvalues are scaled so that the norm-variance is 1 across 32^2 basis functions
  eigen = EIGEN(ga, al, be, ka, r, sigma, max(M,M_max))
  
  #compute rescaling constant for variance (i.e. the variance at the centre of the domain)
  c_ = 0
  for(k in 1:max(M^2,M_max^2)) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    c_ = c_ + eigen$NORM[k] * (C * basis(pi * 0.5 * i) * basis(pi * 0.5 * j))^2
  }
  
  #compute percentage of variance explained by the M^2 basis functions compared to the 32^2 basis functions
  p_acc_mid = 0
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    p_acc_mid = p_acc_mid + eigen$NORM[k] * (C * basis(pi * 0.5 * i) * basis(pi * 0.5 * j))^2
  }
  p_acc_mid = (c_ - p_acc_mid) / c_ * sigma^2
  if(silent == F) {
    print(paste0("Explained error: ", sigma^2 - p_acc_mid))
    print(paste0("Unexplained error: ", p_acc_mid))
    print(paste0("Spc.approx error in norm: ", 1 - sum(eigen$NORM[1:M^2]))) 
  }
  
  #Computation of rational coefficients
  coef = arma_coef(1, ga, h, ratdegree, r_approx_func, T)
  ar_coef = coef$phi
  ma_coef = coef$theta
  if(silent == F) {
    print(paste0("Rat.approx error: ", coef$epsilon))
    cond = (1 - exp(-min(eigen$MU)*h))^(-(ga - floor(ga))) * coef$epsilon
    print(paste0("Stab. cond. ", cond, " < 1 is ", cond < 1))
  }
  p = length(ar_coef)
  q = length(ma_coef)
  
  #currently there are no covariates
  param_n = 0
  
  ### innovation matrix  
  Fm = Matrix(0, nrow = (p + q)*M^2 + param_n, ncol = (p + q) * M^2 + param_n, sparse = T)
  #Fm[(p + q)*M^2 + 1:param_n, (p + q)*M^2 + 1:param_n] = diag(1, param_n)
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    Fe = matrix(0, nrow = p + q, ncol = p + q)
    Fe[1, 1:p] = ar_coef * exp(-(eigen$MU[k]-1)*h*(1:p))
    if(q > 0) {Fe[1, (p + 1):(p + q)] = - ma_coef * exp(-(eigen$MU[k]-1)*h*(1:q))}
    if(p > 1) {for(l in 1:(p - 1)) {Fe[l + 1, l] = 1}}
    if(q > 1) {for(l in 1:(q - 1)) {Fe[l + p + 1, l + p] = 1}}
    n_ = (k - 1) * (p + q) +  1:(p + q)
    Fm[n_, n_] = Fe
  }
  
  ### innovation covariance (not scaled properly)
  Qm = Matrix(0, nrow = (p + q) * M^2 + param_n, ncol = (p + q) *  M^2 + param_n, sparse = T)
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    n_ = (k - 1) * (p + q) +  1:(p + q)
    rs = rep(1, p+q)
    
    Q = matrix(0, nrow = p + q, ncol = p + q)
    
    Q[1,1] = 1
    if(q > 0) {
      Q[p + 1, p + 1] = 1
      Q[p + 1, 1] = 1
      Q[1, p + 1] = 1
    }
    
    Qm[n_, n_] = Q
  }
  
  ### A priori mean and variances
  m_tilde = c(rep(0, M^2 * (p + q) + param_n))
  s_tilde = diag(100, M^2 * (p + q) + param_n)
  Ind = (0:(M^2 - 1))*(p + q) + 1
  
  for(i in 1:min(500,ceiling(r_t/h*5))) {
    s_tilde = Fm %*% s_tilde %*% t(Fm) + Qm
  }
  
  ### rescaling of the innovation covariance and the a priori variance
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    ck = eigen$LAMBDA[k] / (2 * eigen$MU[k])^(2 * ga - 1) * gamma(2*ga - 1) / gamma(ga)^2
    #r = covariance_function_exact(eigen$MU[k], ga, eigen$LAMBDA[k])
    ck = ck / s_tilde[(k - 1)*(p + q) + 1, (k - 1)*(p + q) + 1]
    n_ = (k - 1) * (p + q) +  1:(p + q)
    s_tilde[n_, n_] = ck * s_tilde[n_, n_]
    Qm[n_, n_] = ck * Qm[n_, n_]
  }
  Qm = sigma^2 / c_ * Qm
  s_tilde[1:(M^2*(p + q)), 1:(M^2*(p + q))] = sigma^2 / c_ * s_tilde[1:(M^2*(p + q)), 1:(M^2*(p + q))]
  
  ### Observation matrix, defined inside for loop since the temperature data sometimes have missing data
  Hm = matrix(0, ncol = (p + q) * M^2 + param_n, nrow = n_stations)    
  for(k in 1:M^2) {
    i = eigen$keys[1,k]
    j = eigen$keys[2,k]
    C = sqrt(2)^((i != 0) + (j != 0))
    for(ind in 1:n_stations) {
      n_ = (k - 1) * (p + q) +  1
      #ind = (1:N)[-NA_ind][l]
      Hm[ind, n_] = C * basis(pi * x[ind] * i) * basis(pi * y[ind] * j)
    }
  }
  
  #set up
  #Ind = (0:(M^2 - 1))*(p + q) + 1
  mm_hat = list()
  pm_hat = numeric(Ti)
  ss_hat = list()
  sd_hat = numeric(Ti)
  NA_list = list()
  mm_tilde = list(m_tilde)
  pm_tilde = numeric(Ti + 1)
  pm_tilde[1] = m_tilde[1]
  ss_tilde = list(s_tilde)
  sd_tilde = numeric(Ti + 1)
  sd_tilde[1] = sqrt(s_tilde[1, 1])
  logl_list = numeric(Ti)
  
  #A penalty is added for large temporal ranges
  loglik = 0
  
  #Adjustment of measurement error
  meas_error = tau^2 #+ p_acc_mid 
  
  for(iter in 1:Ti) {
    
    if(verbose) {
      message('\r', "Forward pass: ", round(iter/Ti*100, 2), "%", appendLF = FALSE)
    }
    
    ### Compute index set of non-missing data (if any) (there isnt currently any)
    #N = length(Y[,iter])
    NA_ind = (1:n_stations)*is.na(Y[, iter])
    if(sum(NA_ind) == 0) {
      NA_ind = c(n_stations + 2)
    }
    
    NA_list = c(NA_list, list(NA_ind))
    
    ### measurement error matrix
    big_number = 1e6
    R = diag(meas_error, n_stations) + diag(is.na(Y[,iter])*big_number)
    
    ### Update step
    #prediction mean
    y_hat = Hm %*% m_tilde
    
    
    y = Y[,iter]
    y[is.na(y)] = 0
    
    result <- sequential_kf_fast(
      m_hat = as.vector(m_tilde), 
      s_hat = as.matrix(s_tilde), 
      y = as.vector(y), 
      Hm = as.matrix(Hm), 
      R_diag = as.vector(diag(R)), 
      NA_ind = as.vector(NA_ind)
    )
    m_hat = result$m_hat
    s_hat = result$s_hat
    
    #update likelihood
    if(iter > 0) {
      loglik = loglik + result$log_lik
      #print(dl)
    }
    
    #store updates
    mm_hat = c(mm_hat, list(result$m_hat))
    ss_hat = c(ss_hat, list(result$s_hat))
    
    #forecast
    m_tilde = Fm %*% result$m_hat
    s_tilde = Fm %*% result$s_hat %*% t(Fm) + Qm
    s_tilde = (s_tilde + t(s_tilde)) / 2
    
    mm_tilde = c(mm_tilde, list(Fm %*% result$m_hat))
    ss_tilde = c(ss_tilde, list(Fm %*% result$s_hat %*% t(Fm) + Qm))
    
  }
  
  if(backpass) {
    
    # backwards pass
    m_filt = m_tilde
    s_filt = s_tilde
    mm_filt = list()
    ss_filt = list()
    
    for(iter in Ti:1) {
      
      if(verbose) {
        message("\r", "Backward pass: ", round((Ti - iter + 1)/Ti*100, 2), "%", appendLF = FALSE)
      }
      
      n = dim(s_tilde)[1]
      s_tilde = ss_tilde[[iter + 1]] + diag(rep(1e-8, n))
      s_hat = ss_hat[[iter]]
      m_tilde = mm_tilde[[iter + 1]]
      m_hat = mm_hat[[iter]]
      
      #print(dim(s_tilde))
      
      #print(n)
      #print(dim(s_tilde))
      #print(s_tilde[(277 - 3):(277 + 3), (277 - 3):(277 + 3)])
      
      C = s_hat %*% t(Fm) %*% solve(s_tilde)
      
      m_filt = m_hat + C %*% (m_filt - m_tilde)
      s_filt = s_hat + C %*% (s_filt - s_tilde) %*% t(C)
      
      mm_filt = c(mm_filt, list(m_filt))
      ss_filt = c(ss_filt, list(s_filt))
      
    }
    
    a_hat = matrix(data = 0, nrow = M^2, ncol = Ti)
    param_hat = matrix(data = 0, nrow = param_n, ncol = Ti)
    I = (0:(M^2 - 1)) * (p + q) + 1
    for(i in 1:Ti) {
      a_hat[, Ti - i + 1] = mm_filt[[i]][I]
      param_hat[, Ti - i + 1] = mm_filt[[i]][M^2*(p + q) + 1:param_n]
    }
    
    if(verbose) {
      message("\r", appendLF = FALSE)
      print("-----------------------------------------------")
    }
    
    return(list("logl_list"=logl_list, "loglik" = loglik, "Hm" = Hm_list, "Fm" = Fm, "Qm" = Qm, "m_hat" = mm_hat, "a_hat" = a_hat, "param_hat" = param_hat, "s_hat" = ss_hat, "m_tilde" = mm_tilde, "s_tilde" = ss_tilde, "m_filt" = mm_filt, "s_filt" = ss_filt, "p" = p, "q" = q, "eigen" = eigen))
  }
  
  a_hat = matrix(data = 0, nrow = M^2, ncol = Ti)
  param_hat = matrix(data = 0, nrow = param_n, ncol = Ti)
  I = (0:(M^2 - 1)) * (p + q) + 1
  for(i in 1:Ti) {
    a_hat[, i] = mm_hat[[i]][I]
    param_hat[, Ti - i + 1] = mm_hat[[i]][M^2*(p + q) + 1:param_n]
  }
  
  if(verbose) {
    message("\r", appendLF = FALSE)
    print("-----------------------------------------------")
  }
  
  return(list("logl_list"=logl_list, "loglik" = loglik, "Hm" = Hm, "NA_list" = NA_list, "Fm" = Fm, "Qm" = Qm, "m_hat" = mm_hat, "a_hat" = a_hat, "param_hat" = param_hat, "s_hat" = ss_hat, "m_tilde" = mm_tilde, "s_tilde" = ss_tilde, "p" = p, "q" = q, "eigen" = eigen))
  
}

#These transform between the interpretable parameters (interp-scale) and the ones I use in optim (optim-scale)
eps = 0.001
param_trans = function(param) {
  
  max_v_t = 5 / 2
  v_t = max_v_t * exp(param[1]) / (exp(param[1]) + 1) + 0.25
  v_s = exp(param[2]) + 0.25
  
  #a = v_s / (v_s + 1) #must be <= v_s / (v_s + 1)
  a = 0.5
  k_ = exp(3*param[3]) / (exp(3*param[3]) + 1/a - 1)
  #k_ = g_trans(param[3])
  
  r_t = exp(param[4]) + 5 * eps
  
  a = 0.5
  #r_s = exp(param[5]) / (exp(param[5]) + 1/a - 1)
  r_s = exp(param[5]) + 5 * eps
  
  sigma = exp(param[6]) + 5 * eps
  tau = exp(param[7])
  
  return(c(v_t, v_s, k_, r_t, r_s, sigma, tau))
  
}
param_inv_trans = function(param) {
  max_v_t = 5/2
  #p1 = log((param[1] - 0.5 - 2 * (param[1] - 0.5) * max_v_t)/(param[1] - 0.5  - max_v_t))
  p1 = log((param[1] - 0.25) / max_v_t) - log(1 - (param[1] - 0.25) / max_v_t)
  p2 = log(param[2] - 0.25)
  a = 0.5
  p3 = log(param[3] * (1 - 1/a) / (param[3] - 1)) / 3
  #p3 = g_trans_inv(param[3])
  p4 = log(param[4] - 5 * eps)
  a = 0.5
  #p5 = log(-param[5]/5 * (1/a - 1) / (param[5]/5 - 1))
  p5 = log(param[5] - 5 * eps)
  p6 = log(param[6] - 5 * eps)
  p7 = log(param[7])
  return(c(p1, p2, p3, p4, p5, p6, p7))
}

#This function generates spatial locations for sampling
#Only type = "Uniform" is used in the paper.
generate_station_locations = function(n,type,plotB = F) {
  if(type == "Uniform") {
    Ux = runif(n, 0, 1)
    Uy = runif(n, 0, 1)
  } else if (type == "Neumann-Scott") {
    nclust <- function(x0, y0, radius, n) {
      return(runifdisc(n, radius, centre=c(x0, y0)))
    }
    tst = 0
    while(tst!= n) {
      obj = rNeymanScott(10, 0.2, nclust, radius=0.1, n=5)
      tst = obj$n 
    }
    Ux = obj$x
    Uy = obj$y
  } else if (type == "Diggle-Gratton") {
    tst = 0
    while(tst!= n) {
      obj = rDiggleGratton(75.0, 0.10, 0.12, kappa=0.5, W = owin(), expand=TRUE, nsim=1, drop=TRUE)
      tst = obj$n
    }
    Ux = obj$x
    Uy = obj$y
  }
  Ux = 0.6*Ux + 0.4/2
  Uy = 0.6*Uy + 0.4/2
  if(plotB) {plot(Ux, Uy, xlim = c(0,1), ylim = c(0,1), main = type)}
  return(list("x" = Ux, "y" = Uy))
}


#parameters are in order: 
#1: temporal smoothness
#2: spatial smoothness
#3: non-seperability
#4: temporal range
#5: spatial range
#6: stationary standard deviation of X(0.5, 0.5)
#7: standard deviation of measurement error

sim_n = 250 #number of spatial measurement drawn
M_sim = 32 #(root) number of spatial basis functions used for simulation
Mm = 8 #(root) number of spatial basis functions used for inference
ratio = 1 #order of the rational approximation
h = 1 #temporal step size
Ti = 45 #number of temporal steps



### Example 1

field_param = c(1.0, 1.0, 0.25, 10.0, 1.0, 3.5, 0.75)

#Step 1: Simulate sampling locations
stat_loc = generate_station_locations(sim_n, "Uniform", plotB = F)

#Step 2: Simulate data
simulation_obj = exact_simulate_data(field_param[1], field_param[2], field_param[3], field_param[5], field_param[4], field_param[6], field_param[7], M_sim, h, Ti, N_obs = sim_n, x = stat_loc$x, y = stat_loc$y)

#Step 3: Set-up technical likelihood function for use in optim()
log_lik_sim = function(param) {
  
  #transform parameters into interp-scale 
  param_ = param_trans(param)
  
  #extract parameters
  v_t = param_[1]
  v_s = param_[2]
  k_ = param_[3]
  r_t = param_[4]
  r_s = param_[5]
  sigma = param_[6]
  tau = param_[7]
  
  #run kalman filter
  #model_l = try(KF_clean_seq(simulation_obj$Y[,1:Ti], simulation_obj$x, simulation_obj$y, v_t, v_s, k_, r_s, r_t, sigma, tau, Mm, ratio, h, Ti), silent = F)
  model_l = KF_clean_seq(simulation_obj$Y[,1:Ti], simulation_obj$x, simulation_obj$y, v_t, v_s, k_, r_s, r_t, sigma, tau, Mm, ratio, h, Ti)
  
  #test if an error has occured or +-Inf returned, if so return 1e6, if not return the negative log likelihood
  if(class(model_l) == "try-error") {
    message("\r", "Error in Kalman filter. Returning Inf.", "\n", appendLF = FALSE)
    print("===============================================")
    return(1e6)
  } else {
    l = model_l
    if(abs(l) == Inf) {
      return(1e6)
    }
    else {
      print(paste0("loglik: ", model_l))
      return(-model_l)
    }
  }
}

#Step 4: Compute likelihood of the true parameters
ll_true = log_lik_sim(param_inv_trans(field_param))
print(paste0("Log-likelihood of the truth: ", ll_true))

#Step 5: Run optimParallel() for inference

#initial parameters
init_par = c(0.5, 0.5, 0.5, 5.0, 0.5, 5.0, 0.5)

#cluster set-up
nCores <- detectCores() - 1
cluster <- makeCluster(nCores)

setDefaultCluster(cl=cluster)

clusterEvalQ(cluster, library(pracma))
clusterEvalQ(cluster, library(Matrix))
clusterEvalQ(cluster, library(KFParallel))

clusterExport(cluster, list("init_par", "log_lik_sim", "Mm", "param_trans", "param_inv_trans", "eps", "KF_clean_seq", "simulation_obj", "Ti", "EIGEN", "basis_is_cos", "basis", "arma_coef", "r_approx_func", "ratio", "sup_norm_error", "h"))

#run optimParallel()
t_start = Sys.time()
print(paste0("Starting full inference at ", t_start))
opt_param = optimParallel(init_par, log_lik_sim, parallel=list(loginfo=TRUE) , control = list(fnscale = 1000), hessian = F) #, method = "Nelder-Mead") #, lower = c(0.25, 0.25, 0.1, 0.1, 0.001, 0.001), upper = c(2.5, 2.5, 2.5, 1.0, 5.0, 5.0))
print(param_trans(opt_param$par))
print(paste0("Log-likelihood of est. parameters in full inference: ", opt_param$value))
t_end = Sys.time()
print(paste0("Full inference for sqrt(M) = ", Mm, " completed at ", t_end))

#stop cluster
stopCluster(cluster)

print(param_trans(opt_param$par))
print(field_param)



### The code for the full simulation study is below:

repl = 3 #number of replicates of simulation+inference. 30 is used in the paper

# Matrix of the four sets of parameters considered in the paper:
field_param_matr = matrix(NA, nrow = 4, ncol = 7)
field_param_matr[1,] = c(1.0, 1.0, 0.25, 10.0, 1.0, 3.5, 0.75) #LH
field_param_matr[2,] = c(1.0, 1.0, 0.75, 10.0, 1.0, 3.5, 0.75) #HH
field_param_matr[3,] = c(1.0, 1.0, 0.25, 10.0, 1.0, 3.5, 0.35) #LL
field_param_matr[4,] = c(1.0, 1.0, 0.75, 10.0, 1.0, 3.5, 0.35) #HL

#Set target directory
dir = paste0(getwd(), "/")

### This following code generates empty data-files
### It will delete and overwrite data-files if they exist
### Uncomment and run once, then comment again

for(param_case_iter in c(1,2,3,4))  {
  filepath = paste0(dir, "parameter-inference-full-simulated-param-",param_case_iter,"-250stat.RData")
  saveRDS(list(), file = filepath)
  filepath = paste0(dir, "parameter-inference-simple-simulated-param-",param_case_iter,"-250stat.RData")
  saveRDS(list(), file = filepath)
}

### Inference over the replicates
for(iter in 1:repl) {
  
  for(param_case_iter in c(1,2,3,4)) {
    
    field_param = field_param_matr[param_case_iter,]
    
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    t = Sys.time()
    print(paste0("Iteration ", iter, " out of ", repl))
    print(paste0("True field parameters: ", paste(field_param, collapse = ", ")))
    print(paste0("Starting simulation at ", t))
    stat_loc = generate_station_locations(sim_n, "Uniform", plotB = F)
    #stat_loc = list("x" = x[I], "y" = y[I])
    simulation_obj = exact_simulate_data(field_param[1], field_param[2], field_param[3], field_param[5], field_param[4], field_param[6], field_param[7], M_sim, h, Ti, N_obs = sim_n, x = stat_loc$x, y = stat_loc$y)
    print(paste0("Simulation for iter ", iter, " took ", Sys.time() - t))
    
    filepath1 = paste0(dir, "parameter-inference-full-simulated-param-",param_case_iter,"-250stat.RData")
    ret_list1 = readRDS(filepath1)
    print(paste0("Filepath #1 is: ", filepath1))
    
    filepath2 = paste0(dir, "parameter-inference-simple-simulated-param-",param_case_iter,"-250stat.RData")
    ret_list2 = readRDS(filepath2)
    print(paste0("Filepath #2 is: ", filepath2))
    
    print(paste0("Replicate #", iter))
      
    log_lik_sim = function(param) {
      
      #transform parameters into interp-scale 
      param_ = param_trans(param)
      
      #extract parameters
      v_t = param_[1]
      v_s = param_[2]
      k_ = param_[3]
      r_t = param_[4]
      r_s = param_[5]
      sigma = param_[6]
      tau = param_[7]
      
      #run kalman filter
      model_l = try(KF_clean_seq(simulation_obj$Y[,1:Ti], simulation_obj$x, simulation_obj$y, v_t, v_s, k_, r_s, r_t, sigma, tau, Mm, ratio, h, Ti), silent = F)
      
      #test if an error has occured or +-Inf returned, if so return 1e6, if not return the negative log likelihood
      if(class(model_l) == "try-error") {
        message("\r", "Error in Kalman filter. Returning Inf.", "\n", appendLF = FALSE)
        print("===============================================")
        return(1e6)
      } else {
        l = model_l
        if(abs(l) == Inf) {
          return(1e6)
        }
        else {
          print(paste0("loglik: ", model_l))
          return(-model_l)
        }
      }
    }
      
    restr = c(4,5,6,7)
    ll_restr = function(param) {
      print("----------------------------------------------")
      param_ = c(0.5, 1.0, 0.0, field_param[restr])
      param_ = param_inv_trans(param_)
      param_[restr] = param
      print(param_trans(param_))
      ll = log_lik_sim(param_)
      return(ll)
    }
    
    ll_true = log_lik_sim(param_inv_trans(field_param))
    print(paste0("Log-likelihood of the truth: ", ll_true))
    
    init_par = c(0.5, 0.5, 0.5, 5.0, 0.5, 5.0, 0.5)
    nCores <- detectCores() - 1
    cluster <- makeCluster(nCores)
    
    setDefaultCluster(cl=cluster)
    
    clusterEvalQ(cluster, library(pracma))
    clusterEvalQ(cluster, library(Matrix))
    clusterEvalQ(cluster, library(KFParallel))
    
    clusterExport(cluster, list("iter","init_par", "log_lik_sim", "Mm", "param_trans", "param_inv_trans", "eps", "KF_clean_seq", "simulation_obj", "Ti", "EIGEN", "basis_is_cos", "basis", "arma_coef", "r_approx_func", "ratio", "sup_norm_error", "h"))
    
    
    t_start = Sys.time()
    print(paste0("Starting full inference for iter ", iter, " at ", t_start))
    opt_param = optimParallel(init_par, log_lik_sim, parallel=list(loginfo=TRUE) , control = list(fnscale = 1000), hessian = F) #, method = "Nelder-Mead") #, lower = c(0.25, 0.25, 0.1, 0.1, 0.001, 0.001), upper = c(2.5, 2.5, 2.5, 1.0, 5.0, 5.0))
    print(param_trans(opt_param$par))
    print(paste0("Log-likelihood of est. parameters in full inference: ", opt_param$value))
    t_end = Sys.time()
    print(paste0("Full inference for iter ", iter, " and sqrt(M) = ", Mm, " completed at ", t_end))
    print(paste0("Convergence code: ", opt_param$convergence))
    print("xxxxxxxxxxxxxxxxxxxxxxxxx")
    
    t_start2 = Sys.time()
    print(paste0("Starting inference for iter ", iter, " at ", t_start2))
    opt_param2 = optimParallel(init_par[restr], ll_restr, parallel=list(loginfo=TRUE) , control = list(fnscale = 1000), hessian = F) #, method = "Nelder-Mead") #, lower = c(0.25, 0.25, 0.1, 0.1, 0.001, 0.001), upper = c(2.5, 2.5, 2.5, 1.0, 5.0, 5.0))
    print(param_trans(c(0,0,0,opt_param2$par))[restr])
    print(paste0("Log-likelihood of est. parameters in simple inference: ", opt_param2$value))
    t_end2 = Sys.time()
    print(paste0("Simple inference for iter ", iter, " and sqrt(M) = ", Mm, " completed at ", t_end2))
    print(paste0("Convergence code: ", opt_param2$convergence))
    print("xxxxxxxxxxxxxxxxxxxxxxxxx")

    stopCluster(cluster)
    
    print("-------------------------------------------------------------")
    
    ret_list1[[iter]] = list("field_param" = field_param, "M" = M, "Ti" = Ti, "sim_n" = sim_n, "simulation_obj" = simulation_obj ,"init_par" = init_par, "log_lik_true" = ll_true, "t_start" = t_start, "t_end" = t_end, "opt_param" = opt_param)
    ret_list2[[iter]] = list("field_param" = field_param, "M" = M, "Ti" = Ti, "sim_n" = sim_n, "simulation_obj" = simulation_obj ,"init_par" = init_par[restr], "log_lik_true" = ll_true, "t_start" = t_start2, "t_end" = t_end2, "opt_param" = opt_param2)

    saveRDS(ret_list1, file = filepath1)
    saveRDS(ret_list2, file = filepath2)
  }
  
}

### Predictions (RMSE and CRPS for both filtering and forecasting) over the replicates
for(ind in c(1,2,3,4)) { 
  
  param_case_iter = ind
  filepath1 = paste0(dir, "parameter-inference-full-simulated-param-",param_case_iter,"-250stat.RData")
  ret_list1 = readRDS(filepath1)
  print(paste0("Filepath #1 is: ", filepath1))
  
  filepath2 = paste0(dir, "parameter-inference-simple-simulated-param-",param_case_iter,"-250stat.RData")
  ret_list2 = readRDS(filepath2)
  print(paste0("Filepath #2 is: ", filepath2))
  
  RMSE_filtered_full = matrix(NA, nrow = repl, ncol = Ti)
  RMSE_1st_pred_full = matrix(NA, nrow = repl, ncol = Ti)
  RMSE_filtered_simp = matrix(NA, nrow = repl, ncol = Ti)
  RMSE_1st_pred_simp = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_filtered_full = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_1st_pred_full = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_filtered_simp = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_1st_pred_simp = matrix(NA, nrow = repl, ncol = Ti)
  
  for(iter in 1:repl) {
    
    ### Simulate a new data set
    print(paste0("Iteration #",iter))
    field_param = field_param_matr[ind,]    
    stat_loc = generate_station_locations(sim_n, "Uniform", plotB = F)
    simulation_obj = exact_simulate_data(field_param[1], field_param[2], field_param[3], field_param[5], field_param[4], field_param[6], field_param[7], M_sim, h, Ti, N_obs = sim_n, x = stat_loc$x, y = stat_loc$y)
    
    
    ### Basis functions for prediction
    M_pred = 8
    
    ### Extract estimated parameters
    est_field_param_full = param_trans(ret_list1[[iter]]$opt_param$par)
    est_field_param_simp =  param_trans(c(0,0,0,ret_list2[[iter]]$opt_param$par))
    est_field_param_simp[c(1,2,3)] = c(0.5, 1.0, 0.0)
    
    print(est_field_param_full)
    print(est_field_param_simp)
    
    ### Run Kalman filter
    model_full = KF_seq(simulation_obj$Y, simulation_obj$x, simulation_obj$y, est_field_param_full[1], est_field_param_full[2], est_field_param_full[3], est_field_param_full[5], est_field_param_full[4], est_field_param_full[6], est_field_param_full[7], M_pred, 1, h, Ti, backpass = F, verbose = T)
    model_simp = KF_seq(simulation_obj$Y, simulation_obj$x, simulation_obj$y, est_field_param_simp[1], est_field_param_simp[2], est_field_param_simp[3], est_field_param_simp[5], est_field_param_simp[4], est_field_param_simp[6], est_field_param_simp[7], M_pred, 1, h, Ti, backpass = F, verbose = T)
    
    ### Extract one-step predictions
    a_tilde_full = matrix(data = NA, nrow = M_pred^2, ncol = Ti)
    I = (0:(M_pred^2 - 1)) * (model_full$p + model_full$q) + 1
    for(j in 1:(Ti)) {
      a_tilde_full[,j] = model_full$m_tilde[[j]][I]
    }
    a_tilde_simp = matrix(data = NA, nrow = M_pred^2, ncol = Ti)
    I = (0:(M_pred^2 - 1)) * (model_simp$p + model_simp$q) + 1
    for(j in 1:(Ti)) {
      a_tilde_simp[,j] = model_simp$m_tilde[[j]][I]
    }
    
    ### Compute RMSE
    print("Computing RMSE")
    RMSE_filtered_full[iter,] = colSums((simulation_obj$spec_procs - rbind(model_full$a_hat, matrix(0, nrow = M_sim^2 - M_pred^2, ncol = Ti)))^2)
    RMSE_1st_pred_full[iter,] = colSums((simulation_obj$spec_procs - rbind(a_tilde_full, matrix(0, nrow = M_sim^2 - M_pred^2, ncol = Ti)))^2)
    RMSE_filtered_simp[iter,] = colSums((simulation_obj$spec_procs - rbind(model_simp$a_hat, matrix(0, nrow = M_sim^2 - M_pred^2, ncol = Ti)))^2)
    RMSE_1st_pred_simp[iter,] = colSums((simulation_obj$spec_procs - rbind(a_tilde_simp, matrix(0, nrow = M_sim^2 - M_pred^2, ncol = Ti)))^2)
    
    print(RMSE_filtered_full[iter,1:10])
    print(RMSE_1st_pred_full[iter,1:10])
    
    ### Compute CRPS
    print("Computing CRPS")
    O = full_structure_matrix(simulation_obj$eigen, Nf = 20)
    O_red = O[,1:M_pred^2]
    I_full = (0:(M_pred^2 - 1)) * (model_full$p + model_full$q) + 1
    I_simp = (0:(M_pred^2 - 1)) * (model_simp$p + model_simp$q) + 1
    obs_matr = O %*% simulation_obj$spec_procs
    for(ti in 1:(Ti)) {
      
      m_vec_hat_full = O_red %*% model_full$a_hat[,ti]
      m_vec_tilde_full = O_red %*% a_tilde_full[,ti]
      m_vec_hat_simp = O_red %*% model_simp$a_hat[,ti]
      m_vec_tilde_simp = O_red %*% a_tilde_simp[,ti]
      
      s_vec_hat_full = sqrt(diag(O_red %*% model_full$s_hat[[ti]][I_full,I_full] %*% t(O_red)))
      s_vec_tilde_full = sqrt(diag(O_red %*% model_full$s_tilde[[ti]][I_full,I_full] %*% t(O_red)))
      s_vec_hat_simp = sqrt(diag(O_red %*% model_simp$s_hat[[ti]][I_simp,I_simp] %*% t(O_red)))
      s_vec_tilde_simp = sqrt(diag(O_red %*% model_simp$s_tilde[[ti]][I_simp,I_simp] %*% t(O_red)))
      
      CRPS_filtered_full[iter,ti] = crps(obs_matr[,ti], cbind(m_vec_hat_full, s_vec_hat_full))$CRPS
      CRPS_1st_pred_full[iter,ti] = crps(obs_matr[,ti], cbind(m_vec_tilde_full, s_vec_tilde_full))$CRPS
      CRPS_filtered_simp[iter,ti] = crps(obs_matr[,ti], cbind(m_vec_hat_simp, s_vec_hat_simp))$CRPS
      CRPS_1st_pred_simp[iter,ti] = crps(obs_matr[,ti], cbind(m_vec_tilde_simp, s_vec_tilde_simp))$CRPS
      
    }
    
    ### Store estimates in lists
    ret_list1[[iter]]$RMSE_filtered = RMSE_filtered_full[iter,]
    ret_list1[[iter]]$RMSE_1st_pred = RMSE_1st_pred_full[iter,]
    ret_list1[[iter]]$CRPS_filtered = CRPS_filtered_full[iter,]
    ret_list1[[iter]]$CRPS_1st_pred = CRPS_1st_pred_full[iter,]
    
    ret_list2[[iter]]$RMSE_filtered = RMSE_filtered_simp[iter,]
    ret_list2[[iter]]$RMSE_1st_pred = RMSE_1st_pred_simp[iter,]
    ret_list2[[iter]]$CRPS_filtered = CRPS_filtered_simp[iter,]
    ret_list2[[iter]]$CRPS_1st_pred = CRPS_1st_pred_simp[iter,]
    
    saveRDS(ret_list1, filepath1)
    saveRDS(ret_list2, filepath2)
    
  }
    
}


### Below is the code to generate the plots for the simulation study.

### This function loads all of the data from the simulation study from the data files
### param_case_iter refer to which row in the field_param_matr matrix.
aggr_func = function(param_case_iter) {
  
  filepath = paste0(dir, "parameter-inference-simple-simulated-param-",param_case_iter,"-250stat.RData")
  ret_lst = readRDS(filepath)
  
  print(ret_lst[[1]]$RMSE_filtered)
  repl = 30
  RMSE_filtered_simple = matrix(NA, nrow = repl, ncol = Ti)
  RMSE_1st_pred_simple = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_filtered_simple = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_1st_pred_simple = matrix(NA, nrow = repl, ncol = Ti)
  param_matr_simple = matrix(NA, nrow = repl, ncol = length(ret_lst[[1]]$opt_param$par))
  logl_simple = numeric(repl)
  for(i in 1:repl) {
    #print(paste0(i, " - ", length(ret_lst[[i]]$RMSE_filtered), " - ", length(RMSE_filtered_simple[i,])))
    RMSE_filtered_simple[i,] = ret_lst[[i]]$RMSE_filtered
    RMSE_1st_pred_simple[i,] = ret_lst[[i]]$RMSE_1st_pred
    CRPS_filtered_simple[i,] = ret_lst[[i]]$CRPS_filtered
    CRPS_1st_pred_simple[i,] = ret_lst[[i]]$CRPS_1st_pred
    texh = ret_lst[[i]]$opt_param$par
    param_matr_simple[i,] = param_trans(c(0,0,0,texh))[4:7]
    logl_simple[i] = ret_lst[[i]]$opt_param$value
  }
  
  filepath = paste0(dir, "parameter-inference-full-simulated-param-",param_case_iter,"-250stat.RData")
  ret_lst = readRDS(filepath)
  repl = 30
  RMSE_filtered = matrix(NA, nrow = repl, ncol = Ti)
  RMSE_1st_pred = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_filtered = matrix(NA, nrow = repl, ncol = Ti)
  CRPS_1st_pred = matrix(NA, nrow = repl, ncol = Ti)
  param_matr = matrix(NA, nrow = repl, ncol = length(ret_lst[[1]]$opt_param$par))
  logl = numeric(repl)
  for(i in 1:repl) {
    RMSE_filtered[i,] = ret_lst[[i]]$RMSE_filtered
    RMSE_1st_pred[i,] = ret_lst[[i]]$RMSE_1st_pred
    CRPS_filtered[i,] = ret_lst[[i]]$CRPS_filtered
    CRPS_1st_pred[i,] = ret_lst[[i]]$CRPS_1st_pred
    param_matr[i,] = param_trans(ret_lst[[i]]$opt_param$par)
    logl[i] = ret_lst[[i]]$opt_param$value
  }
  
  return(list(
    "RMSE_filtered_simple" = RMSE_filtered_simple,
    "RMSE_1st_pred_simple" = RMSE_1st_pred_simple,
    "CRPS_filtered_simple" = CRPS_filtered_simple,
    "CRPS_1st_pred_simple" = CRPS_1st_pred_simple,
    "RMSE_filtered" = RMSE_filtered,
    "RMSE_1st_pred" = RMSE_1st_pred,
    "CRPS_filtered" = CRPS_filtered,
    "CRPS_1st_pred" = CRPS_1st_pred,
    "param_matr_simple" = param_matr_simple, 
    "param_matr" = param_matr, 
    "logl_simple" = logl_simple, 
    "logl" = logl))
}

### All data is loaded:
obj1 = aggr_func(1)
obj2 = aggr_func(2)
obj3 = aggr_func(3)
obj4 = aggr_func(4)

################# PLOT OF PREDICTIONS

filepath = paste0(dir,"RMSEfiltering-2.png")
png(filename=filepath, pointsize=10, width=6000, height=3000, res=600)
par(mfrow = c(1,1))
df1 = data.frame("Full" = rowMeans(obj1$RMSE_filtered),"Reduced" = rowMeans(obj1$RMSE_filtered_simple))
df2 = data.frame("Full" = rowMeans(obj2$RMSE_filtered),"Reduced" = rowMeans(obj2$RMSE_filtered_simple))
df3 = data.frame("Full" = rowMeans(obj3$RMSE_filtered),"Reduced" = rowMeans(obj3$RMSE_filtered_simple))
df4 = data.frame("Full" = rowMeans(obj4$RMSE_filtered),"Reduced" = rowMeans(obj4$RMSE_filtered_simple))
boxplot(df1, xlim = c(0.5, 11.5), ylim = c(min(df1, df2, df3, df4),  max(df1, df2, df3, df4)))
boxplot(df2, add = T, at = c(4,5))
boxplot(df3, add = T, at = c(7,8))
boxplot(df4, add = T, at = c(10,11))
txt1 = TeX(paste0("$\\beta_s$ = ", field_param_matr[1,3], ", $\\tau$ = ", field_param_matr[1,7]))
txt2 = TeX(paste0("$\\beta_s$ = ", field_param_matr[2,3], ", $\\tau$ = ", field_param_matr[2,7]))
txt3 = TeX(paste0("$\\beta_s$ = ", field_param_matr[3,3], ", $\\tau$ = ", field_param_matr[3,7]))
txt4 = TeX(paste0("$\\beta_s$ = ", field_param_matr[4,3], ", $\\tau$ = ", field_param_matr[4,7]))
mtext(txt1, at = 1.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt2, at = 4.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt3, at = 7.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt4, at = 10.5, cex = 1.4, line = 0.75, adj = 0.5)
lines(c(3,3), c(-1000, 1000), lwd = 2)
lines(c(6,6), c(-1000, 1000), lwd = 2)
lines(c(9,9), c(-1000, 1000), lwd = 2)
dev.off()

filepath = paste0(dir,"CRPSfiltering-2.png")
png(filename=filepath, pointsize=10, width=6000, height=3000, res=600)
par(mfrow = c(1,1))
df1 = data.frame("Full" = rowMeans(obj1$CRPS_filtered),"Reduced" = rowMeans(obj1$CRPS_filtered_simple))
df2 = data.frame("Full" = rowMeans(obj2$CRPS_filtered),"Reduced" = rowMeans(obj2$CRPS_filtered_simple))
df3 = data.frame("Full" = rowMeans(obj3$CRPS_filtered),"Reduced" = rowMeans(obj3$CRPS_filtered_simple))
df4 = data.frame("Full" = rowMeans(obj4$CRPS_filtered),"Reduced" = rowMeans(obj4$CRPS_filtered_simple))
boxplot(df1, xlim = c(0.5, 11.5), ylim = c(min(df1, df2, df3, df4),  max(df1, df2, df3, df4)))
boxplot(df2, add = T, at = c(4,5))
boxplot(df3, add = T, at = c(7,8))
boxplot(df4, add = T, at = c(10,11))
txt1 = TeX(paste0("$\\beta_s$ = ", field_param_matr[1,3], ", $\\tau$ = ", field_param_matr[1,7]))
txt2 = TeX(paste0("$\\beta_s$ = ", field_param_matr[2,3], ", $\\tau$ = ", field_param_matr[2,7]))
txt3 = TeX(paste0("$\\beta_s$ = ", field_param_matr[3,3], ", $\\tau$ = ", field_param_matr[3,7]))
txt4 = TeX(paste0("$\\beta_s$ = ", field_param_matr[4,3], ", $\\tau$ = ", field_param_matr[4,7]))
mtext(txt1, at = 1.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt2, at = 4.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt3, at = 7.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt4, at = 10.5, cex = 1.4, line = 0.75, adj = 0.5)
lines(c(3,3), c(-1000, 1000), lwd = 2)
lines(c(6,6), c(-1000, 1000), lwd = 2)
lines(c(9,9), c(-1000, 1000), lwd = 2)
dev.off()

filepath = paste0(dir,"RMSEone-step-prediction-2.png")
png(filename=filepath, pointsize=10, width=6000, height=3000, res=600)
par(mfrow = c(1,1))
df1 = data.frame("Full" = rowMeans(obj1$RMSE_1st_pred),"Reduced" = rowMeans(obj1$RMSE_1st_pred_simple))
df2 = data.frame("Full" = rowMeans(obj2$RMSE_1st_pred),"Reduced" = rowMeans(obj2$RMSE_1st_pred_simple))
df3 = data.frame("Full" = rowMeans(obj3$RMSE_1st_pred),"Reduced" = rowMeans(obj3$RMSE_1st_pred_simple))
df4 = data.frame("Full" = rowMeans(obj4$RMSE_1st_pred),"Reduced" = rowMeans(obj4$RMSE_1st_pred_simple))
boxplot(df1, xlim = c(0.5, 11.5), ylim = c(min(df1, df2, df3, df4),  max(df1, df2, df3, df4)))
boxplot(df2, add = T, at = c(4,5))
boxplot(df3, add = T, at = c(7,8))
boxplot(df4, add = T, at = c(10,11))
txt1 = TeX(paste0("$\\beta_s$ = ", field_param_matr[1,3], ", $\\tau$ = ", field_param_matr[1,7]))
txt2 = TeX(paste0("$\\beta_s$ = ", field_param_matr[2,3], ", $\\tau$ = ", field_param_matr[2,7]))
txt3 = TeX(paste0("$\\beta_s$ = ", field_param_matr[3,3], ", $\\tau$ = ", field_param_matr[3,7]))
txt4 = TeX(paste0("$\\beta_s$ = ", field_param_matr[4,3], ", $\\tau$ = ", field_param_matr[4,7]))
mtext(txt1, at = 1.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt2, at = 4.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt3, at = 7.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt4, at = 10.5, cex = 1.4, line = 0.75, adj = 0.5)
lines(c(3,3), c(-1000, 1000), lwd = 2)
lines(c(6,6), c(-1000, 1000), lwd = 2)
lines(c(9,9), c(-1000, 1000), lwd = 2)
dev.off()

filepath = paste0(dir,"CRPS-one-step-prediction-2.png")
png(filename=filepath, pointsize=10, width=6000, height=3000, res=600)
par(mfrow = c(1,1))
df1 = data.frame("Full" = rowMeans(obj1$CRPS_1st_pred),"Reduced" = rowMeans(obj1$CRPS_1st_pred_simple))
df2 = data.frame("Full" = rowMeans(obj2$CRPS_1st_pred),"Reduced" = rowMeans(obj2$CRPS_1st_pred_simple))
df3 = data.frame("Full" = rowMeans(obj3$CRPS_1st_pred),"Reduced" = rowMeans(obj3$CRPS_1st_pred_simple))
df4 = data.frame("Full" = rowMeans(obj4$CRPS_1st_pred),"Reduced" = rowMeans(obj4$CRPS_1st_pred_simple))
boxplot(df1, xlim = c(0.5, 11.5), ylim = c(min(df1, df2, df3, df4),  max(df1, df2, df3, df4)))
boxplot(df2, add = T, at = c(4,5))
boxplot(df3, add = T, at = c(7,8))
boxplot(df4, add = T, at = c(10,11))
txt1 = TeX(paste0("$\\beta_s$ = ", field_param_matr[1,3], ", $\\tau$ = ", field_param_matr[1,7]))
txt2 = TeX(paste0("$\\beta_s$ = ", field_param_matr[2,3], ", $\\tau$ = ", field_param_matr[2,7]))
txt3 = TeX(paste0("$\\beta_s$ = ", field_param_matr[3,3], ", $\\tau$ = ", field_param_matr[3,7]))
txt4 = TeX(paste0("$\\beta_s$ = ", field_param_matr[4,3], ", $\\tau$ = ", field_param_matr[4,7]))
mtext(txt1, at = 1.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt2, at = 4.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt3, at = 7.5, cex = 1.4, line = 0.75, adj = 0.5)
mtext(txt4, at = 10.5, cex = 1.4, line = 0.75, adj = 0.5)
lines(c(3,3), c(-1000, 1000), lwd = 2)
lines(c(6,6), c(-1000, 1000), lwd = 2)
lines(c(9,9), c(-1000, 1000), lwd = 2)
dev.off()

################# PLOT OF V_T ESTIMATES
ind = 1
#low measurement error
df = data.frame("LH" = obj1$param_matr[,ind],
                "HH" = obj3$param_matr[,ind],
                "LL" = obj2$param_matr[,ind],
                "HL" = obj4$param_matr[,ind])

filepath = paste0(dir,"temp-smoothness-plot-2.png")
png(filename=filepath, pointsize=10, width=1500, height=1500, res=450)
vioplot(df, ylim = c(0.5,2.75), main =  TeX("$\\nu_t$"))
lines(c(0.5,4.5), c(1.0,1.0), col = "red", lwd = 2, lty = 2)
dev.off()


################# PLOT OF V_S ESTIMATES
ind = 2
#low measurement error
df = data.frame("LH" = obj1$param_matr[,ind],
                "LL" = obj3$param_matr[,ind],
                "HH" = obj2$param_matr[,ind],
                "HL" = obj4$param_matr[,ind])

filepath = paste0(dir,"spatial-smoothness-plot-2.png")
png(filename=filepath, pointsize=10, width=1500, height=1500, res=450)
vioplot(df, ylim = c(0.4,1.1), main =  TeX("$\\nu_s$"))
lines(c(0.5,4.5), c(1.0,1.0), col = "red", lwd = 2, lty = 2)
dev.off()

################# PLOT OF BETA_S ESTIMATES
ind = 3
#low measurement error
df = data.frame("LH" = obj1$param_matr[,ind],
                "LL" = obj3$param_matr[,ind],
                "HH" = obj2$param_matr[,ind],
                "HL" = obj4$param_matr[,ind])

filepath = paste0(dir,"beta_s_plot-2.png")
png(filename=filepath, pointsize=10, width=1500, height=1500, res=450)
vioplot(df, ylim = c(0,1), main =  TeX("$\\beta_s$"))
lines(c(0.5,2.5), c(0.25,0.25), col = "red", lwd = 2, lty = 2)
lines(c(2.5,4.5), c(0.75,0.75), col = "red", lwd = 2, lty = 2)
dev.off()

############### TEMPORAL RANGE PLOTS
ind = 4

df1 = data.frame("LH" = obj1$param_matr[,ind],
                "LL" = obj3$param_matr[,ind],
                "HH" = obj2$param_matr[,ind],
                "HL" = obj4$param_matr[,ind])


df2 = data.frame("LH" = obj1$param_matr_simple[,ind - 3],
                "LL" = obj3$param_matr_simple[,ind - 3],
                "HH" = obj2$param_matr_simple[,ind - 3],
                "HL" = obj4$param_matr_simple[,ind - 3])

filepath = paste0(dir, "temp-range-plot-2.png")
png(filename=filepath, pointsize=10, width = 2500, height=1500, res=450)
vioplot(df1, xlim = c(0.5,9.5), ylim = c(2,16), main =  TeX("$\\nu_t$"))
vioplot(df2, at = c(6,7,8,9), add = TRUE)
mtext("Full", at = 2.5, cex = 1, line = -1.1, adj = 0.5)
mtext("Simple", at = 7.5, cex = 1, line = -1.1, adj = 0.5)
axis(side = 1, at = c(6,7,8,9), labels = c("LH","LL","HH", "HL"))
lines(c(0.5,4.5), c(10.0,10.0), col = "red", lwd = 2, lty = 2)
lines(c(5.5,9.5), c(10.0,10.0), col = "red", lwd = 2, lty = 2)
lines(c(5,5), c(-1000,1000), lwd = 1.5)
dev.off()

############### SPATIAL RANGE PLOTS
ind = 5

df1 = data.frame("LH" = obj1$param_matr[,ind],
                 "LL" = obj3$param_matr[,ind],
                 "HH" = obj2$param_matr[,ind],
                 "HL" = obj4$param_matr[,ind])


df2 = data.frame("LH" = obj1$param_matr_simple[,ind - 3],
                 "LL" = obj3$param_matr_simple[,ind - 3],
                 "HH" = obj2$param_matr_simple[,ind - 3],
                 "HL" = obj4$param_matr_simple[,ind - 3])

filepath = paste0(dir, "spatial-range-plot-2.png")
png(filename=filepath, pointsize=10, width = 2500, height=1500, res=450)
vioplot(df1, xlim = c(0.5,9.5), ylim = c(0.4,1.7), main =  TeX("$\\nu_s$"))
vioplot(df2, at = c(6,7,8,9), add = TRUE)
mtext("Full", at = 2.5, cex = 1, line = -1.1, adj = 0.5)
mtext("Simple", at = 7.5, cex = 1, line = -1.1, adj = 0.5)
axis(side = 1, at = c(6,7,8,9), labels = c("LH","LL","HH", "HL"))
lines(c(0.5,4.5), c(1.0,1.0), col = "red", lwd = 2, lty = 2)
lines(c(5.5,9.5), c(1.0,1.0), col = "red", lwd = 2, lty = 2)
lines(c(5,5), c(-1000,1000), lwd = 1.5)
dev.off()

############### SD RANGE PLOTS
ind = 6

df1 = data.frame("LH" = obj1$param_matr[,ind],
                 "LL" = obj3$param_matr[,ind],
                 "HH" = obj2$param_matr[,ind],
                 "HL" = obj4$param_matr[,ind])


df2 = data.frame("LH" = obj1$param_matr_simple[,ind - 3],
                 "LL" = obj3$param_matr_simple[,ind - 3],
                 "HH" = obj2$param_matr_simple[,ind - 3],
                 "HL" = obj4$param_matr_simple[,ind - 3])

filepath = paste0(dir, "sd-plot-2.png")
png(filename=filepath, pointsize=10, width = 2500, height=1500, res=450)
vioplot(df1, xlim = c(0.5,9.5), ylim = c(1.5,6.0), main =  TeX("$\\sigma$"))
vioplot(df2, at = c(6,7,8,9), add = TRUE)
mtext("Full", at = 2.5, cex = 1, line = -1.1, adj = 0.5)
mtext("Simple", at = 7.5, cex = 1, line = -1.1, adj = 0.5)
axis(side = 1, at = c(6,7,8,9), labels = c("LH","LL","HH", "HL"))
lines(c(0.5,4.5), c(3.5,3.5), col = "red", lwd = 2, lty = 2)
lines(c(5.5,9.5), c(3.5,3.5), col = "red", lwd = 2, lty = 2)
lines(c(5,5), c(-1000,1000), lwd = 1.5)
dev.off()

###### TAU PLOTS
#library(plotrix)
ind = 7

df1 = data.frame("LH" = obj1$param_matr[,ind],
                 "LL" = obj3$param_matr[,ind],
                 "HH" = obj2$param_matr[,ind],
                 "HL" = obj4$param_matr[,ind])


df2 = data.frame("LH" = obj1$param_matr_simple[,ind - 3],
                 "LL" = obj3$param_matr_simple[,ind - 3],
                 "HH" = obj2$param_matr_simple[,ind - 3],
                 "HL" = obj4$param_matr_simple[,ind - 3])

#filepath = paste0(dir, "temp-range-plot.png")
#png(filename=filepath, pointsize=10, width = 2600, height=1500, res=300)
#gap.plot(c(-10), c(-10), xlim = c(0.5,9.5), ylim = c(0.3,0.85), gap = c(0.45,0.7), gap.axis = "y")
#axis.break(axis = 2, breakpos = 0.453, style = "slash")

filepath = paste0(dir, "tau-plot-2.png")
png(filename=filepath, pointsize=10, width = 2500, height=1500, res=450)
vioplot(df1, at = c(1,2,3,4), xlim = c(0.5,9.5), ylim = c(0.3,0.9), main =  TeX("$\\sigma_{obs}$"))
vioplot(df2, at = c(6,7,8,9), add = TRUE)
mtext("Full", at = 2.5, cex = 1, line = -1.1, adj = 0.5)
mtext("Simple", at = 7.5, cex = 1, line = -1.1, adj = 0.5)
axis(side = 1, at = c(6,7,8,9), labels = c("LH","LL","HH", "HL"))
lines(c(1.5,2.5), c(0.35,0.35), col = "red", lwd = 2, lty = 2)
lines(c(3.5,4.5), c(0.35,0.35), col = "red", lwd = 2, lty = 2)
lines(c(6.5,7.5), c(0.35,0.35), col = "red", lwd = 2, lty = 2)
lines(c(8.5,9.5), c(0.35,0.35), col = "red", lwd = 2, lty = 2)
####
lines(c(0.5,1.5), c(0.75,0.75), col = "red", lwd = 2, lty = 2)
lines(c(2.5,3.5), c(0.75,0.75), col = "red", lwd = 2, lty = 2)
lines(c(5.5,6.5), c(0.75,0.75), col = "red", lwd = 2, lty = 2)
lines(c(7.5,8.5), c(0.75,0.75), col = "red", lwd = 2, lty = 2)
lines(c(5,5), c(-1000,1000), lwd = 1.5)
dev.off()

