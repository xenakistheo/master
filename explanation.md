
The goal here is to explain the algorithm that is used in `ARMA-SPDE-simulation-study.R` and in `sequential_kf.cpp`. 



### Kalman filter update
`sequential_kf.cpp` has a single function: `sequential_kf_fast`

The function does a single step of the Kalman filter to update the model's posterior. 
It takes as input (amongst other things)the prior mean (vector, m_hat) and prior covariance (matrix, s_hat) of the state, the new observations (vector, y). The observations are from $n$ different concurrent sensors i.e. varying spatial locations. 


For $n$ concurrent observations, a standard, text-book, Kalman filter needs one to invert an $n \times n$ matrix. This is very expensive computationally.

This function considers the case where the observations are independent. Hence, the noise in/covariance matrix of the observations is diagonal. 
This is a completely standard approach - typically called a scalar update form of the Kalman filter. Worth pointing out that this method does not require any matrix inversion. 
The covariance matrix of the observations is passed as a vector (R_diag) of the diagonal elements.

The function also takes in an observation matrix (Hm) that maps the state to the observations. For those not too familiar with the Kalman filter, this is the linear transformation that describes the relationship between the state and the observations. As an example, if the state is a 2D position and the observations are 1D distances, then Hm would be a matrix that transforms the 2D position into a 1D distance. A more intricate example is the case of a 2D position and 2D velocity state, with 1D distance observations. In this case, Hm would be a matrix that transforms the 4D state into a 1D distance.

Some sensors may not return a value at a given time point, and hence the function also takes in a vector NA_ind that indicates which sensors are NA. The function will ignore these sensors when updating the posterior.

The function returns an list with three elements: 
- the updated mean vector. 
- the updated covariance matrix.
- the log-likelihood of the observations given the prior. This is useful for model comparison and for calculating the marginal likelihood of the data. (scalar)

### Numerical simulation

Big picture summary is that this script is fitting a spatio-temporal Gaussian random field to noisy observations via maximum likelihood estimation. 

Doing this with 
It does this in the following way: 

1. **Spectral Expansion**
Instead of modelling the field $F(x,y,t)$ directly, model it as a linear combination of basis functions. $$F(x,y,t) = \sum_k a_k(t)\phi_k(x,y)$$

2. **Physically realistic dynamics**
The coefficients $a_k(t)$ are modelled so as to follow physically realistic dynamics. In this, their covariance is derived from a stochastic partial differential equation (SPDE) - the Matérn covariance function - determined by physical parameters such as the diffusion coefficient, the correlation length, and the smoothness parameter.

3. **Transforming to a Kalman filter problem**
