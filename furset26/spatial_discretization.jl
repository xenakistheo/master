using SparseArrays

include("parameters.jl")

function build_full_spatial_matrix(H_spatial; Mx::Int, My::Int, m::Int, m_params::maternParams)
    d = 2*m + floor(Int, m_params.γ)
    M = Mx * My
    N_spatial = size(H_spatial, 1)
    H = spzeros(N_spatial, M*d)

    for k in 1:M
        H[:, (k-1)*d + 1] = H_spatial[:, k]
    end

    return H
end 