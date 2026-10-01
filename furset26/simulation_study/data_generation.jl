include("../parameters.jl")
include("../spectral_discretization.jl")
include("../temporal_discretization.jl")

using JLD2
using Plots

#### experiment

### Define configuration parameters
N_temporal = 45 # Number of temporal observations. 
N_spatial = 250  # Number of spatial locations
Mx_sim, My_sim = 32, 32 # Number of spatial modes to include in simulation
R = 30 # Number of repetitions 
D = Rectangle(1.0, 1.0) # Define spatial domain. 
σ_obs = 0.1 # Standard deviation of observation noise.

# Draw random spatial locations from [0.2, 0.8]^2
spatial_locations = [(rand() * 0.6 + 0.2, rand() * 0.6 + 0.2) for _ in 1:N_spatial]

# Build observation matrix H
H = build_observation_matrix(spatial_locations, Mx_sim, My_sim, D)

ν_s = 1.0
ν_t = 1.0
r_t = 10.0
r_s = 1.0
σ = 3.5

β_s_LL, σ_obs_LL = 0.25, 0.35 
β_s_LH, σ_obs_LH = 0.25, 0.75
β_s_HL, σ_obs_HL = 0.75, 0.35
β_s_HH, σ_obs_HH = 0.75, 0.75




params_LL = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s_LL, σ=σ)
@time y_LL, C_LL, ϵ_LL = sample_GRF(interpretable_to_matern(params_LL),  H, D, Mx_sim, My_sim, N_spatial, N_temporal, σ_obs_LL, R)

params_LH = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s_LH, σ=σ)
@time y_LH, C_LH, ϵ_LH = sample_GRF(interpretable_to_matern(params_LH),  H, D, Mx_sim, My_sim, N_spatial, N_temporal, σ_obs_LH, R)

params_HL = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s_HL, σ=σ)
@time y_HL, C_HL, ϵ_HL = sample_GRF(interpretable_to_matern(params_HL),  H, D, Mx_sim, My_sim, N_spatial, N_temporal, σ_obs_HL, R)

params_HH = maternParams_interpretable(d=2, ν_s=ν_s, ν_t=ν_t, r_s=r_s, r_t=r_t, β_s=β_s_HH, σ=σ)
@time y_HH, C_HH, ϵ_HH = sample_GRF(interpretable_to_matern(params_HH),  H, D, Mx_sim, My_sim, N_spatial, N_temporal, σ_obs_HH, R)

### SAVE y_LL, y_LH, y_HL, y_HH
# @save "furset26/simulation_study/furset26_simulation_data.jld2" y_LL y_LH y_HL y_HH spatial_locations

##### VISUALIZE

# Choose a single of the R random realizations per scenario. 
y_LL_single = y_LL[:, 1, :] # N_temporal x N_spatial matrix. 
y_LH_single = y_LH[:, 1, :]
y_HL_single = y_HL[:, 1, :]
y_HH_single = y_HH[:, 1, :]

# y = Array{Float64}(undef, N_temporal, R, N_spatial)
# y_LL_single[1, :] # N_spatial vector


# Extract coordinates
x = first.(spatial_locations)
y = last.(spatial_locations)

# Extract one realization
y_LL_single = y_LL[:, 1, :]
y_LH_single = y_LH[:, 1, :]
y_HL_single = y_HL[:, 1, :]
y_HH_single = y_HH[:, 1, :]

# Same color scale across all fields and all times
clims = extrema(vcat(
    vec(y_LL_single),
    vec(y_LH_single),
    vec(y_HL_single),
    vec(y_HH_single)
))

anim = @animate for t in axes(y_LL_single, 1)

    p_LL = scatter(
        x, y,
        marker_z = y_LL_single[t, :],
        clims = clims,
        color = :viridis,
        markersize = 8,
        markerstrokewidth = 0,
        aspect_ratio = :equal,
        title = "LL",
        xlabel = "x",
        ylabel = "y",
        colorbar = true,
    )

    p_LH = scatter(
        x, y,
        marker_z = y_LH_single[t, :],
        clims = clims,
        color = :viridis,
        markersize = 8,
        markerstrokewidth = 0,
        aspect_ratio = :equal,
        title = "LH",
        xlabel = "x",
        ylabel = "y",
        colorbar = true,
    )

    p_HL = scatter(
        x, y,
        marker_z = y_HL_single[t, :],
        clims = clims,
        color = :viridis,
        markersize = 8,
        markerstrokewidth = 0,
        aspect_ratio = :equal,
        title = "HL",
        xlabel = "x",
        ylabel = "y",
        colorbar = true,
    )

    p_HH = scatter(
        x, y,
        marker_z = y_HH_single[t, :],
        clims = clims,
        color = :viridis,
        markersize = 8,
        markerstrokewidth = 0,
        aspect_ratio = :equal,
        title = "HH",
        xlabel = "x",
        ylabel = "y",
        colorbar = true,
    )

    plot(
        p_LL, p_LH, p_HL, p_HH,
        layout = (2, 2),
        size = (900, 800),
        plot_title = "t = $t"
    )
end

gif(anim, "fields.gif", fps = 10)