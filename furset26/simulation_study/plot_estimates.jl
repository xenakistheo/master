using JLD2
using StatsPlots

# Violin plots of the estimates across replications, with the true values marked
results_dir = "furset26/simulation_study/results/LL"
files = sort(filter(f -> startswith(f, "estimate_r") && endswith(f, ".jld2"), readdir(results_dir)))
θ_hat = reduce(hcat, [load(joinpath(results_dir, f), "θ_opt") for f in files])' # R × 7

labels = ["ν_s", "ν_t", "r_t", "r_s", "σ", "β_obs", "σ_obs"]
θ_true = [1.0, 1.0, 10.0, 1.0, 3.5, 0.25, 0.35]
log_scale = [false, false, true, true, false, false, false] # r_t, r_s have outliers spanning orders of magnitude

fill_color = "#2a78d6"
true_color = "#eb6834"

panels = map(1:7) do j
    # Log panels: estimate the density on log10 values and label the axis in original units
    x = log_scale[j] ? log10.(θ_hat[:, j]) : θ_hat[:, j]
    y_true = log_scale[j] ? log10(θ_true[j]) : θ_true[j]
    ticks = log_scale[j] ? (k = ceil(Int, minimum(x)):floor(Int, maximum(x)); (k, ["10^$i" for i in k])) : :auto
    p = violin(fill(1, length(x)), x;
        fillcolor = fill_color, fillalpha = 0.35, linecolor = fill_color, linewidth = 1,
        yticks = ticks,
        title = labels[j] * (log_scale[j] ? " (log scale)" : ""),
        legend = false, xticks = false, grid = :y, gridalpha = 0.3)
    dotplot!(p, fill(1, length(x)), x; mode = :density, markersize = 3,
        markercolor = fill_color, markerstrokewidth = 0)
    hline!(p, [y_true]; color = true_color, linewidth = 2, linestyle = :dash)
    p
end

legend_panel = plot(framestyle = :none, legend = :left, legendfontsize = 9)
plot!(legend_panel, [NaN], [NaN]; seriestype = :shape, fillcolor = fill_color, fillalpha = 0.35,
    linecolor = fill_color, label = "Estimates ($(length(files)) runs)")
plot!(legend_panel, [NaN], [NaN]; color = true_color, linewidth = 2, linestyle = :dash, label = "True value")

fig = plot(panels..., legend_panel; layout = (2, 4), size = (1200, 650), margin = 4Plots.mm)
savefig(fig, joinpath(results_dir, "violin_estimates_LL.png"))
savefig(fig, joinpath(results_dir, "violin_estimates_LL.pdf"))
