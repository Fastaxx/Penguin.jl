using Penguin
using CairoMakie
using LibGEOS
using SparseArrays
using Statistics
"""
Simplified comparison of space-time capacities using Space-Time Mesh with VOFI
"""

# 1. Define the mesh parameters
nx, ny = 40, 40
lx, ly = 10.0, 10.0
x0, y0 = -5.0, -5.0
dt = 0.1

# Create the spatial mesh
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2. Create front trackers at two different time steps (expanding circle)
# Time n: Circle at center
front_n = FrontTracker()
nmarkers = 100  # Number of markers for the circle
radius_n = 2.5
center_x, center_y = 0.1, 0.0
create_circle!(front_n, center_x, center_y, radius_n, nmarkers)

# Time n+1: Circle slightly larger
front_np1 = FrontTracker()
radius_np1 = 3.0
create_circle!(front_np1, center_x, center_y, radius_np1, nmarkers)

# 3. Compute space-time capacities using Front Tracking
ft_spacetime_capacities = compute_spacetime_capacities(mesh, front_n, front_np1, dt)
Ax_st_ft = ft_spacetime_capacities[:Ax_spacetime]
Ay_st_ft = ft_spacetime_capacities[:Ay_spacetime]

# 4. Create a space-time level set function
function spacetime_ls(x, y, t)
    # Linear interpolation between circles
    α = t / dt
    radius_t = (1-α) * radius_n + α * radius_np1
    return sqrt((x - center_x)^2 + (y - center_y)^2) - radius_t
end

# 5. Create a SpaceTimeMesh and compute VOFI capacities directly
# Create time interval [0, dt]
times = [0.0, dt]
st_mesh = Penguin.SpaceTimeMesh(mesh, times)

# Get VOFI capacity on the space-time mesh
vofi_st_capacity = Capacity(spacetime_ls, st_mesh, method="VOFI")

# Extract Ax and Ay components (these should be space-time integrated)
Ax_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.A[1]))
Ax_st_vofi = reshape(Ax_st_vofi, (nx+1, ny+1,2))
Ay_st_vofi = Array(SparseArrays.diag(vofi_st_capacity.A[2]))
Ay_st_vofi = reshape(Ay_st_vofi, (nx+1, ny+1,2))

Ax_st_vofi = Ax_st_vofi[:, :, 1]  # Extract Ax component
Ay_st_vofi = Ay_st_vofi[:, :, 1]  # Extract Ay component

# Helper function for safe relative error calculation
function safe_relative_error(ft_val, vofi_val; epsilon=1e-10)
    if abs(vofi_val) < epsilon
        # For near-zero vofi values
        return abs(ft_val) < epsilon ? 0.0 : 1.0  # 0% or 100% error
    else
        return abs(ft_val - vofi_val) / abs(vofi_val)
    end
end

# Apply this function element-wise
function relative_error_matrix(ft_mat, vofi_mat; epsilon=1e-10)
    result = zeros(size(ft_mat))
    for i in eachindex(ft_mat)
        result[i] = safe_relative_error(ft_mat[i], vofi_mat[i], epsilon=epsilon)
    end
    return result
end

# Calculate relative differences between methods
Ax_rel_diff = relative_error_matrix(Ax_st_ft, Ax_st_vofi)
Ay_rel_diff = relative_error_matrix(Ay_st_ft, Ay_st_vofi)

# 6. Create visualization for comparison
fig = Figure(size=(1500, 800))  # Wider figure to accommodate difference plots

# 6.1 Row 1: Show interfaces at both time steps
ax1 = Axis(fig[1, 1:3], title="Interface Movement", 
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot the interfaces
markers_n = get_markers(front_n)
markers_np1 = get_markers(front_np1)

lines!(ax1, first.(markers_n), last.(markers_n), color=:blue, linewidth=2,
      label="Interface at t=0")
lines!(ax1, first.(markers_np1), last.(markers_np1), color=:red, linewidth=2,
      label="Interface at t=dt")

axislegend(ax1, position=:rt)

# 6.2 Row 2: Compare Ax capacities
ax2 = Axis(fig[2, 1], title="Front Tracking Ax_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax3 = Axis(fig[2, 2], title="VOFI Space-Time Ax_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_x = Axis(fig[2, 3], title="Relative Difference (Ax)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Get mesh node positions for plotting
x_nodes = mesh.nodes[1]
y_nodes = mesh.nodes[2]

# Set consistent color range for Ax
max_Ax = maximum([maximum(Ax_st_ft), maximum(Ax_st_vofi)])

# Plot Ax capacities
hm_Ax_ft = heatmap!(ax2, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                  Ax_st_ft', colormap=:viridis, colorrange=(0, max_Ax))
Colorbar(fig[2, 4], hm_Ax_ft, label="Ax_st (Front Tracking)")

hm_Ax_vofi = heatmap!(ax3, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                     Ax_st_vofi', colormap=:viridis, colorrange=(0, max_Ax))
Colorbar(fig[2, 5], hm_Ax_vofi, label="Ax_st (VOFI)")

# Plot relative difference for Ax (capped at 0.2 or 20%)
max_rel_diff_x = min(maximum(filter(isfinite, Ax_rel_diff)), 0.2)  # Cap at 20% for better visualization
hm_Ax_diff = heatmap!(ax_diff_x, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                     Ax_rel_diff', colormap=:plasma, colorrange=(0, max_rel_diff_x))
Colorbar(fig[2, 6], hm_Ax_diff, label="Relative Difference (0-20%)")

# Add interfaces to the Ax plots
lines!(ax2, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax2, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax3, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax3, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax_diff_x, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_x, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# 6.3 Row 3: Compare Ay capacities
ax4 = Axis(fig[3, 1], title="Front Tracking Ay_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax5 = Axis(fig[3, 2], title="VOFI Space-Time Ay_st",
          xlabel="x", ylabel="y", aspect=DataAspect())
ax_diff_y = Axis(fig[3, 3], title="Relative Difference (Ay)",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Set consistent color range for Ay
max_Ay = maximum([maximum(Ay_st_ft), maximum(Ay_st_vofi)])

# Plot Ay capacities
hm_Ay_ft = heatmap!(ax4, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                  Ay_st_ft, colormap=:viridis, colorrange=(0, max_Ay))
Colorbar(fig[3, 4], hm_Ay_ft, label="Ay_st (Front Tracking)")

hm_Ay_vofi = heatmap!(ax5, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                     Ay_st_vofi, colormap=:viridis, colorrange=(0, max_Ay))
Colorbar(fig[3, 5], hm_Ay_vofi, label="Ay_st (VOFI)")

# Plot relative difference for Ay (capped at 0.2 or 20%)
max_rel_diff_y = min(maximum(filter(isfinite, Ay_rel_diff)), 0.2)  # Cap at 20% for better visualization
hm_Ay_diff = heatmap!(ax_diff_y, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                     Ay_rel_diff, colormap=:plasma, colorrange=(0, max_rel_diff_y))
Colorbar(fig[3, 6], hm_Ay_diff, label="Relative Difference (0-20%)")

# Add interfaces to the Ay plots
lines!(ax4, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax4, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax5, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax5, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

lines!(ax_diff_y, first.(markers_n), last.(markers_n), color=:blue, linewidth=1)
lines!(ax_diff_y, first.(markers_np1), last.(markers_np1), color=:red, linewidth=1)

# Add row for statistics of differences
ax_stats = Axis(fig[4, 1:3], title="Relative Error Statistics", 
               xlabel="Relative Error", ylabel="Frequency")

# Create histograms of difference values
hist!(ax_stats, filter(x -> x > 0 && x < 0.5, vec(Ax_rel_diff)), bins=20, color=:blue, label="Ax Relative Error")
hist!(ax_stats, filter(x -> x > 0 && x < 0.5, vec(Ay_rel_diff)), bins=20, color=:red, label="Ay Relative Error")
axislegend(ax_stats, position=:rt)

# Add statistical summary text
# Using valid_values to filter out potential NaN or Inf values
valid_ax = filter(isfinite, vec(Ax_rel_diff))
valid_ay = filter(isfinite, vec(Ay_rel_diff))

stats_text = """
Statistical Summary:
Max Relative Error Ax: $(round(maximum(valid_ax)*100, digits=2))%
Max Relative Error Ay: $(round(maximum(valid_ay)*100, digits=2))%
Mean Relative Error Ax: $(round(mean(valid_ax)*100, digits=2))%
Mean Relative Error Ay: $(round(mean(valid_ay)*100, digits=2))%
Median Relative Error Ax: $(round(median(valid_ax)*100, digits=2))%
Median Relative Error Ay: $(round(median(valid_ay)*100, digits=2))%
"""

Label(fig[4, 4:6], stats_text, tellwidth=false)

# Save and display the figure
save("spacetime_capacities_with_relative_differences.png", fig)
display(fig)

# Print total capacities for comparison
println("Total Ax (Front Tracking): $(sum(Ax_st_ft))")
println("Total Ax (VOFI): $(sum(Ax_st_vofi))")
println("Relative Error: $(round((sum(Ax_st_ft) - sum(Ax_st_vofi))/sum(Ax_st_vofi)*100, digits=2))%")
println()
println("Total Ay (Front Tracking): $(sum(Ay_st_ft))")
println("Total Ay (VOFI): $(sum(Ay_st_vofi))")
println("Relative Error: $(round((sum(Ay_st_ft) - sum(Ay_st_vofi))/sum(Ay_st_vofi)*100, digits=2))%")