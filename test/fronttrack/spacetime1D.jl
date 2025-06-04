using Penguin
using CairoMakie
using SparseArrays
using Statistics

"""
Comparison of 1D space-time capacities: Front Tracking vs VOFI approach
"""

# 1. Define the 1D spatial mesh
nx = 100
lx = 2.0
x0 = 0.0
mesh_1d = Penguin.Mesh((nx,), (lx,), (x0,))
x_nodes = mesh_1d.nodes[1]
dx = lx/nx

# 2. Define time step
dt = 0.1

# 3. Create front trackers at two different time steps (moving interface)
# Linear motion of the interface from x=0.5 to x=1.0
interface_pos_n = 0.5
interface_pos_np1 = 1.0

front_n = FrontTracker1D([interface_pos_n])
front_np1 = FrontTracker1D([interface_pos_np1])

# 4. Compute space-time capacities using Front Tracking
st_capacities = compute_spacetime_capacities_1d(mesh_1d, front_n, front_np1, dt)

# 5. Define a space-time level set function for comparison with VOFI
# Time becomes the second dimension (y)
function spacetime_ls(x, y, z=0)
    # Convert y to normalized time [0,1]
    t_normalized = y / dt
    
    # Linear interpolation of interface position
    interface_t = interface_pos_n + t_normalized * (interface_pos_np1 - interface_pos_n)
    
    # Return signed distance to interface
    return -(x - interface_t)
end

# 6. Create a 2D space-time mesh for VOFI
STmesh = Penguin.SpaceTimeMesh(mesh_1d, [0.0, dt], tag=mesh_1d.tag)

# 7. Compute VOFI capacities on the space-time mesh
vofi_capacity = Capacity(spacetime_ls, STmesh, method="VOFI")

# Extract the Ax component from VOFI (integrated over time)
Ax_vofi = Array(SparseArrays.diag(vofi_capacity.A[1]))
Ax_vofi = reshape(Ax_vofi, (nx+1, 2))[:, 1]  # Extract Ax component at t=0

# 8. Visualize and compare results
fig = Figure(size=(1000, 800), fontsize=12)

# 8.1 Plot showing the interface movement
ax1 = Axis(fig[1, 1:2], 
          title="Interface Movement",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Plot the interface positions
scatter!(ax1, [interface_pos_n], [0.0], color=:blue, markersize=10, label="t = 0")
scatter!(ax1, [interface_pos_np1], [dt], color=:red, markersize=10, label="t = dt")

# Connect with line to show movement
lines!(ax1, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:black, linestyle=:dash)

# Add horizontal lines at both time steps
lines!(ax1, [x0, lx], [0.0, 0.0], color=:blue, linestyle=:dash, alpha=0.5)
lines!(ax1, [x0, lx], [dt, dt], color=:red, linestyle=:dash, alpha=0.5)

# Add legend
axislegend(ax1)

# 8.2 Plot the space-time Ax capacities from both methods
ax2 = Axis(fig[2, 1:2], 
          title="Space-Time Capacities: Ax",
          xlabel="Position (x)",
          ylabel="Capacity Value")

lines!(ax2, x_nodes, st_capacities[:Ax_spacetime], color=:blue, linewidth=2, label="Front Tracking")
lines!(ax2, x_nodes, Ax_vofi, color=:red, linewidth=2, linestyle=:dash, label="VOFI")

# Add vertical markers at interface positions
vlines!(ax2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, label="Interface t=0")
vlines!(ax2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, label="Interface t=dt")

# Add legend
axislegend(ax2, position=:lt)

# 8.3 Plot the difference between methods
ax3 = Axis(fig[3, 1:2], 
          title="Absolute Difference: |Front Tracking - VOFI|",
          xlabel="Position (x)",
          ylabel="Difference")

abs_diff = abs.(st_capacities[:Ax_spacetime] - Ax_vofi)
lines!(ax3, x_nodes, abs_diff, color=:purple, linewidth=2)

# Add vertical markers at interface positions
vlines!(ax3, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax3, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)


# 9. Print statistics
mean_diff = mean(abs_diff)
max_diff = maximum(abs_diff)
total_ft = sum(st_capacities[:Ax_spacetime])
total_vofi = sum(Ax_vofi)
rel_total_diff = abs(total_ft - total_vofi) / total_vofi * 100

stats_text = """
Comparison Statistics:
Mean Absolute Difference: $(round(mean_diff, digits=6))
Maximum Absolute Difference: $(round(max_diff, digits=6))
Total Front Tracking: $(round(total_ft, digits=6))
Total VOFI: $(round(total_vofi, digits=6))
Relative Difference in Total: $(round(rel_total_diff, digits=2))%
"""

# Add stats text to the figure
Label(fig[5, 1:2], stats_text, tellwidth=false)

# Save the figure
save("spacetime_1d_comparison.png", fig)

# Display the figure
display(fig)

# Print detailed statistics
println("Space-Time Capacities Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_diff)")
println("Max Absolute Difference: $(max_diff)")
println("Total Front Tracking: $(total_ft)")
println("Total VOFI: $(total_vofi)")
println("Relative Difference in Total: $(rel_total_diff)%")
println()

# Print table of values at key positions
println("Values at key positions:")
println("=" ^ 30)
println("x\tFront Tracking\tVOFI\tDiff")
println("-" ^ 30)

for i in 1:length(x_nodes)
    # Print values around the interfaces
    if abs(x_nodes[i] - interface_pos_n) < 0.1 || abs(x_nodes[i] - interface_pos_np1) < 0.1
        println("$(round(x_nodes[i], digits=3))\t$(round(st_capacities[:Ax_spacetime][i], digits=6))\t$(round(Ax_vofi[i], digits=6))\t$(round(abs_diff[i], digits=6))")
    end
end

# 10. Compute space-time volumes using Front Tracking
V_spacetime_ft = st_capacities[:V_spacetime]
edge_types = st_capacities[:edge_types]

# 11. Extract volumes from VOFI
# In the SpaceTimeMesh, the volumes are stored in the D component
V_spacetime_vofi = Array(SparseArrays.diag(vofi_capacity.V))
V_spacetime_vofi = reshape(V_spacetime_vofi, (nx+1, 2))[:, 1]  # Extract Ax component at t=0

# 12. Compare volumes
abs_vol_diff = abs.(V_spacetime_ft - V_spacetime_vofi)
cell_centers = mesh_1d.nodes[1]

# Create a new figure for volume comparison
fig2 = Figure(size=(1000, 800), fontsize=12)

# 12.1 Plot the space-time volumes from both methods
ax1 = Axis(fig2[1, 1:2], 
          title="Space-Time Volumes",
          xlabel="Position (x)",
          ylabel="Volume")

scatter!(ax1, cell_centers, V_spacetime_ft, color=:blue, markersize=8, label="Front Tracking")
scatter!(ax1, cell_centers, V_spacetime_vofi, color=:red, markersize=4, label="VOFI")
stairs!(ax1, [x_nodes[1]; cell_centers; x_nodes[end]], [0; V_spacetime_ft; 0], color=:blue, alpha=0.5)
stairs!(ax1, [x_nodes[1]; cell_centers; x_nodes[end]], [0; V_spacetime_vofi; 0], color=:red, alpha=0.3, linestyle=:dash)

# Add vertical markers at interface positions
vlines!(ax1, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5, label="Interface t=0")
vlines!(ax1, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5, label="Interface t=dt")

# Add legend
axislegend(ax1)

# 12.2 Plot the absolute difference in volumes
ax2 = Axis(fig2[2, 1:2], 
          title="Absolute Difference in Space-Time Volumes",
          xlabel="Position (x)",
          ylabel="Difference")

scatter!(ax2, cell_centers, abs_vol_diff, color=:purple, markersize=8)
stairs!(ax2, [x_nodes[1]; cell_centers; x_nodes[end]], [0; abs_vol_diff; 0], color=:purple, alpha=0.5)

# Add vertical markers at interface positions
vlines!(ax2, [interface_pos_n], color=:blue, linestyle=:dash, alpha=0.5)
vlines!(ax2, [interface_pos_np1], color=:red, linestyle=:dash, alpha=0.5)

# 12.3 Display edge types
ax3 = Axis(fig2[3, 1:2],
          title="Edge Types for Front Tracking",
          xlabel="Position (x)",
          ylabel="Type")

# Map edge types to numerical values for visualization
edge_type_values = zeros(length(edge_types))
for i in 1:length(edge_types)
    if edge_types[i] == :empty
        edge_type_values[i] = 0
    elseif edge_types[i] == :dead
        edge_type_values[i] = 1
    elseif edge_types[i] == :fresh
        edge_type_values[i] = 2
    else # :full
        edge_type_values[i] = 3
    end
end

scatter!(ax3, x_nodes, edge_type_values, color=edge_type_values, colormap=:plasma, markersize=10)
#text!(ax3, x_nodes, edge_type_values .+ 0.2, text=string.(Symbol.(edge_types)), textsize=8, align=(:center, :bottom))

# 12.4 Print volume statistics
mean_vol_diff = mean(abs_vol_diff)
max_vol_diff = maximum(abs_vol_diff)
total_ft_vol = sum(V_spacetime_ft)
total_vofi_vol = sum(V_spacetime_vofi)
rel_total_vol_diff = abs(total_ft_vol - total_vofi_vol) / total_vofi_vol * 100

vol_stats_text = """
Volume Comparison Statistics:
Mean Absolute Difference: $(round(mean_vol_diff, digits=6))
Maximum Absolute Difference: $(round(max_vol_diff, digits=6))
Total Front Tracking Volume: $(round(total_ft_vol, digits=6))
Total VOFI Volume: $(round(total_vofi_vol, digits=6))
Relative Difference in Total: $(round(rel_total_vol_diff, digits=2))%
"""

Label(fig2[4, 1:2], vol_stats_text, tellwidth=false)

# Save the volume comparison figure
save("spacetime_volume_comparison.png", fig2)

# Display the figure
display(fig2)

# 13. Print detailed volume statistics
println("\nSpace-Time Volume Comparison (Front Tracking vs VOFI)")
println("=" ^ 50)
println("Mean Absolute Difference: $(mean_vol_diff)")
println("Max Absolute Difference: $(max_vol_diff)")
println("Total Front Tracking Volume: $(total_ft_vol)")
println("Total VOFI Volume: $(total_vofi_vol)")
println("Relative Difference in Total: $(rel_total_vol_diff)%")

# Print table of volumes at key positions
println("\nVolume Values at key positions:")
println("=" ^ 40)
println("x (cell center)\tFront Tracking\tVOFI\tDiff")
println("-" ^ 40)

for i in 1:nx
    # Print values around the interfaces
    cell_center = (x_nodes[i] + x_nodes[i+1])/2
    if abs(cell_center - interface_pos_n) < 0.1 || abs(cell_center - interface_pos_np1) < 0.1
        println("$(round(cell_center, digits=3))\t$(round(V_spacetime_ft[i], digits=6))\t$(round(V_spacetime_vofi[i], digits=6))\t$(round(abs_vol_diff[i], digits=6))")
    end
end

# 14. Optional: Visualize the space-time fluid regions
fig3 = Figure(size=(800, 400), fontsize=12)
ax = Axis(fig3[1, 1], 
          title="Space-Time Fluid Regions",
          xlabel="Position (x)",
          ylabel="Time (t)")

# Draw cell boundaries
for i in 1:nx+1
    lines!(ax, [x_nodes[i], x_nodes[i]], [0, dt], color=:gray, alpha=0.5)
end
lines!(ax, [x_nodes[1], x_nodes[end]], [0, 0], color=:gray, alpha=0.5)
lines!(ax, [x_nodes[1], x_nodes[end]], [dt, dt], color=:gray, alpha=0.5)

# Draw fluid areas with color intensity proportional to the normalized volume
for i in 1:nx
    x_min, x_max = x_nodes[i], x_nodes[i+1]
    cell_width = x_max - x_min
    rect = Rect(x_min, 0, cell_width, dt)
    
    # Normalize the volume by the maximum possible volume (cell_width * dt)
    normalized_volume = V_spacetime_ft[i] / (cell_width * dt)
    
    # Draw the rectangle with opacity based on fluid volume
    poly!(ax, rect, color=(:blue, normalized_volume))
end

# Show interface positions
scatter!(ax, [interface_pos_n], [0.0], color=:red, markersize=10)
scatter!(ax, [interface_pos_np1], [dt], color=:red, markersize=10)
lines!(ax, [interface_pos_n, interface_pos_np1], [0.0, dt], color=:red, linestyle=:dash)

# Save and display the third figure
save("spacetime_fluid_regions.png", fig3)
display(fig3)