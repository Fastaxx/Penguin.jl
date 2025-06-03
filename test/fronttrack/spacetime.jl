using Penguin
using CairoMakie
using LibGEOS

"""
Visualization of Space-Time Capacities (Ax_st and Ay_st)
Shows how front movement between two time steps affects space-time capacities
"""

# 1. Define the mesh parameters
nx, ny = 20, 20        # Number of cells (higher resolution for better visualization)
lx, ly = 10.0, 10.0    # Domain size
x0, y0 = -5.0, -5.0    # Domain origin
dt = 0.1              # Time step size

# Create the mesh using ranges
x_range = range(x0, x0 + lx, length=nx+1)
y_range = range(y0, y0 + ly, length=ny+1)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# 2. Create front trackers at two different time steps
# Time n: Circle at the center
front_n = FrontTracker()
radius_n = 2.5
center_x_n, center_y_n = 0.0, 0.0
create_circle!(front_n, center_x_n, center_y_n, radius_n, 60)  # More markers for smoother interface

# Time n+1: Circle slightly larger and shifted
front_np1 = FrontTracker()
radius_np1 = 3.0
center_x_np1, center_y_np1 = 0.0, 0.0
create_circle!(front_np1, center_x_np1, center_y_np1, radius_np1, 60)

# 3. Compute space-time capacities
spacetime_capacities = compute_spacetime_capacities(mesh, front_n, front_np1, dt)
Ax_st = spacetime_capacities[:Ax_st]
Ay_st = spacetime_capacities[:Ay_st]

# 4. Create the visualization
fig = Figure(size=(1200, 1000))

# Plot the interfaces and mesh
ax1 = Axis(fig[1, 1:2], title="Front Movement over Time",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot mesh grid lines
for x in x_range
    lines!(ax1, [x, x], [y_range[1], y_range[end]], 
          color=:lightgray, linestyle=:dash, linewidth=0.5)
end

for y in y_range
    lines!(ax1, [x_range[1], x_range[end]], [y, y], 
          color=:lightgray, linestyle=:dash, linewidth=0.5)
end

# Plot the front tracking interfaces
markers_n = get_markers(front_n)
marker_x_n = [m[1] for m in markers_n]
marker_y_n = [m[2] for m in markers_n]

markers_np1 = get_markers(front_np1)
marker_x_np1 = [m[1] for m in markers_np1]
marker_y_np1 = [m[2] for m in markers_np1]

# Draw the interface lines
lines!(ax1, marker_x_n, marker_y_n, color=:blue, linewidth=2,
      label="Interface at t_n")
lines!(ax1, marker_x_np1, marker_y_np1, color=:red, linewidth=2,
      label="Interface at t_n+1")

# Add arrows to show movement direction
n_arrows = 10
for i in 1:n_arrows
    idx = div(length(markers_n), n_arrows) * i
    if idx > 0 && idx <= length(markers_n) && idx <= length(markers_np1)
        arrows!([marker_x_n[idx]], [marker_y_n[idx]], 
               [marker_x_np1[idx] - marker_x_n[idx]], [marker_y_np1[idx] - marker_y_n[idx]], 
               arrowsize=10, linewidth=1.5, color=:purple)
    end
end

axislegend(ax1, position=:rt)

# Create heatmap for space-time Ax capacity
ax2 = Axis(fig[2, 1], title="Space-Time Ax Capacity",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Get mesh node positions for plotting
x_nodes = mesh.nodes[1]
y_nodes = mesh.nodes[2]

# Plot Ax_st as a heatmap (vertically centered at cell faces)
Ax_st_clipped = copy(Ax_st)
max_val = maximum(abs.(Ax_st)) * 0.9  # Clip for better color scale
Ax_st_clipped = clamp.(Ax_st, 0.0, max_val)

# For vertically-aligned heatmap (Ax is on vertical faces)
hm_ax = heatmap!(ax2, x_nodes, y_nodes[1:end-1] .+ diff(y_nodes)/2, 
                Ax_st_clipped', colormap=:viridis)
Colorbar(fig[2, 2], hm_ax, label="Ax_st (Space-Time Capacity)")

# Plot the interfaces on the Ax map too
lines!(ax2, marker_x_n, marker_y_n, color=:blue, linewidth=1.5)
lines!(ax2, marker_x_np1, marker_y_np1, color=:red, linewidth=1.5)

# Create heatmap for space-time Ay capacity
ax3 = Axis(fig[3, 1], title="Space-Time Ay Capacity",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot Ay_st as a heatmap (horizontally centered at cell faces)
Ay_st_clipped = copy(Ay_st)
max_val = maximum(abs.(Ay_st)) * 0.9  # Clip for better color scale
Ay_st_clipped = clamp.(Ay_st, 0.0, max_val)

# For horizontally-aligned heatmap (Ay is on horizontal faces)
hm_ay = heatmap!(ax3, x_nodes[1:end-1] .+ diff(x_nodes)/2, y_nodes, 
                Ay_st_clipped, colormap=:plasma)
Colorbar(fig[3, 2], hm_ay, label="Ay_st (Space-Time Capacity)")

# Plot the interfaces on the Ay map too
lines!(ax3, marker_x_n, marker_y_n, color=:blue, linewidth=1.5)
lines!(ax3, marker_x_np1, marker_y_np1, color=:red, linewidth=1.5)

# Add a combined view with cell marching squares classification
ax4 = Axis(fig[2:3, 3:4], title="Marching Squares Classification",
          xlabel="x", ylabel="y", aspect=DataAspect())

# Plot mesh grid
for x in x_range
    lines!(ax4, [x, x], [y_range[1], y_range[end]], 
          color=:lightgray, linestyle=:dash, linewidth=0.5)
end

for y in y_range
    lines!(ax4, [x_range[1], x_range[end]], [y, y], 
          color=:lightgray, linestyle=:dash, linewidth=0.5)
end

# Draw interfaces
lines!(ax4, marker_x_n, marker_y_n, color=:blue, linewidth=2, label="t_n")
lines!(ax4, marker_x_np1, marker_y_np1, color=:red, linewidth=2, label="t_n+1")

# Color cells based on marching squares case
# (This is a simplified visualization - for each vertical face, check if it has non-zero capacity)
for i in 1:nx+1
    for j in 1:ny
        # For vertical faces (Ax)
        if Ax_st[i,j] > 0.01
            scatter!(ax4, [x_nodes[i]], [y_nodes[j] + diff(y_nodes)[1]/2], 
                    color=:green, marker='∣', markersize=20, alpha=0.7)
        end
    end
end

# For horizontal faces (Ay)
for i in 1:nx
    for j in 1:ny+1
        if Ay_st[i,j] > 0.01
            scatter!(ax4, [x_nodes[i] + diff(x_nodes)[1]/2], [y_nodes[j]], 
                    color=:orange, marker='—', markersize=20, alpha=0.7)
        end
    end
end

axislegend(ax4, position=:rt)

# Adjust layout
fig[1, 1:2] = ax1
fig[2, 1:2] = ax2
fig[3, 1:2] = ax3
fig[2:3, 3:4] = ax4

# Display the figure
display(fig)

# Save the figure
save("spacetime_capacities_visualization.png", fig)
println("Visualization saved as 'spacetime_capacities_visualization.png'")