using Penguin
using LinearAlgebra
using CairoMakie
using LibGEOS

# Create a simple mesh
nx, ny = 20, 20
lx, ly = 2.0, 2.0
x0, y0 = -1.0, -1.0
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Create a vertical line (unclosed interface)
front = FrontTracker()
nmarkers = 10
y_points = range(y0, y0+ly, nmarkers)
vertical_markers = [(0.0, y) for y in y_points]
set_markers!(front, vertical_markers, false)  # false indicates non-closed

# Print interface info
println("Created unclosed interface with $(length(vertical_markers)) markers")
println("Is closed: $(front.is_closed)")

# Override get_fluid_polygon for unclosed interfaces
function Penguin.get_fluid_polygon(ft::FrontTracker)
    if length(ft.markers) < 3
        return nothing
    end
    
    if ft.is_closed
        # Use existing implementation for closed interfaces
        if !LibGEOS.isValid(ft.interface_poly)
            return LibGEOS.buffer(ft.interface_poly, 0.0)
        end
        return ft.interface_poly
    else
        # For vertical interfaces (like our test case), create a polygon for the left side
        coords = [marker for marker in ft.markers]
        min_y = minimum(m[2] for m in coords) - 0.1
        max_y = maximum(m[2] for m in coords) + 0.1
        
        # Create geometry coordinates
        geo_coords = Vector{Vector{Float64}}()
        
        # Interface points from top to bottom
        for i in length(coords):-1:1
            push!(geo_coords, [coords[i][1], coords[i][2]])
        end
        
        # Close on the left side - use x0 from mesh
        push!(geo_coords, [x0, min_y])
        push!(geo_coords, [x0, max_y])
        push!(geo_coords, [coords[length(coords)][1], coords[length(coords)][2]])  # Back to start
        
        # Create a LinearRing from our coordinates
        ring = LibGEOS.LinearRing(geo_coords)
        
        # Create a Polygon from the LinearRing
        return LibGEOS.Polygon(ring)
    end
end

# Calculate actual volume Jacobian
println("Computing volume Jacobian...")
volJ = compute_volume_jacobian(mesh, front, 1e-6)
println("Found $(length(volJ)) cells with non-zero Jacobian entries")

# Visualize the Jacobian matrix
function visualize_jacobian_matrix(volume_jacobian, mesh, markers, front)
    # Extract necessary dimensions
    n_markers = length(markers) - (front.is_closed ? 1 : 0)
    
    # Create an index for each active cell (those with non-zero derivatives)
    active_cells = sort([(i, j) for (i, j) in keys(volume_jacobian) if !isempty(volume_jacobian[(i, j)])])
    n_active_cells = length(active_cells)
    
    if n_active_cells == 0
        println("No active cells found with non-zero derivatives.")
        return nothing
    end
    
    # Create a full matrix and fill it with Jacobian values
    J = zeros(n_active_cells, n_markers)
    
    for (cell_idx, (i, j)) in enumerate(active_cells)
        for (marker_idx, jac_value) in volume_jacobian[(i, j)]
            if 0 <= marker_idx < n_markers  # Check that marker index is valid
                J[cell_idx, marker_idx+1] = jac_value  # +1 because Julia is 1-indexed
            end
        end
    end
    
    # Create the figure
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
              xlabel = "Marker Index",
              ylabel = "Cell Index",
              title = "Volume Jacobian Matrix")
    
    # Determine symmetric color scale
    max_val = maximum(abs.(J))
    
    # Create the heatmap
    heatmap!(ax, 1:n_markers, 1:n_active_cells, J, 
            colormap = :RdBu, colorrange = (-max_val, max_val))
    
    # Add a colorbar
    Colorbar(fig[1, 2], colormap = :RdBu, limits = (-max_val, max_val),
            label = "∂V/∂n (normal direction)")
    
    # Add annotations for cell indices
    cell_labels = ["($(i),$(j))" for (i, j) in active_cells]
    ax.yticks = (1:n_active_cells, cell_labels)
    
    # Add ticks for markers
    ax.xticks = (1:n_markers, string.(0:n_markers-1))
    
    return fig
end

# Visualize marker influence
function visualize_marker_influence(volume_jacobian, mesh, markers, front, marker_idx)
    # Extract mesh dimensions
    nx, ny = size(mesh.centers[1])[1], size(mesh.centers[2])[1]
    
    # Create a matrix for visualization
    marker_impact = zeros(nx, ny)
    
    # Fill the matrix with the selected marker's impact values
    for ((i, j), jac_values) in volume_jacobian
        for (m_idx, jac_value) in jac_values
            if m_idx == marker_idx
                marker_impact[i, j] = jac_value
            end
        end
    end
    
    # Calculate marker normals
    normals = compute_marker_normals(front, markers)  # false for unclosed interface
    
    # Create the figure
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
              xlabel = "x",
              ylabel = "y",
              title = "Cells affected by marker $marker_idx",
              aspect = DataAspect())
    
    # Determine symmetric color scale
    max_val = maximum(abs.(marker_impact))
    if max_val == 0
        max_val = 1.0  # Default value if no impact
    end
    
    # Extract face coordinates for display
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Create the heatmap
    heatmap!(ax, x_faces, y_faces, marker_impact, 
            colormap = :RdBu, colorrange = (-max_val, max_val))
    
    # Add a colorbar
    Colorbar(fig[1, 2], colormap = :RdBu, limits = (-max_val, max_val),
            label = "Volume rate of change")
    
    # Draw the interface
    interface_x = [m[1] for m in markers]
    interface_y = [m[2] for m in markers]
    lines!(ax, interface_x, interface_y, color = :black, linewidth = 1)
    
    # Highlight the selected marker (0-indexed but 1-indexed for access)
    mx, my = markers[marker_idx+1]
    scatter!(ax, [mx], [my], color = :red, markersize = 12)
    
    # Draw the normal vector at the marker
    nx, ny = normals[marker_idx+1]
    arrow_scale = (maximum(x_faces) - minimum(x_faces)) * 0.05
    arrows!(ax, [mx], [my], [nx * arrow_scale], [ny * arrow_scale],
           color = :red, arrowsize = 15)
    
    return fig
end

# Get markers and create visualization
markers = get_markers(front)

# Plot Jacobian matrix
fig_matrix = visualize_jacobian_matrix(volJ, mesh, markers, front)
display(fig_matrix)

# Plot influence of a specific marker (middle one)
middle_marker = div(length(markers), 2)
fig_influence = visualize_marker_influence(volJ, mesh, markers, front, middle_marker)
display(fig_influence)

println("\nJacobian visualization complete. The matrix shows how each marker affects cell volumes.")