using Test
using LinearAlgebra
using Random
using Statistics
using LibGEOS
using Penguin

# Plot interface and normal vectors using CairoMakie
using CairoMakie
# Create a test mesh for the Jacobian visualization
using Penguin

# Create a circular interface for testing
front = FrontTracker()
create_circle!(front, 0.5, 0.5, 0.3, 32)

# Create a proper mesh object
nx, ny = 20, 20
x0, y0 = 0.0, 0.0
lx, ly = 1.0, 1.0
x_nodes = range(x0, stop=lx, length=nx+1)
y_nodes = range(y0, stop=ly, length=ny+1)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Compute the volume Jacobian
volume_jacobian = compute_volume_jacobian(front, x_nodes, y_nodes)
function unzip(pairs)
    return (getindex.(pairs, 1), getindex.(pairs, 2))
end
function visualize_jacobian_matrix(volume_jacobian, mesh::Penguin.Mesh{2}, front::FrontTracker, markers=nothing)
    # Import Makie if needed
    
    # Use provided markers or get them from the front tracker
    if isnothing(markers)
        markers = get_markers(front)
    end
    
    # Extract dimensions
    n_markers = length(markers) - (front.is_closed ? 1 : 0)  # Don't count duplicated closing marker
    
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
                J[cell_idx, marker_idx+1] = jac_value  # Julia is 1-indexed, so add 1
            end
        end
    end
    
    # Create the figure
    fig = Figure(size=(1000, 700))
    
    # Determine the color scale symmetrically
    max_val = maximum(abs.(J))
    
    # Create the heatmap
    ax = Axis(fig[1, 1],
              xlabel="Marker Index",
              ylabel="Cell Index",
              title="Volume Jacobian Matrix with respect to marker displacements")
    
    hm = heatmap!(ax, 1:n_markers, 1:n_active_cells, J, 
                  colormap=:RdBu, 
                  colorrange=(-max_val, max_val))
    
    # Add a colorbar
    Colorbar(fig[1, 2], hm, label="∂V/∂n (normal direction)")
    
    # Add cell index labels
    cell_labels = ["($i,$j)" for (i, j) in active_cells]
    ax.yticks = (1:n_active_cells, cell_labels)
    
    # Add marker index labels
    ax.xticks = (1:n_markers, string.(0:n_markers-1))
    
    # Add additional diagnostics
    # Count how many cells each marker influences
    marker_influence = zeros(Int, n_markers)
    for (_, jac_entries) in volume_jacobian
        for (marker_idx, _) in jac_entries
            if 0 <= marker_idx < n_markers
                marker_influence[marker_idx+1] += 1
            end
        end
    end
    
    # Display markers with zero or low influence
 
    
    # Add visualization of marker influence below the heatmap
    ax2 = Axis(fig[3, 1:2], 
               xlabel="Marker Index", 
               ylabel="# Cells Influenced",
               title="Number of cells influenced by each marker")
    
    barplot!(ax2, 0:n_markers-1, marker_influence, color=:steelblue)
    ax2.xticks = (0:n_markers-1, string.(0:n_markers-1))
    
    # Add vertical lines to indicate top and bottom markers
    if length(markers) >= 3
        # Find markers at top and bottom
        y_values = last.(markers[1:n_markers])
        top_idx = argmax(y_values) - 1  # Convert to 0-indexed
        bottom_idx = argmin(y_values) - 1
        
        vlines!(ax2, [top_idx], color=:red, linestyle=:dash, 
                label="Top marker")
        vlines!(ax2, [bottom_idx], color=:orange, linestyle=:dash, 
                label="Bottom marker")
        
        Legend(fig[3, 3], ax2)
    end
    
    # Add overall figure title with summary
    supertitle = "Volume Jacobian Analysis - $(n_active_cells) cells × $(n_markers) markers"
    Label(fig[0, :], supertitle, fontsize=20, font=:bold)
    
    return fig
end
# Generate and display the visualization
markers = get_markers(front)
fig = visualize_jacobian_matrix(volume_jacobian, mesh, front, markers)

# Save the figure to a file
save("jacobian_heatmap.png", fig)

# Display the figure in the REPL/notebook
display(fig)

# Also display a figure showing the interface and mesh together
fig2 = Figure(resolution=(800, 600))
ax = Axis(fig2[1, 1], aspect=DataAspect(), title="Interface and Mesh")

# Plot mesh grid
for x in x_nodes
    lines!(ax, [x, x], [y_nodes[1], y_nodes[end]], color=:lightgray)
end
for y in y_nodes
    lines!(ax, [x_nodes[1], x_nodes[end]], [y, y], color=:lightgray)
end

# Plot interface
marker_xs, marker_ys = unzip(markers)
lines!(ax, marker_xs, marker_ys, color=:blue, linewidth=2)
scatter!(ax, marker_xs, marker_ys, color=:blue, markersize=5)

# Highlight cells with non-zero Jacobian entries
active_cells = [(i,j) for (i,j) in keys(volume_jacobian) if !isempty(volume_jacobian[(i,j)])]

for (i,j) in active_cells
    x_min = x_nodes[i]
    x_max = x_nodes[i+1]
    y_min = y_nodes[j]
    y_max = y_nodes[j+1]
    poly!(ax, [Point2f(x_min, y_min), Point2f(x_max, y_min), 
              Point2f(x_max, y_max), Point2f(x_min, y_max)], 
          color=(:red, 0.2))
end

display(fig2)

@testset "Julia FrontTracking Tests" begin
    
    @testset "Basic Construction" begin
        # Create an empty interface
        front = FrontTracker()
        @test isempty(front.markers)
        @test front.is_closed == true
        @test front.interface === nothing
        @test front.interface_poly === nothing
        
        # Create interface with markers
        markers = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        front = FrontTracker(markers)
        # The implementation adds a closing point
        @test length(front.markers) == 5  
        @test front.is_closed == true
        @test front.interface !== nothing
        @test front.interface_poly !== nothing
    end
    
    @testset "Shape Creation" begin
        # Test circle creation
        front = FrontTracker()
        create_circle!(front, 0.5, 0.5, 0.3, 20)
        markers = get_markers(front)
        @test length(markers) == 21  # With closing point
        
        # Check that markers are approximately on a circle
        for (x, y) in markers
            distance = sqrt((x - 0.5)^2 + (y - 0.5)^2)
            @test isapprox(distance, 0.3, atol=1e-10)
        end
        
        # Test rectangle creation
        front = FrontTracker()
        create_rectangle!(front, 0.1, 0.2, 0.8, 0.9)
        markers = get_markers(front)
        @test length(markers) == 5  # With closing point
        @test markers[1] == (0.1, 0.2)
        @test markers[2] == (0.8, 0.2)
        @test markers[3] == (0.8, 0.9)
        @test markers[4] == (0.1, 0.9)
        
        # Test ellipse creation
        front = FrontTracker()
        create_ellipse!(front, 0.5, 0.5, 0.3, 0.2, 20)
        markers = get_markers(front)
        @test length(markers) == 21  # With closing point
        
        # Check that markers are approximately on an ellipse
        for (x, y) in markers
            normalized_distance = ((x - 0.5)/0.3)^2 + ((y - 0.5)/0.2)^2
            @test isapprox(normalized_distance, 1.0, atol=1e-10)
        end
    end
    
    @testset "Point Inside Tests" begin
        # Create a square
        front = FrontTracker()
        create_rectangle!(front, 0.0, 0.0, 1.0, 1.0)
        
        # Test points inside
        @test is_point_inside(front, 0.5, 0.5) == true
        @test is_point_inside(front, 0.1, 0.1) == true
        @test is_point_inside(front, 0.9, 0.9) == true
        
        # Test points outside
        @test is_point_inside(front, -0.5, 0.5) == false
        @test is_point_inside(front, 1.5, 0.5) == false
        @test is_point_inside(front, 0.5, -0.5) == false
        @test is_point_inside(front, 0.5, 1.5) == false
    end
    
    @testset "SDF Calculation" begin
        # Create a square centered at origin
        front = FrontTracker()
        create_rectangle!(front, -1.0, -1.0, 1.0, 1.0)
        
        # Test points inside (should be negative)
        @test sdf(front, 0.0, 0.0) < 0
        @test isapprox(sdf(front, 0.0, 0.0), -1.0, atol=1e-2)
        
        # Test points outside (should be positive)
        @test sdf(front, 2.0, 0.0) > 0
        @test isapprox(sdf(front, 2.0, 0.0), 1.0, atol=1e-2)
        
        # Test points on the boundary (should be approximately zero)
        @test isapprox(sdf(front, 1.0, 0.0), 0.0, atol=1e-2)
        @test isapprox(sdf(front, 0.0, 1.0), 0.0, atol=1e-2)
        
        # Create a circle for additional tests
        front = FrontTracker()
        create_circle!(front, 0.0, 0.0, 1.0)
        
        # Test points at various distances from circle
        @test isapprox(sdf(front, 0.0, 0.0), -1.0, atol=1e-2)  # Center should be -radius
        @test isapprox(sdf(front, 2.0, 0.0), 1.0, atol=1e-2)   # Outside, distance = 1
        @test isapprox(sdf(front, 0.5, 0.0), -0.5, atol=1e-2)  # Inside, distance = 0.5
    end
    
    @testset "Perturbed Interface" begin
        # Create a circular interface
        front = FrontTracker()
        create_circle!(front, 0.5, 0.5, 0.3)
        
        # Get original markers
        original_markers = get_markers(front)
        
        # Create perturbed markers
        rng = MersenneTwister(42)  # For reproducibility
        perturbed_markers = []
        for (x, y) in original_markers
            # Apply random perturbation to each marker
            dx = rand(rng) * 0.01
            dy = rand(rng) * 0.01
            push!(perturbed_markers, (x + dx, y + dy))
        end
        
        # Update interface with perturbed markers
        set_markers!(front, perturbed_markers)
        
        # Compute displacements between original and perturbed markers
        displacements = [sqrt((x2-x1)^2 + (y2-y1)^2) for ((x1, y1), (x2, y2)) in zip(original_markers, perturbed_markers)]
        
        # Check displacement statistics
        @test 0.0 < mean(displacements) < 0.01
        @test maximum(displacements) < 0.015
        
        # Test several points to ensure SDF is still reasonable after perturbation
        @test is_point_inside(front, 0.5, 0.5) == true  # Center point should still be inside
        
        # Point on the original interface might now be inside or outside but should be close to zero
        for (x, y) in original_markers
            @test abs(sdf(front, x, y)) < 0.02  # Should be close to interface
        end
    end

    @testset "Normals and Curvature" begin
        @testset "Circle Normals" begin
            # Create a circle centered at origin
            radius = 0.5
            center_x, center_y = 0.0, 0.0
            front = FrontTracker()
            create_circle!(front, center_x, center_y, radius, 32)
            
            # Calculate normals
            markers = get_markers(front)
            normals = compute_marker_normals(front, markers)
            
            # For a circle, normals should point away from center with unit length
            for i in 1:length(markers)-1 # Skip last point (duplicate of first for closed shape)
                x, y = markers[i]
                nx, ny = normals[i]
                
                # Expected normal: unit vector from center to point
                dist = sqrt((x - center_x)^2 + (y - center_y)^2)
                expected_nx = (x - center_x) / dist
                expected_ny = (y - center_y) / dist
                
                # Check that normal is a unit vector
                @test isapprox(nx^2 + ny^2, 1.0, atol=1e-6)
                
                # Check normal direction
                @test isapprox(nx, expected_nx, atol=1e-6)
                @test isapprox(ny, expected_ny, atol=1e-6)
            end
        end
    end
    
    @testset "Volume Jacobian" begin
        # Create a circular interface
        front = FrontTracker()
        create_circle!(front, 0.5, 0.5, 0.3)
        
        # Create a simple mesh grid
        x_faces = 0.0:0.1:1.0
        y_faces = 0.0:0.1:1.0
        
        # Compute the volume Jacobian
        jacobian = compute_volume_jacobian(front, x_faces, y_faces)
        
        # Basic checks
        @test isa(jacobian, Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}})
        
        # At least some cells should have non-zero Jacobian entries
        @test any(length(values) > 0 for (_, values) in jacobian)
        
        # Cells far from the interface should have zero Jacobian entries
        # Cells at corner (1,1) and (10,10) should be outside the circle
        @test length(get(jacobian, (1, 1), [])) == 0
        @test length(get(jacobian, (10, 10), [])) == 0
        
        # Cells near the interface should have non-zero Jacobian entries
        # Find where the interface crosses cells
        has_nonzero_entries = false
        for i in 1:length(x_faces)-1
            for j in 1:length(y_faces)-1
                if haskey(jacobian, (i, j)) && !isempty(jacobian[(i, j)])
                    has_nonzero_entries = true
                    break
                end
            end
            if has_nonzero_entries
                break
            end
        end
        @test has_nonzero_entries

    end
end
