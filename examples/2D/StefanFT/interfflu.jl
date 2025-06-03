using Penguin
using SparseArrays
using LinearAlgebra
using CairoMakie
using Statistics

"""
Plot and analyze interface flux symmetry for different temperature fields
"""
function analyze_interface_flux_symmetry()
    # Output directory for plots
    plot_dir = joinpath(pwd(), "flux_symmetry_analysis")
    if !isdir(plot_dir)
        mkdir(plot_dir)
    end
    
    println("Analyzing interface flux symmetry for different temperature fields...")
    
    #----- 1. Set up mesh and parameters -----#
    # Create mesh
    nx, ny = 32, 32
    lx, ly = 16.0, 16.0
    x0, y0 = -8.0, -8.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Get mesh coordinates directly from the mesh
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Create circular interface
    center = (0.0, 0.0)
    radius = 4.0
    n_markers = 100
    markers = [(center[1] + radius*cos(θ), center[2] + radius*sin(θ)) 
               for θ in range(0, 2π, length=n_markers+1)]
    front = FrontTracker(markers, true)
    
    # Interface conditions
    ρL = 1.0  # Latent heat parameter
    _ic = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρL))
    
    # Time parameters
    Δt = 1.0
    
    # Create body function for mesh
    body_func = (x, y, t) -> sdf(front, x, y)
    
    # Create space-time mesh and capacity
    time_interval = [0.0, Δt]
    STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
    capacity = Capacity(body_func, STmesh; compute_centroids=false)
    
    # Phase properties
    diffusion_coeff = (_x,_y,_z) -> 1.0  # Thermal conductivity
    source_term = (_x,_y,_z,_t=0) -> 0.0  # No internal heat generation
    operator = DiffusionOps(capacity)
    phase = Phase(capacity, operator, source_term, diffusion_coeff)
    
    # Get capacity matrices
    cap_index = length(phase.operator.size)
    V_matrices = phase.capacity.A[cap_index]
    
     # Get operators for flux calculation
    n = (nx+1)*(ny+1)  # Taille correcte pour les vecteurs de température
    W! = phase.operator.Wꜝ[1:n, 1:n]
    G = phase.operator.G[1:n, 1:n]
    H = phase.operator.H[1:n, 1:n]
    Id = build_I_D(phase.operator, diffusion_coeff, phase.capacity)
    Id = Id[1:n, 1:n]
    
    #----- 2. Create different temperature fields -----#
    temperature_fields = Dict{String, Vector{Float64}}()
    
    # Case 1: Uniform temperature field (perfect symmetry)
    Tw_uniform = ones(n)  # Températures bulk uniformes
    Tgamma_uniform = zeros(n)  # Températures interfaciales nulles
    T_uniform = vcat(Tw_uniform, Tgamma_uniform)
    temperature_fields["Uniform"] = T_uniform
    
    # Case 2: Linear gradient along x (horizontal asymmetry)
    Tw_x_gradient = zeros(n)
    Tgamma_x_gradient = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            Tw_x_gradient[idx] = (x_nodes[i] - x0) / lx  # Normalized x coordinate
            # Tgamma reste à zéro
        end
    end
    T_x_gradient = vcat(Tw_x_gradient, Tgamma_x_gradient)
    temperature_fields["X Gradient"] = T_x_gradient
    
    # Case 3: Linear gradient along y (vertical asymmetry)
    Tw_y_gradient = zeros(n)
    Tgamma_y_gradient = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            Tw_y_gradient[idx] = (y_nodes[j] - y0) / ly  # Normalized y coordinate
            # Tgamma reste à zéro
        end
    end
    T_y_gradient = vcat(Tw_y_gradient, Tgamma_y_gradient)
    temperature_fields["Y Gradient"] = T_y_gradient
    
    # Case 4: Radial gradient centered at domain center
    Tw_radial = zeros(n)
    Tgamma_radial = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            x_pos = x_nodes[i]
            y_pos = y_nodes[j]
            dist_from_center = sqrt((x_pos - center[1])^2 + (y_pos - center[2])^2)
            # Normalized distance (1 at edges, 0 at center)
            Tw_radial[idx] = max(0, 1.0 - dist_from_center / (max(lx, ly)/2))
            # Tgamma reste à zéro
        end
    end
    T_radial = vcat(Tw_radial, Tgamma_radial)
    temperature_fields["Radial Gradient"] = T_radial
    
    # Case 5: Checkerboard pattern (mixed symmetry)
    Tw_checker = zeros(n)
    Tgamma_checker = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            Tw_checker[idx] = (i+j) % 2 == 0 ? 1.0 : 0.0
            # Tgamma reste à zéro
        end
    end
    T_checker = vcat(Tw_checker, Tgamma_checker)
    temperature_fields["Checkerboard"] = T_checker
    #----- 3. Calculate and analyze interface flux for each field -----#
    # Figure for comparing all fields
    fig_comparison = Figure(size=(1200, 900), title="Interface Flux Symmetry Comparison")
    
    # Process each temperature field
    for (idx, (field_name, T)) in enumerate(temperature_fields)
        println("Processing field: $field_name")
        
        # Extract internal and boundary temperatures
        Tₒ, Tᵧ = T[1:n], T[n+1:end]
        
        # Calculate interface flux
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Reshape to get flux per cell
        interface_flux_2d = reshape(interface_flux, (nx+1, ny+1))
        interface_flux_2d = reshape(interface_flux, (nx+1, ny+1)) + 
                              reshape(interface_flux, (nx+1, ny+1))'  # Symmetric flux

        # Compute volume Jacobian (to find cells affected by interface)
        volume_jacobian = compute_volume_jacobian(mesh, front)
        
        # Collect cells affected by interface
        cells_idx = []
        for i in 1:nx
            for j in 1:ny
                if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                    push!(cells_idx, (i, j))
                end
            end
        end
        
        # Extract flux values for cells along interface
        interface_cells_flux = [(i, j, interface_flux_2d[i,j]) for (i,j) in cells_idx]
        
        # Sort cells by angle around interface center
        sorted_cells = sort(interface_cells_flux, by=cell -> begin
            i, j, _ = cell
            # Use actual node coordinates for angle calculation
            x_pos = x_nodes[i]
            y_pos = y_nodes[j]
            atan(y_pos - center[2], x_pos - center[1])
        end)
        
        # Extract flux values in angular order
        flux_values = [flux for (_, _, flux) in sorted_cells]
        
        # Calculate symmetry metrics
        mean_flux = mean(flux_values)
        std_flux = std(flux_values)
        max_dev = maximum(abs.(flux_values .- mean_flux))
        symmetry_score = 1.0 - min(1.0, std_flux / (abs(mean_flux) + 1e-10))
        
        println("  Mean flux: $mean_flux")
        println("  Std flux: $std_flux")
        println("  Max deviation: $max_dev")
        println("  Symmetry score (0-1): $symmetry_score")
        
        # Create detailed plot for this field
        fig_detailed = Figure(size=(1000, 800), 
                             title="Interface Flux Analysis: $field_name")
        
        # Temperature field heatmap
        ax_temp = Axis(fig_detailed[1, 1], 
                      title="Temperature Field",
                      aspect=DataAspect())
        
        # Using mesh.nodes directly for heatmap
        temp_heatmap = heatmap!(ax_temp, x_nodes, y_nodes, reshape(Tₒ, (nx+1, ny+1))',
                               colormap=:thermal)
        Colorbar(fig_detailed[1, 2], temp_heatmap)
        
        # Add interface contour using mesh nodes for proper scaling
        # Create a finer grid for smooth contour
        x_fine = range(minimum(x_nodes), maximum(x_nodes), length=100)
        y_fine = range(minimum(y_nodes), maximum(y_nodes), length=100)
        phi = [sdf(front, x, y) for y in y_fine, x in x_fine]
        
        contour!(ax_temp, x_fine, y_fine, phi, levels=[0], 
                color=:black, linewidth=2)
        
        # Interface flux heatmap
        ax_flux = Axis(fig_detailed[2, 1], 
                      title="Interface Flux",
                      aspect=DataAspect())
        
        # Using mesh.nodes directly for heatmap
        flux_heatmap = heatmap!(ax_flux, x_nodes, y_nodes, interface_flux_2d', 
                              colormap=:viridis)
        Colorbar(fig_detailed[2, 2], flux_heatmap)
        
        # Add interface contour
        contour!(ax_flux, x_fine, y_fine, phi, levels=[0], 
                color=:black, linewidth=2)
        
        # Mark cells used for interface flux calculation
        scatter!(ax_flux, 
                [x_nodes[i] for (i,_,_) in sorted_cells], 
                [y_nodes[j] for (_,j,_) in sorted_cells], 
                color=:red, markersize=5)
        
        # Flux around interface plot
        ax_angular = Axis(fig_detailed[3, 1:2], 
                         title="Flux Around Interface (by angle)",
                         xlabel="Angle (radians)",
                         ylabel="Flux")
        
        # Plot flux vs angle - Generate angles based on actual number of cells
        θ_vals = range(0, 2π, length=length(flux_values))
        
        scatter!(ax_angular, θ_vals, flux_values, label="Raw flux")
        
        # Add horizontal line for mean
        hlines!(ax_angular, mean_flux, linewidth=2, 
               color=:red, linestyle=:dash, label="Mean flux")
        
        # Add standard deviation band
        band!(ax_angular, θ_vals, 
             fill(mean_flux - std_flux, length(θ_vals)), 
             fill(mean_flux + std_flux, length(θ_vals)), 
             color=(:blue, 0.2), label="±1 std dev")
        
        # Add symmetry score text
        text!(ax_angular, 0.1, maximum(flux_values) * 0.9,
             text="Symmetry Score: $(round(symmetry_score, digits=4))", 
             fontsize=14)
        
        Legend(fig_detailed[3, 1:2], ax_angular, position=:rt)
        
        # Save detailed plot
        save(joinpath(plot_dir, "flux_analysis_$(lowercase(replace(field_name, " " => "_"))).png"), 
            fig_detailed)
        
        # Add to comparison figure
        if idx <= 5  # Limit to 5 fields for comparison
            ax_comp = Axis(fig_comparison[div(idx-1,3)+1, mod(idx-1,3)+1],
                          title=field_name,
                          xlabel="Angle (radians)",
                          ylabel="Flux")
            
            # Generate angles based on actual number of cells
            θ_vals_comp = range(0, 2π, length=length(flux_values))
            
            scatter!(ax_comp, θ_vals_comp, flux_values)
            hlines!(ax_comp, mean_flux, linewidth=2, color=:red, linestyle=:dash)
            text!(ax_comp, 0.1, maximum(flux_values) * 0.9,
                 text="Sym Score: $(round(symmetry_score, digits=2))", 
                 fontsize=12)
        end
    end
    
    # Save comparison figure
    save(joinpath(plot_dir, "flux_symmetry_comparison.png"), fig_comparison)
    
    println("Analysis complete. Plots saved to: $plot_dir")
end

# Run the analysis
analyze_interface_flux_symmetry()