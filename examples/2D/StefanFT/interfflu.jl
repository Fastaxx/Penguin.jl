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

# Add this function to your file
function analyze_direct_2d_flux_symmetry()
    # Output directory for plots
    plot_dir = joinpath(pwd(), "direct_2d_flux_analysis")
    if !isdir(plot_dir)
        mkdir(plot_dir)
    end
    
    println("Analyzing interface flux symmetry using direct 2D mesh (no space-time)...")
    
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
    
    # Create body function for mesh (direct SDF)
    body_func = (x, y,_=0) -> sdf(front, x, y)
    
    # Create capacity directly with 2D mesh (no space-time component)
    capacity = Capacity(body_func, mesh; compute_centroids=false)
    
    # Set up operators directly on 2D mesh
    diffusion_coeff = (_x,_y,_z) -> 1.0  # Thermal conductivity
    operator = DiffusionOps(capacity)
    
    # Get direct 2D operators for flux calculation
    # These are not reshaped since we're working directly in 2D
    n = (nx+1)*(ny+1)  # Size for temperature vectors
    W! = operator.Wꜝ
    G = operator.G
    H = operator.H
    Id = build_I_D(operator, diffusion_coeff, capacity)
    
    #----- 2. Create different temperature fields -----#
    temperature_fields = Dict{String, Vector{Float64}}()
    
    # Case 1: Uniform temperature field (perfect symmetry)
    Tw_uniform = ones(n)  # Bulk temperatures (uniform)
    Tgamma_uniform = zeros(n)  # Interface temperatures (zero)
    T_uniform = vcat(Tw_uniform, Tgamma_uniform)
    temperature_fields["Uniform"] = T_uniform
    
    # Case 2: Linear gradient along x (horizontal asymmetry)
    Tw_x_gradient = zeros(n)
    Tgamma_x_gradient = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            Tw_x_gradient[idx] = (x_nodes[i] - x0) / lx  # Normalized x coordinate
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
        end
    end
    T_radial = vcat(Tw_radial, Tgamma_radial)
    temperature_fields["Radial Gradient"] = T_radial
    
    # Figure for comparing all fields
    fig_comparison = Figure(size=(1200, 900), title="Direct 2D Interface Flux Comparison")
    
    # Process each temperature field
    for (idx, (field_name, T)) in enumerate(temperature_fields)
        println("Processing field: $field_name")
        
        # Extract internal and boundary temperatures
        Tₒ, Tᵧ = T[1:n], T[n+1:end]
        
        # Calculate interface flux directly (no space-time transformation needed)
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Get flux as 2D matrix
        interface_flux_2d = reshape(interface_flux, (nx+1, ny+1))
        
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
                             title="Direct 2D Interface Flux: $field_name")
        
        # Temperature field heatmap
        ax_temp = Axis(fig_detailed[1, 1], 
                      title="Temperature Field",
                      aspect=DataAspect())
        
        temp_heatmap = heatmap!(ax_temp, x_nodes, y_nodes, reshape(Tₒ, (nx+1, ny+1))',
                              colormap=:thermal)
        Colorbar(fig_detailed[1, 2], temp_heatmap)
        
        # Add interface contour
        x_fine = range(minimum(x_nodes), maximum(x_nodes), length=100)
        y_fine = range(minimum(y_nodes), maximum(y_nodes), length=100)
        phi = [sdf(front, x, y) for y in y_fine, x in x_fine]
        
        contour!(ax_temp, x_fine, y_fine, phi, levels=[0], 
                color=:black, linewidth=2)
        
        # Interface flux heatmap
        ax_flux = Axis(fig_detailed[2, 1], 
                      title="Interface Flux",
                      aspect=DataAspect())
        
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
        
        # Plot flux vs angle
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
        save(joinpath(plot_dir, "direct_2d_flux_$(lowercase(replace(field_name, " " => "_"))).png"), 
            fig_detailed)
        
        # Add to comparison figure
        if idx <= 4  # Limit to 4 fields for comparison
            ax_comp = Axis(fig_comparison[div(idx-1,2)+1, mod(idx-1,2)+1],
                          title=field_name,
                          xlabel="Angle (radians)",
                          ylabel="Flux")
            
            θ_vals_comp = range(0, 2π, length=length(flux_values))
            
            scatter!(ax_comp, θ_vals_comp, flux_values)
            hlines!(ax_comp, mean_flux, linewidth=2, color=:red, linestyle=:dash)
            text!(ax_comp, 0.1, maximum(flux_values) * 0.9,
                 text="Sym Score: $(round(symmetry_score, digits=2))", 
                 fontsize=12)
        end
    end
    
    # Save comparison figure
    save(joinpath(plot_dir, "direct_2d_flux_comparison.png"), fig_comparison)
    
    println("Direct 2D analysis complete. Plots saved to: $plot_dir")
end

# Call this at the end of your script to run both analyses

function compare_sdf_approaches()
    # Configuration du dossier de sortie pour les graphiques
    plot_dir = joinpath(pwd(), "sdf_comparison_analysis")
    if !isdir(plot_dir)
        mkdir(plot_dir)
    end
    
    println("Comparaison des approches de SDF pour le calcul de flux d'interface...")
    
    #----- 1. Configuration du maillage et paramètres -----#
    # Créer maillage
    nx, ny = 32, 32
    lx, ly = 16.0, 16.0
    x0, y0 = -8.0, -8.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Coordonnées du maillage
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Définir une interface circulaire
    center = (0.0, 0.0)
    radius = 4.04
    n_markers = 100
    
    # Créer des marqueurs pour l'approche FrontTracker
    markers = [(center[1] + radius*cos(θ), center[2] + radius*sin(θ)) 
               for θ in range(0, 2π, length=n_markers+1)]
    front = FrontTracker(markers, true)
    
    # Définir la SDF analytique 
    analytical_sdf(x, y, t=0) = sqrt((x-center[1])^2 + (y-center[2])^2) - radius
    
    # Paramètres d'interface
    ρL = 1.0  # Chaleur latente
    
    # Paramètres temporels
    Δt = 1.0
    time_interval = [0.0, Δt]
    
    # Créer les fonctions body pour les deux approches
    body_front_tracking = (x, y, t) -> -sdf(front, x, y)  # Négation pour convention intérieur/extérieur
    body_analytical = (x, y, t) -> -analytical_sdf(x, y, t)
    
    # Créer les deux maillages espace-temps et capacités
    STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
    
    # Capacités pour les deux approches
    capacity_ft = Capacity(body_front_tracking, STmesh; compute_centroids=false)
    capacity_analytical = Capacity(body_analytical, STmesh; compute_centroids=false)
    
    # Propriétés physiques communes
    diffusion_coeff = (_x,_y,_z) -> 1.0  # Conductivité thermique
    source_term = (_x,_y,_z,_t=0) -> 0.0  # Pas de source interne
    
    # Opérateurs pour les deux approches
    operator_ft = DiffusionOps(capacity_ft)
    operator_analytical = DiffusionOps(capacity_analytical)
    
    # Phases pour les deux approches
    phase_ft = Phase(capacity_ft, operator_ft, source_term, diffusion_coeff)
    phase_analytical = Phase(capacity_analytical, operator_analytical, source_term, diffusion_coeff)
    
    # Configuration pour le calcul du flux
    n = (nx+1)*(ny+1)
    
    # Récupérer les matrices pour le calcul du flux (FrontTracker)
    W_ft = phase_ft.operator.Wꜝ[1:n, 1:n]
    G_ft = phase_ft.operator.G[1:n, 1:n]
    H_ft = phase_ft.operator.H[1:n, 1:n]
    Id_ft = build_I_D(phase_ft.operator, diffusion_coeff, phase_ft.capacity)
    Id_ft = Id_ft[1:n, 1:n]
    
    # Récupérer les matrices pour le calcul du flux (Analytique)
    W_an = phase_analytical.operator.Wꜝ[1:n, 1:n]
    G_an = phase_analytical.operator.G[1:n, 1:n]
    H_an = phase_analytical.operator.H[1:n, 1:n]
    Id_an = build_I_D(phase_analytical.operator, diffusion_coeff, phase_analytical.capacity)
    Id_an = Id_an[1:n, 1:n]
    
    #----- 2. Créer des champs de température pour les tests -----#
    temperature_fields = Dict{String, Vector{Float64}}()
    
    # Cas 1: Champ uniforme (symétrie parfaite)
    Tw_uniform = ones(n)
    Tgamma_uniform = zeros(n)
    T_uniform = vcat(Tw_uniform, Tgamma_uniform)
    temperature_fields["Uniforme"] = T_uniform
    
    # Cas 2: Gradient linéaire selon x
    Tw_x_gradient = zeros(n)
    Tgamma_x_gradient = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            Tw_x_gradient[idx] = (x_nodes[i] - x0) / lx
        end
    end
    T_x_gradient = vcat(Tw_x_gradient, Tgamma_x_gradient)
    temperature_fields["Gradient X"] = T_x_gradient
    
    # Cas 3: Gradient radial
    Tw_radial = zeros(n)
    Tgamma_radial = zeros(n)
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (j-1)*(nx+1) + i
            x_pos = x_nodes[i]
            y_pos = y_nodes[j]
            dist_from_center = sqrt((x_pos - center[1])^2 + (y_pos - center[2])^2)
            Tw_radial[idx] = max(0, 1.0 - dist_from_center / (max(lx, ly)/2))
        end
    end
    T_radial = vcat(Tw_radial, Tgamma_radial)
    temperature_fields["Gradient Radial"] = T_radial
    
    #----- 3. Analyser et comparer les flux pour chaque champ -----#
    for (field_name, T) in temperature_fields
        println("Analyse du champ: $field_name")
        
        # Extraire les températures internes et à l'interface
        Tₒ, Tᵧ = T[1:n], T[n+1:end]
        
        # Calculer les flux avec les deux approches
        flux_ft = Id_ft * H_ft' * W_ft * G_ft * Tₒ + Id_ft * H_ft' * W_ft * H_ft * Tᵧ
        flux_analytical = Id_an * H_an' * W_an * G_an * Tₒ + Id_an * H_an' * W_an * H_an * Tᵧ
        
        # Mettre en forme 2D
        flux_ft_2d = reshape(flux_ft, (nx+1, ny+1))
        flux_analytical_2d = reshape(flux_analytical, (nx+1, ny+1))
        
        # Calculer la différence de flux
        flux_diff = flux_ft_2d - flux_analytical_2d
        
        # Créer une figure de comparaison
        fig = Figure(size=(1200, 800), title="Comparaison de flux: $field_name")
        
        # Champ de température
        ax_temp = Axis(fig[1, 1], title="Champ de température", aspect=DataAspect())
        temp_hm = heatmap!(ax_temp, x_nodes, y_nodes, reshape(Tₒ, (nx+1, ny+1))', colormap=:thermal)
        Colorbar(fig[1, 2], temp_hm, label="Température")
        
        # Tracer le contour de l'interface
        x_fine = range(minimum(x_nodes), maximum(x_nodes), length=100)
        y_fine = range(minimum(y_nodes), maximum(y_nodes), length=100)
        phi_ft = [sdf(front, x, y) for y in y_fine, x in x_fine]
        phi_an = [analytical_sdf(x, y) for y in y_fine, x in x_fine]
        
        contour!(ax_temp, x_fine, y_fine, phi_ft, levels=[0], color=:black, linewidth=2, linestyle=:solid, label="Front Tracking")
        contour!(ax_temp, x_fine, y_fine, phi_an, levels=[0], color=:red, linewidth=1, linestyle=:dash, label="Analytique")
        
        # Flux avec Front Tracking
        ax_ft = Axis(fig[2, 1], title="Flux (Front Tracking)", aspect=DataAspect())
        flux_ft_hm = heatmap!(ax_ft, x_nodes, y_nodes, flux_ft_2d', colormap=:viridis)
        Colorbar(fig[2, 2], flux_ft_hm, label="Flux")
        contour!(ax_ft, x_fine, y_fine, phi_ft, levels=[0], color=:black, linewidth=2)
        
        # Flux avec SDF Analytique
        ax_an = Axis(fig[2, 3], title="Flux (SDF Analytique)", aspect=DataAspect())
        flux_an_hm = heatmap!(ax_an, x_nodes, y_nodes, flux_analytical_2d', colormap=:viridis)
        Colorbar(fig[2, 4], flux_an_hm, label="Flux")
        contour!(ax_an, x_fine, y_fine, phi_an, levels=[0], color=:red, linewidth=2)
        
        # Différence de flux
        ax_diff = Axis(fig[3, 2], title="Différence de flux", aspect=DataAspect())
        diff_hm = heatmap!(ax_diff, x_nodes, y_nodes, flux_diff', colormap=:balance)
        Colorbar(fig[3, 3], diff_hm, label="Différence")
        contour!(ax_diff, x_fine, y_fine, phi_ft, levels=[0], color=:black, linewidth=2)
        contour!(ax_diff, x_fine, y_fine, phi_an, levels=[0], color=:red, linewidth=1, linestyle=:dash)
        
        # Calculer les statistiques sur les différences
        max_diff = maximum(abs.(flux_diff))
        mean_diff = mean(abs.(flux_diff))
        
        # Ajouter des statistiques à la figure
        Label(fig[3, 1], "Différence maximale: $(round(max_diff, digits=6))\nDifférence moyenne: $(round(mean_diff, digits=6))")
        
        # Légende pour l'interface
        Legend(fig[1, 3], ax_temp)
        
        # Sauvegarder la figure
        save(joinpath(plot_dir, "flux_comparison_$(lowercase(replace(field_name, " " => "_"))).png"), fig)
        
        # Analyse de la symétrie pour les deux méthodes
        # Identifier les cellules près de l'interface pour l'analyse angulaire
        volume_jacobian_ft = compute_volume_jacobian(mesh, front)
        
        cells_idx = []
        for i in 1:nx
            for j in 1:ny
                if haskey(volume_jacobian_ft, (i,j)) && !isempty(volume_jacobian_ft[(i,j)])
                    push!(cells_idx, (i, j))
                end
            end
        end
        
        # Extraire les valeurs de flux le long de l'interface
        interface_flux_ft = [(i, j, flux_ft_2d[i,j]) for (i,j) in cells_idx]
        interface_flux_an = [(i, j, flux_analytical_2d[i,j]) for (i,j) in cells_idx]
        
        # Trier par angle autour du centre
        sort_by_angle = cell -> begin
            i, j, _ = cell
            atan(y_nodes[j] - center[2], x_nodes[i] - center[1])
        end
        
        sorted_flux_ft = sort(interface_flux_ft, by=sort_by_angle)
        sorted_flux_an = sort(interface_flux_an, by=sort_by_angle)
        
        # Extraire les valeurs de flux triées
        flux_values_ft = [flux for (_, _, flux) in sorted_flux_ft]
        flux_values_an = [flux for (_, _, flux) in sorted_flux_an]
        
        # Figure pour la comparaison angulaire
        fig_angular = Figure(size=(900, 600), title="Flux le long de l'interface: $field_name")
        
        ax_angular = Axis(fig_angular[1, 1], 
                         title="Flux autour de l'interface",
                         xlabel="Angle (radians)",
                         ylabel="Flux")
        
        θ_vals = range(0, 2π, length=length(flux_values_ft))
        
        scatter!(ax_angular, θ_vals, flux_values_ft, color=:black, label="Front Tracking")
        scatter!(ax_angular, θ_vals, flux_values_an, color=:red, label="SDF Analytique")
        
        # Calculer et afficher les scores de symétrie
        std_ft = std(flux_values_ft)
        mean_ft = mean(flux_values_ft)
        sym_score_ft = 1.0 - min(1.0, std_ft / (abs(mean_ft) + 1e-10))
        
        std_an = std(flux_values_an)
        mean_an = mean(flux_values_an)
        sym_score_an = 1.0 - min(1.0, std_an / (abs(mean_an) + 1e-10))
        
        # Ajouter les scores à la figure
        text!(ax_angular, 0.1, maximum(vcat(flux_values_ft, flux_values_an)) * 0.9,
             text="Score de symétrie:\nFront Tracking: $(round(sym_score_ft, digits=4))\nSDF Analytique: $(round(sym_score_an, digits=4))",
             fontsize=12)
        
        Legend(fig_angular[1, 1], ax_angular, position=:rt)
        
        # Sauvegarder la comparaison angulaire
        save(joinpath(plot_dir, "angular_comparison_$(lowercase(replace(field_name, " " => "_"))).png"), fig_angular)
    end
    
    println("Analyse terminée. Graphiques sauvegardés dans: $plot_dir")
end

# Exécuter la fonction
compare_sdf_approaches()