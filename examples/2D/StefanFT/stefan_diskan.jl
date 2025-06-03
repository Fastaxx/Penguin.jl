using Penguin
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Statistics

function smooth_displacements!(displacements::Vector{Float64}, 
                              markers::Vector{Tuple{Float64,Float64}}, 
                              is_closed::Bool=true,
                              smoothing_factor::Float64=0.5,
                              window_size::Int=2)
    n = length(displacements)
    if n <= 1
        return displacements  # Nothing to smooth with single marker
    end
    
    # Create a copy of original displacements
    original_displacements = copy(displacements)
    
    for i in 1:n
        # Calculate weighted sum of neighbors
        neighbor_sum = 0.0
        weight_sum = 0.0
        
        for j in -window_size:window_size
            if j == 0
                continue  # Skip the marker itself
            end
            
            # Handle wrapping for closed curves
            idx = i + j
            if is_closed
                # Apply modulo to wrap around for closed curves
                idx = mod1(idx, n)
            else
                # Skip out of bounds indices for open curves
                if idx < 1 || idx > n
                    continue
                end
            end
            
            # Calculate weight based on distance (closer markers have higher weight)
            distance = sqrt((markers[i][1] - markers[idx][1])^2 + 
                           (markers[i][2] - markers[idx][2])^2)
            weight = 1.0 / (distance + 1e-10)  # Avoid division by zero
            
            # Add weighted contribution
            neighbor_sum += weight * original_displacements[idx]
            weight_sum += weight
        end
        
        # Calculate weighted average
        if weight_sum > 0
            neighbor_avg = neighbor_sum / weight_sum
            
            # Apply smoothing with blend factor
            displacements[i] = (1.0 - smoothing_factor) * original_displacements[i] + 
                               smoothing_factor * neighbor_avg
        end
    end
    
    return displacements
end
function visualize_flux_and_interface(mesh, interface_flux_2d, volume_jacobian, 
                                     markers, iter=0, updated_markers=nothing)
    # Créer la figure
    fig = Figure(size=(1000, 800))
    
    # Extraire les coordonnées du maillage
    nx, ny = size(interface_flux_2d) .- 1
    x0, y0 = mesh.nodes[1][1], mesh.nodes[2][1]
    lx, ly = mesh.nodes[1][end] - x0, mesh.nodes[2][end] - y0
    
    # Grille pour le plotting
    x_faces = range(x0, x0+lx, nx+1)
    y_faces = range(y0, y0+ly, ny+1)
    
    # Axe principal
    ax = Axis(fig[1, 1], aspect=DataAspect(),
             title="Interface Flux & Jacobian Cells (Iter $iter)",
             xlabel="x", ylabel="y")
    
    # 1. Tracer le flux comme heatmap
    hm = heatmap!(ax, range(x0, x0+lx, nx), range(y0, y0+ly, ny), 
                 interface_flux_2d[1:end-1, 1:end-1]', colormap=:plasma)
    Colorbar(fig[1, 2], hm, label="Interface Flux")
    
    # 2. Marquer les cellules actives (avec Jacobien)
    active_cells = [(i,j) for (i,j) in keys(volume_jacobian) if !isempty(volume_jacobian[(i,j)])]
    for (i,j) in active_cells
        poly!(ax, [
            Point2f(x_faces[i], y_faces[j]),
            Point2f(x_faces[i+1], y_faces[j]),
            Point2f(x_faces[i+1], y_faces[j+1]),
            Point2f(x_faces[i], y_faces[j+1])
        ], color=(:green, 0.2), strokewidth=0)
    end
    
    # 3. Tracer l'interface initiale
    lines!(ax, first.(markers), last.(markers), 
          color=:blue, linewidth=2, label="Initial Interface")
    scatter!(ax, first.(markers), last.(markers), 
            color=:blue, markersize=6)
    
    # 4. Tracer l'interface mise à jour si disponible
    if updated_markers !== nothing
        lines!(ax, first.(updated_markers), last.(updated_markers), 
              color=:red, linewidth=2, label="Updated Interface")
        scatter!(ax, first.(updated_markers), last.(updated_markers), 
                color=:red, markersize=6)
    end
    
    # Legend
    Legend(fig[2, 1:2], ax, orientation=:horizontal)
    
    # Stats section - Plot the flux along the circle
    if length(markers) > 2
        # Get angle for each marker
        angles = Float64[]
        flux_values = Float64[]
        
        for (idx, (x, y)) in enumerate(markers[1:end-1])  # Skip last duplicated point
            θ = atan(y, x)  # atan2 equivalent
            push!(angles, θ)
            
            # Find closest grid point and get flux
            i = clamp(round(Int, (x - x0) / (lx/nx)) + 1, 1, nx)
            j = clamp(round(Int, (y - y0) / (ly/ny)) + 1, 1, ny)
            push!(flux_values, interface_flux_2d[i, j])
        end
        
        # Sort by angle for proper plotting
        sorted_idx = sortperm(angles)
        sorted_angles = angles[sorted_idx]
        sorted_flux = flux_values[sorted_idx]
        
        # Plot flux vs angle
        ax_polar = Axis(fig[3, 1:2], xlabel="Angle (radians)", 
                       ylabel="Interface Flux", 
                       title="Flux Distribution Around Interface")
        
        lines!(ax_polar, sorted_angles, sorted_flux, linewidth=2)
        scatter!(ax_polar, sorted_angles, sorted_flux, markersize=6)
        
        # Add average line
        avg_flux = mean(sorted_flux)
        hlines!(ax_polar, avg_flux, color=:red, linestyle=:dash, 
               label="Average: $(round(avg_flux, digits=5))")
        
        # Add stats
        std_flux = std(sorted_flux)
        Label(fig[4, :], "Flux Statistics: Mean = $(round(avg_flux, digits=5)), " * 
                        "Std = $(round(std_flux, digits=5)), " * 
                        "Max/Min Ratio = $(round(maximum(sorted_flux)/minimum(sorted_flux), digits=3))", 
             fontsize=12)
    end
    
    return fig
end
"""
Test d'un unique pas de temps pour le problème de Stefan
- Utilise la température analytique
- Calcule le flux d'interface avec G, H, et W!
- Résout la condition de Stefan avec l'algorithme de Gauss-Newton
"""
function test_analytical_stefan_first_step()
    println("Test du premier pas de temps avec température analytique et flux calculé numériquement")
    
    # Paramètres du problème
    ρL = 1.0
    
    # Solution parameter from Almgren [75]
    Λ = 1.56
    T_infinity = -0.5
    println("Solution parameter Λ = $Λ")
    
    # Time parameters
    t0 = 1.0  # Initial time
    Δt = 0.1 # Time step
    t = t0    # Start simulation at t0, not at t=0 
    timestep = 1  # Define timestep for visualization
    
    # Calculate initial radius from similarity solution
    initial_radius = Λ * sqrt(t0)
    println("Initial radius = $initial_radius")
    
    # IMPORTANT: Increase mesh domain to properly contain the interface
    nx, ny = 64, 64  # More cells for better resolution
    lx, ly = 4.0, 4.0  # Larger domain to fit circle with radius ~1.56
    x0, y0 = -2.0, -2.0
    Δx, Δy = lx/nx, ly/ny  # Cell sizes
    
    # Create mesh
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    println("Mesh domain: [$(x0), $(x0+lx)] × [$(y0), $(y0+ly)]")
    
    # Function for exact radius
    function exact_radius(t)
        return Λ*sqrt(t)
    end
    
    # Fonction de température analytique from E1 function
    function temperature(r, t)
        s = r / exact_radius(t)
        
        if s <= 1.0  # Inside solid (r < R(t))
            return 1.0

        else  # In liquid (r > R(t))
            # Solution for 2D Stefan: T = T∞ * (1 - F2(s)/F2(S))
            return T_infinity * (1.0 - F_similarity(s) / F_similarity(Λ))
        end
    end
    
    function F_similarity(s)
        return expint(s^2/4)  # E₁ function from SpecialFunctions
    end

    
    # Create front tracker with circle at origin
    front = FrontTracker()
    create_circle!(front, 0.0, 0.0, initial_radius, 30)
    
    # Store initial markers and compute normals
    markers_initial = get_markers(front)
    normals = compute_marker_normals(front, markers_initial)
    n_markers = length(markers_initial) - (front.is_closed ? 1 : 0)
    println("Nombre de marqueurs: $n_markers")
    
    # DEBUG: Check the position of some markers
    println("Sample markers positions:")
    for i in [1, 25, 50, 75, 100]
        if i <= length(markers_initial)
            println("  Marker $i: $(markers_initial[i])")
        end
    end

    # Plot initial markers, temperature.
    fig_initial = Figure(size=(800, 600))
    ax_initial = Axis(fig_initial[1, 1], 
                      title="Initial Markers and Temperature",
                      xlabel="x", ylabel="y",
                      aspect=DataAspect())
    # Plot initial markers
    x_initial = [m[1] for m in markers_initial]
    y_initial = [m[2] for m in markers_initial]
    scatter!(ax_initial, x_initial, y_initial, 
             color=:blue, markersize=5, label="Initial Markers")
    # Plot initial temperature field
    x_range = range(x0, stop=x0 + lx, length=100)
    y_range = range(y0, stop=y0 + ly, length=100)
    phi_initial = [temperature(sqrt(x^2 + y^2), t0) for y in y_range, x in x_range]
    c = contourf!(ax_initial, x_range, y_range, phi_initial, 
              levels=20, colormap=:viridis, label="Initial Temperature")
    Colorbar(fig_initial[1, 2], c, label="Temperature")
    println("Initial markers and temperature field plotted.")

    display(fig_initial)

    
    # Calculer la jacobienne de volume pour le maillage
    volume_jacobian = compute_volume_jacobian(mesh, front)
    
    # DEBUG: Check if volume Jacobian has entries
    affected_cells = 0
    for i in 1:nx
        for j in 1:ny
            if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                affected_cells += 1
            end
        end
    end
    println("Number of cells affected by interface: $affected_cells")
    
    if affected_cells == 0
        println("ERROR: No cells are affected by the interface!")
        println("Try adjusting the mesh size or interface position.")
        # Optional: Visualize the issue
        fig_debug = Figure(size=(800, 600))
        ax_debug = Axis(fig_debug[1, 1], aspect=DataAspect())
        
        # Plot mesh grid
        for i in 1:nx+1
            x_line = x0 + (i-1) * (lx/nx)
            lines!(ax_debug, [x_line, x_line], [y0, y0+ly], color=:gray, alpha=0.5)
        end
        for j in 1:ny+1
            y_line = y0 + (j-1) * (ly/ny)
            lines!(ax_debug, [x0, x0+lx], [y_line, y_line], color=:gray, alpha=0.5)
        end
        
        # Plot interface
        x_front = [m[1] for m in markers_initial]
        y_front = [m[2] for m in markers_initial]
        lines!(ax_debug, x_front, y_front, color=:red, linewidth=2)
        
        display(fig_debug)
        return fig_debug, zeros(n_markers), Float64[]
    end
    
     # Parameters for Newton algorithm
    max_iter = 100
    tol = 1e-6
    reltol = 1e-6
    α = 1.0
    residual_norm_history = Float64[]
    
    # Initialize displacement vector
    displacements = zeros(n_markers)
    
    # Define the body function - Fix unused parameter
    function body(x, y, t_local)
        return -sdf(front, x, y)
    end
    
    # Create space-time mesh and capacity
    t_next = t + Δt
    time_interval = [t, t_next]
    STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
    capacity = Capacity(body, STmesh; compute_centroids=false)
    
    # Create diffusion operators
    operator = DiffusionOps(capacity)
    
    # Source term and conductivity - Fix unused parameters
    f = (_x,_y,_z,_t) -> 0.0  # Prefix unused parameters with _
    K = (_x,_y,_z) -> 1.0     # Prefix unused parameters with _
    
    # Create phase
    phase = Phase(capacity, operator, f, K)
    
    # Total number of nodes
    n = (nx+1) * (ny+1)
    cap_index = 3  # For 2D problem
            # 1. Calculer le champ de température analytique au temps t+Δt
        Tₒ = zeros(n)
        Tᵧ = zeros(n)
        Δx, Δy = mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]
        for i in 1:nx
            for j in 1:ny
                idx = (i-1)*ny + j
                x = x0 + (i-0.5) * Δx
                y = y0 + (j-0.5) * Δy
                
                r = sqrt(x^2 + y^2)  # Distance à l'origine
                Tₒ[idx] = temperature(r, t)  # Temps t+Δt
                Tᵧ[idx] = temperature(r, t)  # Temps t+Δt, même température pour le problème de Stefan
            end
        end
        
    # Boucle d'itérations Gauss-Newton
    for iter in 1:max_iter

        
        # 2. Calculer le flux à l'interface à l'aide des opérateurs de diffusion
        W! = operator.Wꜝ[1:n, 1:n]
        G = operator.G[1:n, 1:n]
        H = operator.H[1:n, 1:n]
        Id = build_I_D(operator, phase.Diffusion_coeff, capacity)
        Id = Id[1:n, 1:n]
        
        # Calculer le flux d'interface
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Reshape en 2D pour faciliter l'accès par indice
        interface_flux_2d = reshape(interface_flux, (nx+1, ny+1))

        # Plot interface flux and jacobian for debugging
        # Après le calcul de interface_flux_2d
        flux_viz_fig = visualize_flux_and_interface(mesh, interface_flux_2d, volume_jacobian, 
                                                  markers_initial, iter)
        display(flux_viz_fig)
        
        # 3. Construire le système aux moindres carrés
        row_indices = Int[]
        col_indices = Int[]
        values = Float64[]
        cells_idx = []
        
        # Pré-calculer les cellules affectées
        for i in 1:nx
            for j in 1:ny
                if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                    push!(cells_idx, (i, j))
                end
            end
        end
        
        # Nombre d'équations dans notre système
        m = length(cells_idx)
        println("Iteration $iter: $m équations pour $n_markers inconnues")
        
        # Construire la matrice jacobienne pour les changements de volume
        for (eq_idx, (i, j)) in enumerate(cells_idx)
            for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                if 0 <= marker_idx < n_markers
                    push!(row_indices, eq_idx)
                    push!(col_indices, marker_idx + 1)  # Indexation 1-based pour Julia
                    push!(values, ρL * jac_value)
                end
            end
        end
        
        # Créer la matrice jacobienne J
        J = sparse(row_indices, col_indices, values, m, n_markers)
        
        # Extraire les matrices de capacité
        V_matrices = capacity.A[cap_index]
        Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]  # Volumes au temps t+Δt
        Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]  # Volumes au temps t
        
        # Calculer le vecteur résidu F
        F = zeros(m)
        for (eq_idx, (i, j)) in enumerate(cells_idx)
            # Obtenir le changement de volume souhaité basé sur le flux d'interface
            volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
            F[eq_idx] = ρL * volume_change - interface_flux_2d[i,j]
        end
        
        # Résoudre le système avec régularisation
        JTJ = J' * J
        
        # Diagnostic du système
        used_columns = unique(col_indices)
        println("  Info matrice: size(J)=$(size(J)), marqueurs utilisés: $(length(used_columns))/$n_markers")
        
        # Vérifier si JTJ est singulière et gérer en conséquence
        newton_step = zeros(n_markers)
        try
            # Ajouter un terme de régularisation
            λ = 1e-1  # Paramètre de régularisation
            reg_JTJ = JTJ #+ λ * diagm(0 => diag(JTJ))  # λ est un paramètre de régularisation
            newton_step = reg_JTJ \ (J' * F)
        catch e
            if isa(e, SingularException)
                println("  JTJ est singulière, utilisation de SVD")
                
                # Utiliser SVD pour résoudre
                F_svd = svd(Matrix(JTJ))
                tol_svd = eps(Float64) * max(size(JTJ)...) * maximum(F_svd.S)
                S_inv = [s > tol_svd ? 1/s : 0.0 for s in F_svd.S]
                
                # Calculer la solution par pseudo-inverse
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            else
                rethrow(e)
            end
        end
        
        # Mise à jour des déplacements avec sous-relaxation
        displacements -= α * newton_step 
        println("  Displacements updated: $(displacements)")

        # check which markers number have a displacement equal to NaN or zero
        for i in 1:n_markers
            if isnan(displacements[i]) || displacements[i] == 0.0
                println("Maker $i has NaN or zero displacement, resetting to zero.")
            end
        end

        smooth_displacements!(displacements, markers_initial, front.is_closed, 1.0, 3)


        # Check which markers number don't have a displacement
        if any(isnan, displacements) || any(iszero, displacements)
            # If any displacement is NaN or zero, reset all displacements to zero
            println("  Attention: des déplacemens sont NaN ou zéro, réinitialisation des déplacements.")
        end
        
        # Calculer la norme du résidu pour vérifier la convergence
        residual_norm = norm(F)
        push!(residual_norm_history, residual_norm)
        
        println("  Iteration $iter | Residual = $residual_norm | Max disp = $(maximum(abs.(displacements))) | Min disp = $(minimum(abs.(displacements)))")
        
        # Vérifier la convergence
        if residual_norm < tol 
            println("  Converged after $iter iterations with residual $residual_norm")
            break
        end
        
        # Mise à jour des positions des marqueurs
        new_markers = copy(markers_initial)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers_initial[i][1] + displacements[i] * normal[1],
                markers_initial[i][2] + displacements[i] * normal[2]
            )
        end
        
        # Si l'interface est fermée, mettre à jour le marqueur dupliqué à la fin
        if front.is_closed && markers_initial[1] == markers_initial[end]
            new_markers[end] = new_markers[1]
        end
        
        # Créer un nouvel objet front tracking avec les marqueurs mis à jour
        updated_front = FrontTracker(new_markers, front.is_closed)

        # Mettre à jour la jacobienne de volume avec la position actualisée de l'interface
        volume_jacobian = compute_volume_jacobian(mesh, updated_front)

        # Vérifier si volume Jacobian a des entrées
        affected_cells = 0
        for i in 1:nx
            for j in 1:ny
                if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                    affected_cells += 1
                end
            end
        end
        println("  After update: Cells affected by interface: $affected_cells")

        if affected_cells == 0
            println("  WARNING: No cells are affected by updated interface! Retaining original markers.")
            # Revert to original markers if no cells are affected
            displacements = zeros(n_markers)  # Reset displacements to zero
            continue  # Skip to next iteration
        end
        
        # Créer une nouvelle fonction body pour l'interpolation - Fix unused parameter
        function updated_body(x, y, t_local)
            # Temps normalisé dans [0,1]
            τ = (t_local - t) / Δt
            
            # Interpolation linéaire entre les SDFs
            sdf1 = -sdf(front, x, y)
            sdf2 = -sdf(updated_front, x, y)
            return (1-τ) * sdf1 + τ * sdf2
        end
        
        # Plot the sdf for debugging
        try
            # Create output directory if it doesn't exist
            debug_dir = joinpath(pwd(), "sdf_debug")
            if !isdir(debug_dir)
                mkdir(debug_dir)
            end
            
            # Get domain bounds from mesh
            x_min, x_max = extrema(mesh.nodes[1])
            y_min, y_max = extrema(mesh.nodes[2])
            
            # Generate grid points
            x_range = range(x_min, x_max, length=100)
            y_range = range(y_min, y_max, length=100)
            
            # Calculate SDF values
            phi_original = [sdf(front, x, y) for y in y_range, x in x_range]
            phi_updated = [sdf(updated_front, x, y) for y in y_range, x in x_range]
            
            # Create figure with expanded size for multiple plots
            fig = Figure(size=(1200, 1000))
            
            # 2D contour comparison (top row)
            ax = Axis(fig[1, 1:2], 
                     title="SDF Comparison - Timestep $(timestep), Iteration $(iter)",
                     xlabel="x", 
                     ylabel="y",
                     aspect=DataAspect())
            
            # Plot SDF contours
            hm = contourf!(ax, x_range, y_range, phi_original, 
                          levels=20)
            
            # Zero contours for both interfaces
            contour!(ax, x_range, y_range, phi_original, 
                   levels=[0], 
                   color=:black, 
                   linewidth=2,
                   label="Original Interface")
            
            contour!(ax, x_range, y_range, phi_updated, 
                   levels=[0], 
                   color=:red, 
                   linewidth=2,
                   label="Updated Interface")
            
            # Original markers
            scatter!(ax, 
                   first.(markers_initial), # Use markers_initial instead of undefined markers
                   last.(markers_initial),  # Use markers_initial instead of undefined markers
                   color=:black, 
                   markersize=7, 
                   label="Original Markers")
            
            # Updated markers
            scatter!(ax, 
                   first.(new_markers), 
                   last.(new_markers), 
                   color=:red, 
                   markersize=9,
                   marker=:cross,
                   strokewidth=1.5,
                   label="Updated Markers")
            
            # Add colorbar and legend for 2D plot
            Colorbar(fig[1, 3], hm, label="Signed Distance")
            Legend(fig[2, 1:3], ax, orientation=:horizontal)
            
            # 3D surface plots for SDFs - Fix unused variables
            # Instead of creating unused X and Y variables, use them directly in the surface calls
            
            # 3D plot for SDF at t_n (original)
            ax3d_1 = Axis3(fig[3, 1], 
                          title="SDF at t_n",
                          xlabel="x", ylabel="y", zlabel="ϕ",
                          aspect=(1, 1, 0.5))
            
            # Use _ to capture the return value but not create unused variable
            _ = surface!(ax3d_1, 
                      [x for x in x_range, y in y_range],  # X grid directly inline
                      [y for x in x_range, y in y_range],  # Y grid directly inline
                      phi_original, 
                      colormap=:viridis,
                      shading=false)
            
            # Add zero level contour on 3D plot
            contour3d!(ax3d_1, x_range, y_range, phi_original, 
                      levels=[0], 
                      color=:black,
                      linewidth=3)
            
            # 3D plot for SDF at t_{n+1} (updated)
            ax3d_2 = Axis3(fig[3, 2], 
                          title="SDF at t_{n+1}",
                          xlabel="x", ylabel="y", zlabel="ϕ",
                          aspect=(1, 1, 0.5))
            
            # Use surf2 since it's used in Colorbar below
            surf2 = surface!(ax3d_2, 
                           [x for x in x_range, y in y_range],  # X grid directly inline
                           [y for x in x_range, y in y_range],  # Y grid directly inline
                           phi_updated, 
                           colormap=:viridis,
                           shading=false)
            
            # Add zero level contour on 3D plot
            contour3d!(ax3d_2, x_range, y_range, phi_updated,
                      levels=[0], 
                      color=:red, 
                      linewidth=3)
            
            # Add colorbars for 3D plots
            Colorbar(fig[3, 3], surf2, label="Signed Distance")
            
            # Save the figure
            filename = joinpath(debug_dir, "sdf_timestep_$(timestep)_iteration_$(iter).png")
            save(filename, fig)
            
            println("SDF visualization saved to $filename")
        catch e
            println("Failed to create SDF visualization: $e")
            println("Error: ", e)
            # Print the traceback
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end

        # Mettre à jour le maillage spatio-temporel et la capacité
        STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
        capacity = Capacity(updated_body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, f, K)

        # Update the front tracker with new markers
        front = updated_front
    end
    
    # Calculer les positions finales des marqueurs
    final_markers = copy(markers_initial)
    for i in 1:n_markers
        normal = normals[i]
        final_markers[i] = (
            markers_initial[i][1] + displacements[i] * normal[1],
            markers_initial[i][2] + displacements[i] * normal[2]
        )
    end
    
    # Si l'interface est fermée, mettre à jour le marqueur dupliqué
    if front.is_closed && markers_initial[1] == markers_initial[end]
        final_markers[end] = final_markers[1]
    end
    
    # Visualiser les résultats
    fig = visualize_results(markers_initial, final_markers, residual_norm_history, t+Δt, Λ)
    
    return fig, displacements, residual_norm_history
end

"""
Visualise les résultats du pas de temps
"""
function visualize_results(initial_markers, final_markers, residuals, final_time, Λ)
    fig = Figure(size=(1000, 800))
    
    # Plot de l'interface
    ax1 = Axis(fig[1, 1], title="Interface Position", 
              xlabel="x", ylabel="y", 
              aspect=DataAspect())
    
    # Extraire les coordonnées x et y des marqueurs initiaux et finals
    x_initial = [m[1] for m in initial_markers]
    y_initial = [m[2] for m in initial_markers]
    x_final = [m[1] for m in final_markers]
    y_final = [m[2] for m in final_markers]
    
    # Tracer l'interface initiale
    lines!(ax1, x_initial, y_initial, 
          color=:blue, linewidth=2, label="Initial")
    scatter!(ax1, x_initial, y_initial, 
           color=:blue, markersize=5)
    
    # Tracer l'interface finale
    lines!(ax1, x_final, y_final, 
          color=:red, linewidth=2, label="Final")
    scatter!(ax1, x_final, y_final, 
           color=:red, markersize=5)
    
    # Tracer la solution analytique
    theta = range(0, 2π, length=100)
    r_analytical = Λ*sqrt(final_time)
    x_analytical = r_analytical .* cos.(theta)
    y_analytical = r_analytical .* sin.(theta)
    
    lines!(ax1, x_analytical, y_analytical, 
          color=:green, linestyle=:dash, 
          linewidth=2, label="Analytical")
    
    # Calculer l'erreur
    r_final = [sqrt(x^2 + y^2) for (x,y) in final_markers]
    r_avg = mean(r_final[1:end-1])  # Exclure le point dupliqué
    rel_error = abs(r_avg - r_analytical) / r_analytical
    
    # Ajouter une légende avec l'erreur
    error_text = "Relative error: $(round(rel_error * 100, digits=4))%"
    Legend(fig[1, 2], ax1, title=error_text)
    
    # Plot de la convergence
    ax2 = Axis(fig[2, 1:2], 
              title="Residual Norm Convergence", 
              xlabel="Iteration", ylabel="Residual Norm",
              yscale=log10)
    
    lines!(ax2, 1:length(residuals), residuals, 
          linewidth=2, color=:purple)
    scatter!(ax2, 1:length(residuals), residuals, 
           markersize=6, color=:purple)
    
    # Ajouter des étiquettes avec les informations du problème
    Label(fig[0, :], "Stefan Problem First Timestep: Λ=$Λ, t_final=$final_time",
         fontsize=16)
    
    return fig
end

# Exécuter le test et afficher les résultats
fig, displacements, residuals = test_analytical_stefan_first_step()
display(fig)
save("analytical_stefan_first_step.png", fig)