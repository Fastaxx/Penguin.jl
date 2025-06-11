function compute_volume_jacobian(mesh::Penguin.Mesh{2}, front::FrontTracker, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Call Julia function directly
    return compute_volume_jacobian(front, x_faces, y_faces, epsilon)
end


"""
Smooth marker displacements using weighted averaging of neighbors
This helps maintain interface regularity and stability

Parameters:
- displacements: Vector of displacement values to smooth
- markers: Vector of marker positions (Tuple{Float64,Float64})
- is_closed: Boolean indicating if the interface is a closed curve
- smoothing_factor: Weight given to neighbor values (0.0-1.0)
- window_size: Number of neighbors to consider on each side
"""
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

function StefanMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])    
    if scheme == "CN"
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "CN")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "CN")
    else # BE
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, "BE")
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, "BE")
    end
    
    BC_border_mono!(s.A, s.b, bc_b, mesh)
    
    s.x = Tᵢ
    return s
end

function solve_StefanMono2D!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{2}, scheme::String; 
                            method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            jacobian_epsilon=1e-6, smooth_factor=0.5, window_size=10, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving Stefan problem with Front Tracking (Julia implementation):")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")

    # Unpack parameters
    max_iter = Newton_params[1]
    tol = Newton_params[2]
    reltol = Newton_params[3]
    α = Newton_params[4]
    
    # Extract interface flux parameter
    ρL = ic.flux.value
    
    # Initialize tracking variables
    t = Tₛ
    residuals = Dict{Int, Vector{Float64}}()
    xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Tuple{Float64, Float64}[]
    push!(timestep_history, (t, Δt))
    position_increments = Dict{Int, Vector{Float64}}()  # Nouveau dictionnaire pour les incréments de position

    
    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    # Create the 1D or 2D indices
    nx, ny, _ = dims
    n = nx * ny  # Total number of nodes in the mesh
    
    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    
    # Store initial interface position
    xf_log[1] = markers
    
    # Boucle principale pour tous les pas de temps
    timestep = 1
    phase_2d = phase  # Initialize phase_2d for later use
    while t < Tₑ
        # Affichage du pas de temps actuel
        if timestep == 1
            println("\nFirst time step: t = $(round(t, digits=6))")
        else
            println("\nTime step $(timestep), t = $(round(t, digits=6))")
        end

        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Update time for this step
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        time_interval = [tₙ, tₙ₊₁]
        
        # Calculate total number of markers (excluding duplicated closing point if interface is closed)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]
        position_increment_history = Float64[]  # Nouveau vecteur pour les incréments de position

        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            Vₙ₊₁_matrix = diag(Vₙ₊₁_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ_matrix = diag(Vₙ_matrix)  # Convert to diagonal matrix for easier handling
            Vₙ₊₁_matrix = reshape(Vₙ₊₁_matrix, (nx, ny))  # Reshape to 2D matrix
            Vₙ_matrix = reshape(Vₙ_matrix, (nx, ny))  # Reshape to 2D matrix
            
            # 2. Calculate the interface flux
            W! = phase.operator.Wꜝ[1:end÷2, 1:end÷2]
            G = phase.operator.G[1:end÷2, 1:end÷2]
            H = phase.operator.H[1:end÷2, 1:end÷2]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:end÷2, 1:end÷2]  # Adjust for 2D case
            
            Tₒ, Tᵧ = Tᵢ[1:end÷2], Tᵢ[end÷2+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell - IMPORTANT: use the same reshape consistently
            interface_flux_2d = reshape(interface_flux, (nx, ny)) #+ reshape(interface_flux, (nx, ny))'
            
            """
            # Debug visualizations
            if timestep <= 2 || mod(timestep, 5) == 0  # Limit visualizations
                # Plot temperature field
                fig = Figure()
                ax = Axis(fig[1, 1], title="Temperature Field", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, reshape(Tₒ, (nx, ny)), colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Temperature")
                display(fig)
                
                # Plot interface flux
                fig = Figure()
                ax = Axis(fig[1, 1], title="Interface Flux", xlabel="x", ylabel="y", aspect=DataAspect())
                hm = heatmap!(ax, interface_flux_2d, colormap=:viridis)
                Colorbar(fig[1, 2], hm, label="Interface Flux")
                display(fig)
                
                if iter > 1  # Only when we have updated markers
                    # Calculate updated marker positions
                    updated_markers = Vector{Tuple{Float64, Float64}}(undef, length(markers))
                    
                    # Use eachindex instead of 1:length()
                    for (i, marker) in enumerate(markers)
                        if i <= length(displacements) && i <= length(normals)  # Check bounds for both arrays
                            updated_markers[i] = (
                                marker[1] + displacements[i] * normals[i][1],
                                marker[2] + displacements[i] * normals[i][2]
                            )
                        else
                            updated_markers[i] = marker  # Keep the same if no normal or displacement
                        end

                        if front.is_closed && length(updated_markers) > 1
                            # Ensure the last marker matches the first for closed curves
                            updated_markers[end] = updated_markers[1]
                        end
                    end
                    
                    # Create the comparison plot
                    fig = Figure()
                    ax = Axis(fig[1, 1], title="Interface Evolution (Iteration $iter)", 
                              xlabel="x", ylabel="y", aspect=DataAspect())
                    
                    # Plot original markers
                    orig_x = [m[1] for m in markers]
                    orig_y = [m[2] for m in markers]
                    lines!(ax, orig_x, orig_y, color=:blue, linewidth=1.5, 
                           label="Original", linestyle=:dash)
                    scatter!(ax, orig_x, orig_y, color=:blue, markersize=3)
                    
                    # Plot updated markers
                    new_x = [m[1] for m in updated_markers]
                    new_y = [m[2] for m in updated_markers]
                    lines!(ax, new_x, new_y, color=:red, linewidth=1.5, 
                           label="Updated")
                    scatter!(ax, new_x, new_y, color=:red, markersize=3)
                    
                    # Add legend
                    axislegend(ax, position=:rt)
                    
                    display(fig)
                end
            end
            """

            # Compute volume Jacobian for the mesh
            volume_jacobian = compute_volume_jacobian(mesh, front, jacobian_epsilon)
            
            # 3. Build least squares system
            row_indices = Int[]
            col_indices = Int[]
            values = Float64[]
            cells_idx = []
            
            # Precompute affected cells and their indices for residual vector
            for i in 1:nx
                for j in 1:ny
                    if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                        push!(cells_idx, (i, j))
                    end
                end
            end
            
            # Number of equations (cells) in our system
            m = length(cells_idx)
            
            # Now build the Jacobian matrix for volume changes
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Handle each marker affecting this cell
                for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)  # 1-based indexing
                        # Volume Jacobian is multiplied by ρL to match the Stefan condition
                        push!(values, ρL * jac_value)
                    end
                end
            end
            
            # Create Jacobian matrix J for the system
            J = sparse(row_indices, col_indices, values, m, n_markers)
            
            # Calculate current residual vector F
            F = zeros(m)
            mismatches = 0
            total_cells = 0
            
            # Diagnostic arrays
            flux_nonzero = zeros(Bool, nx, ny)
            volume_nonzero = zeros(Bool, nx, ny)
            both_nonzero = zeros(Bool, nx, ny)
            
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Get the desired volume change based on interface flux
                volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
                flux = interface_flux_2d[i,j]
                
                # Record diagnostics
                total_cells += 1
                volume_nonzero[i,j] = abs(volume_change) > 1e-10
                flux_nonzero[i,j] = abs(flux) > 1e-10
                both_nonzero[i,j] = volume_nonzero[i,j] && flux_nonzero[i,j]
                
                # Detect mismatches (volume change but no flux or vice versa)
                if (volume_nonzero[i,j] && !flux_nonzero[i,j]) || (!volume_nonzero[i,j] && flux_nonzero[i,j])
                    mismatches += 1
                    if mismatches <= 5  # Limiter le nombre de messages
                        println("Mismatch at ($i,$j): volume_change = $volume_change, flux = $flux")
                    end
                end
                
                # F_i = ρL * volume_change - interface_flux
                F[eq_idx] = ρL * volume_change - flux
            end
            
            # Print diagnostic summary
            nonzero_vol = count(volume_nonzero)
            nonzero_flux = count(flux_nonzero)
            nonzero_both = count(both_nonzero)
            println("Cells with nonzero volume change: $nonzero_vol")
            println("Cells with nonzero interface flux: $nonzero_flux")
            println("Cells with both nonzero: $nonzero_both")
            println("Mismatches: $mismatches out of $total_cells cells")
            
            """
            # Visualization of matches/mismatches
            if iter == 1 || mismatches > 0
                mismatch_plot = Figure()
                ax1 = Axis(mismatch_plot[1, 1], title="Volume Change", aspect=DataAspect())
                heatmap!(ax1, [abs(Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]) for i=1:nx, j=1:ny], colormap=:viridis)
                
                ax2 = Axis(mismatch_plot[1, 2], title="Interface Flux", aspect=DataAspect()) 
                heatmap!(ax2, interface_flux_2d, colormap=:viridis)
                
                ax3 = Axis(mismatch_plot[2, 1:2], title="Mismatch Map", aspect=DataAspect())
                mismatch_map = zeros(Int, nx, ny)
                for i=1:nx, j=1:ny
                    if volume_nonzero[i,j] && !flux_nonzero[i,j]
                        mismatch_map[i,j] = 1  # Volume sans flux
                    elseif !volume_nonzero[i,j] && flux_nonzero[i,j]
                        mismatch_map[i,j] = 2  # Flux sans volume
                    elseif volume_nonzero[i,j] && flux_nonzero[i,j]
                        mismatch_map[i,j] = 3  # Les deux
                    end
                end
                heatmap!(ax3, mismatch_map, colormap=[:transparent, :red, :blue, :green])
                
                display(mismatch_plot)
            end
            """
            
            # Visualiser le résidual F sur la grille
            fig_residual = Figure()
            ax_residual = Axis(fig_residual[1, 1], 
                              title="Residual F (Iteration $iter)", 
                              xlabel="x", ylabel="y",
                              aspect=DataAspect())
            
            # Créer une matrice 2D pour représenter le résidual sur la grille
            residual_2d = zeros(nx, ny)
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                residual_2d[i, j] = F[eq_idx]
            end
            
            """
            # Tracer le heatmap du résidual
            hm_residual = heatmap!(ax_residual, residual_2d, colormap=:balance)
            Colorbar(fig_residual[1, 2], hm_residual, label="Residual value")
            
            # Ajouter les contours de l'interface pour référence
            markers_x = [m[1] for m in markers]
            markers_y = [m[2] for m in markers]
            lines!(ax_residual, markers_x, markers_y, color=:black, linewidth=1.5)
            
            display(fig_residual)
            """
            
            # 4. Implement the Gauss-Newton formula: X^{n+1} = X^n - (J^T J)^{-1} J^T F
            JTJ = J' * J
            
            # Diagnose the system
            used_columns = unique(col_indices)
            println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
            println("Used marker indices: $(length(used_columns)) of $n_markers")
            
            # Add regularization using JTJ diagonal (Levenberg-Marquardt style)
            reg_param = 1e-6
            diag_JTJ = diag(JTJ)  # Extract the diagonal of JTJ
            
            # Ensure diagonal elements aren't too small (protect against near-zero entries)
            min_diag = 1e-10 * maximum(diag_JTJ)
            for i in 1:length(diag_JTJ)
                if diag_JTJ[i] < min_diag
                    diag_JTJ[i] = min_diag
                end
            end
            
            # Apply Levenberg-Marquardt style regularization
            reg_JTJ = JTJ + reg_param * Diagonal(diag_JTJ)
            
            # Solve the system using robust method
            newton_step = zeros(n_markers)
            try
                newton_step = reg_JTJ \ (J' * F)
            catch e
                println("Matrix solver failed, using SVD as backup")
                # SVD-based pseudoinverse for robust solving
                F_svd = svd(Matrix(reg_JTJ))
                svd_tol = eps(Float64) * max(size(reg_JTJ)...) * maximum(F_svd.S)
                S_inv = [s > svd_tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Compute pseudoinverse solution
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            end
            # Calculer la norme de l'incrément de position avant de l'appliquer
            position_increment_norm = α * norm(newton_step)
            push!(position_increment_history, position_increment_norm)
            
            # Apply the step with adjustment factor
            displacements -= α * newton_step
            
            # For closed curves, match first and last displacement to ensure continuity
            if front.is_closed
                displacements[end] = displacements[1]
            end
            
            # Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, smooth_factor, window_size)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("Maximum displacement (after smoothing): $max_disp")
            
            # Optional: Limit maximum displacement if needed
            #max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.1
           # if max_disp > max_allowed
            #    scale = max_allowed / max_disp
           #     displacements .*= scale
           #     println("Scaling displacements by $scale to limit maximum displacement")
            #end
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("Iteration $iter | Residual = $residual_norm")
            
            # Check convergence
            if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("Converged after $iter iterations with residual $residual_norm and position increment $position_increment_norm")
                break
            end
            
            # 5. Update marker positions
            new_markers = copy(markers)
            for i in 1:n_markers
                normal = normals[i]
                new_markers[i] = (
                    markers[i][1] + displacements[i] * normal[1],
                    markers[i][2] + displacements[i] * normal[2]
                )
            end
            
            # If interface is closed, update the duplicated last marker
            if front.is_closed
                new_markers[end] = new_markers[1]
            end
            
            # Print mean radius for diagnostic
            if front.is_closed
                # Calculer le centre approximatif
                center_x = sum(m[1] for m in new_markers) / length(new_markers)
                center_y = sum(m[2] for m in new_markers) / length(new_markers)
                
                # Calculer le rayon moyen
                mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                
                # Afficher le rayon moyen
                println("Mean radius: $(round(mean_radius, digits=6))")
                
                # Vérifier la régularité
                std_radius = std([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers])
                if std_radius / mean_radius > 0.05
                    println("⚠️ Warning: Interface irregularity detected ($(round(100*std_radius/mean_radius, digits=2))% variation)")
                end
            end
            
            # 6. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            # 7. Create space-time level set for capacity calculation
            function body(x, y, t_local, _=0)
                # Normalized time in [0,1]
                τ = (t_local - tₙ) / Δt

                # Linear interpolation between SDFs
                sdf1 = -sdf(front, x, y)
                sdf2 = -sdf(updated_front, x, y)
                return (1-τ) * sdf1 + τ * sdf2
            end
            
            # 8. Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # 9. Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh)
            
            # 10. Update phase for next iteration
            phase = phase_updated

            body_2d(x,y,_=0) = body(x, y, tₙ₊₁)
            capacity_2d = Capacity(body_2d, mesh; compute_centroids=false)
            operator_2d = DiffusionOps(capacity_2d)
            phase_2d = Phase(capacity_2d, operator_2d, phase.source, phase.Diffusion_coeff)
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        position_increments[timestep] = position_increment_history  # Stocker l'historique des incréments de position

        
        # Update front with new marker positions
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        
        # If interface is closed, update the duplicated last marker
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        
        # Update front with new markers
        set_markers!(front, new_markers)
        
        # Store updated interface position
        xf_log[timestep+1] = new_markers
        
        # Store solution
        push!(s.states, s.x)
        
        println("Time: $(round(t, digits=6))")
        println("Max temperature: $(maximum(abs.(s.x)))")
        
        # Increment timestep counter
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history, phase_2d, position_increments
end
