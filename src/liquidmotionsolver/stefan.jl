function compute_volume_jacobian(mesh::Penguin.Mesh{2}, front::FrontTracker, epsilon::Float64=1e-6)
    # Extract mesh data
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Call Julia function directly
    return compute_volume_jacobian(front, x_faces, y_faces, epsilon)
end

function StefanMono2D(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary, Δt::Float64, Tᵢ::Vector{Float64}, mesh::AbstractMesh, scheme::String)
    println("Solver Creation:")
    println("- Stefan problem")
    println("- Monophasic problem")
    println("- Phase change with moving interface")
    println("- Unsteady problem")
    println("- Diffusion problem")
    
    s = Solver(Unsteady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])    
    s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i, scheme)
    s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity, phase.Diffusion_coeff, phase.source, bc_i, Tᵢ, Δt, 0.0, scheme)
    
    BC_border_mono!(s.A, s.b, bc_b, mesh)
    
    s.x = Tᵢ
    return s
end

"""
Crée une fonction body pour représenter l'interface en SDF
- front: l'interface actuelle
- updated_front: l'interface mise à jour (optionnel)
- tₙ: temps au début du pas de temps
- Δt: pas de temps
"""
function create_interface_body_function(front::FrontTracker, updated_front::Union{FrontTracker,Nothing}=nothing, 
                                       tₙ::Float64=0.0, Δt::Float64=1.0)
    if updated_front === nothing
        # Pas d'interpolation, utiliser uniquement front actuel
        return function(x, y, t_local, _=0)
            return sdf(front, x, y)
        end
    else
        # Interpolation entre deux interfaces
        return function(x, y, t_local, _=0)
            # Temps normalisé entre 0 et 1
            τ = (t_local - tₙ) / Δt
            
            # Interpolation linéaire entre les SDFs
            sdf1 = sdf(front, x, y)
            sdf2 = sdf(updated_front, x, y)
            return (1-τ) * sdf1 + τ * sdf2
        end
    end
end

function solve_StefanMono2D!(s::Solver, phase::Phase, front::FrontTracker, Δt::Float64, Tₛ::Float64, Tₑ::Float64, bc_b::BorderConditions, bc::AbstractBoundary, ic::InterfaceConditions, mesh::Penguin.Mesh{2}, scheme::String; 
                            adaptive_timestep=true, method=Base.:\, 
                            Newton_params=(100, 1e-6, 1e-6, 1.0),
                            cfl_target=0.5, Δt_min=1e-4, Δt_max=1.0, kwargs...)
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
    
    # Determine how many dimensions
    dims = phase.operator.size
    len_dims = length(dims)
    cap_index = len_dims

    nx, ny, nt = dims
    n = nx*ny

    # Store initial state
    Tᵢ = s.x
    push!(s.states, s.x)
    
    # Get initial interface markers
    markers = get_markers(front)
    normals = compute_marker_normals(front, markers)

    # Store initial interface position
    xf_log[1] = markers
    
    # Initialize variables for first time step
    timestep = 1
    println("\nFirst time step: t = $(round(t, digits=6))")
    
    # Create time interval for space-time mesh
    t += Δt
    tₙ = t - Δt
    tₙ₊₁ = t
    time_interval = [tₙ, tₙ₊₁]
    
    # Calculate total number of markers (excluding duplicated closing point if interface is closed)
    n_markers = length(markers) - (front.is_closed ? 1 : 0)
    
    # Initialize displacement vector and residual vector
    displacements = zeros(n_markers)
    residual_norm_history = Float64[]
    
    # Compute volume Jacobian for the mesh
    volume_jacobian = compute_volume_jacobian(mesh, front)
    
    # First time step: Gauss-Newton iterations to solve least squares problem
    for iter in 1:max_iter
        # 1. Solve temperature field with current interface position
        solve_system!(s; method=method, kwargs...)
        Tᵢ = s.x
        
        # Get capacity matrices
        V_matrices = phase.capacity.A[cap_index]
        Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
        Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
        
        # 2. Calculate the interface flux
        W! = phase.operator.Wꜝ[1:n, 1:n]
        G = phase.operator.G[1:n, 1:n]
        H = phase.operator.H[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:n, 1:n]
        
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Reshape to get flux per cell
        interface_flux = reshape(interface_flux, (nx, ny))
        
        # Plot the interface flux for debugging
        try
            fig_flux = Figure(size=(800, 600))
            ax_flux = Axis(fig_flux[1, 1], 
                           title="Interface Flux at Iteration $iter",
                           xlabel="x", ylabel="y",
                           aspect=DataAspect())
            hm_flux = heatmap!(ax_flux, interface_flux', colormap=:viridis)
            Colorbar(fig_flux[1, 2], hm_flux, label="Flux")
            save(joinpath(pwd(), "interface_flux_iter_$iter.png"), fig_flux)
            println("Interface flux visualization saved for iteration $iter")
        catch e
            println("Failed to create interface flux visualization: $e")
        end

        volume_jacobian = compute_volume_jacobian(mesh, front)
        
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
        for (eq_idx, (i, j)) in enumerate(cells_idx)
            # Get the desired volume change based on interface flux
            # Use matrices directly instead of diagonals
            volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
            
            # F_i = ρL * volume_change - interface_flux
            F[eq_idx] = ρL * volume_change - interface_flux[i,j]
        end
        
        # 4. Implement the Gauss-Newton formula: X^{n+1} = X^n - (J^T J)^{-1} J^T F
        JTJ = J' * J
        
        # Diagnose the system
        used_columns = unique(col_indices)
        println("Matrix info: size(J)=$(size(J)), n_markers=$n_markers")
        println("Used marker indices: $(length(used_columns)) of $n_markers")
        
        # Check if JTJ is singular and handle appropriately
        singular = false
        newton_step = zeros(n_markers)
        try
            # Try adding a small regularization term
            reg_JTJ = JTJ #+ 1e-12 * I(size(JTJ, 1))
            newton_step = reg_JTJ \ (J' * F)
        catch e
            if isa(e, SingularException)
                singular = true
                println("JTJ is singular even with regularization, using SVD solver")
                
                # Use SVD-based pseudoinverse for robust solving
                F_svd = svd(Matrix(JTJ))
                tol = eps(Float64) * max(size(JTJ)...) * maximum(F_svd.S)
                S_inv = [s > tol ? 1/s : 0.0 for s in F_svd.S]
                
                # Compute pseudoinverse solution
                JTF = J' * F
                newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
            else
                rethrow(e)
            end
        end
        
        # Apply the step with the adjusted factor and propagated displacements
        displacements -= α * newton_step 
        displacements[1] = displacements[end]

        # Smooth the displacements for stability
        smooth_displacements!(displacements, markers, front.is_closed, 1.0, 4)

        # Print maximum displacement for diagnostics
        max_disp = maximum(abs.(displacements))
        println("Maximum displacement (after smoothing): $max_disp")

        """
        # Optional: Limit maximum displacement if needed
        max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.2
        if max_disp > max_allowed
            scale = max_allowed / max_disp
            displacements .*= scale
            println("Scaling displacements by scale to limit maximum displacement")
        end
        """

        # Calculate residual norm for convergence check
        residual_norm = norm(F)
        push!(residual_norm_history, residual_norm)
        
        # Report progress
        println("Iteration $iter | Residual = $residual_norm")

        # Check convergence
        if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
            println("Converged after $iter iterations with residual $residual_norm")
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
        
        # Print mean radius
        # Calculate geometric center of the interface
        center_x = sum(m[1] for m in new_markers) / length(new_markers)
        center_y = sum(m[2] for m in new_markers) / length(new_markers)
        
        # Calculate radius for each marker
        radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in new_markers]
        mean_radius = sum(radii) / length(radii)
        radius_std = sqrt(sum((r - mean_radius)^2 for r in radii) / length(radii))
        
        println("Mean radius: $(round(mean_radius, digits=6)), std: $(round(radius_std, digits=6))")

        
        # If interface is closed, update the duplicated last marker
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        

        # 6. Create updated front tracking object
        updated_front = FrontTracker(new_markers, front.is_closed)
        
        # 7. Create space-time level set for capacity calculation
        body_func = create_interface_body_function(front, updated_front, tₙ, Δt)

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
                   first.(markers), 
                   last.(markers), 
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
            
            # 3D surface plots for SDFs at t_n and t_{n+1} (bottom row)
            # Create mesh grid for 3D plotting
            X = [x for x in x_range, y in y_range]
            Y = [y for x in x_range, y in y_range]
            
            # 3D plot for SDF at t_n (original)
            ax3d_1 = Axis3(fig[3, 1], 
                          title="SDF at t_n",
                          xlabel="x", ylabel="y", zlabel="ϕ",
                          aspect=(1, 1, 0.5))
            
            surf1 = surface!(ax3d_1, X, Y, phi_original, 
                            colormap=:viridis)
            
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
            
            surf2 = surface!(ax3d_2, X, Y, phi_updated, 
                            colormap=:viridis)
            
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
        
        # 8. Update space-time mesh and capacity
        STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
        capacity = Capacity(body_func, STmesh; compute_centroids=false)
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

        front = updated_front
    end
    
    # Store residuals from first time step
    residuals[timestep] = residual_norm_history
    
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
    println("Max temperature: $(maximum(abs.(s.x[1:n])))")
    
    # Main time-stepping loop
    timestep = 2
    while t < Tₑ
        println("\n--- Timestep $timestep, t = $(round(t, digits=6)) ---")
        
        # Get current markers and calculate normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        
        # Calculate interface flux using diffusion operators
        W! = phase.operator.Wꜝ[1:n, 1:n]
        G = phase.operator.G[1:n, 1:n]
        H = phase.operator.H[1:n, 1:n]
        Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id[1:n, 1:n]
        
        Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        interface_velocities = 1/(ρL) * abs.(interface_flux)
        
        # Adapt timestep if needed based on CFL condition
        if adaptive_timestep
            time_left = Tₑ - t
            Δt_max_current = min(Δt_max, time_left)
            
            # Use maximum interface velocity for CFL
            max_velocity = maximum(interface_velocities)
            if max_velocity > 0
                # Calculate cell size
                Δx = mesh.nodes[1][2] - mesh.nodes[1][1]
                Δy = mesh.nodes[2][2] - mesh.nodes[2][1]
                min_cell_size = min(Δx, Δy)
                
                # CFL-based timestep
                cfl = max_velocity * Δt / min_cell_size
                
                if cfl > cfl_target
                    # Reduce timestep if CFL exceeds target
                    Δt = cfl_target * min_cell_size / max_velocity * 0.9
                    Δt = max(Δt, Δt_min)
                else
                    # Grow timestep if CFL is small
                    Δt = min(Δt * 1.1, Δt_max_current)
                end
            end
            
            push!(timestep_history, (t, Δt))
            println("  Adaptive timestep: Δt = $(round(Δt, digits=6))")
        end
        
        # Update time
        t += Δt
        tₙ = t - Δt
        tₙ₊₁ = t
        
        # Create time interval for space-time mesh
        time_interval = [tₙ, tₙ₊₁]
        
        # Compute volume Jacobian
        volume_jacobian = compute_volume_jacobian(mesh, front)
        
        # Check number of affected cells
        affected_cells = 0
        for i in 1:nx
            for j in 1:ny
                if haskey(volume_jacobian, (i,j)) && !isempty(volume_jacobian[(i,j)])
                    affected_cells += 1
                end
            end
        end
        println("  Number of cells affected by interface: $affected_cells")
        
        if affected_cells == 0
            println("  ERROR: No cells are affected by the interface!")
            println("  Interface may have moved outside computational domain.")
            break  # Exit time stepping loop or implement recovery mechanism
        end
        
        # Total markers count (excluding duplicated point if closed)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacement vector and residual vector
        displacements = zeros(n_markers)
        residual_norm_history = Float64[]

        body_func = create_interface_body_function(front)
        STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
        capacity = Capacity(body_func, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)

        # Rebuild the matrix system
        s.A = A_mono_unstead_diff_moving(phase.operator, phase.capacity, 
                                        phase.Diffusion_coeff, bc, scheme)
        s.b = b_mono_unstead_diff_moving(phase.operator, phase.capacity,
                                        phase.Diffusion_coeff, phase.source, 
                                        bc, Tᵢ, Δt, tₙ, scheme)
        BC_border_mono!(s.A, s.b, bc_b, mesh)
        
        # Gauss-Newton iterations
        for iter in 1:max_iter
            # 1. Solve temperature field with current interface position
            solve_system!(s; method=method, kwargs...)
            Tᵢ = s.x
            
            # Get capacity matrices
            V_matrices = phase.capacity.A[cap_index]
            Vₙ₊₁_matrix = V_matrices[1:end÷2, 1:end÷2]
            Vₙ_matrix = V_matrices[end÷2+1:end, end÷2+1:end]
            
            # 2. Calculate the interface flux with properly extracted matrices
            W! = phase.operator.Wꜝ[1:n, 1:n]
            G = phase.operator.G[1:n, 1:n]
            H = phase.operator.H[1:n, 1:n]
            Id = build_I_D(phase.operator, phase.Diffusion_coeff, phase.capacity)
            Id = Id[1:n, 1:n]
            
            Tₒ, Tᵧ = Tᵢ[1:n], Tᵢ[n+1:end]
            interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
            
            # Reshape to get flux per cell
            interface_flux_2d = reshape(interface_flux, (nx, ny))
            
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
            println("  Iteration $iter: $m equations for $n_markers unknowns")
            
            # Build Jacobian matrix for volume changes
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Handle each marker affecting this cell
                for (marker_idx, jac_value) in volume_jacobian[(i,j)]
                    if 0 <= marker_idx < n_markers
                        push!(row_indices, eq_idx)
                        push!(col_indices, marker_idx + 1)  # 1-based indexing
                        push!(values, ρL * jac_value)
                    end
                end
            end
            
            # Create Jacobian matrix J
            J = sparse(row_indices, col_indices, values, m, n_markers)
            
            # Calculate residual vector F
            F = zeros(m)
            for (eq_idx, (i, j)) in enumerate(cells_idx)
                # Get desired volume change based on interface flux
                volume_change = Vₙ₊₁_matrix[i,j] - Vₙ_matrix[i,j]
                F[eq_idx] = ρL * volume_change - interface_flux_2d[i,j]
            end
            
            # 4. Solve the Gauss-Newton system
            JTJ = J' * J
            
            # Diagnostic info
            used_columns = unique(col_indices)
            println("    Matrix info: size(J)=$(size(J)), markers used: $(length(used_columns))/$n_markers")
            
            # Check if JTJ is singular and handle appropriately
            newton_step = zeros(n_markers)
            try
                # Add regularization for numerical stability
                λ = 1e-1  # Regularization parameter
                reg_JTJ = JTJ + 1e-12 * I(size(JTJ, 1))
                newton_step = reg_JTJ \ (J' * F)
            catch e
                if isa(e, SingularException)
                    println("    JTJ is singular, using SVD")
                    
                    # Use SVD-based pseudoinverse
                    F_svd = svd(Matrix(JTJ))
                    tol_svd = eps(Float64) * max(size(JTJ)...) * maximum(F_svd.S)
                    S_inv = [s > tol_svd ? 1/s : 0.0 for s in F_svd.S]
                    
                    # Calculate solution via pseudoinverse
                    JTF = J' * F
                    newton_step = F_svd.V * (S_inv .* (F_svd.U' * JTF))
                else
                    rethrow(e)
                end
            end
            
            # Update displacements with relaxation
            displacements -= α * newton_step
            
            # Smooth the displacements for stability
            smooth_displacements!(displacements, markers, front.is_closed, 0.5, 4)
            
            # Print maximum displacement for diagnostics
            max_disp = maximum(abs.(displacements))
            println("    Maximum displacement (after smoothing): $max_disp")
            
            # Optional: Limit maximum displacement if needed
            max_allowed = min(mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]) * 0.2
            if max_disp > max_allowed
                scale = max_allowed / max_disp
                displacements .*= scale
                println("    Scaling displacements by $scale to limit maximum displacement")
            end
            
            # Calculate residual norm for convergence check
            residual_norm = norm(F)
            push!(residual_norm_history, residual_norm)
            
            # Report progress
            println("    Iteration $iter | Residual = $residual_norm | Max disp = $(maximum(abs.(displacements))) | Min disp = $(minimum(abs.(displacements)))")
            
            # Check convergence
            if residual_norm < tol || (iter > 1 && abs(residual_norm_history[end] - residual_norm_history[end-1]) < reltol)
                println("    Converged after $iter iterations with residual $residual_norm")
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
            if front.is_closed && markers[1] == markers[end]
                new_markers[end] = new_markers[1]
            end
            
            # 6. Create updated front tracking object
            updated_front = FrontTracker(new_markers, front.is_closed)
            
            
            body_func = create_interface_body_function(front, updated_front, tₙ, Δt)

            # 8. Update space-time mesh and capacity
            STmesh = Penguin.SpaceTimeMesh(mesh, time_interval, tag=mesh.tag)
            capacity = Capacity(body_func, STmesh; compute_centroids=false)
            operator = DiffusionOps(capacity)
            phase_updated = Phase(capacity, operator, phase.source, phase.Diffusion_coeff)
            
            # 9. Rebuild the matrix system
            s.A = A_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, bc, scheme)
            s.b = b_mono_unstead_diff_moving(phase_updated.operator, phase_updated.capacity, 
                                            phase_updated.Diffusion_coeff, phase_updated.source, 
                                            bc, Tᵢ, Δt, tₙ, scheme)
            
            BC_border_mono!(s.A, s.b, bc_b, mesh)
            
            # 10. Update phase and front for next iteration
            phase = phase_updated
            front = updated_front
        end
        
        # Store residuals from this time step
        residuals[timestep] = residual_norm_history
        
        # Store updated interface position
        xf_log[timestep+1] = get_markers(front)
        
        # Store solution
        push!(s.states, s.x)
        
        println("  Time: $(round(t, digits=6)), Max temperature: $(maximum(abs.(s.x[1:n])))")
        timestep += 1
    end
    
    return s, residuals, xf_log, timestep_history
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