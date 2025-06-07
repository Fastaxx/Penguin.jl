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
    # ...existing code...
    # Garder la fonction de lissage existante
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

function visualize_results(initial_markers, final_markers, times, final_time, Λ)
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
    
    # Ajouter des étiquettes avec les informations du problème
    Label(fig[0, :], "Stefan Problem - Direct Advection: Λ=$Λ, t_final=$final_time",
         fontsize=16)
    
    return fig
end

function visualize_flux_and_interface(mesh, interface_flux_2d, markers, 
                                     iter=0, updated_markers=nothing)
    # Créer la figure
    fig = Figure(size=(1000, 800))
    
    # Extraire les coordonnées du maillage
    nx, ny = size(interface_flux_2d)
    x0, y0 = mesh.nodes[1][1], mesh.nodes[2][1]
    lx, ly = mesh.nodes[1][end] - x0, mesh.nodes[2][end] - y0
    
    # Axe principal
    ax = Axis(fig[1, 1], aspect=DataAspect(),
             title="Interface Flux & Advection (Iter $iter)",
             xlabel="x", ylabel="y")
    
    # 1. Tracer le flux comme heatmap
    hm = heatmap!(ax, range(x0, x0+lx, nx), range(y0, y0+ly, ny), 
                 interface_flux_2d', colormap=:plasma)
    Colorbar(fig[1, 2], hm, label="Interface Flux")
    
    # 2. Tracer l'interface initiale
    lines!(ax, first.(markers), last.(markers), 
          color=:blue, linewidth=2, label="Initial Interface")
    scatter!(ax, first.(markers), last.(markers), 
            color=:blue, markersize=6)
    
    # 3. Tracer l'interface mise à jour si disponible
    if updated_markers !== nothing
        lines!(ax, first.(updated_markers), last.(updated_markers), 
              color=:red, linewidth=2, label="Updated Interface")
        scatter!(ax, first.(updated_markers), last.(updated_markers), 
                color=:red, markersize=6)
    end
    
    # Legend
    Legend(fig[2, 1:2], ax, orientation=:horizontal)
    
    return fig
end

"""
Test du problème de Stefan avec advection directe des marqueurs
- Utilise le flux interfacial calculé par les opérateurs de diffusion
- Déplace chaque marqueur selon dx/dt = flux/rhoL
"""
function solve_stefan_direct_advection()
    println("Test du problème de Stefan avec advection directe")
    
    # Paramètres du problème
    ρL = 1.0          # Chaleur latente
    
    # Solution parameter from Almgren [75]
    Λ = 1.56
    T_infinity = -0.5
    println("Solution parameter Λ = $Λ")
    
    # Time parameters
    t0 = 1.0         # Initial time
    t_end = 1.5       # End time
    Δt = 0.002          # Initial time step
    
    # Calculate initial radius from similarity solution
    initial_radius = Λ * sqrt(t0)
    println("Initial radius = $initial_radius")
    
    # Mesh setup
    nx, ny = 128, 128   # Number of cells
    lx, ly = 4.0, 4.0 # Domain size
    x0, y0 = -2.0, -2.0
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
    create_circle!(front, 0.0, 0.0, initial_radius, 200)  # Use more markers for better resolution
    
    # Store initial markers for plotting at the end
    initial_markers = get_markers(front)
    
    # Initialize variables for tracking
    markers_history = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    markers_history[0] = initial_markers
    time_history = Float64[t0]
    
    # Define the body function
    function body(x, y, t_local=0)
        return -sdf(front, x, y)
    end
    
    # Initialize time counter
    t = t0
    timestep = 0
    
    # Source term and conductivity
    f = (x,y,z,t) -> 0.0  # No source term
    K = (x,y,z) -> 1.0    # Unit conductivity
    
    # Define boundary conditions
    bc_b = Dirichlet(T_infinity)  # Far field temperature
    bc = Dirichlet(0.0)          # Temperature at the interface (melting point)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => bc_b, :top => bc_b, 
                                                          :left => bc_b, :right => bc_b))
    
    # Stefan condition at the interface
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρL))
    
    # Main time-stepping loop
    while t < t_end
        timestep += 1
        println("\n--- Timestep $timestep, t = $(round(t, digits=6)) ---")
        
        # Adjust time step to hit exactly t_end if necessary
        if t + Δt > t_end
            Δt = t_end - t
        end
        
        # 1. Create capacity, operator with the current front position
        capacity = Capacity(body, mesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        phase = Phase(capacity, operator, f, K)
        
        # 2. Set up initial condition for the temperature field
        n = (nx+1) * (ny+1)
        Tₒ = zeros(n)
        Tᵧ = zeros(n)
        
        Δx, Δy = mesh.nodes[1][2] - mesh.nodes[1][1], mesh.nodes[2][2] - mesh.nodes[2][1]
        for i in 1:nx+1
            for j in 1:ny+1
                idx = (i-1)*ny + j
                x = x0 + (i-0.5) * Δx
                y = y0 + (j-0.5) * Δy
                
                r = sqrt(x^2 + y^2)  # Distance from origin
                Tₒ[idx] = temperature(r, t)  # Current temperature
            end
        end
        
        # 3. Calculate the interface flux using the diffusion operators
        W! = operator.Wꜝ
        G = operator.G
        H = operator.H
        Id = build_I_D(operator, phase.Diffusion_coeff, phase.capacity)
        Id = Id
        
        # Calculate interface flux
        interface_flux = Id * H' * W! * G * Tₒ + Id * H' * W! * H * Tᵧ
        
        # Reshape to 2D for easier access by grid coordinates
        interface_flux_2d = reshape(interface_flux, (nx+1, ny+1)) #+ reshape(interface_flux, (nx+1, ny+1))'
        
        # 4. DIRECT ADVECTION - Move markers based on local flux
        # Get current markers and normals
        markers = get_markers(front)
        normals = compute_marker_normals(front, markers)
        n_markers = length(markers) - (front.is_closed ? 1 : 0)
        
        # Initialize displacements vector
        displacements = zeros(n_markers)
        
        # For each marker, calculate local flux and velocity
        for i in 1:n_markers
            x, y = markers[i]
            
            # Convert to grid indices with better interpolation
            # First convert to cell coordinates
            cell_x = (x - x0) / Δx
            cell_y = (y - y0) / Δy
            
            # Find the four nearest cell centers for bilinear interpolation
            i1 = floor(Int, cell_x) + 1
            j1 = floor(Int, cell_y) + 1
            i2 = i1 + 1
            j2 = j1 + 1
            
            # Clamp indices to grid boundaries
            i1 = clamp(i1, 1, nx)
            j1 = clamp(j1, 1, ny)
            i2 = clamp(i2, 1, nx)
            j2 = clamp(j2, 1, ny)
            
            # Compute interpolation weights
            wx = cell_x - floor(cell_x)
            wy = cell_y - floor(cell_y)
            
            # Fetch flux values from the four neighboring cells
            f11 = interface_flux_2d[i1, j1]
            f21 = interface_flux_2d[i2, j1]
            f12 = interface_flux_2d[i1, j2]
            f22 = interface_flux_2d[i2, j2]
            
            # Perform bilinear interpolation
            f1 = (1 - wx) * f11 + wx * f21
            f2 = (1 - wx) * f12 + wx * f22
            local_flux = (1 - wy) * f1 + wy * f2
            
            # If flux is very small, use neighborhood average to avoid stagnation
            if abs(local_flux) < 1e-10
                # Look at nearby cells in a small window
                flux_sum = 0.0
                count = 0
                for di in -1:1
                    for dj in -1:1
                        ni = clamp(i1 + di, 1, nx)
                        nj = clamp(j1 + dj, 1, ny)
                        flux_sum += interface_flux_2d[ni, nj]
                        count += 1
                    end
                end
                if count > 0
                    local_flux = flux_sum / count
                end
            end
            
            # Calculate displacement for this time step: Δx = (flux/ρL) * Δt
            # Note: The negative sign is because positive flux means melting (moving inward)
            displacements[i] = -(local_flux / ρL) * Δt
            
            # Ensure minimum movement for numerical stability (prevents stagnant markers)
            if abs(displacements[i]) < 1e-10
                # Use a small fraction of average displacement from previous timestep
                # or a minimum based on mesh size if this is first timestep
                min_movement = timestep > 1 ? 0.01 * avg_disp_previous : 0.001 * Δx
                displacements[i] = sign(displacements[i]) * max(abs(displacements[i]), min_movement)
            end
        end
        
        # Store average displacement for next timestep
        avg_disp = mean(abs.(displacements))
        avg_disp_previous = avg_disp
        
        # Apply smoothing to displacements for numerical stability
        smoothing_factor = 0.5
        window_size = 20
        smooth_displacements!(displacements, markers, front.is_closed, smoothing_factor, window_size)
        
        # Calculate new marker positions
        new_markers = copy(markers)
        for i in 1:n_markers
            normal = normals[i]
            new_markers[i] = (
                markers[i][1] + displacements[i] * normal[1],
                markers[i][2] + displacements[i] * normal[2]
            )
        end
        
        # If closed interface, ensure first and last markers match
        if front.is_closed
            new_markers[end] = new_markers[1]
        end
        
        # Visualize flux and interface movement
        flux_viz = visualize_flux_and_interface(mesh, interface_flux_2d, markers, timestep, new_markers)
        display(flux_viz)
        
        # Print diagnostic information
        max_disp = maximum(abs.(displacements))
        min_disp = minimum(abs.(displacements))
        avg_disp = mean(abs.(displacements))
        println("Displacement statistics:")
        println("  Maximum: $max_disp")
        println("  Minimum: $min_disp")
        println("  Average: $avg_disp")
        
        # Update front with new markers
        set_markers!(front, new_markers)
        
        # Store results for this timestep
        markers_history[timestep] = copy(new_markers)
        push!(time_history, t + Δt)
        
        # Calculate error compared to analytical solution
        t_next = t + Δt
        r_analytical = exact_radius(t_next)
        r_numerical = mean([sqrt(x^2 + y^2) for (x,y) in new_markers[1:end-1]])
        rel_error = abs(r_numerical - r_analytical) / r_analytical
        
        println("Time: $(round(t_next, digits=4))")
        println("  Analytical radius: $(round(r_analytical, digits=6))")
        println("  Numerical radius: $(round(r_numerical, digits=6))")
        println("  Relative error: $(round(rel_error * 100, digits=4))%")
        
        # Update time
        t += Δt
    end
    
    # Visualize final results
    fig = visualize_results(initial_markers, markers_history[timestep], time_history, t, Λ)
    
    return fig, markers_history, time_history
end

# Execute the simulation and display the results
fig, markers_history, time_history = solve_stefan_direct_advection()
display(fig)
save("stefan_direct_advection.png", fig)