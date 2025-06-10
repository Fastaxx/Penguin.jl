using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Colors
using Statistics
using FFTW
using DSP
using Roots
using NLsolve  # For improved solver

### 2D Test Case: Melting Circle (Stefan Problem with Circular Interface)
### Hot solid sphere melting in cooler liquid with similarity solution

# Define physical parameters
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
TM = 0.0    # Melting temperature (interface)
T_hot = 1.0  # Hot temperature inside sphere
T∞ = 1.0     # Far field temperature (same as melting temp for simplicity)

# Calculate the Stefan number (negative for melting)
Ste = (c * (TM - T_hot)) / L
println("Stefan number: $Ste")

# Define similarity parameter for melting case
# Note: For melting, similarity parameter is negative of freezing case
S = 1.1  # Will need to solve for this value based on physical parameters
println("Similarity parameter S = $S")

# Set initial conditions
R0 = 2.5      # Initial radius (larger than final)
t_init = 1.0  # Initial time
t_final = 1.5 # Final time

# Analytical temperature function for melting sphere
function analytical_temperature(r, t)
    # Calculate similarity variable
    s = r / sqrt(t)
    
    if s < S
        # Inside the hot solid, temperature is uniform
        return T_hot
    else
        # In liquid region, use similarity solution
        # For melting case, we adapt the solution
        return TM + (T∞ - TM) * (1.0 - F(s)/F(S))
    end
end

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Function to calculate the interface position at time t
function interface_position(t)
    return S * sqrt(t)
end

# Calculate analytical flux at interface
function analytical_flux(t)
    return (k * (T_hot - TM) / F(S)) * exp(-S^2) / sqrt(t)
end

# Print information about the simulation
println("Initial radius at t=$t_init: R=$(interface_position(t_init))")
println("Final radius at t=$t_final: R=$(interface_position(t_final))")

# Define the spatial mesh
nx, ny = 32, 32
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy)")

# Create the front-tracking body
nmarkers = 100
front = FrontTracker() 
create_circle!(front, 0.0, 0.0, R0, nmarkers)

# Define the initial position of the front
body = (x, y, t, _=0) -> -sdf(front, x, y)

# Define the Space-Time mesh
Δt = 1.0*(lx / nx)^2  # Time step size based on mesh spacing
t_final = t_init + 6Δt
println("Final radius at t=$(t_init + Δt): R=$(interface_position(t_init + Δt))")

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature
bc = Dirichlet(TM)    # Temperature at the interface (melting point)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
    :left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))

# Stefan condition at the interface (negative for melting)
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Phase1 = Phase(capacity, operator, f, K)

# Set up initial condition
u0ₒ = zeros((nx+1)*(ny+1))
body_init = (x,y,_=0) -> -sdf(front, x, y)
cap_init = Capacity(body_init, mesh; compute_centroids=false)
centroids = cap_init.C_ω

# Initialize the temperature
for idx in 1:length(centroids)
    centroid = centroids[idx]
    x, y = centroid[1], centroid[2]
    r = sqrt(x^2 + y^2)
    u0ₒ[idx] = analytical_temperature(r, t_init)
end
u0ₒ = ones((nx+1)*(ny+1))  # Reset to zero for initial condition
u0ᵧ = ones((nx+1)*(ny+1))*T_hot  # Initial temperature inside (hot)
u0 = vcat(u0ₒ, u0ᵧ)

# Visualize initial temperature field
fig_init = Figure(size=(800, 600))
ax_init = Axis(fig_init[1, 1], 
               title="Initial Temperature Field", 
               xlabel="x", ylabel="y",
               aspect=DataAspect())
hm = heatmap!(ax_init, mesh.nodes[1], mesh.nodes[2], 
              reshape(u0ₒ, (nx+1, ny+1)),
              colormap=:thermal)
Colorbar(fig_init[1, 2], hm, label="Temperature")

# Add interface contour
markers = get_markers(front)
marker_x = [m[1] for m in markers]
marker_y = [m[2] for m in markers]
lines!(ax_init, marker_x, marker_y, color=:black, linewidth=2)

display(fig_init)

# Newton parameters with slightly better convergence settings
Newton_params = (20, 1e-7, 1e-7, 0.8)  # max_iter, tol, reltol, α

# Create solver
solver = StefanMono2D(Phase1, bc_b, bc, Δt, u0, mesh, "BE")

# Modify the system solver to use NLsolve for better performance
function residual_func!(F_out, delta_displacements, J, F, markers, normals, displacements, n_markers, front)
    # Calculate proposed displacements
    test_displacements = displacements + delta_displacements
    
    # Calculate new marker positions
    test_markers = Vector{Tuple{Float64, Float64}}(undef, length(markers))
    for i in 1:n_markers
        test_markers[i] = (
            markers[i][1] + test_displacements[i] * normals[i][1],
            markers[i][2] + test_displacements[i] * normals[i][2]
        )
    end
    
    # If closed, ensure continuity
    if front.is_closed && length(test_markers) > 1
        test_markers[end] = test_markers[1]
    end
    
    # Use current Jacobian approximation for speed
    F_out .= F + J * delta_displacements
    return F_out
end

# Solve the problem
solver, residuals, xf_log, timestep_history, phase, position_increments = solve_StefanMono2D!(
    solver, Phase1, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
    Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\)

# Plot results
function plot_results(solver, mesh, xf_log, timestep_history)
    results_dir = joinpath(pwd(), "melting_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # Extract and calculate analytics
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    
    # Plot interface evolution
    fig_interface = Figure(size=(800, 800))
    ax = Axis(fig_interface[1, 1], 
              title="Melting Circle Interface Evolution", 
              xlabel="x", ylabel="y",
              aspect=DataAspect())
    
    # Generate color gradient
    colors = cgrad(:thermal, num_timesteps)
    
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        marker_x = [m[1] for m in markers]
        marker_y = [m[2] for m in markers]
        
        # Plot interface
        lines!(ax, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2)
        
        # Add timestamp
        if i == 1 || i == num_timesteps || i % max(1, div(num_timesteps, 5)) == 0
            time_value = timestep_history[min(timestep, length(timestep_history))][1]
            text!(ax, mean(marker_x), mean(marker_y), 
                 text="t=$(round(time_value, digits=2))",
                 fontsize=10)
        end
    end
    
    # Add a colorbar
    Colorbar(fig_interface[1, 2], limits=(1, num_timesteps),
            colormap=:thermal, label="Time progression")
    
    save(joinpath(results_dir, "melting_interface.png"), fig_interface)
    
    # Calculate radii
    times = [hist[1] for hist in timestep_history]
    radii = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        
        # Calculate geometric center
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        
        # Calculate mean radius
        mean_radius = mean([sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers])
        push!(radii, mean_radius)
    end
    
    # Plot radius vs time
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1], 
                    title="Melting Interface Radius Evolution", 
                    xlabel="Time", 
                    ylabel="Radius")
    
    # Create correct time values for each radius
    times_for_plot = Float64[]
    for ts in all_timesteps
        if ts == 1
            push!(times_for_plot, timestep_history[1][1])
        else
            time_index = min(ts, length(timestep_history))
            push!(times_for_plot, timestep_history[time_index][1])
        end
    end
    
    # Add analytical solution
    analytical_times = range(t_init, stop=t_final, length=100)
    analytical_radii = [interface_position(t) for t in analytical_times]
    
    # Plot results
    scatter!(ax_radius, times_for_plot, radii, 
            label="Simulation", markersize=6)
    
    lines!(ax_radius, analytical_times, analytical_radii,
          label="Analytical", linewidth=2, color=:red, linestyle=:dash)
    
    Legend(fig_radius[1, 2], ax_radius)
    save(joinpath(results_dir, "radius_evolution.png"), fig_radius)
    
    # Create animation
    create_animation(solver, mesh, xf_log, results_dir)
    
    return results_dir
end

results_dir = plot_results(solver, mesh, xf_log, timestep_history)
println("\nResults saved to: $results_dir")

# Function to create animation
function create_animation(solver, mesh, xf_log, results_dir)
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Find temperature range
    all_temps = Float64[]
    for Tstate in solver.states
        Tw = Tstate[1:npts]
        push!(all_temps, extrema(Tw)...)
    end
    temp_limits = (minimum(all_temps), maximum(all_temps))
    
    all_timesteps = sort(collect(keys(xf_log)))
    
    # Create animation
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], 
            title="Melting Circle Evolution", 
            xlabel="x", ylabel="y", aspect=DataAspect())
    
    # Create recording
    record(fig, joinpath(results_dir, "melting_animation.mp4"), 1:length(solver.states)) do i
        empty!(ax)
        
        current_time = i
        ax.title = "Temperature & Melting Interface, t=$(round(current_time, digits=3))"
        
        Tstate = solver.states[i]
        Tw = Tstate[1:npts]
        Tmat = reshape(Tw, (nx1, ny1))
        
        hm = heatmap!(ax, xi, yi, Tmat, colormap=:thermal, colorrange=temp_limits)
        Colorbar(fig[1, 2], hm, label="Temperature")
        
        ts = i
        if ts <= length(all_timesteps)
            markers = xf_log[all_timesteps[ts]]
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
            lines!(ax, marker_x, marker_y, color=:white, linewidth=3)
        end
    end
    
    return joinpath(results_dir, "melting_animation.mp4")
end

println("Simulation and visualization complete!")