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

### 2D Test Case: Frank Sphere (Stefan Problem with Circular Interface)
### Ice sphere growing in undercooled liquid with self-similar solution

# Define physical parameters
L = 1.0      # Latent heat
c = 1.0      # Specific heat capacity
TM = 0.0     # Melting temperature (inside sphere)
T∞ = -0.5    # Far field temperature (undercooled liquid)

# Calculate the Stefan number
Ste = (c * (TM - T∞)) / L
println("Stefan number: $Ste")

# Define F(s) function using the exponential integral E₁
function F(s)
    return expint(s^2/4)  # E₁(s²/4)
end

# Calculate the similarity parameter S
S = 1.56
println("Similarity parameter S = $S")

# Set initial conditions as specified
R0 = 1.56      # Initial radius
t_init = 1.0   # Initial time
t_final = 1.05  # Final time

# Analytical temperature function
function analytical_temperature(r, t)
    # Calculate similarity variable
    s = r / sqrt(t)
    
    if s < S
        return TM  # Inside the solid (ice)
    else
        # In liquid region, use the similarity solution
        return T∞ * (1.0 - F(s)/F(S))
    end
end

# Function to calculate the interface position at time t
function interface_position(t)
    return S * sqrt(t)
end

# Print information about the simulation
println("Initial radius at t=$t_init: R=$(interface_position(t_init))")

# Plot the analytical solution
radii = [interface_position(t) for t in range(t_init, stop=t_final, length=100)]
temperatures = [analytical_temperature(r, t_final) for r in radii]

# Define the spatial mesh
nx, ny = 64, 64
lx, ly = 16.0, 16.0
x0, y0 = -8.0, -8.0
Δx, Δy = lx/(nx), ly/(ny)
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

println("Mesh created with dimensions: $(nx) x $(ny), Δx=$(Δx), Δy=$(Δy), domain=[$x0, $(x0+lx)], [$y0, $(y0+ly)]")

# Create the front-tracking body
nmarkers = 30
front = FrontTracker() 
create_circle!(front, 0.0, 0.0, R0, nmarkers)

# Define the initial position of the front
body = (x, y, t, _=0) -> sdf(front, x, y)

# Define the Space-Time mesh
Δt = 0.05
println("Final radius at t=$(t_init + Δt): R=$(interface_position(t_init + Δt))")

STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; compute_centroids=false)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc_b = Dirichlet(T∞)  # Far field temperature
bc = Dirichlet(TM)  # Temperature at the interface
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => bc_b, :top => bc_b, :left => bc_b, :right => bc_b))

# Stefan condition at the interface
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Fluide = Phase(capacity, operator, f, K)

# Set up initial condition
u0ₒ = zeros((nx+1)*(ny+1))
for i in 1:nx
    for j in 1:ny
        idx = i + (j - 1) * (nx + 1)
        x = mesh.centers[1][i]
        y = mesh.centers[2][j]
        r = sqrt(x^2 + y^2)  # Distance from origin
        u0ₒ[idx] = analytical_temperature(r, t_init)  # Initial temperature
    end
end
u0ᵧ = ones((nx+1)*(ny+1))*TM # Initial temperature at the interface
u0 = vcat(u0ₒ, u0ᵧ)

# Plot the initial temperature field
fig_init = Figure(size=(800, 600))
ax_init = Axis(fig_init[1, 1], 
               title="Initial Temperature Field", 
               xlabel="x", 
               ylabel="y",
               aspect=DataAspect())
hm = heatmap!(ax_init, mesh.centers[1], mesh.centers[2], reshape(u0ₒ, (nx+1, ny+1)),
            colormap=:thermal)
Colorbar(fig_init[1, 2], hm, label="Temperature")
display(fig_init)
# Newton parameters
Newton_params = (10000, 1e-6, 1e-6, 1.0) # max_iter, tol, reltol, α

# Run the simulation
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history = solve_StefanMono2D!(solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
   Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\)

# Plot the results
function plot_simulation_results(residuals, xf_log, timestep_history, Ste=nothing)
    # Create results directory
    results_dir = joinpath(pwd(), "simulation_results")
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # 1. Plot residuals for each timestep
    fig_residuals = Figure(size=(900, 600))
    ax_residuals = Axis(fig_residuals[1, 1], 
                        title="Convergence History", 
                        xlabel="Iteration", 
                        ylabel="Residual Norm (log scale)",
                        yscale=log10)
    
    for (timestep, residual_vec) in sort(collect(residuals))
        lines!(ax_residuals, 1:length(residual_vec), residual_vec, 
              label="Timestep $timestep", 
              linewidth=2)
    end
    
    Legend(fig_residuals[1, 2], ax_residuals)
    save(joinpath(results_dir, "residuals_$(timestep_history[1][1]).png"), fig_residuals)
    
    # 2. Plot interface positions at different timesteps
    fig_interface = Figure(size=(800, 800))
    ax_interface = Axis(fig_interface[1, 1], 
                       title="Interface Evolution", 
                       xlabel="x", 
                       ylabel="y",
                       aspect=DataAspect())
    
    # Get all timesteps and sort them
    all_timesteps = sort(collect(keys(xf_log)))
    num_timesteps = length(all_timesteps)
    
    # Generate color gradient based on timestep
    colors = cgrad(:viridis, num_timesteps)
    
    # Modify the plotting loop in the interface evolution section:
    for (i, timestep) in enumerate(all_timesteps)
        markers = xf_log[timestep]
        
        # For closed interfaces, make sure first and last markers match
        # This ensures proper visualization of closed curves
        if length(markers) > 1 && isapprox(markers[1][1], markers[end][1], atol=1e-10) && 
           isapprox(markers[1][2], markers[end][2], atol=1e-10)
            # It's a closed interface
            is_closed = true
        else
            is_closed = false
        end
        
        # Extract marker coordinates for plotting
        if is_closed
            # For closed interfaces, exclude the last marker (which should be a duplicate)
            marker_x = [m[1] for m in markers[1:end-1]]
            marker_y = [m[2] for m in markers[1:end-1]]
            
            # Add the first marker again to close the loop properly
            push!(marker_x, markers[1][1])
            push!(marker_y, markers[1][2])
        else
            # For open interfaces, use all markers
            marker_x = [m[1] for m in markers]
            marker_y = [m[2] for m in markers]
        end
        
        # Plot interface as a closed curve
        lines!(ax_interface, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2)
        
        # Add timestamp label near the interface
        if i == 1 || i == num_timesteps || i % max(1, div(num_timesteps, 5)) == 0
            time_value = timestep_history[min(timestep, length(timestep_history))][1]
            text!(ax_interface, mean(marker_x), mean(marker_y), 
                 text="t=$(round(time_value, digits=2))",
                 align=(:center, :center),
                 fontsize=10)
        end
    end
    
    # Add colorbar to show timestep progression
    Colorbar(fig_interface[1, 2], limits=(1, num_timesteps),
            colormap=:viridis, label="Timestep")
    
    save(joinpath(results_dir, "interface_evolution.png"), fig_interface)
    
    # 3. Plot radius evolution over time
    times = [hist[1] for hist in timestep_history]
    radii = Float64[]
    radius_stds = Float64[]
    
    for timestep in all_timesteps
        markers = xf_log[timestep]
        
        # Calculate geometric center
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        
        # Calculate radii and statistics
        marker_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in markers]
        mean_radius = sum(marker_radii) / length(marker_radii)
        radius_std = sqrt(sum((r - mean_radius)^2 for r in marker_radii) / length(marker_radii))
        
        push!(radii, mean_radius)
        push!(radius_stds, radius_std)
    end
    
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1], 
                    title="Interface Radius Evolution", 
                    xlabel="Time", 
                    ylabel="Mean Radius")
    
# Plot radius vs time
    # The issue is that timestep_history is accessed incorrectly
    # Create a mapping from timestep numbers to corresponding times
    timestep_to_time = Dict{Int, Float64}()
    for (i, hist) in enumerate(timestep_history)
        timestep_to_time[i] = hist[1]  # Store time for each timestep
    end
    
    # Use the actual times vector that corresponds to our radii
    times_for_plot = [timestep_to_time[min(ts, length(timestep_history))] for ts in all_timesteps]
    
    scatter!(ax_radius, times_for_plot, radii, 
            label="Simulation")
    

    # Plot analytical solution if we have Stefan number
    if Ste !== nothing
        # Calculate similarity parameter from Stefan number
        # For axisymmetric case: S = 2λ where λ satisfies equation with Ste
        λ = 1.2012  # This should be calculated from Ste or passed in
        S = λ
        
        # Plot analytical solution R = S√t
        t_range = range(minimum(times), maximum(times), length=100)
        analytical_radii = [S * sqrt(t) for t in t_range]
        
        lines!(ax_radius, t_range, analytical_radii, 
              linewidth=2, 
              color=:red, 
              linestyle=:dash,
              label="Analytical: R = S√t")
        
        # Calculate and display error metrics
        interpolated_analytical = [S * sqrt(timestep_history[ts][1]) for ts in all_timesteps]
        rel_errors = abs.(radii .- interpolated_analytical) ./ interpolated_analytical
        max_rel_error = maximum(rel_errors)
        mean_rel_error = mean(rel_errors)
        
        error_text = "Max relative error: $(round(max_rel_error*100, digits=2))%\n" *
                    "Mean relative error: $(round(mean_rel_error*100, digits=2))%"
        
        text!(ax_radius, 0.05, 0.95, text=error_text,
             align=(:left, :top),
             space=:relative,
             fontsize=12)
    end
    
    Legend(fig_radius[1, 2], ax_radius)
    save(joinpath(results_dir, "radius_evolution.png"), fig_radius)
    
    # 4. Plot adaptive timestep history
    fig_dt = Figure(size=(800, 400))
    ax_dt = Axis(fig_dt[1, 1], 
                title="Adaptive Timestep History", 
                xlabel="Time", 
                ylabel="Δt")
    
    dt_times = [hist[1] for hist in timestep_history]
    dt_values = [hist[2] for hist in timestep_history]
    
    lines!(ax_dt, dt_times, dt_values, 
          linewidth=2, 
          marker=:circle)
    
    save(joinpath(results_dir, "timestep_history.png"), fig_dt)
    
    # 5. Plot interface shape evolution (deviation from circularity)
    fig_shape = Figure(size=(900, 500))
    ax_shape = Axis(fig_shape[1, 1], 
                   title="Interface Shape Evolution", 
                   xlabel="Angle (degrees)", 
                   ylabel="Normalized Radius")
    
    # Select a subset of timesteps to avoid overcrowding
    plot_timesteps = all_timesteps[1:max(1, div(length(all_timesteps), 10)):end]
    
    for (i, timestep) in enumerate(plot_timesteps)
        markers = xf_log[timestep]
        
        # Calculate geometric center
        center_x = sum(m[1] for m in markers) / length(markers)
        center_y = sum(m[2] for m in markers) / length(markers)
        
        # Calculate angles and radii
        angles = Float64[]
        marker_radii = Float64[]
        
        for marker in markers
            angle = atan(marker[2] - center_y, marker[1] - center_x)
            radius = sqrt((marker[1] - center_x)^2 + (marker[2] - center_y)^2)
            push!(angles, angle)
            push!(marker_radii, radius)
        end
        
        # Sort by angle
        p = sortperm(angles)
        angles = angles[p]
        marker_radii = marker_radii[p]
        
        # Convert angles to degrees
        angles_deg = rad2deg.(angles)
        
        # Normalize radii by mean
        mean_radius = mean(marker_radii)
        norm_radii = marker_radii ./ mean_radius
        
        # Plot normalized radius vs angle
        lines!(ax_shape, angles_deg, norm_radii,
              linewidth=2,
              color=colors[findfirst(x -> x == timestep, all_timesteps)],
              label="t=$(round(timestep_history[timestep][1], digits=2))")
    end
    
    # Add perfect circle reference
    hlines!(ax_shape, [1.0], color=:black, linestyle=:dash, label="Perfect Circle")
    
    Legend(fig_shape[1, 2], ax_shape)
    save(joinpath(results_dir, "interface_shape.png"), fig_shape)
    
    return results_dir
end

results_dir = plot_simulation_results(residuals, xf_log, timestep_history, Ste)
println("\nSimulation results visualization saved to: $results_dir")
