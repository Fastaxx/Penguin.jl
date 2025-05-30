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


### 2D Test Case: Frank Sphere (Stefan Problem with Circular Interface)
# Define parameters
Stefan_number = 2.5
ρL = 1.0
Pe = 1.0
initial_radius = 0.5

# Find Lambda (interface position in similarity coordinates)

function F_similarity(s)
    # E₁ function from SpecialFunctions
    return expint(s^2/4)
end

function solve_for_lambda(S)
    # Solve using a simple bisection method
    f(λ) = S*λ^2/4*exp(λ^2/4)*F_similarity(λ) - 1.0
    
    λ_min, λ_max = 0.1, 10.0
    while λ_max - λ_min > 1e-10
        λ_mid = (λ_min + λ_max)/2
        if f(λ_mid) > 0
            λ_max = λ_mid
        else
            λ_min = λ_mid
        end
    end
    return (λ_min + λ_max)/2
end

Λ = solve_for_lambda(Stefan_number)
println("Solution parameter Λ = $Λ")

# Interface radius function
function exact_radius(t)
    if t <= 0.0
        return 0.1  # Small initial radius
    else
        return Λ*sqrt(t/Pe)
    end
end

# Temperature field function
function temperature(r, t)
    if t <= 0.0
        return 1.0
    end
    
    s = r * sqrt(Pe/t)
    if s < 1e-10  # Avoid singularity
        return 1.0
    end
    return F_similarity(s) / F_similarity(Λ)
end

# Define the spatial mesh
nx, ny = 32, 32
lx, ly = 2.0, 2.0
x0, y0 = -1.0, -1.0
Δx, Δy = lx/nx, ly/ny
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Time settings
Δt = 0.01
t_end = 1.0

# Create a global front tracking object to reuse
# Using the Julia FrontTracker implementation
global_front = FrontTracker()
create_circle!(global_front, 0.0, 0.0, initial_radius, 50)

# Compute volume Jacobian for the mesh
# Extract x and y face positions from mesh
x_faces = mesh.nodes[1]
y_faces = mesh.nodes[2]
jacobian = compute_volume_jacobian(global_front, x_faces, y_faces)

# Modify the body function to use the Julia implementation
function body(x, y, t, _ = 0)
    # Use the Julia sdf implementation
    return sdf(global_front, x, y)
end

# Define the Space-Time mesh
t0 = 0.0
STmesh = Penguin.SpaceTimeMesh(mesh, [t0, t0+Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh; compute_centroids=false)

# Calculate initial heights
Vₙ₊₁ = capacity.A[3][1:end÷2, 1:end÷2]
Vₙ = capacity.A[3][end÷2+1:end, end÷2+1:end]
Vₙ = reshape(diag(Vₙ), (nx+1, ny+1))
Vₙ₊₁ = reshape(diag(Vₙ₊₁), (nx+1, ny+1))

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(1.0)  # Temperature = 1.0 at the boundary
bc_b = Dirichlet(-0.5)  # Temperature = 0.0 at the boundary
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:bottom => bc_b, 
                                                       :top => bc_b, 
                                                       :left => bc_b, 
                                                       :right => bc_b))

# Stefan condition at the interface
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρL))

# Define the source term (no source)
f = (x,y,z,t) -> 0.0
K = (x,y,z) -> 1.0  # Thermal conductivity

Fluide = Phase(capacity, operator, f, K)

# Create initial temperature field using the analytical solution
function initial_temperature_field()
    t_init = t0  # Initial time
    temp = zeros((nx+1)*(ny+1))
    
    for i in 1:nx+1
        for j in 1:ny+1
            idx = (i-1)*(ny+1) + j
            x = x0 + (i-1) * Δx
            y = y0 + (j-1) * Δy
            
            r = sqrt(x^2 + y^2)  # Distance from origin
            if r <= initial_radius
                temp[idx] = 0.0  # Solid phase at melting temperature
            else
                temp[idx] = temperature(r, t_init)  # Liquid phase temperature
            end
        end
    end
    
    return temp
end

# Plot initial interface position and temperature field
function plot_initial_conditions()
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Initial Interface and Temperature Field")
    
    # Plot initial interface
    markers = get_markers(global_front)
    
    # Extract x and y coordinates from the tuples
    x_coords = [m[1] for m in markers]
    y_coords = [m[2] for m in markers]
    
    scatter!(ax, x_coords, y_coords, color=:blue, label="Initial Interface", markersize=5)
    
    # Plot initial temperature field
    temp_field = initial_temperature_field()
    temp_grid = reshape(temp_field, (nx+1, ny+1))
    hm = heatmap!(ax, temp_grid', colormap=:viridis, label="Temperature Field")
    Colorbar(fig[1, 2], hm, label="Temperature")
    return fig
end

# Set up initial condition
u0ₒ = initial_temperature_field()
u0ₒ = zeros((nx+1)*(ny+1))  # Initial temperature field
u0ᵧ = ones((nx+1)*(ny+1))  # Auxiliary variable
u0 = vcat(u0ₒ, u0ᵧ)

# Define the solver
Newton_params = (10000, 1e-6, 1e-6, 1.0)  # (max_iter, tol, reltol, α)
solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log, timestep_history = solve_StefanMono2D!(solver, Fluide, global_front, Δt, t_end, bc_b, bc, stef_cond, mesh, "BE";
   Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\)

# Pb front at n and n+1 are in some place at the same value, so VOFI crashes

# Plot the results
function plot_interface_evolution(xf_log::Dict{Int, Vector{Tuple{Float64, Float64}}})
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1], 
             title="Interface Evolution Over Time", 
             xlabel="x", ylabel="y",
             aspect=DataAspect())
    
    # Get all timesteps sorted
    timesteps = sort(collect(keys(xf_log)))
    n_steps = length(timesteps)
    
    # Create a color gradient
    colors = cgrad(:viridis, n_steps, categorical=true)
    
    # Plot each timestep's interface
    for (i, step) in enumerate(timesteps)
        markers = xf_log[step]
        
        # Extract x and y coordinates
        x_coords = [m[1] for m in markers]
        y_coords = [m[2] for m in markers]
        
        # For closed curves, add the first point again at the end
        if sqrt((x_coords[1] - x_coords[end])^2 + (y_coords[1] - y_coords[end])^2) < 1e-10
            push!(x_coords, x_coords[1])
            push!(y_coords, y_coords[1])
        end
        
        # Plot the interface line
        lines!(ax, x_coords, y_coords, 
              color=colors[i], 
              linewidth=2,
              label=i == 1 ? "Initial" : (i == n_steps ? "Final" : ""))
        
        # Plot markers with smaller size
        scatter!(ax, x_coords, y_coords,
                color=colors[i],
                markersize=3)
    end
       # Add analytical solution for comparison
    times = timestep_history  # This is a list of Tuple{Float64, Float64}
    theta = range(0, 2π, length=100)
    
    for (i, time_tuple) in enumerate(times)
        if i == 1 || i == n_steps || i % (n_steps ÷ 5) == 0 # Only plot some analytical circles
            t = time_tuple[1]  # Extract the actual time value from the tuple
            r = exact_radius(t)
            x_circle = r .* cos.(theta)
            y_circle = r .* sin.(theta)
            
            lines!(ax, x_circle, y_circle, 
                  color=:black, 
                  linestyle=:dash,
                  linewidth=1.5,
                  label=i == 1 ? "Analytical" : "")
        end
    end
    
     # Extract the actual time values from the tuples
    time_values = [t[1] for t in times]  # Assuming time is stored in first element of tuple

    # Add legend
    axislegend(position=:rt)
    
    # Add title with problem parameters
    Label(fig[0, :], "Stefan Problem: Stefan Number = $(Stefan_number), Initial Radius = $(initial_radius)",
         fontsize=16)
    
    return fig
end

# Create and display the interface evolution plot
interface_evolution = plot_interface_evolution(xf_log)
display(interface_evolution)

# Save the figure
save("interface_evolution.png", interface_evolution)
