using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie

### 1D Test Case : One-phase Stefan Problem
# Define the spatial mesh
nx = 40
lx = 1.
x0 = 0.
domain = ((x0, lx),)
mesh = Penguin.Mesh((nx,), (lx,), (x0,))

# Define the body
xf = 0.05*lx   # Interface position
body = (x,t, _=0)->(x - xf)

# Define the Space-Time mesh
Δt = 0.001
Tend = 0.1
STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, Δt], tag=mesh.tag)

# Define the capacity
capacity = Capacity(body, STmesh)

# Define the diffusion operator
operator = DiffusionOps(capacity)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc1 = Dirichlet(0.0)

bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => Dirichlet(0.0), :bottom => Dirichlet(1.0)))
ρ, L = 1.0, 1.0
stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, ρ*L))

# Define the source term
f = (x,y,z,t)-> 0.0 #sin(x)*cos(10*y)
K = (x,y,z)-> 1.0

# Define the phase
Fluide = Phase(capacity, operator, f, K)

# Initial condition
u0ₒ = zeros((nx+1))
u0ᵧ = zeros((nx+1))
u0 = vcat(u0ₒ, u0ᵧ)

# Newton parameters
max_iter = 1000
tol = 1e-5
reltol = 1e-5
α = 1.0
Newton_params = (max_iter, tol, reltol, α)

# Define the solver
solver = MovingLiquidDiffusionUnsteadyMono(Fluide, bc_b, bc, Δt, u0, mesh, "BE")

# Solve the problem
solver, residuals, xf_log = solve_MovingLiquidDiffusionUnsteadyMono!(solver, Fluide, xf, Δt, Tend, bc_b, bc, stef_cond, mesh, "BE"; Newton_params=Newton_params, method=Base.:\)

# Animation
animate_solution(solver, mesh, body)

# Plot residuals   
#residuals[i] might be empty, remove them
residuals = filter(x -> !isempty(x), residuals)

figure = Figure()
ax = Axis(figure[1,1], xlabel = "Newton Iterations", ylabel = "Residuals", title = "Residuals")
for i in 1:length(residuals)
    lines!(ax, log10.(residuals[i]), label = "Time = $(i*Δt)")
end
#axislegend(ax)
display(figure)

# Plot the position
figure = Figure()
ax = Axis(figure[1,1], xlabel = "Time", ylabel = "Interface position", title = "Interface position")
lines!(ax, xf_log, label = "Interface position")
display(figure)

# Add convergence rate analysis for residuals
function analyze_convergence_rates(residuals)
    figure = Figure(resolution=(1000, 800))
    ax1 = Axis(figure[1, 1], 
               xlabel="Newton Iteration (k)", 
               ylabel="log₁₀(Residual)",
               title="Residual Convergence Analysis")
    
    # For rate estimates
    ax2 = Axis(figure[2, 1], 
               xlabel="Time Step", 
               ylabel="Convergence Rate (-slope)",
               title="Convergence Rate Evolution")
    
    rates = Float64[]
    times = Float64[]
    
    for (i, res) in enumerate(residuals)
        if length(res) < 3
            println("Skipping time step $i (not enough iterations)")
            continue
        end
        
        # Take log of residuals
        log_res = log10.(res)
        
        # Create iteration indices
        iterations = collect(1:length(log_res))
        
        # Only fit the linear part of the convergence (after initial iterations)
        # Try to automatically detect where linear convergence begins
        start_idx = 1
        for j in 2:length(log_res)-1
            # Check if we have steady decrease for 3 consecutive points
            if log_res[j] < log_res[j-1] && log_res[j+1] < log_res[j]
                start_idx = j
                break
            end
        end
        
        # Ensure we have enough points for a meaningful fit
        if length(log_res) - start_idx + 1 < 3
            println("Skipping time step $i (not enough data for linear fit)")
            continue
        end
        
        # Fit linear model log(residual) = A*iteration + B
        model(x, p) = p[1] .* x .+ p[2]
        linear_fit = curve_fit(model, iterations[start_idx:end], log_res[start_idx:end], [-1.0, 0.0])
        
        # Extract convergence rate
        rate = -linear_fit.param[1]  # Negative slope is the convergence rate
        push!(rates, rate)
        push!(times, i*Δt)
        
        # Plot data points and fit on first plot
        scatter!(ax1, iterations, log_res, label="Time = $(i*Δt)")
        
        # Plot fitted line
        fit_line = model(collect(start_idx:length(log_res)), linear_fit.param)
        lines!(ax1, start_idx:length(log_res), fit_line, 
               linestyle=:dash, linewidth=2, 
               label="Rate = $(round(rate, digits=2))")
        
        if mod(i, 15) == 0
            text!(ax1, 
                  iterations[end], log_res[end]-0.5, 
                  text="$(round(rate, digits=2))", 
                  fontsize=12, 
                  align=(:left, :bottom))
        end
    end
    
    # Plot the evolution of convergence rates
    scatter!(ax2, times, rates, markersize=10)
    lines!(ax2, times, rates)
    
    # Add average rate line
    avg_rate = sum(rates) / length(rates)
    hlines!(ax2, [avg_rate], color=:red, linestyle=:dash, 
            label="Average: $(round(avg_rate, digits=2))")
    
    # Format plots
    axislegend(ax2, position=:lb)
    
    # Calculate statistics
    println("Average convergence rate: $(avg_rate)")
    println("Minimum convergence rate: $(minimum(rates))")
    println("Maximum convergence rate: $(maximum(rates))")
    
    display(figure)
    return rates, times
end

# Call the analysis function
convergence_rates, time_steps = analyze_convergence_rates(residuals)

# Plot log-log of convergence rates if desired
figure = Figure()
ax = Axis(figure[1, 1],
         xlabel="Residual at iteration k",
         ylabel="Residual at iteration k+1",
         title="Convergence Order Analysis")

# For each time step, plot residual[k+1] vs residual[k] in log-log scale
for (i, res) in enumerate(residuals)
    if length(res) < 4  # Need at least 4 points for this analysis
        continue
    end
    
    # Skip the first iteration which might be far off
    x_vals = res[2:end-1]
    y_vals = res[3:end]
    
    scatter!(ax, x_vals, y_vals, label="Time = $(i*Δt)")
    
    # For reference, add lines showing quadratic and linear convergence
    if i == 1
        x_range = 10.0 .^ range(log10(minimum(x_vals)), log10(maximum(x_vals)), length=100)
        # Quadratic convergence: y ∝ x²
        quad_factor = y_vals[1] / x_vals[1]^2
        lines!(ax, x_range, quad_factor .* x_range.^2, 
              linestyle=:dash, color=:red, linewidth=2,
              label="Quadratic: r[k+1] ∝ r[k]²")
        
        # Linear convergence: y ∝ x
        lin_factor = y_vals[1] / x_vals[1]
        lines!(ax, x_range, lin_factor .* x_range, 
              linestyle=:dot, color=:blue, linewidth=2,
              label="Linear: r[k+1] ∝ r[k]")
    end
end

ax.xscale = log10
ax.yscale = log10
#axislegend(ax, position=:lt)
display(figure)

# save xf_log
open("xf_log_$nx.txt", "w") do io
    for i in 1:length(xf_log)
        println(io, xf_log[i])
    end
end

# Plot the solution
plot_solution(solver, mesh, body, capacity; state_i=10)

# create a directory to save solver.states[i]
if !isdir("solver_states_$nx")
    mkdir("solver_states_$nx")
end
# save solver.states[i]
for i in 1:length(solver.states)
    open("solver_states_$nx/solver_states_$i.txt", "w") do io
        for j in 1:length(solver.states[i])
            println(io, solver.states[i][j])
        end
    end
end