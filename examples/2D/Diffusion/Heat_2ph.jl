using Penguin
using IterativeSolvers
using WriteVTK
using CairoMakie

### 2D Test Case : Diphasic Unsteady Diffusion Equation with a Disk
# Define the mesh
nx, ny = 160, 160
lx, ly = 8., 8.
x0, y0 = 0., 0.
domain = ((x0, lx), (y0, ly))
mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

# Define the body
radius, center = ly/4, (lx/2, ly/2)
circle = (x,y,_=0)->sqrt((x-center[1])^2 + (y-center[2])^2) - radius
circle_c = (x,y,_=0)->-(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

# Define the capacity
capacity = Capacity(circle, mesh)
capacity_c = Capacity(circle_c, mesh)

# Define the operators
operator = DiffusionOps(capacity)
operator_c = DiffusionOps(capacity_c)

# Define the boundary conditions
bc = Dirichlet(0.0)
bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:left => bc, :right => bc, :top => bc, :bottom => bc))

ic = InterfaceConditions(ScalarJump(1.0, 2.0, 0.0), FluxJump(1.0, 1.0, 0.0))

# Define the source term
f1 = (x,y,z,t)->0.0
f2 = (x,y,z,t)->0.0

# Define the phases
Fluide_1 = Phase(capacity, operator, f1, (x,y,z)->1.0)
Fluide_2 = Phase(capacity_c, operator_c, f2, (x,y,z)->1.0)

# Initial condition
u0ₒ1 = ones((nx+1)*(ny+1))
u0ᵧ1 = ones((nx+1)*(ny+1))
u0ₒ2 = zeros((nx+1)*(ny+1))
u0ᵧ2 = ones((nx+1)*(ny+1))
u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)

# Define the solver
Δt = 0.1
Tend = 5.0
solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "BE")

# Solve the problem
solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "BE"; method=Base.:\)

# Write the solution to a VTK file
#write_vtk("solution", mesh, solver)

# Plot the solution
#plot_solution(solver, mesh, circle, capacity, state_i=101)

# Plot the Profile
#plot_profile(solver, mesh; x=lx/2.01)

# Animation
#animate_solution(solver, mesh, circle)

function animate_diphasic_solution(
    solver, 
    mesh, 
    body1,
    body2,
    nx, ny, lx, ly, x0, y0;
    filename="diphasic_solution_animation.mp4",
    fps=10,
    title="Diphasic Heat Transfer",
    colorrange_bulk=(0,1),
    colorrange_interface=(0,1),
    colormap1=:thermal,
    colormap2=:viridis,
)
    body2 = isnothing(body2) ? body1 : body2

    
    # Create meshgrid for plotting
    x = range(x0, stop=x0+lx, length=nx+1)
    y = range(y0, stop=y0+ly, length=ny+1)
    
    # Number of frames
    num_frames = length(solver.states)
    
    # Create a figure with 2x2 layout
    fig = Figure(size=(1200, 900))
    
    # Create titles for each subplot
    titles = [
        "Bulk Field - Phase 1", 
        "Interface Field - Phase 1",
        "Bulk Field - Phase 2", 
        "Interface Field - Phase 2"
    ]
    
    # Create axes for each subplot
    ax_bulk1 = Axis3(fig[1, 1], 
                    title=titles[1],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface1 = Axis3(fig[1, 2], 
                        title=titles[2], 
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_bulk2 = Axis3(fig[2, 1], 
                    title=titles[3],
                    xlabel="x", ylabel="y", zlabel="Temperature")
    
    ax_interface2 = Axis3(fig[2, 2], 
                        title=titles[4],
                        xlabel="x", ylabel="y", zlabel="Temperature")
    
    # Add a main title
    Label(fig[0, :], title, fontsize=20)
    
    # Create colorbar for each phase
    Colorbar(fig[1, 3], colormap=colormap1, limits=colorrange_bulk, label="Temperature (Phase 1)")
    Colorbar(fig[2, 3], colormap=colormap2, limits=colorrange_bulk, label="Temperature (Phase 2)")
    
    # Set common view angles for 3D plots
    """
    viewangle = (45, 30)
    for ax in [ax_bulk1, ax_interface1, ax_bulk2, ax_interface2]
        ax.azimuth = viewangle[1]
        ax.elevation = viewangle[2]
    end
    """

    # Create a time label
    time_label = Label(fig[3, :], "t = 0.00", fontsize=16)
    
    # Create surface plots - will be updated in the animation
    bulk1_surface = surface!(ax_bulk1, x, y, zeros(nx+1, ny+1), 
                          colormap=colormap1, colorrange=colorrange_bulk)
    
    interface1_surface = surface!(ax_interface1, x, y, zeros(nx+1, ny+1), 
                               colormap=colormap1, colorrange=colorrange_interface)
    
    bulk2_surface = surface!(ax_bulk2, x, y, zeros(nx+1, ny+1), 
                          colormap=colormap2, colorrange=colorrange_bulk)
    
    interface2_surface = surface!(ax_interface2, x, y, zeros(nx+1, ny+1), 
                               colormap=colormap2, colorrange=colorrange_interface)
    
    # Create record of the animation
    println("Creating animation with $num_frames frames...")
    record(fig, filename, 1:num_frames; framerate=fps) do frame_idx
        # Extract the state at the current frame
        state = solver.states[frame_idx]
        
        # Extract solutions for each field
        u1_bulk = reshape(state[1:(nx+1)*(ny+1)], (nx+1, ny+1))
        u1_interface = reshape(state[(nx+1)*(ny+1)+1:2*(nx+1)*(ny+1)], (nx+1, ny+1))
        u2_bulk = reshape(state[2*(nx+1)*(ny+1)+1:3*(nx+1)*(ny+1)], (nx+1, ny+1))
        u2_interface = reshape(state[3*(nx+1)*(ny+1)+1:end], (nx+1, ny+1))
        
        # Compute phase indicators for masking
        phase1_indicator = zeros(nx+1, ny+1)
        phase2_indicator = zeros(nx+1, ny+1)
        
        # Compute mask for each phase
        for i in 1:nx+1
            for j in 1:ny+1
                phase1_indicator[i,j] = body1(x[i], y[j]) <= 0 ? 1.0 : NaN
                phase2_indicator[i,j] = body2(x[i], y[j]) <= 0 ? 1.0 : NaN
            end
        end
        
        # Apply masks to the solutions
        u1_bulk_masked = u1_bulk #.* phase1_indicator
        u1_interface_masked = u1_interface #.* phase1_indicator
        u2_bulk_masked = u2_bulk #.* phase2_indicator
        u2_interface_masked = u2_interface #.* phase2_indicator
        
        # Update surface plots with current data
        bulk1_surface[3] = u1_bulk_masked
        interface1_surface[3] = u1_interface_masked
        bulk2_surface[3] = u2_bulk_masked
        interface2_surface[3] = u2_interface_masked
        
        # Update time label
        time_t = round((frame_idx-1)*(Δt), digits=3)
        time_label.text = "t = $time_t"
        
        if frame_idx % 10 == 0
            println("Processing frame $frame_idx / $num_frames")
        end
    end
    
    println("Animation saved to $filename")
end

# Example usage
function create_diphasic_animation(solver, mesh)
    # Define the body functions
    circle = (x,y,_=0) -> sqrt((x-center[1])^2 + (y-center[2])^2) - radius
    circle_c = (x,y,_=0) -> -(sqrt((x-center[1])^2 + (y-center[2])^2) - radius)
    
    # Create animation
    animate_diphasic_solution(
        solver, 
        mesh, 
        circle,
        circle_c,
        nx, ny, lx, ly, x0, y0,
        filename="diphasic_heat_transfer.mp4",
        fps=15,
        colorrange_bulk=(0, 0.5),
        colorrange_interface=(0, 0.5),
        colormap1=:viridis,
        colormap2=:viridis,
    )
end

# Run the animation function
create_diphasic_animation(solver, mesh)


"""

function compute_sherwood_all(solver, capacity, capacity_c, Δt, He, L, D)
    nx = size(capacity.V, 1) - 1

    # Precompute volumes
    Vg   = sum(capacity.V)
    Vl   = sum(capacity_c.V)
    Vg_i = [capacity.V[i, i] for i in 1:nx+1]
    Vl_i = [capacity_c.V[i, i] for i in 1:nx+1]
    Γ    = sum(capacity.Γ)

    # Store Sherwood numbers
    Sh_values = Float64[]

    # Loop over consecutive states
    for i in 2:length(solver.states)
        u_nm1 = solver.states[i-1]
        u_n   = solver.states[i]

        cgω_nm1 = u_nm1[1:nx+1]                  # gas in ω at t_{n-1}
        clω_nm1 = u_nm1[2*(nx+1)+1:3*(nx+1)]     # liquid in ω at t_{n-1}

        cgω_n   = u_n[1:nx+1]                    # gas in ω at t_n
        clω_n   = u_n[2*(nx+1)+1:3*(nx+1)]       # liquid in ω at t_n

        # Average concentrations
        cgω̅_nm1 = sum(cgω_nm1 .* Vg_i) / Vg
        clω̅_nm1 = sum(clω_nm1 .* Vl_i) / Vl
        cgω̅_n   = sum(cgω_n   .* Vg_i) / Vg
        clω̅_n   = sum(clω_n   .* Vl_i) / Vl

        # Average concentrations at t_{n+1/2}
        cgω̅_n2 = 0.5 * (cgω̅_n + cgω̅_nm1)
        clω̅_n2 = 0.5 * (clω̅_n + clω̅_nm1)

        # Mass transfer rate
        # (Difference between latest and previous step) / (Γ * Δt * (He * cgω̅_{n+1/2} - clω̅_{n+1/2}))
        numerator   = (cgω̅_n - cgω̅_nm1)
        denominator = Γ * Δt * (He*cgω̅_n2 - clω̅_n2)
        k = numerator / denominator

        # Sherwood
        Sh = k * L / D
        push!(Sh_values, Sh)
    end

    return abs.(Sh_values)
end

# Compute Sherwood number
Sh_val = compute_sherwood_all(solver, capacity, capacity_c, Δt, 2.0, lx, 1.0)

# Save Sherwood number in a file
open("Sherwood_number_$nx.txt", "w") do io
    for i in 1:length(Sh_val)
        println(io, Sh_val[i])
    end
end

# Plot Sherwood number
fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="Sh", title="Sherwood number")
scatter!(ax, Sh_val, color=:blue, label="Sherwood number")
axislegend(ax, position=:rt)
display(fig)

# Analytical solution
using QuadGK
using SpecialFunctions

Dg,Dl = 1.0, 1.0
R0 = radius
cg0, cl0 = 1.0, 0.0
He = 0.5

D = sqrt(Dg/Dl)

function Phi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*R0)*bessely0(D*u*R0)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*R0)*bessely1(D*u*R0)
    return term1 - term2
end

function Psi(u)
    term1 = Dg*sqrt(Dl)*besselj1(u*R0)*besselj0(D*u*R0)
    term2 = He*Dl*sqrt(Dg)*besselj0(u*R0)*besselj1(D*u*R0)
    return term1 - term2
end

function cg_integrand(u, r, t)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u^2*(Φu^2 + Ψu^2)
    num   = exp(-Dg*u^2*t)*besselj0(u*r)*besselj1(u*R0)
    return iszero(denom) ? 0.0 : num/denom
end

function cl_integrand(u, r, t)
    Φu = Phi(u)
    Ψu = Psi(u)
    denom = u*(Φu^2 + Ψu^2)
    term1 = besselj0(D*u*r)*Φu
    term2 = bessely0(D*u*r)*Ψu
    num   = exp(-Dg*u^2*t)*besselj1(u*R0)*(term1 - term2)
    return iszero(denom) ? 0.0 : num/denom
end

function compute_cg(r_values, t_values)
    prefactor = (4*cg0*Dg*Dl*Dl*He)/(π^2*R0)
    cg_results = Matrix{Float64}(undef, length(t_values), length(r_values))
    for (i, t) in pairs(t_values)
        Umax = 5.0/sqrt(Dg*t)
        for (j, r) in pairs(r_values)
            val, _ = quadgk(u->cg_integrand(u, r, t), 0, Umax; atol=1e-6, rtol=1e-6)
            cg_results[i, j] = prefactor*val
        end
    end
    return cg_results
end

function compute_cl(r_values, t_values)
    prefactor = (2*cg0*Dg*sqrt(Dl)*He)/π
    cl_results = Matrix{Float64}(undef, length(t_values), length(r_values))
    for (i, t) in pairs(t_values)
        Umax = 5.0/sqrt(Dg*t)
        for (j, r) in pairs(r_values)
            val, _ = quadgk(u->cl_integrand(u, r, t), 0, Umax; atol=1e-6, rtol=1e-6)
            cl_results[i, j] = prefactor*val
        end
    end
    return cl_results
end

r_values_inside = range(1e-6, stop=R0, length=100)
r_values_outside = range(R0, stop=4*R0, length=100)
r_values = range(1e-6, stop=4*R0, length=nx+1)
t_values = [1.0]

cg_vals = compute_cg(collect(r_values), t_values)
cl_vals = compute_cl(collect(r_values), t_values)

# Plot the analytical solution and the numerical solution (profile)
x=2.01
y = range(mesh.x0[2], stop=mesh.x0[2]+mesh.h[2][1]*length(mesh.h[2]), length=length(mesh.h[2])+1)
x_idx = round(Int, (x - mesh.x0[1]) / mesh.h[1][1])
state = solver.states[end]
u1ₒ = reshape(state[1:length(state) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))[x_idx, 1:div(length(mesh.centers[2])+1, 2)]
u2ₒ = reshape(state[2*length(state) ÷ 4 + 1:3*length(state) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))[x_idx, 1:div(length(mesh.centers[2])+1, 2)]

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="r", ylabel="c", title="Analytical solution")
lines!(ax, r_values, cg_vals[1, :], color=:blue, linewidth=2, label="Analytical solution - Phase 1")
scatter!(ax, r_values, u1ₒ, color=:red, label="Numerical solution - Phase 1")
lines!(ax, r_values, cl_vals[1, :], color=:red, linewidth=2, label="Analytical solution - Phase 2")
scatter!(ax, r_values,u2ₒ, color=:blue, label="Numerical solution - Phase 2")
axislegend(ax)
display(fig)
"""
