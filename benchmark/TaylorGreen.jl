using Penguin
using LinearAlgebra
using Printf

# Convergence test: volume-integrated L2 norm of error (exclude boundary values)
# Taylor–Green vortex on [0, 2π]×[0, 2π]
# IMPORTANT: This script uses StokesMono (unsteady Stokes, no advection term).
# The classical Taylor–Green pressure p_TG(x,y,t) = -(ρ/4)(cos 2kx + cos 2ky) e^{-4νk²t}
# is the Navier–Stokes pressure that balances the convective term ρ(u·∇)u.
# For the unforced Stokes equations with the same velocity field,
# ρ ∂u/∂t - μ Δu = 0, so -∇p = 0 and p is constant.
# Comparing StokesMono to p_TG therefore yields a non-convergent pressure error.
# Use `use_NS_pressure = false` (default) to compare against the correct Stokes pressure (constant).
# If you want to compare against the NS pressure, you must either
#   - solve the full NS equations, or
#   - add a manufactured body force f = -ρ (u·∇)u so that (u,p_TG) solves forced Stokes.
# The current Fluid API passes a single scalar fᵤ to both components, so that change
# is not wired here.
# Keep t_end small enough that temporal accuracy is not the dominant effect.

Lx = 2π
Ly = 2π
x0 = 0.0
y0 = 0.0

μ = 1.0      # dynamic viscosity
ρ = 1.0
k = 1.0      # fundamental wavenumber
ν = μ/ρ

t_end = 0.1
Δt = 0.01
scheme = :CN

# analytical fields
u_exact = (x,y,t) ->  sin(k*x) * cos(k*y) * exp(-2.0*ν*k^2*t)
v_exact = (x,y,t) -> -cos(k*x) * sin(k*y) * exp(-2.0*ν*k^2*t)
use_NS_pressure = false
if use_NS_pressure
    # NS pressure (balances convection); not appropriate for unforced Stokes
    p_exact = (x,y,t) -> -(ρ/4.0) * (cos(2k*x) + cos(2k*y)) * exp(-4.0*ν*k^2*t)
else
    # Correct Stokes pressure for this manufactured velocity: constant (set to 0)
    p_exact = (x,y,t) -> 0.0
end

# Notes on forcing:
# - For unforced Stokes with the chosen (u,v), the required body force is f ≡ 0 and p is constant.
# - To recover the NS Taylor–Green pair (u, p_TG) using the Stokes solver, the manufactured body
#   force must be f = -ρ (u·∇)u (NOT ∇p). The current script leaves f = 0.0
#   so that (u,v,p) with p constant is the correct solution.

# Helper: exclude boundary indices
interior_indices(n) = 2:(n-1)

# list of resolutions (staggered grids use same counts for ux, uy, p here)
ns = [8, 16, 32, 64, 128]

errors_u = Float64[]
errors_v = Float64[]
errors_p = Float64[]
hs = Float64[]
xs_ux_plot = nothing; ys_ux_plot = nothing
xs_uy_plot = nothing; ys_uy_plot = nothing
Xp_plot = nothing; Yp_plot = nothing
Ux_plot = nothing; Uy_plot = nothing; P_plot = nothing

for n in ns
    nx = n; ny = n
    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5*dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5*dy))

    body = (x, y, _=0) -> -1.0  # entire domain is fluid

    capacity_ux = Capacity(body, mesh_ux)
    capacity_uy = Capacity(body, mesh_uy)
    capacity_p  = Capacity(body, mesh_p)

    operator_ux = DiffusionOps(capacity_ux)
    operator_uy = DiffusionOps(capacity_uy)
    operator_p  = DiffusionOps(capacity_p)

    # time-dependent Dirichlet BCs (assumes API accepts (x,y,t)->val)
    ux_bc_left   = Dirichlet((x,y,t)->u_exact(x,y,t))
    ux_bc_right  = Dirichlet((x,y,t)->u_exact(x,y,t))
    ux_bc_bottom = Dirichlet((x,y,t)->u_exact(x,y,t))
    ux_bc_top    = Dirichlet((x,y,t)->u_exact(x,y,t))

    uy_bc_left   = Dirichlet((x,y,t)->v_exact(x,y,t))
    uy_bc_right  = Dirichlet((x,y,t)->v_exact(x,y,t))
    uy_bc_bottom = Dirichlet((x,y,t)->v_exact(x,y,t))
    uy_bc_top    = Dirichlet((x,y,t)->v_exact(x,y,t))

    bc_ux = BorderConditions(Dict(
        :left=>ux_bc_left, :right=>ux_bc_right, :bottom=>ux_bc_bottom, :top=>ux_bc_top
    ))
    bc_uy = BorderConditions(Dict(
        :left=>uy_bc_left, :right=>uy_bc_right, :bottom=>uy_bc_bottom, :top=>uy_bc_top
    ))
    bc_p = BorderConditions(Dict{Symbol,AbstractBoundary}())

    u_bc = Dirichlet(0.0)  # pressure

    # build fluid; pass forcing functions that accept (x,y,t)
    f_ux = (x,y,z=0.0) -> 0.0 # unforced Stokes (see note above)
    f_p = (x,y,z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (capacity_ux, capacity_uy),
                  (operator_ux, operator_uy),
                  mesh_p,
                  capacity_p,
                  operator_p,
                  μ, ρ, f_ux, f_p)

    # initial state vector: use analytic at t=0 sampled on staggered nodes
    global xs_ux = mesh_ux.nodes[1]; global ys_ux = mesh_ux.nodes[2]
    global xs_uy = mesh_uy.nodes[1]; global ys_uy = mesh_uy.nodes[2]
    global Xp = mesh_p.nodes[1]; global Yp = mesh_p.nodes[2]

    nu = prod(operator_ux.size)
    np = prod(operator_p.size)
    x0_vec = zeros(4*nu + np)

    # place analytic initial condition into x0_vec (staggered layout matching solver expectation)
    # uωx then something then uωy then something then p (indices follow example layout)
    uvec = zeros(nu); vvec = zeros(nu); pvec = zeros(np)

    # fill u (ux) on mesh_ux grid
    ix = 1
    for j in 1:length(ys_ux), i in 1:length(xs_ux)
        uvec[ix] = u_exact(xs_ux[i], ys_ux[j], 0.0)
        ix += 1
    end

    # fill v (uy) on mesh_uy grid
    iy = 1
    for j in 1:length(ys_uy), i in 1:length(xs_uy)
        vvec[iy] = v_exact(xs_uy[i], ys_uy[j], 0.0)
        iy += 1
    end

    # fill p on pressure grid
    ip = 1
    for j in 1:length(Yp), i in 1:length(Xp)
        pvec[ip] = p_exact(Xp[i], Yp[j], 0.0)
        ip += 1
    end

    # assemble initial state matching solver's expected layout (as in example)
    x0_vec[1:nu] .= uvec
    x0_vec[2nu+1:3nu] .= vvec
    x0_vec[4nu+1:end] .= pvec

    solver = StokesMono(fluid, (bc_ux, bc_uy), bc_p, u_bc; x0=x0_vec)

    @printf("Running resolution %d×%d ...\n", nx, ny)
    times, states = solve_StokesMono_unsteady!(solver; Δt=Δt, T_end=t_end, scheme=scheme, method=Base.:\)

    final = states[end]
    u_num = final[1:nu]
    v_num = final[2nu+1:3nu]
    p_num = final[4nu+1:end]

    # reshape to 2D arrays for easier interior masking
    global Ux = reshape(u_num, (length(xs_ux), length(ys_ux)))
    global Uy = reshape(v_num, (length(xs_uy), length(ys_uy)))
    global P  = reshape(p_num,  (length(Xp), length(Yp)))
    println(P[1:5,1:5])  # print a few values to check
    # compute analytic samples at final time on the same staggered nodes
    Ux_ex = [u_exact(xs_ux[i], ys_ux[j], t_end) for i in 1:length(xs_ux), j in 1:length(ys_ux)]
    Uy_ex = [v_exact(xs_uy[i], ys_uy[j], t_end) for i in 1:length(xs_uy), j in 1:length(ys_uy)]
    P_ex  = [p_exact(Xp[i], Yp[j], t_end) for i in 1:length(Xp), j in 1:length(Yp)]

    
    # exclude boundary nodes (first and last indices in each direction)
    ix_range_u = interior_indices(length(xs_ux))
    iy_range_u = interior_indices(length(ys_ux))
    ix_range_v = interior_indices(length(xs_uy))
    iy_range_v = interior_indices(length(ys_uy))
    ix_range_p = interior_indices(length(Xp))
    iy_range_p = interior_indices(length(Yp))

    # vectorize interior errors and weight by capacity volumes (operator.V diagonal)
    Vux = diag(operator_ux.V)
    Vuy = diag(operator_uy.V)
    Vp  = diag(operator_p.V)

    # function to sum (error^2 * volume) over interior only
    function weighted_L2_grid(num, ex, mask_i, mask_j, Vdiag; remove_mean=false)
        ni, nj = size(num, 1), size(num, 2)
        total_w = 0.0
        weighted_sum = 0.0
        for j in 1:nj, i in 1:ni
            if (i in mask_i) && (j in mask_j)
                lin = (j-1)*ni + i
                w = Vdiag[lin]
                err = num[i,j] - ex[i,j]
                total_w += w
                weighted_sum += w * err
            end
        end
        mean_err = (remove_mean && total_w > 0) ? (weighted_sum / total_w) : 0.0

        accum = 0.0
        for j in 1:nj, i in 1:ni
            if (i in mask_i) && (j in mask_j)
                lin = (j-1)*ni + i
                w = Vdiag[lin]
                err = (num[i,j] - ex[i,j]) - mean_err
                accum += w * err^2
            end
        end
        return sqrt(accum)
    end

    err_u = weighted_L2_grid(Ux, Ux_ex, ix_range_u, iy_range_u, Vux)
    err_v = weighted_L2_grid(Uy, Uy_ex, ix_range_v, iy_range_v, Vuy)
    # When use_NS_pressure=false, p_exact≡0 and removing the mean leaves pure numerical pressure fluctuations.
    err_p = weighted_L2_grid(P,  P_ex,  ix_range_p, iy_range_p, Vp; remove_mean=true)

    push!(errors_u, err_u)
    push!(errors_v, err_v)
    push!(errors_p, err_p)
    push!(hs, max(Lx/(nx), Ly/(ny)))

    # store current (will end up holding last iteration's) arrays for plotting later
    xs_ux_plot = xs_ux; ys_ux_plot = ys_ux
    xs_uy_plot = xs_uy; ys_uy_plot = ys_uy
    Xp_plot = Xp; Yp_plot = Yp
    Ux_plot = Ux; Uy_plot = Uy; P_plot = P

    @printf("  h=%.5e  ||u||_L2=%.5e  ||v||_L2=%.5e  ||p||_L2=%.5e\n", hs[end], err_u, err_v, err_p)
end

# compute convergence rates (slope of log(error) vs log(h))
function rate(h, e)
    r = []
    for i in 2:length(e)
        push!(r, log(e[i]/e[i-1]) / log(h[i]/h[i-1]))
    end
    return r
end

r_u = rate(hs, errors_u)
r_v = rate(hs, errors_v)
r_p = rate(hs, errors_p)

println("\nEstimated convergence rates (between successive resolutions):")
for i in 1:length(r_u)
    @printf("  between %d and %d: u rate=%.2f, v rate=%.2f, p rate=%.2f\n",
            ns[i], ns[i+1], r_u[i], r_v[i], r_p[i])
end

println("\nFinal errors:")
for (i,n) in enumerate(ns)
    @printf("  %4d: h=%.3e  ||u||=%.5e  ||v||=%.5e  ||p||=%.5e\n",
            n, hs[i], errors_u[i], errors_v[i], errors_p[i])
end

# Save data as CSV
using CSV
using DataFrames
df = DataFrame(h=hs, error_u=errors_u, error_v=errors_v, error_p=errors_p)
CSV.write("taylor_green_convergence.csv", df)

# Plotting with CairoMakie
using CairoMakie

# Plot convergence (log-log)
fig = Figure(resolution=(900,500))
ax = Axis(fig[1,1], xscale = log10, yscale = log10,
          xlabel = "h", ylabel = "volume-integrated L2 error",
          title = "Taylor–Green convergence (t = $(t_end))")

lines!(ax, hs, errors_u; label="u", color=:tomato)
scatter!(ax, hs, errors_u; color=:tomato)
lines!(ax, hs, errors_v; label="v", color=:royalblue)
scatter!(ax, hs, errors_v; color=:royalblue)

# reference slope (second order)
p_ref = 2.0
h_ref = [minimum(hs), maximum(hs)]
ref_line = errors_u[1] * (h_ref ./ hs[1]).^p_ref
lines!(ax, h_ref, ref_line; color=:black, linestyle=:dash, label="O(h²)")

axislegend(ax, position = :rb)
save("taylor_green_convergence.png", fig)
display(fig)

# Plot highest-resolution fields (use arrays from the last loop iteration)
fig_snap = Figure(resolution=(1200, 400))
# u_x
ax1 = Axis(fig_snap[1,1], title = "u_x (n=$(last(ns)), t=$(t_end))", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, xs_ux, ys_ux, Ux; colormap = :viridis)
Colorbar(fig_snap[1,2], hm1)

# u_y
ax2 = Axis(fig_snap[1,3], title = "u_y (n=$(last(ns)), t=$(t_end))", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, xs_uy, ys_uy, Uy; colormap = :thermal)
Colorbar(fig_snap[1,4], hm2)

# pressure
ax3 = Axis(fig_snap[1,5], title = "p (n=$(last(ns)), t=$(t_end))", xlabel="x", ylabel="y")
hm3 = heatmap!(ax3, Xp, Yp, P; colormap = :balance)
Colorbar(fig_snap[1,6], hm3)

save("taylor_green_highest_resolution_fields.png", fig_snap)
display(fig_snap)
