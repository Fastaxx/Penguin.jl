using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie

function run_mesh_convergence(
    nx_list::Vector{Int},
    ny_list::Vector{Int},
    radius::Float64,
    center::Tuple{Float64,Float64},
    u_analytical::Function;
    lx::Float64=4.0,
    ly::Float64=4.0,
    norm,
    relative::Bool=false
)

    h_vals = Float64[]
    err_vals = Float64[]
    err_full_vals = Float64[]
    err_cut_vals = Float64[]
    err_empty_vals = Float64[]

    for (nx, ny) in zip(nx_list, ny_list)
        # Build mesh
        x0, y0 = 0.0, 0.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))

        # Define the body
        circle = (x,y,_=0) -> (sqrt((x-center[1])^2 + (y-center[2])^2) - radius)

        # Define capacity/operator
        capacity = Capacity(circle, mesh)
        operator = DiffusionOps(capacity)

        # BC + solver
        bc_boundary = Dirichlet(0.0)
        bc_b = BorderConditions(Dict(
            :left   => bc_boundary,
            :right  => bc_boundary,
            :top    => bc_boundary,
            :bottom => bc_boundary
        ))
        phase = Phase(capacity, operator, (x,y,_)->4.0, (x,y,_)->1.0)
        solver = DiffusionSteadyMono(phase, bc_b, Dirichlet(0.0))

        solve_DiffusionSteadyMono!(solver; method=Base.:\)

        # Compute errors
        u_ana, u_num, global_err, full_err, cut_err, empty_err =
            check_convergence(u_analytical, solver, capacity, norm, relative)

        # Representative mesh size ~ 1 / min(nx, ny)
        push!(h_vals, 1.0 / min(nx, ny))

        push!(err_vals,       global_err)
        push!(err_full_vals,  full_err)
        push!(err_cut_vals,   cut_err)
        push!(err_empty_vals, empty_err)
    end

    # Model for curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end

    # Fit each on log scale: log(err) = p*log(h) + c
    log_h = log.(h_vals)

    function do_fit(log_err)
        fit_result = curve_fit(fit_model, log_h, log_err, [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end

    p_global, _ = do_fit(log.(err_vals))
    p_full,   _ = do_fit(log.(err_full_vals))
    p_cut,    _ = do_fit(log.(err_cut_vals))

    # Round
    p_global = round(p_global, digits=2)
    p_full   = round(p_full, digits=2)
    p_cut    = round(p_cut, digits=2)

    println("Estimated order of convergence (global) = ", p_global)
    println("Estimated order of convergence (full)   = ", p_full)
    println("Estimated order of convergence (cut)    = ", p_cut)

    # Plot in log-log scale
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "h",
        ylabel = "L$norm error",
        title  = "Convergence plot",
        xscale = log10,
        yscale = log10
    )

    scatter!(ax, h_vals, err_vals,       label="All cells ($p_global)", markersize=12)
    lines!(ax, h_vals, err_vals,         color=:black)
    scatter!(ax, h_vals, err_full_vals,  label="Full cells ($p_full)",   markersize=12)
    lines!(ax, h_vals, err_full_vals,    color=:black)
    scatter!(ax, h_vals, err_cut_vals,   label="Cut cells ($p_cut)",     markersize=12)
    lines!(ax, h_vals, err_cut_vals,     color=:black)

    lines!(ax, h_vals, 10.0*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax, h_vals, 1.0*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    axislegend(ax, position=:rb)
    display(fig)

    return (
        h_vals,
        err_vals,
        err_full_vals,
        err_cut_vals,
        err_empty_vals,
        p_global,
        p_full,
        p_cut,
    )
end

# Example usage:
nx_list = [20, 40, 80, 160, 320, 640]
ny_list = [20, 40, 80, 160, 320, 640]
radius, center = 1.0, (2.0, 2.0)
u_analytical(x,y) = 1.0 - (x-center[1])^2 - (y-center[2])^2
h_vals, err_vals, err_full_vals, err_cut_vals, err_empty_vals, p_global, p_full, p_cut = run_mesh_convergence(nx_list, ny_list, radius, center, u_analytical, norm=2, relative=false)

# Write the output to a file
open("convergence.txt", "w") do io
    println(io, "h, err, err_full, err_cut, err_empty")
    for (h, err, err_full, err_cut, err_empty) in zip(h_vals, err_vals, err_full_vals, err_cut_vals, err_empty_vals)
        println(io, "$h, $err, $err_full, $err_cut, $err_empty")
    end
end
