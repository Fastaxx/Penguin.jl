using Penguin, LsqFit, SparseArrays, LinearAlgebra
using IterativeSolvers
using CairoMakie
using SpecialFunctions
using Roots
using DelimitedFiles
using Printf

"""
Run mesh convergence study for diphasic heat transfer problem
"""
function run_diphasic_mesh_convergence(
    nx_list::Vector{Int},
    u1_analytical::Function,
    u2_analytical::Function;
    lx::Float64=8.0,
    x0::Float64=0.0,
    xint::Float64=4.0,
    Tend::Float64=0.5,
    He::Float64=0.5,
    D1::Float64=1.0,
    D2::Float64=1.0,
    norm::Real=2,
    relative::Bool=false
)
    # Initialize storage arrays
    h_vals = Float64[]
    
    # Global errors (all cells)
    err1_vals = Float64[]  # Phase 1
    err2_vals = Float64[]  # Phase 2
    err_combined_vals = Float64[]  # Combined phases
    
    # Full cell errors
    err1_full_vals = Float64[]  # Phase 1
    err2_full_vals = Float64[]  # Phase 2
    err_full_combined_vals = Float64[]  # Combined phases
    
    # Cut cell errors
    err1_cut_vals = Float64[]  # Phase 1
    err2_cut_vals = Float64[]  # Phase 2
    err_cut_combined_vals = Float64[]  # Combined phases

    # For each mesh resolution
    for nx in nx_list
        println("\n===== Testing mesh size nx = $nx =====")
        
        # Build mesh
        mesh = Penguin.Mesh((nx,), (lx,), (x0,))
        
        # Define the body
        body = (x, _=0) -> (x - xint)
        body_c = (x, _=0) -> -(x - xint)
        
        # Define the capacity
        capacity = Capacity(body, mesh)
        capacity_c = Capacity(body_c, mesh)
        
        # Define the operators
        operator = DiffusionOps(capacity)
        operator_c = DiffusionOps(capacity_c)
        
        # Define the boundary conditions
        bc1 = Dirichlet(0.0)
        bc0 = Dirichlet(1.0)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(:top => bc0, :bottom => bc1))
        
        # Interface conditions
        ic = InterfaceConditions(ScalarJump(1.0, 0.5, 0.0), FluxJump(1.0, 1.0, 0.0))
        
        # Define the source term
        f1 = (x,y,z,t)->0.0
        f2 = (x,y,z,t)->0.0
        
        # Define diffusion coefficients
        D1_func = (x,y,z)->D1
        D2_func = (x,y,z)->D2
        
        # Define the phases
        Fluide_1 = Phase(capacity, operator, f1, D1_func)
        Fluide_2 = Phase(capacity_c, operator_c, f2, D2_func)
        
        # Initial condition
        u0ₒ1 = zeros(nx+1)
        u0ᵧ1 = zeros(nx+1)
        u0ₒ2 = ones(nx+1)
        u0ᵧ2 = ones(nx+1)
        
        u0 = vcat(u0ₒ1, u0ᵧ1, u0ₒ2, u0ᵧ2)
        
        # Time step based on mesh size
        Δt = 0.5 * (lx/nx)^2
        
        # Define the solver
        solver = DiffusionUnsteadyDiph(Fluide_1, Fluide_2, bc_b, ic, Δt, u0, "CN")
        
        # Solve the problem
        solve_DiffusionUnsteadyDiph!(solver, Fluide_1, Fluide_2, Δt, Tend, bc_b, ic, "CN"; method=Base.:\)
        
        # Compute errors
        (ana_sols, num_sols, global_errs, full_errs, cut_errs, empty_errs) = 
            check_convergence_diph(u1_analytical, u2_analytical, solver, capacity, capacity_c, norm, relative)
        
        # Unpack error values
        (err1, err2, err_combined) = global_errs
        (err1_full, err2_full, err_full_combined) = full_errs
        (err1_cut, err2_cut, err_cut_combined) = cut_errs
        
        # Store mesh size and errors
        push!(h_vals, lx / nx)
        
        # Store global errors
        push!(err1_vals, err1)
        push!(err2_vals, err2)
        push!(err_combined_vals, err_combined)
        
        # Store full cell errors
        push!(err1_full_vals, err1_full)
        push!(err2_full_vals, err2_full)
        push!(err_full_combined_vals, err_full_combined)
        
        # Store cut cell errors
        push!(err1_cut_vals, err1_cut)
        push!(err2_cut_vals, err2_cut)
        push!(err_cut_combined_vals, err_cut_combined)
        
        # For the largest mesh, save a plot of the solution and error
        if nx == maximum(nx_list)
            # Create x coordinates
            x = range(x0, stop = lx, length = nx+1)
            
            # Extract solutions
            (u1_ana, u2_ana) = ana_sols
            (u1_num, u2_num) = num_sols
            
            # Mask empty cells with NaN
            u1_ana[capacity.cell_types .== 0] .= NaN
            u2_ana[capacity_c.cell_types .== 0] .= NaN
            u1_num[capacity.cell_types .== 0] .= NaN
            u2_num[capacity_c.cell_types .== 0] .= NaN
            
            # Plot solutions
            fig_sol = Figure()
            ax_sol = Axis(fig_sol[1, 1], 
                         xlabel="x", 
                         ylabel="u", 
                         title="Diphasic Solution (nx=$nx)")
            
            scatter!(ax_sol, x, u1_ana, color=:blue, label="Phase 1 Analytical")
            scatter!(ax_sol, x, u1_num, color=:red, label="Phase 1 Numerical")
            scatter!(ax_sol, x, u2_ana, color=:green, label="Phase 2 Analytical")
            scatter!(ax_sol, x, u2_num, color=:orange, label="Phase 2 Numerical")
            
            axislegend(ax_sol, position=:rb)
            display(fig_sol)
            
            # Plot errors
            err1 = abs.(u1_ana .- u1_num)
            err2 = abs.(u2_ana .- u2_num)
            
            fig_err = Figure()
            ax_err = Axis(fig_err[1, 1], 
                         xlabel="x", 
                         ylabel="Absolute Error", 
                         title="Diphasic Error (nx=$nx)",
                         yscale=log10)
            
            scatter!(ax_err, x, err1, color=:blue, label="Phase 1 Error")
            scatter!(ax_err, x, err2, color=:red, label="Phase 2 Error")
            
            axislegend(ax_err, position=:rt)
            display(fig_err)
        end
    end
    
    # Fit convergence rates
    # Model for curve_fit
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end
    
    # Fit each on log scale: log(err) = p*log(h) + c
    log_h = log.(h_vals)
    
    function do_fit(log_err, use_last_n=3)
        # Use only the last n points (default 3)
        n = min(use_last_n, length(log_h))
        idx = length(log_h) - n + 1 : length(log_h)
        
        # Fit using only those points
        fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
        return fit_result.param[1], fit_result.param[2]  # (p_est, c_est)
    end
    
    # Fit convergence rates for each phase and cell type (all points)
    p1_global_all, _ = do_fit(log.(err1_vals), length(err1_vals))
    p2_global_all, _ = do_fit(log.(err2_vals), length(err2_vals))
    p_combined_all, _ = do_fit(log.(err_combined_vals), length(err_combined_vals))
    
    p1_full_all, _ = do_fit(log.(err1_full_vals), length(err1_full_vals))
    p2_full_all, _ = do_fit(log.(err2_full_vals), length(err2_full_vals))
    p_full_combined_all, _ = do_fit(log.(err_full_combined_vals), length(err_full_combined_vals))
    
    p1_cut_all, _ = do_fit(log.(err1_cut_vals), length(err1_cut_vals))
    p2_cut_all, _ = do_fit(log.(err2_cut_vals), length(err2_cut_vals))
    p_cut_combined_all, _ = do_fit(log.(err_cut_combined_vals), length(err_cut_combined_vals))
    
    # Fit convergence rates using only last 3 points
    p1_global, _ = do_fit(log.(err1_vals), 3)
    p2_global, _ = do_fit(log.(err2_vals), 3)
    p_combined, _ = do_fit(log.(err_combined_vals), 3)
    
    p1_full, _ = do_fit(log.(err1_full_vals), 3)
    p2_full, _ = do_fit(log.(err2_full_vals), 3)
    p_full_combined, _ = do_fit(log.(err_full_combined_vals), 3)
    
    p1_cut, _ = do_fit(log.(err1_cut_vals), 3)
    p2_cut, _ = do_fit(log.(err2_cut_vals), 3)
    p_cut_combined, _ = do_fit(log.(err_cut_combined_vals), 3)
    
    # Round for display
    p1_global_all = round(p1_global_all, digits=2)
    p2_global_all = round(p2_global_all, digits=2)
    p_combined_all = round(p_combined_all, digits=2)
    
    p1_full_all = round(p1_full_all, digits=2)
    p2_full_all = round(p2_full_all, digits=2)
    p_full_combined_all = round(p_full_combined_all, digits=2)
    
    p1_cut_all = round(p1_cut_all, digits=2)
    p2_cut_all = round(p2_cut_all, digits=2)
    p_cut_combined_all = round(p_cut_combined_all, digits=2)
    
    p1_global = round(p1_global, digits=2)
    p2_global = round(p2_global, digits=2)
    p_combined = round(p_combined, digits=2)
    
    p1_full = round(p1_full, digits=2)
    p2_full = round(p2_full, digits=2)
    p_full_combined = round(p_full_combined, digits=2)
    
    p1_cut = round(p1_cut, digits=2)
    p2_cut = round(p2_cut, digits=2)
    p_cut_combined = round(p_cut_combined, digits=2)
    
    # Print convergence rates
    println("\n===== Convergence Rates =====")
    println("\n--- Global Errors (All Cells) ---")
    println("Phase 1: p = $p1_global (last 3), p = $p1_global_all (all)")
    println("Phase 2: p = $p2_global (last 3), p = $p2_global_all (all)")
    println("Combined: p = $p_combined (last 3), p = $p_combined_all (all)")
    
    println("\n--- Full Cell Errors ---")
    println("Phase 1: p = $p1_full (last 3), p = $p1_full_all (all)")
    println("Phase 2: p = $p2_full (last 3), p = $p2_full_all (all)")
    println("Combined: p = $p_full_combined (last 3), p = $p_full_combined_all (all)")
    
    println("\n--- Cut Cell Errors ---")
    println("Phase 1: p = $p1_cut (last 3), p = $p1_cut_all (all)")
    println("Phase 2: p = $p2_cut (last 3), p = $p2_cut_all (all)")
    println("Combined: p = $p_cut_combined (last 3), p = $p_cut_combined_all (all)")
    
    # Plot global errors
    fig_global = Figure()
    ax_global = Axis(
        fig_global[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Global Errors (All Cells)",
        xscale = log10,
        yscale = log10
    )
    
    scatter!(ax_global, h_vals, err1_vals, 
             label="Phase 1 (p = $p1_global)", 
             markersize=12, color=:blue)
    lines!(ax_global, h_vals, err1_vals, color=:blue)
    
    scatter!(ax_global, h_vals, err2_vals, 
             label="Phase 2 (p = $p2_global)", 
             markersize=12, color=:red)
    lines!(ax_global, h_vals, err2_vals, color=:red)
    
    scatter!(ax_global, h_vals, err_combined_vals, 
             label="Combined (p = $p_combined)", 
             markersize=12, color=:black)
    lines!(ax_global, h_vals, err_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_global, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_global, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    # Fix the error by getting the c_est value properly from the do_fit function
    # The issue is that you're trying to use "_" which is a write-only placeholder
    
    # Add fitted line for last 3 points (for Phase 1)
    last_3_idx = length(h_vals)-2:length(h_vals)
    h_range = exp.(range(log(h_vals[last_3_idx[1]]), log(h_vals[last_3_idx[end]]), length=100))
    
    # Get the coefficient for the fitted curve
    _, c_est_p1 = do_fit(log.(err1_vals), 3)
    c_est_p1 = exp(c_est_p1)  # Convert from log space
    
    # Use the coefficient to generate the fitted line
    err_fit1 = c_est_p1 * h_range.^p1_global
    lines!(ax_global, h_range, err_fit1, 
           color=:blue, linestyle=:dot, linewidth=2, 
           label="Last 3 fit (p = $p1_global)")
    
    axislegend(ax_global, position=:rb)
    display(fig_global)
    
    # Plot full cell errors
    fig_full = Figure()
    ax_full = Axis(
        fig_full[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Full Cell Errors",
        xscale = log10,
        yscale = log10
    )
    
    scatter!(ax_full, h_vals, err1_full_vals, label="Phase 1 ($p1_full)", markersize=12, color=:blue)
    lines!(ax_full, h_vals, err1_full_vals, color=:blue)
    
    scatter!(ax_full, h_vals, err2_full_vals, label="Phase 2 ($p2_full)", markersize=12, color=:red)
    lines!(ax_full, h_vals, err2_full_vals, color=:red)
    
    scatter!(ax_full, h_vals, err_full_combined_vals, label="Combined ($p_full_combined)", markersize=12, color=:black)
    lines!(ax_full, h_vals, err_full_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_full, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_full, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    axislegend(ax_full, position=:rb)
    display(fig_full)
    
    # Plot cut cell errors
    fig_cut = Figure()
    ax_cut = Axis(
        fig_cut[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Cut Cell Errors",
        xscale = log10,
        yscale = log10
    )
    
    scatter!(ax_cut, h_vals, err1_cut_vals, label="Phase 1 ($p1_cut)", markersize=12, color=:blue)
    lines!(ax_cut, h_vals, err1_cut_vals, color=:blue)
    
    scatter!(ax_cut, h_vals, err2_cut_vals, label="Phase 2 ($p2_cut)", markersize=12, color=:red)
    lines!(ax_cut, h_vals, err2_cut_vals, color=:red)
    
    scatter!(ax_cut, h_vals, err_cut_combined_vals, label="Combined ($p_cut_combined)", markersize=12, color=:black)
    lines!(ax_cut, h_vals, err_cut_combined_vals, color=:black)
    
    # Add reference lines
    lines!(ax_cut, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
    lines!(ax_cut, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    
    axislegend(ax_cut, position=:rb)
    display(fig_cut)
    
    # Create a comprehensive convergence plot for both phases
    fig_comp = Figure(resolution=(1200, 800))
    
    # Global errors panel
    ax_comp_global = Axis(
        fig_comp[1, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Global Errors",
        xscale = log10,
        yscale = log10
    )
    
    # Full cell errors panel
    ax_comp_full = Axis(
        fig_comp[1, 2],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Full Cell Errors",
        xscale = log10,
        yscale = log10
    )
    
    # Cut cell errors panel
    ax_comp_cut = Axis(
        fig_comp[2, 1],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Cut Cell Errors",
        xscale = log10,
        yscale = log10
    )
    
    # Combined errors panel
    ax_comp_combined = Axis(
        fig_comp[2, 2],
        xlabel = "Mesh size (h)",
        ylabel = "L$norm error",
        title  = "Combined Errors",
        xscale = log10,
        yscale = log10
    )
    
    # Plot in global panel
    scatter!(ax_comp_global, h_vals, err1_vals, label="Phase 1 ($p1_global)", markersize=10, color=:blue)
    lines!(ax_comp_global, h_vals, err1_vals, color=:blue)
    scatter!(ax_comp_global, h_vals, err2_vals, label="Phase 2 ($p2_global)", markersize=10, color=:red)
    lines!(ax_comp_global, h_vals, err2_vals, color=:red)
    
    # Plot in full cell panel
    scatter!(ax_comp_full, h_vals, err1_full_vals, label="Phase 1 ($p1_full)", markersize=10, color=:blue)
    lines!(ax_comp_full, h_vals, err1_full_vals, color=:blue)
    scatter!(ax_comp_full, h_vals, err2_full_vals, label="Phase 2 ($p2_full)", markersize=10, color=:red)
    lines!(ax_comp_full, h_vals, err2_full_vals, color=:red)
    
    # Plot in cut cell panel
    scatter!(ax_comp_cut, h_vals, err1_cut_vals, label="Phase 1 ($p1_cut)", markersize=10, color=:blue)
    lines!(ax_comp_cut, h_vals, err1_cut_vals, color=:blue)
    scatter!(ax_comp_cut, h_vals, err2_cut_vals, label="Phase 2 ($p2_cut)", markersize=10, color=:red)
    lines!(ax_comp_cut, h_vals, err2_cut_vals, color=:red)
    
    # Plot in combined panel
    scatter!(ax_comp_combined, h_vals, err_combined_vals, label="Global ($p_combined)", markersize=10, color=:black)
    lines!(ax_comp_combined, h_vals, err_combined_vals, color=:black)
    scatter!(ax_comp_combined, h_vals, err_full_combined_vals, label="Full ($p_full_combined)", markersize=10, color=:green)
    lines!(ax_comp_combined, h_vals, err_full_combined_vals, color=:green)
    scatter!(ax_comp_combined, h_vals, err_cut_combined_vals, label="Cut ($p_cut_combined)", markersize=10, color=:purple)
    lines!(ax_comp_combined, h_vals, err_cut_combined_vals, color=:purple)
    
    # Add reference slopes to all panels
    for ax in [ax_comp_global, ax_comp_full, ax_comp_cut, ax_comp_combined]
        lines!(ax, h_vals, 0.1*h_vals.^2.0, label="O(h²)", color=:black, linestyle=:dash)
        lines!(ax, h_vals, 0.1*h_vals.^1.0, label="O(h¹)", color=:black, linestyle=:dashdot)
    end
    
    # Add legends
    axislegend(ax_comp_global, position=:rb)
    axislegend(ax_comp_full, position=:rb)
    axislegend(ax_comp_cut, position=:rb)
    axislegend(ax_comp_combined, position=:rb)
    
    display(fig_comp)
    
        # Save errors and rates to a file
    open("diphasic_convergence_results.txt", "w") do io
        write(io, "# Diphasic Heat Transfer Convergence Study\n\n")
        
        # Write parameters
        write(io, "Parameters:\n")
        write(io, "  Mesh sizes: $nx_list\n")
        write(io, "  Norm: L$norm\n")
        write(io, "  Relative error: $relative\n")
        write(io, "  Interface position: $xint\n")
        write(io, "  Final time: $Tend\n")
        write(io, "  Henry coefficient: $He\n")
        write(io, "  Diffusion coefficients: D1=$D1, D2=$D2\n\n")
        
        # Write convergence rates
        write(io, "Convergence Rates:\n")
        write(io, "  Global Errors:\n")
        write(io, "    Phase 1: $p1_global (last 3), $p1_global_all (all)\n")
        write(io, "    Phase 2: $p2_global (last 3), $p2_global_all (all)\n")
        write(io, "    Combined: $p_combined (last 3), $p_combined_all (all)\n\n")
        
        write(io, "  Full Cell Errors:\n")
        write(io, "    Phase 1: $p1_full (last 3), $p1_full_all (all)\n")
        write(io, "    Phase 2: $p2_full (last 3), $p2_full_all (all)\n")
        write(io, "    Combined: $p_full_combined (last 3), $p_full_combined_all (all)\n\n")
        
        write(io, "  Cut Cell Errors:\n")
        write(io, "    Phase 1: $p1_cut (last 3), $p1_cut_all (all)\n")
        write(io, "    Phase 2: $p2_cut (last 3), $p2_cut_all (all)\n")
        write(io, "    Combined: $p_cut_combined (last 3), $p_cut_combined_all (all)\n\n")
        
        # Write error data
        write(io, "Raw Data:\n")
        write(io, "h,err1,err2,err_combined,err1_full,err2_full,err_full_combined,err1_cut,err2_cut,err_cut_combined\n")
        
        for i in 1:length(h_vals)
            write(io, @sprintf("%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                h_vals[i],
                err1_vals[i], err2_vals[i], err_combined_vals[i],
                err1_full_vals[i], err2_full_vals[i], err_full_combined_vals[i],
                err1_cut_vals[i], err2_cut_vals[i], err_cut_combined_vals[i]
            ))
        end
    end
    
    return (
        h_vals,
        (err1_vals, err2_vals, err_combined_vals),
        (err1_full_vals, err2_full_vals, err_full_combined_vals),
        (err1_cut_vals, err2_cut_vals, err_cut_combined_vals),
        (p1_global, p2_global, p_combined),     # Last 3 points
        (p1_full, p2_full, p_full_combined),    # Last 3 points
        (p1_cut, p2_cut, p_cut_combined),       # Last 3 points
        (p1_global_all, p2_global_all, p_combined_all),  # All points
        (p1_full_all, p2_full_all, p_full_combined_all), # All points
        (p1_cut_all, p2_cut_all, p_cut_combined_all)     # All points
    )
end

# Run the convergence study
nx_list = [40, 80, 160, 320, 640]
xint = 4.01
Tend = 0.5
He = 0.5
D1 = 1.0
D2 = 1.0

# Define analytical solutions
function T1(x)
    t = Tend
    x = x - xint
    return - He/(1+He*sqrt(D1/D2))*(erfc(x/(2*sqrt(D1*t))) - 2)
end

function T2(x)
    t = Tend
    x = x - xint
    return - He/(1+He*sqrt(D1/D2))*erfc(x/(2*sqrt(D2*t))) + 1
end

# Run convergence study
results = run_diphasic_mesh_convergence(
    nx_list,
    T1,
    T2,
    lx=8.0,
    x0=0.0,
    xint=xint,
    Tend=Tend,
    He=He,
    D1=D1,
    D2=D2,
    norm=2,
    relative=false
)

println("\nDiphasic heat transfer convergence study completed!")