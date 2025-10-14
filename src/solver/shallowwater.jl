"""
    ShallowWater2D

Linearized shallow water solver on a staggered velocity / cell centred height
grid. The formulation mirrors the Navier–Stokes scaffold: velocities live on
edge-aligned control volumes while the free-surface displacement is stored at
cell centres. Time integration uses a θ-scheme (Crank–Nicolson by default) for
the implicit gravity / divergence coupling while the wave propagation speed is
controlled through the user supplied mean depth `H` and gravity `g`.

The unknown ordering is `[uωₓ, uγₓ, uωᵧ, uγᵧ, η]`, i.e. identical to the
Navier–Stokes layout with the pressure block replaced by the free-surface
displacement.
"""
mutable struct ShallowWater2D
    fluid::Fluid{2}
    bc_u::NTuple{2,BorderConditions}
    height_gauge::AbstractPressureGauge
    bc_cut::AbstractBoundary
    depth::Float64
    gravity::Float64

    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    x::Vector{Float64}
    ch::Vector{Any}
end

function ShallowWater2D(fluid::Fluid{2},
                        bc_u::NTuple{2,BorderConditions},
                        height_gauge::AbstractPressureGauge,
                        bc_cut::AbstractBoundary;
                        depth::Float64,
                        gravity::Float64=9.81,
                        x0=zeros(0))
    depth > 0 || throw(ArgumentError("Shallow water depth must be positive, got $(depth)"))
    gravity != 0 || throw(ArgumentError("Gravity must be non-zero for the shallow water model"))

    nu_x = prod(fluid.operator_u[1].size)
    nu_y = prod(fluid.operator_u[2].size)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * (nu_x + nu_y) + np

    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Float64, Ntot)

    return ShallowWater2D(fluid, bc_u, height_gauge, bc_cut,
                          depth, gravity,
                          A, b, x_init, Any[])
end

function ShallowWater2D(fluid::Fluid{2},
                        bc_u_x::BorderConditions,
                        bc_u_y::BorderConditions,
                        height_gauge::AbstractPressureGauge,
                        bc_cut::AbstractBoundary;
                        depth::Float64,
                        gravity::Float64=9.81,
                        x0=zeros(0))
    return ShallowWater2D(fluid, (bc_u_x, bc_u_y), height_gauge, bc_cut;
                           depth=depth, gravity=gravity, x0=x0)
end

function ShallowWater2D(fluid::Fluid{2},
                        bc_u_args::Vararg{BorderConditions,2};
                        height_gauge::AbstractPressureGauge=MeanPressureGauge(),
                        bc_cut::AbstractBoundary=Dirichlet(0.0),
                        depth::Float64,
                        gravity::Float64=9.81,
                        x0=zeros(0))
    return ShallowWater2D(fluid, Tuple(bc_u_args), height_gauge, bc_cut;
                           depth=depth, gravity=gravity, x0=x0)
end

function shallowwater2D_blocks(s::ShallowWater2D)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    np = prod(op_p.size)

    grad_full = (op_p.G + op_p.H)
    total_grad_rows = size(grad_full, 1)
    @assert total_grad_rows == nu_x + nu_y "Height gradient rows $(total_grad_rows) must match velocity DOFs $(nu_x + nu_y)."

    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]

    Gp = op_p.G
    Hp = op_p.H
    Gp_x = Gp[x_rows, :]
    Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]
    Hp_y = Hp[y_rows, :]

    div_x_ω = - (Gp_x' + Hp_x')
    div_x_γ =   (Hp_x')
    div_y_ω = - (Gp_y' + Hp_y')
    div_y_γ =   (Hp_y')

    # Depth-averaged shallow-water momentum has unit inertia; keep density out of the scheme
    mass_x = build_I_D(ops_u[1], 1.0, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], 1.0, caps_u[2]) * ops_u[2].V
    mass_height = build_I_D(op_p, 1.0, cap_p) * op_p.V

    return (; nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p,
            cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = I(nu_x), tie_y = I(nu_y),
            mass_x, mass_y, mass_h = mass_height,
            Vx = ops_u[1].V, Vy = ops_u[2].V, Vp = op_p.V)
end

function assemble_shallowwater2D_unsteady!(s::ShallowWater2D, data, Δt::Float64,
                                           x_prev::AbstractVector{<:Real},
                                           t_prev::Float64, t_next::Float64,
                                           θ::Float64)
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    mass_h_dt = (1.0 / Δt) * data.mass_h
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_h   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_height = 2 * sum_nu

    g = s.gravity
    depth = s.depth

    # Momentum rows
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt
    A[row_uωx+1:row_uωx+nu_x, off_h+1:off_h+np] = θ * g .* data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt
    A[row_uωy+1:row_uωy+nu_y, off_h+1:off_h+np] = θ * g .* data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Height rows
    h_rows = row_height+1:row_height+np
    A[h_rows, off_uωx+1:off_uωx+nu_x] = depth * θ .* data.div_x_ω
    A[h_rows, off_uγx+1:off_uγx+nu_x] = depth * θ .* data.div_x_γ
    A[h_rows, off_uωy+1:off_uωy+nu_y] = depth * θ .* data.div_y_ω
    A[h_rows, off_uγy+1:off_uγy+nu_y] = depth * θ .* data.div_y_γ
    A[h_rows, off_h+1:off_h+np] = mass_h_dt

    # Previous state
    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)
    η_prev   = view(x_prev, off_h+1:off_h+np)

    # Forcing
    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .-= θc * g .* (data.grad_x * Vector{Float64}(η_prev))
    rhs_mom_x .+= load_x

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .-= θc * g .* (data.grad_y * Vector{Float64}(η_prev))
    rhs_mom_y .+= load_y

    f_prev_h = safe_build_source(data.op_p, s.fluid.fₚ, data.cap_p, t_prev)
    f_next_h = safe_build_source(data.op_p, s.fluid.fₚ, data.cap_p, t_next)
    load_h = data.Vp * (θ .* f_next_h .+ θc .* f_prev_h)

    div_prev = depth .* (
        data.div_x_ω * Vector{Float64}(uωx_prev) .+
        data.div_x_γ * Vector{Float64}(uγx_prev) .+
        data.div_y_ω * Vector{Float64}(uωy_prev) .+
        data.div_y_γ * Vector{Float64}(uγy_prev))

    rhs_height = mass_h_dt * Vector{Float64}(η_prev)
    rhs_height .-= θc .* div_prev
    rhs_height .+= load_h

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, rhs_height)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.height_gauge, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_h, np=np, row_start=row_height+1)

    s.A = A
    s.b = b
    return nothing
end

function solve_shallowwater_linear_system!(s::ShallowWater2D; method=Base.:\,
                                           algorithm=nothing, kwargs...)
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(s.A, s.b)

    kwargs_nt = (; kwargs...)
    precond_builder = haskey(kwargs_nt, :precond_builder) ? kwargs_nt.precond_builder : nothing
    if precond_builder !== nothing
        kwargs_nt = Base.structdiff(kwargs_nt, (precond_builder=precond_builder,))
    end

    precond_kwargs = (;)
    if precond_builder !== nothing
        precond_result = try
            precond_builder(Ared, s)
        catch err
            if err isa MethodError
                precond_builder(Ared)
            else
                rethrow(err)
            end
        end
        precond_kwargs = _preconditioner_kwargs(precond_result)
    end

    solve_kwargs = merge(kwargs_nt, precond_kwargs)

    xred = nothing
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm; solve_kwargs...)
        xred = sol.u
    elseif method === Base.:\
        try
            xred = Ared \ bred
        catch e
            if e isa SingularException
                @warn "Direct solver hit SingularException; falling back to bicgstabl" sizeA=size(Ared)
                xred = IterativeSolvers.bicgstabl(Ared, bred)
            else
                rethrow(e)
            end
        end
    else
        log = get(solve_kwargs, :log, false)
        if log
            xred, ch = method(Ared, bred; solve_kwargs...)
            push!(s.ch, ch)
        else
            xred = method(Ared, bred; solve_kwargs...)
        end
    end

    N = size(s.A, 2)
    s.x = zeros(N)
    s.x[keep_idx_cols] = xred
    return s
end

function solve_ShallowWater2D_unsteady!(s::ShallowWater2D; Δt::Float64, T_end::Float64,
                                        scheme::Symbol=:CN, method=Base.:\,
                                        algorithm=nothing, store_states::Bool=true,
                                        kwargs...)
    θ = scheme_to_theta(scheme)
    data = shallowwater2D_blocks(s)

    sum_nu = data.nu_x + data.nu_y
    np = data.np
    Ntot = 2 * sum_nu + np

    x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

    histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
    if store_states
        push!(histories, copy(x_prev))
    end
    times = Float64[0.0]

    t = 0.0
    println("[ShallowWater2D] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        assemble_shallowwater2D_unsteady!(s, data, dt_step, x_prev, t, t_next, θ)
        solve_shallowwater_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_prev = copy(s.x)

        push!(times, t_next)
        if store_states
            push!(histories, x_prev)
        end

        max_state = maximum(abs, x_prev)
        println("[ShallowWater2D] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

        t = t_next
    end

    return times, histories
end

function extract_shallowwater_fields(s::ShallowWater2D)
    data = shallowwater2D_blocks(s)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np

    uωx = view(s.x, 1:nu_x)
    uγx = view(s.x, nu_x+1:2nu_x)
    uωy = view(s.x, 2nu_x+1:2nu_x+nu_y)
    uγy = view(s.x, 2nu_x+nu_y+1:2*(nu_x+nu_y))
    η = view(s.x, 2*(nu_x+nu_y)+1:2*(nu_x+nu_y)+np)

    return uωx, uγx, uωy, uγy, η
end

"""
    interpolate_height(s::ShallowWater2D)

Return the free-surface displacement reshaped on the pressure mesh grid.
"""
function interpolate_height(s::ShallowWater2D)
    _, _, _, _, η = extract_shallowwater_fields(s)
    op_p = s.fluid.operator_p
    dims = op_p.size
    return reshape(Vector{Float64}(η), dims)
end
