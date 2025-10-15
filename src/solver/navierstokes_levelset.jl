using LevelSetMethods
using StaticArrays

"""
    struct NavierStokesVelocitySampler

Cache that interpolates the staggered-grid velocity components produced by
`NavierStokesMono{2}` onto arbitrary spatial points. Each component keeps its
own grid because the x- and y-velocities live on shifted meshes.
"""
mutable struct NavierStokesVelocitySampler
    xs_x::Vector{Float64}
    ys_x::Vector{Float64}
    xs_y::Vector{Float64}
    ys_y::Vector{Float64}
    Ux::Matrix{Float64}
    Uy::Matrix{Float64}
end

"""
    NavierStokesVelocitySampler(solver::NavierStokesMono{2})

Allocate a velocity sampler compatible with the staggered meshes stored in
`solver`. The sampler is initialized with zero velocities; call
[`update_velocity_sampler!`](@ref) to populate it with a given state.
"""
function NavierStokesVelocitySampler(solver::NavierStokesMono{2})
    mesh_x = solver.fluid.mesh_u[1]
    mesh_y = solver.fluid.mesh_u[2]

    xs_x = Float64.(collect(mesh_x.nodes[1]))
    ys_x = Float64.(collect(mesh_x.nodes[2]))
    xs_y = Float64.(collect(mesh_y.nodes[1]))
    ys_y = Float64.(collect(mesh_y.nodes[2]))

    Ux = zeros(Float64, length(xs_x), length(ys_x))
    Uy = zeros(Float64, length(xs_y), length(ys_y))

    return NavierStokesVelocitySampler(xs_x, ys_x, xs_y, ys_y, Ux, Uy)
end

@inline function _interp_index(coords::AbstractVector{<:Real}, x::Real)
    n = length(coords)
    @assert n ≥ 2 "Need at least two points to interpolate"

    if x ≤ coords[1]
        return 1, 0.0
    elseif x ≥ coords[end]
        return n - 1, 1.0
    else
        idx = searchsortedlast(coords, x)
        idx = clamp(idx, 1, n - 1)
        x0 = coords[idx]
        x1 = coords[idx + 1]
        frac = x1 == x0 ? 0.0 : (x - x0) / (x1 - x0)
        return idx, clamp(frac, 0.0, 1.0)
    end
end

@inline function _bilinear_sample(xs::AbstractVector{<:Real},
                                  ys::AbstractVector{<:Real},
                                  field::AbstractMatrix{<:Real},
                                  x::Real, y::Real)
    ix, fx = _interp_index(xs, x)
    iy, fy = _interp_index(ys, y)

    f00 = field[ix, iy]
    f10 = field[ix + 1, iy]
    f01 = field[ix, iy + 1]
    f11 = field[ix + 1, iy + 1]

    return (1 - fx) * (1 - fy) * f00 +
           fx * (1 - fy) * f10 +
           (1 - fx) * fy * f01 +
           fx * fy * f11
end

@inline function levelset_axes(ϕ::LevelSet)
    mesh = ϕ.mesh
    xs = Float64.(collect(range(mesh.lc[1], mesh.hc[1], length=mesh.n[1])))
    ys = Float64.(collect(range(mesh.lc[2], mesh.hc[2], length=mesh.n[2])))
    return xs, ys
end

function body_function_from_levelset(ϕ::LevelSet; invert_sign::Bool=true)
    xs, ys = levelset_axes(ϕ)
    vals = LevelSetMethods.values(ϕ)

    function body(x::Real, y::Real, z::Real=0.0)
        val = _bilinear_sample(xs, ys, vals, x, y)
        return invert_sign ? -val : val
    end

    return body
end

function update_solver_geometry!(solver::NavierStokesMono{2},
                                 ϕ::LevelSet;
                                 invert_sign::Bool=true)
    mesh_u = solver.fluid.mesh_u
    mesh_p = solver.fluid.mesh_p

    body = body_function_from_levelset(ϕ; invert_sign=invert_sign)

    cap_ux = Capacity(body, mesh_u[1]; compute_centroids=false)
    cap_uy = Capacity(body, mesh_u[2]; compute_centroids=false)
    cap_p  = Capacity(body, mesh_p; compute_centroids=false)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)

    fluid_old = solver.fluid
    μ = fluid_old.μ
    ρ = fluid_old.ρ
    fᵤ = fluid_old.fᵤ
    fₚ = fluid_old.fₚ

    new_fluid = Fluid(mesh_u,
                      (cap_ux, cap_uy),
                      (op_ux, op_uy),
                      mesh_p,
                      cap_p,
                      op_p,
                      μ, ρ, fᵤ, fₚ)

    solver.fluid = new_fluid
    solver.convection = build_convection_data(new_fluid)
    solver.prev_conv = nothing

    return solver
end

"""
    (sampler::NavierStokesVelocitySampler)(x, t)

Bilinearly interpolate the stored velocity field at spatial position `x`. The
time `t` argument is ignored so the sampler can be used directly inside
`AdvectionTerm` closures.
"""
function (sampler::NavierStokesVelocitySampler)(x, t)
    x₁ = x[1]
    x₂ = x[2]
    vx = _bilinear_sample(sampler.xs_x, sampler.ys_x, sampler.Ux, x₁, x₂)
    vy = _bilinear_sample(sampler.xs_y, sampler.ys_y, sampler.Uy, x₁, x₂)
    return SVector(vx, vy)
end

"""
    update_velocity_sampler!(sampler, solver, state)

Fill `sampler` with velocities extracted from a Navier–Stokes state vector.
Only the primary (`ω`) velocity DOFs are used—the interpolated values already
enforce the tie constraints in the solver.
"""
function update_velocity_sampler!(sampler::NavierStokesVelocitySampler,
                                  solver::NavierStokesMono{2},
                                  state::AbstractVector{<:Real})
    size_x = solver.fluid.operator_u[1].size
    size_y = solver.fluid.operator_u[2].size
    nu_x = prod(size_x)
    nu_y = prod(size_y)

    @assert length(state) ≥ 2 * (nu_x + nu_y) "State vector too short for Navier–Stokes layout"

    @assert size(sampler.Ux) == size_x "Sampler Ux shape mismatch with operator size"
    @assert size(sampler.Uy) == size_y "Sampler Uy shape mismatch with operator size"

    sampler.Ux .= reshape(@view(state[1:nu_x]), size_x...)
    sampler.Uy .= reshape(@view(state[2 * nu_x + 1:2 * nu_x + nu_y]), size_y...)
    return sampler
end

"""
    build_levelset_advection_equation(levelset, sampler;
                                      bc=PeriodicBC(),
                                      scheme=WENO5())

Create a `LevelSetEquation` driven purely by advection using the velocity field
provided through `sampler`. The returned equation owns a deep copy of the
input level set so that callers can reuse `levelset` elsewhere.
"""
function build_levelset_advection_equation(levelset::LevelSet,
                                           sampler::NavierStokesVelocitySampler;
                                           bc::LevelSetMethods.BoundaryCondition=PeriodicBC(),
                                           scheme=LevelSetMethods.WENO5())
    ϕ₀ = deepcopy(levelset)
    vel = (x, t) -> sampler(x, t)
    term = AdvectionTerm(vel, scheme)
    return LevelSetEquation(; terms=(term,), levelset=ϕ₀, bc=bc)
end
"""
    solve_NavierStokesLevelSet_unsteady!(solver, levelset;
                                         Δt, T_end;
                                         scheme=:CN,
                                         bc=PeriodicBC(),
                                         levelset_scheme=WENO5(),
                                         store_levelsets=true,
                                         store_states=true,
                                         invert_sign=true,
                                         method=Base.:,
                                         algorithm=nothing,
                                         kwargs...)

Advance the Navier–Stokes solver while transporting a level-set interface with
the computed velocity. After every time step the interface is advected, and
the cell capacities/operators are rebuilt from the updated geometry before the
next momentum solve.
"""
function solve_NavierStokesLevelSet_unsteady!(solver::NavierStokesMono{2},
                                              levelset::LevelSet;
                                              Δt::Float64,
                                              T_end::Float64,
                                              scheme::Symbol=:CN,
                                              bc::LevelSetMethods.BoundaryCondition=PeriodicBC(),
                                              levelset_scheme=LevelSetMethods.WENO5(),
                                              store_levelsets::Bool=true,
                                              store_states::Bool=true,
                                              invert_sign::Bool=true,
                                              method=Base.:\,
                                              algorithm=nothing,
                                              kwargs...)
    θ = scheme_to_theta(scheme)

    sampler = NavierStokesVelocitySampler(solver)
    equation = build_levelset_advection_equation(levelset, sampler; bc=bc, scheme=levelset_scheme)

    update_solver_geometry!(solver, equation.state; invert_sign=invert_sign)

    data = navierstokes2D_blocks(solver)
    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    p_offset = 2 * (nu_x + nu_y)
    Ntot = p_offset + np

    x_prev = length(solver.x) == Ntot ? copy(solver.x) : zeros(Ntot)

    p_half_prev = zeros(np)
    if length(solver.x) == Ntot && !isempty(solver.x)
        p_half_prev .= solver.x[p_offset+1:p_offset+np]
    end

    times = Float64[0.0]
    states = Vector{Vector{Float64}}()
    if store_states
        push!(states, copy(x_prev))
    end

    levelsets = Vector{typeof(equation.state)}()
    if store_levelsets
        push!(levelsets, deepcopy(equation.state))
    end

    conv_prev = nothing
    t = 0.0

    println("[NavierStokesLevelSet] Starting coupled solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")

    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        data = navierstokes2D_blocks(solver)
        nu_x = data.nu_x
        nu_y = data.nu_y
        np = data.np
        p_offset = 2 * (nu_x + nu_y)

        conv_curr = assemble_navierstokes2D_unsteady!(solver, data, dt_step,
                                                      x_prev, p_half_prev,
                                                      t, t_next, θ, conv_prev)

        solve_navierstokes_linear_system!(solver; method=method,
                                          algorithm=algorithm, kwargs...)

        x_prev = copy(solver.x)
        p_half_prev .= solver.x[p_offset+1:p_offset+np]

        conv_prev = ntuple(Val(length(data.nu_components))) do i
            copy(conv_curr[Int(i)])
        end

        update_velocity_sampler!(sampler, solver, solver.x)
        LevelSetMethods.integrate!(equation, t_next)

        push!(times, t_next)
        if store_states
            push!(states, copy(x_prev))
        end
        if store_levelsets
            push!(levelsets, deepcopy(equation.state))
        end

        update_solver_geometry!(solver, equation.state; invert_sign=invert_sign)
        conv_prev = nothing

        max_state = maximum(abs, x_prev)
        println("[NavierStokesLevelSet] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

        t = t_next
    end

    solver.prev_conv = conv_prev

    return (times=times,
            states=states,
            levelsets=levelsets,
            equation=equation,
            sampler=sampler)
end
