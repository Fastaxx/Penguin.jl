"""
    MovingStokesSolver(body, interface_velocity, mesh_u, mesh_p, bc_u;
                       μ=1.0, ρ=1.0, fᵤ=(args...)->0.0, fₚ=(args...)->0.0,
                       pressure_gauge=DEFAULT_PRESSURE_GAUGE, t0=0.0, x0=zeros(0))

Helper that orchestrates a sequence of steady Stokes solves while the embedded
geometry prescribed by `body(x, y, t)` moves in time. The interface velocity is
imposed directly on the cut degrees of freedom through the function
`interface_velocity(x, y, t)` (returning either a scalar applied to all
components or a tuple/vector whose entries correspond to each velocity
component). Currently only 2D configurations are supported.

Typical usage:
```julia
mesh_p  = Mesh((nx, ny), domain, origin)
dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
mesh_ux = Mesh((nx, ny), domain, (origin[1] - 0.5*dx, origin[2]))
mesh_uy = Mesh((nx, ny), domain, (origin[1], origin[2] - 0.5*dy))
bc_ux = BorderConditions(...)
bc_uy = BorderConditions(...)

solver = MovingStokesSolver(body, interface_velocity,
                            (mesh_ux, mesh_uy), mesh_p, (bc_ux, bc_uy);
                            μ=1.0, ρ=1.0, t0=0.0)

times, states = solve_MovingStokesSolver!(solver; Δt=0.05, T_end=1.0)
```
"""
mutable struct MovingStokesSolver{N}
    body::Function
    interface_velocity::Function
    meshes_u::NTuple{N,AbstractMesh}
    mesh_p::AbstractMesh
    μ::Union{Float64,Function}
    ρ::Union{Float64,Function}
    fᵤ::Function
    fₚ::Function
    bc_u::NTuple{N,BorderConditions}
    pressure_gauge::AbstractPressureGauge
    solver::StokesMono{N}
    current_time::Float64
    scheme::Symbol
    states::Vector{Vector{Float64}}
    times::Vector{Float64}
end

MovingStokesSolver(body::Function,
                   interface_velocity::Function,
                   mesh_ux::AbstractMesh,
                   mesh_uy::AbstractMesh,
                   mesh_p::AbstractMesh,
                   bc_ux::BorderConditions,
                   bc_uy::BorderConditions;
                   kwargs...) = MovingStokesSolver(body, interface_velocity,
                                                   (mesh_ux, mesh_uy),
                                                   mesh_p,
                                                   (bc_ux, bc_uy);
                                                   kwargs...)

function MovingStokesSolver(body::Function,
                            interface_velocity::Function,
                            meshes_u::NTuple{2,AbstractMesh},
                            mesh_p::AbstractMesh,
                            bc_u::NTuple{2,BorderConditions};
                            μ::Union{Float64,Function}=1.0,
                            ρ::Union{Float64,Function}=1.0,
                            fᵤ::Function=(args...)->0.0,
                            fₚ::Function=(args...)->0.0,
                            pressure_gauge::AbstractPressureGauge=DEFAULT_PRESSURE_GAUGE,
                            t0::Float64=0.0,
                            scheme::Symbol=:CN,
                            x0=zeros(0))
    fluid = _moving_stokes_fluid(body, meshes_u, mesh_p, μ, ρ, fᵤ, fₚ, t0)
    solver = _init_snapshot_solver(fluid, bc_u, pressure_gauge; x0=x0)
    return MovingStokesSolver{2}(body, interface_velocity, meshes_u, mesh_p,
                                 μ, ρ, fᵤ, fₚ, bc_u, pressure_gauge,
                                 solver, t0, scheme, Vector{Vector{Float64}}(),
                                 Float64[])
end

"""
    assemble_moving_stokes!(s::MovingStokesSolver, t)

Update the embedded geometry to time `t`, rebuild the algebraic system, and
populate `s.solver.A`/`b`.
"""
function assemble_moving_stokes!(s::MovingStokesSolver, t_prev::Float64, t_next::Float64, scheme::Symbol)
    if length(s.meshes_u) != 2
        error("MovingStokesSolver currently supports only 2D configurations.")
    end
    assemble_moving_stokes2D!(s, t_prev, t_next, scheme)
end

"""
    solve_MovingStokesSolver!(s; Δt, T_end, method=Base.:\\, algorithm=nothing,
                              store_states=true, kwargs...)

Advance the moving Stokes solve from the solver's current time up to `T_end`
with fixed time step `Δt`. Each step rebuilds the embedded geometry, assembles a
steady Stokes system, and solves it using the backend specified by `method` or
`algorithm`. When `store_states=true`, solution snapshots are collected in
`solver.states`.
"""
function solve_MovingStokesSolver!(s::MovingStokesSolver;
                                   Δt::Float64,
                                   T_end::Float64,
                                   method=Base.:\,
                                   algorithm=nothing,
                                   store_states::Bool=true,
                                   scheme::Union{Symbol,Nothing}=nothing,
                                   kwargs...)
    Δt > 0 || error("Δt must be positive.")
    target_scheme = scheme === nothing ? s.scheme : Symbol(scheme)
    t_prev = s.current_time
    final_time = max(T_end, t_prev)
    tol = 1e-12 * max(1.0, abs(final_time))
    while t_prev < final_time - tol
        t_next = min(t_prev + Δt, final_time)
        assemble_moving_stokes!(s, t_prev, t_next, target_scheme)
        solve_stokes_linear_system!(s.solver; method=method, algorithm=algorithm, kwargs...)
        if store_states
            push!(s.states, copy(s.solver.x))
            push!(s.times, t_next)
        end
        t_prev = t_next
    end
    s.current_time = final_time
    return s.times, s.states
end

# -----------------------------------------------------------------------------#
# Helper utilities
# -----------------------------------------------------------------------------#

_DEFAULT_CUT_BC = Dirichlet(0.0)

function _moving_stokes_fluid(body::Function,
                              meshes_u::NTuple{2,AbstractMesh},
                              mesh_p::AbstractMesh,
                              μ, ρ, fᵤ, fₚ,
                              t::Float64)
    body_t = _freeze_body(body, t)
    capacity_u = ntuple(i -> Capacity(body_t, meshes_u[i]; compute_centroids=true), 2)
    operators_u = ntuple(i -> DiffusionOps(capacity_u[i]), 2)
    capacity_p = Capacity(body_t, mesh_p; compute_centroids=true)
    operator_p = DiffusionOps(capacity_p)
    return Fluid(meshes_u, capacity_u, operators_u,
                 mesh_p, capacity_p, operator_p,
                 μ, ρ, fᵤ, fₚ)
end

function _init_snapshot_solver(fluid::Fluid{N},
                               bc_u::NTuple{N,BorderConditions},
                               pressure_gauge::AbstractPressureGauge;
                               x0=zeros(0)) where {N}
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * sum(nu_components) + np
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)
    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)
    return StokesMono{N}(fluid, bc_u, pressure_gauge, _DEFAULT_CUT_BC,
                         A, b, x_init, Any[])
end

_freeze_body(body::Function, t::Float64) = (coords...) -> begin
    try
        body(coords..., t)
    catch err
        if err isa MethodError
            body(coords...)
        else
            rethrow(err)
        end
    end
end

_scheme_functions(scheme::Symbol) = scheme === :CN ? (psip_cn, psim_cn) : (psip_be, psim_be)

function _build_spacetime_capacity(body::Function,
                                   mesh::AbstractMesh,
                                   t_prev::Float64,
                                   t_next::Float64)
    SpaceTimeMesh(mesh, [t_prev, t_next], tag=mesh.tag) |>
        stmesh -> Capacity(body, stmesh; compute_centroids=true)
end

function _select_half_matrix(M::SparseMatrixCSC{Float64,Int}, half::Symbol)
    nrows = size(M, 1)
    nrows == 0 && return M
    half_n = nrows ÷ 2
    half_n == 0 && return M
    if half === :previous
        rows = 1:half_n
    else
        rows = half_n+1:nrows
    end
    return M[rows, rows]
end

function _truncate_centroid(v::SVector{M,Float64}, N::Int) where M
    N == M && return v
    return SVector{N,Float64}(ntuple(i -> v[i], N))
end

function _select_centroid_vector(coords::Vector{SVector{M,Float64}},
                                 N::Int,
                                 half::Symbol) where M
    len = length(coords)
    len == 0 && return Vector{SVector{N,Float64}}()
    half_len = len ÷ 2
    if half === :previous
        range_idx = 1:half_len
    else
        range_idx = half_len+1:len
    end
    return [ _truncate_centroid(coords[i], N) for i in range_idx ]
end

function _select_scalar_vector(vec::Vector{Float64}, half::Symbol)
    len = length(vec)
    len == 0 && return Float64[]
    half_len = len ÷ 2
    if half === :previous
        return vec[1:half_len]
    else
        return vec[half_len+1:end]
    end
end

function _slice_capacity(cap_st::Capacity,
                         mesh::AbstractMesh,
                         half::Symbol,
                         body_fn::Function)
    N = length(mesh.nodes)
    A = ntuple(i -> _select_half_matrix(cap_st.A[i], half), N)
    B = ntuple(i -> _select_half_matrix(cap_st.B[i], half), N)
    V = _select_half_matrix(cap_st.V, half)
    W = ntuple(i -> _select_half_matrix(cap_st.W[i], half), N)
    C_ω = _select_centroid_vector(cap_st.C_ω, N, half)
    C_γ = _select_centroid_vector(cap_st.C_γ, N, half)
    Γ = _select_half_matrix(cap_st.Γ, half)
    cell_types = _select_scalar_vector(cap_st.cell_types, half)
    return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body_fn)
end

function _blend_diag_matrix(prev::SparseMatrixCSC{Float64,Int},
                            curr::SparseMatrixCSC{Float64,Int},
                            psip, psim)
    diag_prev = Vector(diag(prev))
    diag_curr = Vector(diag(curr))
    λ = psip.(diag_curr, diag_prev)
    μ = psim.(diag_curr, diag_prev)
    diag_blend = λ .* diag_curr .+ μ .* diag_prev
    return spdiagm(0 => diag_blend)
end

function _blend_diag_matrix_with_weights(prev::SparseMatrixCSC{Float64,Int},
                                         curr::SparseMatrixCSC{Float64,Int},
                                         psip, psim)
    diag_prev = Vector(diag(prev))
    diag_curr = Vector(diag(curr))
    λ = psip.(diag_curr, diag_prev)
    μ = psim.(diag_curr, diag_prev)
    diag_blend = λ .* diag_curr .+ μ .* diag_prev
    return spdiagm(0 => diag_blend), λ, μ
end

function _blend_centroids(prev::Vector{SVector{N,Float64}},
                          curr::Vector{SVector{N,Float64}},
                          λ::Vector{Float64},
                          μ::Vector{Float64}) where N
    len = min(length(curr), length(prev), length(λ))
    result = Vector{SVector{N,Float64}}(undef, len)
    for i in 1:len
        result[i] = λ[i] .* curr[i] .+ μ[i] .* prev[i]
    end
    return result
end

function _blend_cell_vector(prev::Vector{Float64},
                            curr::Vector{Float64},
                            λ::Vector{Float64},
                            μ::Vector{Float64})
    len = min(length(curr), length(prev), length(λ))
    result = zeros(Float64, len)
    for i in 1:len
        result[i] = λ[i] * curr[i] + μ[i] * prev[i]
    end
    return result
end

function _blend_capacities(cap_prev::Capacity,
                           cap_curr::Capacity,
                           scheme::Symbol)
    psip, psim = _scheme_functions(scheme)
    N = length(cap_curr.A)
    A = ntuple(i -> _blend_diag_matrix(cap_prev.A[i], cap_curr.A[i], psip, psim), N)
    B = ntuple(i -> _blend_diag_matrix(cap_prev.B[i], cap_curr.B[i], psip, psim), N)
    W = ntuple(i -> _blend_diag_matrix(cap_prev.W[i], cap_curr.W[i], psip, psim), N)
    V, λ_vol, μ_vol = _blend_diag_matrix_with_weights(cap_prev.V, cap_curr.V, psip, psim)
    C_ω = _blend_centroids(cap_prev.C_ω, cap_curr.C_ω, λ_vol, μ_vol)
    cell_types = _blend_cell_vector(cap_prev.cell_types, cap_curr.cell_types, λ_vol, μ_vol)
    Γ, λ_Γ, μ_Γ = _blend_diag_matrix_with_weights(cap_prev.Γ, cap_curr.Γ, psip, psim)
    C_γ = _blend_centroids(cap_prev.C_γ, cap_curr.C_γ, λ_Γ, μ_Γ)
    return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types,
                       cap_curr.mesh, cap_curr.body)
end

function _build_capacity_interval(body::Function,
                                  mesh::AbstractMesh,
                                  t_prev::Float64,
                                  t_next::Float64,
                                  scheme::Symbol)
    st_cap = _build_spacetime_capacity(body, mesh, t_prev, t_next)
    cap_prev = _slice_capacity(st_cap, mesh, :previous, _freeze_body(body, t_prev))
    cap_curr = _slice_capacity(st_cap, mesh, :current, _freeze_body(body, t_next))
    return _blend_capacities(cap_prev, cap_curr, scheme)
end

function _interface_component_vector(capacity::Capacity,
                                     body::Function,
                                     velocity_fn::Function,
                                     t_prev::Float64,
                                     t_next::Float64,
                                     scheme::Symbol,
                                     component_index::Int)
    coords = get_all_coordinates(capacity.C_γ)
    n = length(coords)
    n == 0 && return zeros(Float64, n)
    N = length(capacity.mesh.nodes)
    inside_prev = [ _evaluate_body(body, _spatial_coords(coord, N), t_prev) <= 0 ? 1.0 : 0.0
                    for coord in coords ]
    inside_next = [ _evaluate_body(body, _spatial_coords(coord, N), t_next) <= 0 ? 1.0 : 0.0
                    for coord in coords ]
    psip, psim = _scheme_functions(scheme)
    λ = psip.(inside_next, inside_prev)
    μ = psim.(inside_next, inside_prev)
    g_prev = [Float64(_interface_value(velocity_fn, _spatial_coords(coord, N), t_prev, component_index)) for coord in coords]
    g_next = [Float64(_interface_value(velocity_fn, _spatial_coords(coord, N), t_next, component_index)) for coord in coords]
    return λ .* g_next .+ μ .* g_prev
end

function assemble_moving_stokes2D!(s::MovingStokesSolver{2},
                                   t_prev::Float64,
                                   t_next::Float64,
                                   scheme::Symbol)
    cap_ux = _build_capacity_interval(s.body, s.meshes_u[1], t_prev, t_next, scheme)
    cap_uy = _build_capacity_interval(s.body, s.meshes_u[2], t_prev, t_next, scheme)
    cap_p  = _build_capacity_interval(s.body, s.mesh_p,       t_prev, t_next, scheme)
    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)
    fluid = Fluid(s.meshes_u, (cap_ux, cap_uy), (op_ux, op_uy),
                  s.mesh_p, cap_p, op_p,
                  s.μ, s.ρ, s.fᵤ, s.fₚ)
    s.solver.fluid = fluid
    data = stokes2D_blocks(s.solver)

    nu_x = data.nu_x
    nu_y = data.nu_y
    np = data.np
    sum_nu = nu_x + nu_y
    rows = 2 * sum_nu + np
    cols = rows
    A = spzeros(Float64, rows, cols)

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_con = 2 * sum_nu

    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    fₒx = safe_build_source(data.op_ux, s.solver.fluid.fᵤ, data.cap_px, t_next)
    fₒy = safe_build_source(data.op_uy, s.solver.fluid.fᵤ, data.cap_py, t_next)
    b_mom_x = data.Vx * fₒx
    b_mom_y = data.Vy * fₒy
    g_cut_x = _interface_component_vector(data.cap_px, s.body, s.interface_velocity,
                                          t_prev, t_next, scheme, 1)
    g_cut_y = _interface_component_vector(data.cap_py, s.body, s.interface_velocity,
                                          t_prev, t_next, scheme, 2)
    b_con = zeros(np)
    b = vcat(b_mom_x, g_cut_x, b_mom_y, g_cut_y, b_con)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.meshes_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.pressure_gauge, s.mesh_p, s.solver.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.solver.A = A
    s.solver.b = b
    _resize_state!(s.solver, rows)
end

function _interface_value(velocity_fn::Function,
                          coord::NTuple{M,Float64},
                          t::Float64,
                          component_index::Int) where {M}
    val = _call_with_optional_time(velocity_fn, coord, t)
    if val isa Number
        return Float64(val)
    elseif val isa AbstractVector || val isa Tuple
        length(val) >= component_index ||
            error("Interface velocity function returned $(length(val)) components, " *
                  "but component $(component_index) was requested.")
        return Float64(val[component_index])
    else
        error("Interface velocity function must return a scalar, tuple, or vector.")
    end
end

function _call_with_optional_time(f::Function, coords::NTuple, t::Float64)
    try
        return f(coords..., t)
    catch err
        if err isa MethodError
            return f(coords...)
        else
            rethrow(err)
        end
    end
end

function _evaluate_body(body::Function, coords::NTuple, t::Float64)
    try
        return body(coords..., t)
    catch err
        if err isa MethodError
            return body(coords...)
        else
            rethrow(err)
        end
    end
end

_spatial_coords(coord::NTuple, N::Int) = ntuple(i -> coord[i], N)

function _resize_state!(solver::StokesMono, Ntot::Int)
    if length(solver.x) != Ntot
        solver.x = zeros(Ntot)
    end
end
