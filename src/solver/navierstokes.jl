struct ConvectiveStencil{N}
    primary_dim::Int
    D_plus::NTuple{N,SparseMatrixCSC{Float64,Int}}
    S_minus::NTuple{N,SparseMatrixCSC{Float64,Int}}
    S_plus_primary::SparseMatrixCSC{Float64,Int}
    A::NTuple{N,SparseMatrixCSC{Float64,Int}}
    Ht::SparseMatrixCSC{Float64,Int}
end

struct NavierStokesConvection{N}
    stencils::NTuple{N,ConvectiveStencil{N}}
end

"""
    NavierStokesMono

Prototype incompressible Navier–Stokes solver that reuses the staggered Stokes
layout but augments the momentum equation with an explicit, skew-symmetric
convection operator. The convection matrices (`Cx`, `Cy`) and their interface
counterparts (`Kx`, `Ky`) follow the discrete flux-form described in the
project notes. Time integration is implicit for viscous/pressure terms while
convection is advanced with an Adams–Bashforth extrapolation.
"""
mutable struct NavierStokesMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    bc_p::BorderConditions
    bc_cut::AbstractBoundary
    convection::Union{Nothing,NavierStokesConvection{N}}  # Populated for N>=2 currently
    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    prev_conv::Union{Nothing,NTuple{N,Vector{Float64}}}
    last_conv_ops::Union{Nothing,NamedTuple}
    ch::Vector{Any}
end

"""
    NavierStokesMono(fluid, bc_u, bc_p, bc_cut; x0=zeros(0))

Construct a Navier–Stokes solver scaffold. Unknown ordering matches the Stokes
setup: `[uω₁, uγ₁, ..., uωₙ, uγₙ, pω]`.
"""
function NavierStokesMono(fluid::Fluid{N},
                          bc_u::NTuple{N,BorderConditions},
                          bc_p::BorderConditions,
                          bc_cut::AbstractBoundary;
                          x0=zeros(0)) where {N}
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    Ntot = 2 * sum(nu_components) + np
    x_init = length(x0) == Ntot ? copy(x0) : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    convection = build_convection_data(fluid)

    return NavierStokesMono{N}(fluid, bc_u, bc_p, bc_cut,
                               convection, A, b, x_init,
                               nothing, nothing, Any[])
end

NavierStokesMono(fluid::Fluid{1},
                 bc_u::BorderConditions,
                 bc_p::BorderConditions,
                 bc_cut::AbstractBoundary;
                 x0=zeros(0)) = NavierStokesMono(fluid, (bc_u,), bc_p, bc_cut; x0=x0)

function NavierStokesMono(fluid::Fluid{N},
                          bc_u_args::Vararg{BorderConditions,N};
                          bc_p::BorderConditions,
                          bc_cut::AbstractBoundary,
                          x0=zeros(0)) where {N}
    return NavierStokesMono(fluid, Tuple(bc_u_args), bc_p, bc_cut; x0=x0)
end

function build_convection_data(fluid::Fluid{N}) where {N}
    stencils = ntuple(Val(N)) do i
        build_convective_stencil(fluid.capacity_u[Int(i)], fluid.operator_u[Int(i)], Int(i))
    end
    return NavierStokesConvection{N}(stencils)
end

function build_convective_stencil(capacity::AbstractCapacity,
                                  op::AbstractOperators,
                                  primary_dim::Int)
    _, _, _, _, _, D_p, S_m, S_p = compute_base_operators(capacity)
    N = length(D_p)
    @assert 1 ≤ primary_dim ≤ N "Primary dimension $(primary_dim) out of bounds for N=$(N)"

    D_plus = ntuple(Val(N)) do i
        D_p[Int(i)]
    end

    S_minus = ntuple(Val(N)) do i
        S_m[Int(i)]
    end

    return ConvectiveStencil{N}(primary_dim,
                                 D_plus,
                                 S_minus,
                                 S_p[primary_dim],
                                 capacity.A,
                                 op.H')
end

# Safe sparse products -------------------------------------------------------

@inline function safe_mul(A::SparseMatrixCSC{Float64,Int}, v::AbstractVector{<:Real})
    size(A, 2) == 0 && return zeros(Float64, size(A, 1))
    @assert size(A, 2) == length(v) "Dimension mismatch: size(A,2)=$(size(A,2)) length(v)=$(length(v))"
    return A * Vector{Float64}(v)
end

@inline function build_convection_matrix(stencil::ConvectiveStencil{N},
                                         u_components::NTuple{N,AbstractVector{<:Real}}) where {N}
    primary = stencil.primary_dim

    flux_primary = stencil.S_minus[primary] * safe_mul(stencil.A[primary], u_components[primary])
    term = stencil.D_plus[primary] * spdiagm(0 => flux_primary) * stencil.S_minus[primary]

    for j in 1:N
        j == primary && continue
        A_cross = stencil.A[j]
        if size(A_cross, 2) == 0 || length(u_components[j]) == 0
            continue
        end
        flux_cross = stencil.S_minus[primary] * safe_mul(A_cross, u_components[j])
        term += stencil.D_plus[j] * spdiagm(0 => flux_cross) * stencil.S_minus[j]
    end

    return term
end

@inline function build_K_matrix(stencil::ConvectiveStencil,
                                uγ::AbstractVector{<:Real})
    size(stencil.Ht, 2) == 0 && return spzeros(Float64, size(stencil.S_plus_primary, 1), size(stencil.S_plus_primary, 1))
    @assert size(stencil.Ht, 2) == length(uγ) "Dimension mismatch for interface velocities"
    weights = stencil.S_plus_primary * (stencil.Ht * Vector{Float64}(uγ))
    return spdiagm(0 => weights)
end

@inline function rotated_interfaces(uγ_tuple::NTuple{N,Vector{Float64}}, idx::Int) where {N}
    total = 0
    for j in 1:N
        total += length(uγ_tuple[j])
    end
    result = Vector{Float64}(undef, total)
    pos = 1
    for shift in 0:N-1
        comp = mod1(idx + shift, N)
        vals = uγ_tuple[comp]
        len = length(vals)
        if len > 0
            copyto!(result, pos, vals, 1, len)
        end
        pos += len
    end
    return result
end

# Block builders -------------------------------------------------------------

function navierstokes2D_blocks(s::NavierStokesMono)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    nu_x = prod(ops_u[1].size)
    nu_y = prod(ops_u[2].size)
    np = prod(op_p.size)

    μ = s.fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹_x = build_I_D(ops_u[1], μinv, caps_u[1])
    Iμ⁻¹_y = build_I_D(ops_u[2], μinv, caps_u[2])

    WGx_Gx = ops_u[1].Wꜝ * ops_u[1].G
    WGx_Hx = ops_u[1].Wꜝ * ops_u[1].H
    visc_x_ω = -(Iμ⁻¹_x * ops_u[1].G' * WGx_Gx)
    visc_x_γ = -(Iμ⁻¹_x * ops_u[1].G' * WGx_Hx)

    WGy_Gy = ops_u[2].Wꜝ * ops_u[2].G
    WGy_Hy = ops_u[2].Wꜝ * ops_u[2].H
    visc_y_ω = -(Iμ⁻¹_y * ops_u[2].G' * WGy_Gy)
    visc_y_γ = -(Iμ⁻¹_y * ops_u[2].G' * WGy_Hy)

    grad_full = (op_p.G + op_p.H)
    total_grad_rows = size(grad_full, 1)
    @assert total_grad_rows == nu_x + nu_y "Pressure gradient rows ($(total_grad_rows)) must match velocity DOFs ($(nu_x + nu_y))."

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

    ρ = s.fluid.ρ
    mass_x = build_I_D(ops_u[1], ρ, caps_u[1]) * ops_u[1].V
    mass_y = build_I_D(ops_u[2], ρ, caps_u[2]) * ops_u[2].V

    return (; nu_components=(nu_x, nu_y),
            nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p, cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = I(nu_x), tie_y = I(nu_y),
            mass_x, mass_y,
            Vx = ops_u[1].V, Vy = ops_u[2].V)
end

# Convection helpers ---------------------------------------------------------

function compute_convection_vectors!(s::NavierStokesMono,
                                     data,
                                     advecting_state::AbstractVector{<:Real},
                                     advected_state::AbstractVector{<:Real}=advecting_state)
    convection = s.convection
    convection === nothing && error("Convection data not available for the current Navier–Stokes configuration.")

    nu_components = data.nu_components
    N = length(nu_components)

    uω_adv = Vector{Vector{Float64}}(undef, N)
    uγ_adv = Vector{Vector{Float64}}(undef, N)
    offset = 0
    for i in 1:N
        n = nu_components[i]
        uω_adv[i] = Vector{Float64}(view(advecting_state, offset+1:offset+n))
        offset += n
        uγ_adv[i] = Vector{Float64}(view(advecting_state, offset+1:offset+n))
        offset += n
    end
    uω_adv_tuple = Tuple(uω_adv)
    uγ_adv_tuple = Tuple(uγ_adv)

    same_state = advected_state === advecting_state
    qω_tuple = nothing
    qγ_tuple = nothing
    if same_state
        qω_tuple = uω_adv_tuple
        qγ_tuple = uγ_adv_tuple
    else
        uω_advected = Vector{Vector{Float64}}(undef, N)
        uγ_advected = Vector{Vector{Float64}}(undef, N)
        offset = 0
        for i in 1:N
            n = nu_components[i]
            uω_advected[i] = Vector{Float64}(view(advected_state, offset+1:offset+n))
            offset += n
            uγ_advected[i] = Vector{Float64}(view(advected_state, offset+1:offset+n))
            offset += n
        end
        qω_tuple = Tuple(uω_advected)
        qγ_tuple = Tuple(uγ_advected)
    end

    qω_tuple = qω_tuple::NTuple{N,Vector{Float64}}
    qγ_tuple = qγ_tuple::NTuple{N,Vector{Float64}}

    bulk = ntuple(Val(N)) do i
        idx = Int(i)
        build_convection_matrix(convection.stencils[idx], uω_adv_tuple)
    end

    K_adv = ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(uγ_adv_tuple, idx))
    end

    K_advected = same_state ? K_adv : ntuple(Val(N)) do i
        idx = Int(i)
        build_K_matrix(convection.stencils[idx], rotated_interfaces(qγ_tuple, idx))
    end

    SplusHt = ntuple(Val(N)) do i
        idx = Int(i)
        convection.stencils[idx].S_plus_primary * convection.stencils[idx].Ht
    end

    conv_vectors = ntuple(Val(N)) do i
        idx = Int(i)
        bulk[idx] * qω_tuple[idx] - 0.5 * (K_adv[idx] * qω_tuple[idx] + K_advected[idx] * uω_adv_tuple[idx])
    end

    K_gamma = ntuple(Val(N)) do i
        idx = Int(i)
        spdiagm(0 => uω_adv_tuple[idx]) * SplusHt[idx]
    end

    s.last_conv_ops = (bulk=bulk,
                       K_adv=K_adv,
                       K_advected=K_advected,
                       K_mean=ntuple(Val(N)) do i
                           idx = Int(i)
                           0.5 * (K_adv[idx] + K_advected[idx])
                       end,
                       K_gamma=K_gamma,
                       SplusHt=SplusHt,
                       uω_adv=uω_adv_tuple,
                       uγ_adv=uγ_adv_tuple)

    return conv_vectors
end

# Assembly ------------------------------------------------------------------

function assemble_navierstokes2D_unsteady!(s::NavierStokesMono, data, Δt::Float64,
                                           x_prev::AbstractVector{<:Real},
                                           p_half_prev::AbstractVector{<:Real},
                                           t_prev::Float64, t_next::Float64,
                                           θ::Float64,
                                           conv_prev::Union{Nothing,NTuple{2,Vector{Float64}}})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    θc = 1.0 - θ

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

    # Momentum blocks
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt - θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = -θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt - θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = -θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_prev)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, t_next)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_prev)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, t_next)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    rhs_mom_x = mass_x_dt * Vector{Float64}(uωx_prev)
    rhs_mom_x .+= θc * (data.visc_x_ω * Vector{Float64}(uωx_prev) + data.visc_x_γ * Vector{Float64}(uγx_prev))

    rhs_mom_y = mass_y_dt * Vector{Float64}(uωy_prev)
    rhs_mom_y .+= θc * (data.visc_y_ω * Vector{Float64}(uωy_prev) + data.visc_y_γ * Vector{Float64}(uγy_prev))

    grad_prev_coeff = θ == 1.0 ? 0.0 : (1.0 - θ) / θ
    if grad_prev_coeff != 0.0
        rhs_mom_x .+= grad_prev_coeff * (data.grad_x * p_half_prev)
        rhs_mom_y .+= grad_prev_coeff * (data.grad_y * p_half_prev)
    end

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y

    conv_curr = compute_convection_vectors!(s, data, x_prev)
    if conv_prev === nothing
        rhs_mom_x .-= conv_curr[1]
        rhs_mom_y .-= conv_curr[2]
    else
        rhs_mom_x .-= 1.5 .* conv_curr[1] .- 0.5 .* conv_prev[1]
        rhs_mom_y .-= 1.5 .* conv_curr[2] .- 0.5 .* conv_prev[2]
    end

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return conv_curr
end

function assemble_navierstokes2D_steady_picard!(s::NavierStokesMono,
                                                data,
                                                advecting_state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    sum_nu = nu_x + nu_y
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
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

    # Build convection linearization for the current iterate
    compute_convection_vectors!(s, data, advecting_state)
    ops = s.last_conv_ops
    @assert ops !== nothing
    bulk = ops.bulk
    K_diag = ops.K_adv
    K_gamma = ops.K_gamma

    # Momentum rows with Picard linearized convection
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω + bulk[1] - 0.5 * K_diag[1]
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ - 0.5 * K_gamma[1]
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω + bulk[2] - 0.5 * K_diag[2]
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ - 0.5 * K_gamma[2]
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    # Forcing (steady)
    f_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing)
    f_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing)
    load_x = data.Vx * f_x
    load_y = data.Vy * f_y

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)

    b = vcat(load_x, g_cut_x, load_y, g_cut_y, zeros(np))

    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 t=nothing)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

# Linear solve ---------------------------------------------------------------

function solve_navierstokes_linear_system!(s::NavierStokesMono; method=Base.:\, algorithm=nothing, kwargs...)
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(s.A, s.b)

    xred = nothing
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
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
        kwargs_nt = (; kwargs...)
        log = get(kwargs_nt, :log, false)
        if log
            xred, ch = method(Ared, bred; kwargs...)
            push!(s.ch, ch)
        else
            xred = method(Ared, bred; kwargs...)
        end
    end

    N = size(s.A, 2)
    s.x = zeros(N)
    s.x[keep_idx_cols] = xred
    return s
end

# Time integration ----------------------------------------------------------

function solve_NavierStokesMono_unsteady!(s::NavierStokesMono; Δt::Float64, T_end::Float64,
                                          scheme::Symbol=:CN, method=Base.:\,
                                          algorithm=nothing, store_states::Bool=true,
                                          kwargs...)
    θ = scheme_to_theta(scheme)
    N = length(s.fluid.operator_u)
    N == 2 || error("Navier–Stokes prototype currently only implemented for 2D (found N=$(N)).")

    data = navierstokes2D_blocks(s)

    p_offset = 2 * (data.nu_x + data.nu_y)
    np = data.np
    Ntot = p_offset + np

    x_prev = length(s.x) == Ntot ? copy(s.x) : zeros(Ntot)

    p_half_prev = zeros(np)
    if length(s.x) == Ntot && !isempty(s.x)
        p_half_prev .= s.x[p_offset+1:p_offset+np]
    end

    histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
    if store_states
        push!(histories, copy(x_prev))
    end
    times = Float64[0.0]

    conv_prev = s.prev_conv

    t = 0.0
    println("[NavierStokesMono] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        conv_curr = assemble_navierstokes2D_unsteady!(s, data, dt_step, x_prev, p_half_prev, t, t_next, θ, conv_prev)
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_prev = copy(s.x)
        p_half_prev .= s.x[p_offset+1:p_offset+np]
        N_comp = length(data.nu_components)
        conv_prev = ntuple(Val(N_comp)) do i
            copy(conv_curr[Int(i)])
        end

        push!(times, t_next)
        if store_states
            push!(histories, x_prev)
        end
        max_state = maximum(abs, x_prev)
        println("[NavierStokesMono] t=$(round(t_next; digits=6)) max|state|=$(max_state)")

        t = t_next
    end

    s.prev_conv = conv_prev
    return times, histories
end

function build_convection_operators(s::NavierStokesMono, state::AbstractVector{<:Real})
    data = navierstokes2D_blocks(s)
    conv_vectors = compute_convection_vectors!(s, data, state)
    return s.last_conv_ops, conv_vectors
end

function solve_NavierStokesMono_steady!(s::NavierStokesMono; tol=1e-8, maxiter::Int=25,
                                        relaxation::Float64=1.0, method=Base.:\,
                                        algorithm=nothing, kwargs...)
    θ_relax = clamp(relaxation, 0.0, 1.0)
    N = length(s.fluid.operator_u)
    N == 2 || error("Steady Navier–Stokes Picard solver currently implemented for 2D (N=$(N)).")

    data = navierstokes2D_blocks(s)
    x_iter = copy(s.x)
    residual = Inf
    iter = 0

    println("[NavierStokesMono] Starting steady Picard iterations (tol=$(tol), maxiter=$(maxiter), relaxation=$(θ_relax))")

    while iter < maxiter && residual > tol
        assemble_navierstokes2D_steady_picard!(s, data, x_iter)
        solve_navierstokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_new = θ_relax .* s.x .+ (1.0 - θ_relax) .* x_iter
        residual = maximum(abs, x_new .- x_iter)

        x_iter .= x_new
        s.x .= x_new

        iter += 1
        println("[NavierStokesMono] Picard iter=$(iter) max|Δx|=$(residual)")
    end

    if residual > tol
        @warn "Navier–Stokes steady Picard did not reach tolerance" final_residual=residual iterations=iter tol=tol
    end

    s.prev_conv = nothing
    return s.x, iter, residual
end
