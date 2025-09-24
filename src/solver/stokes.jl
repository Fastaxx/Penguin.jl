"""
    StokesMono

Prototype solver scaffold for monophasic Stokes (u, p) with separate grids.
Velocity boundary conditions are provided per component (e.g., `(bc_ux, bc_uy)` in 2D).
This is a placeholder: it builds a trivial identity system so examples can run.
Actual discretization assembly (coupled momentum + continuity) will be added later.
"""
mutable struct StokesMono{N}
    fluid::Fluid{N}
    bc_u::NTuple{N, BorderConditions}
    bc_p::BorderConditions
    bc_cut::AbstractBoundary  # cut-cell/interface BC for uγ

    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    ch::Vector{Any}
end

@inline function safe_build_source(op::AbstractOperators, f::Function, cap::Capacity, t::Union{Nothing,Float64})
    if t === nothing
        return build_source(op, f, cap)
    end
    try
        return build_source(op, f, t, cap)
    catch err
        if err isa MethodError
            return build_source(op, f, cap)
        else
            rethrow(err)
        end
    end
end

@inline function safe_build_g(op::AbstractOperators, bc::AbstractBoundary, cap::Capacity, t::Union{Nothing,Float64})
    if t === nothing
        return build_g_g(op, bc, cap)
    end
    try
        return build_g_g(op, bc, cap, t)
    catch err
        if err isa MethodError
            return build_g_g(op, bc, cap)
        else
            rethrow(err)
        end
    end
end

function stokes1D_blocks(s::StokesMono)
    op_u = s.fluid.operator_u[1]
    op_p = s.fluid.operator_p
    cap_u = s.fluid.capacity_u[1]
    cap_p = s.fluid.capacity_p

    nu = prod(op_u.size)
    np = prod(op_p.size)

    μ = s.fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹ = build_I_D(op_u, μinv, cap_u)

    WG_uG = op_u.Wꜝ * op_u.G
    WG_uH = op_u.Wꜝ * op_u.H

    visc_uω = -(Iμ⁻¹ * op_u.G' * WG_uG)
    visc_uγ = -(Iμ⁻¹ * op_u.G' * WG_uH)
    grad = -((op_p.G + op_p.H))

    div_uω = - (op_p.G' + op_p.H')
    div_uγ =   (op_p.H')

    ρ = s.fluid.ρ
    Iρ = build_I_D(op_u, ρ, cap_u)
    mass = Iρ * op_u.V

    return (; nu, np, op_u, op_p, cap_u, cap_p,
            visc_uω, visc_uγ, grad, div_uω, div_uγ,
            tie = I(nu), mass, V = op_u.V)
end

function stokes2D_blocks(s::StokesMono)
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

    return (; nu_x, nu_y, np,
            op_ux = ops_u[1], op_uy = ops_u[2], op_p, cap_px = caps_u[1], cap_py = caps_u[2], cap_p,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ,
            grad_x, grad_y,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ,
            tie_x = I(nu_x), tie_y = I(nu_y),
            mass_x, mass_y,
            Vx = ops_u[1].V, Vy = ops_u[2].V)
end

function stokes3D_blocks(s::StokesMono)
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    op_ux, op_uy, op_uz = ops_u
    cap_ux, cap_uy, cap_uz = caps_u

    nu_x = prod(op_ux.size)
    nu_y = prod(op_uy.size)
    nu_z = prod(op_uz.size)
    np = prod(op_p.size)
    sum_nu = nu_x + nu_y + nu_z

    μ = s.fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹_x = build_I_D(op_ux, μinv, cap_ux)
    Iμ⁻¹_y = build_I_D(op_uy, μinv, cap_uy)
    Iμ⁻¹_z = build_I_D(op_uz, μinv, cap_uz)

    WGx_Gx = op_ux.Wꜝ * op_ux.G
    WGx_Hx = op_ux.Wꜝ * op_ux.H
    visc_x_ω = -(Iμ⁻¹_x * op_ux.G' * WGx_Gx)
    visc_x_γ = -(Iμ⁻¹_x * op_ux.G' * WGx_Hx)

    WGy_Gy = op_uy.Wꜝ * op_uy.G
    WGy_Hy = op_uy.Wꜝ * op_uy.H
    visc_y_ω = -(Iμ⁻¹_y * op_uy.G' * WGy_Gy)
    visc_y_γ = -(Iμ⁻¹_y * op_uy.G' * WGy_Hy)

    WGz_Gz = op_uz.Wꜝ * op_uz.G
    WGz_Hz = op_uz.Wꜝ * op_uz.H
    visc_z_ω = -(Iμ⁻¹_z * op_uz.G' * WGz_Gz)
    visc_z_γ = -(Iμ⁻¹_z * op_uz.G' * WGz_Hz)

    grad_full = (op_p.G + op_p.H)
    total_grad_rows = size(grad_full, 1)
    @assert total_grad_rows == sum_nu "Pressure gradient rows ($(total_grad_rows)) must match velocity DOFs ($(sum_nu))."

    x_rows = 1:nu_x
    y_rows = nu_x + 1:nu_x + nu_y
    z_rows = nu_x + nu_y + 1:sum_nu

    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]
    grad_z = -grad_full[z_rows, :]

    Gp = op_p.G
    Hp = op_p.H
    Gp_x = Gp[x_rows, :]
    Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]
    Hp_y = Hp[y_rows, :]
    Gp_z = Gp[z_rows, :]
    Hp_z = Hp[z_rows, :]

    div_x_ω = -(Gp_x' + Hp_x')
    div_x_γ =  (Hp_x')
    div_y_ω = -(Gp_y' + Hp_y')
    div_y_γ =  (Hp_y')
    div_z_ω = -(Gp_z' + Hp_z')
    div_z_γ =  (Hp_z')

    ρ = s.fluid.ρ
    mass_x = build_I_D(op_ux, ρ, cap_ux) * op_ux.V
    mass_y = build_I_D(op_uy, ρ, cap_uy) * op_uy.V
    mass_z = build_I_D(op_uz, ρ, cap_uz) * op_uz.V

    return (; nu_x, nu_y, nu_z, np,
            op_ux, op_uy, op_uz, op_p,
            cap_ux, cap_uy, cap_uz, cap_p,
            cap_px = cap_ux, cap_py = cap_uy, cap_pz = cap_uz,
            visc_x_ω, visc_x_γ, visc_y_ω, visc_y_γ, visc_z_ω, visc_z_γ,
            grad_x, grad_y, grad_z,
            div_x_ω, div_x_γ, div_y_ω, div_y_γ, div_z_ω, div_z_γ,
            tie_x = I(nu_x), tie_y = I(nu_y), tie_z = I(nu_z),
            mass_x, mass_y, mass_z,
            Vx = op_ux.V, Vy = op_uy.V, Vz = op_uz.V)
end

@inline function enforce_dirichlet!(A::SparseMatrixCSC{Float64, Int}, b, row::Int, col::Int, value)
    val = Float64(value)
    for ptr in nzrange(A, col)
        r = A.rowval[ptr]
        r == row && continue
        coeff = A.nzval[ptr]
        if coeff != 0.0
            b[r] -= coeff * val
            A.nzval[ptr] = 0.0
        end
    end
    A[row, :] .= 0.0
    A[row, col] = 1.0
    b[row] = val
    return nothing
end

function StokesMono(fluid::Fluid{N},
                    bc_u::NTuple{N,BorderConditions},
                    bc_p::BorderConditions,
                    bc_cut::AbstractBoundary;
                    x0=zeros(0)) where {N}
    # Number of velocity dofs per component
    nu_components = ntuple(i -> prod(fluid.operator_u[i].size), N)
    np = prod(fluid.operator_p.size)
    # Unknowns: [uω¹, uγ¹, ..., uωᴺ, uγᴺ, pω]
    Ntot = 2 * sum(nu_components) + np
    x_init = length(x0) == Ntot ? x0 : zeros(Ntot)

    # Allocate empty system; assembled later
    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    s = StokesMono{N}(fluid, bc_u, bc_p, bc_cut,
                      A, b, x_init, Any[])
    assemble_stokes!(s)
    return s
end

StokesMono(fluid::Fluid{1},
           bc_u::BorderConditions,
           bc_p::BorderConditions,
           bc_cut::AbstractBoundary;
           x0=zeros(0)) = StokesMono(fluid, (bc_u,), bc_p, bc_cut; x0=x0)

function StokesMono(fluid::Fluid{N},
                    bc_u_args::Vararg{BorderConditions,N};
                    bc_p::BorderConditions,
                    bc_cut::AbstractBoundary,
                    x0=zeros(0)) where {N}
    return StokesMono(fluid, Tuple(bc_u_args), bc_p, bc_cut; x0=x0)
end

"""
    assemble_stokes!(s::StokesMono)

Assemble the steady Stokes system.
Dispatches to 1D, 2D, or 3D assembly based on operator dimensionality.
"""
function assemble_stokes!(s::StokesMono)
    # Number of velocity components (1D:1, 2D:2, 3D:3)
    N = length(s.fluid.operator_u)
    if N == 1
        return assemble_stokes1D!(s)
    elseif N == 2
        return assemble_stokes2D!(s)
    elseif N == 3
        return assemble_stokes3D!(s)
    else
        error("StokesMono assembly not implemented for N=$(N)")
    end
end

"""
    assemble_stokes1D!(s::StokesMono)

Assemble the steady 1D Stokes system with unknowns [uω; uγ; pω]:
Momentum (n): -(1/μ) G' Wꜝ G uω -(1/μ) G' Wꜝ H uγ - Wꜝ (G+H) pω = V fᵤ
Continuity(n):-(G' + H') uω + H' uγ = 0
Also applies Dirichlet BC on velocity at the two domain boundaries and fixes one pressure DOF (gauge).
"""
function assemble_stokes1D!(s::StokesMono)
    data = stokes1D_blocks(s)
    nu = data.nu
    np = data.np

    rows = 3 * nu
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    # Momentum block rows
    A[1:nu, 1:nu]         = data.visc_uω
    A[1:nu, nu+1:2nu]     = data.visc_uγ
    A[1:nu, 2nu+1:2nu+np] = data.grad

    # Tie rows enforce uγ = g_cut
    A[nu+1:2nu, 1:nu]   .= 0.0
    A[nu+1:2nu, nu+1:2nu] = data.tie

    # Continuity rows
    A[2nu+1:3nu, 1:nu]     = data.div_uω
    A[2nu+1:3nu, nu+1:2nu] = data.div_uγ

    f_vec = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, nothing)
    b_mom = data.V * f_vec
    g_cut = safe_build_g(data.op_u, s.bc_cut, data.cap_u, nothing)
    b_con = zeros(np)
    b = vcat(b_mom, g_cut, b_con)

    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=0, uγ_offset=nu)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=2nu, np=np, row_start=2nu+1)

    s.A = A
    s.b = b
    return nothing
end

"""
    assemble_stokes2D!(s::StokesMono)

Assemble the steady 2D Stokes system with unknowns [uωx; uγx; uωy; uγy; pω].
Momentum for each component uses μ∇²; continuity enforces ∇·u = 0.
"""
function assemble_stokes2D!(s::StokesMono)
    data = stokes2D_blocks(s)
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

    # Momentum x-component rows
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    # Tie x rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    # Momentum y-component rows
    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie y rows
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ

    fₒx = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing)
    fₒy = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing)
    b_mom_x = data.Vx * fₒx
    b_mom_y = data.Vy * fₒy
    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)
    b_con = zeros(np)
    b = vcat(b_mom_x, g_cut_x, b_mom_y, g_cut_y, b_con)

    # Apply Dirichlet velocity BCs at domain boundaries for both components
    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy)

    # Fix pressure gauge or apply pressure Dirichlet at boundaries if provided
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes3D!(s::StokesMono)
    data = stokes3D_blocks(s)
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    # Momentum x-component rows
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    # Tie x rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    # Momentum y-component rows
    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie y rows
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Momentum z-component rows
    A[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = data.visc_z_ω
    A[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = data.visc_z_γ
    A[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    # Tie z rows
    A[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    A[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    A[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    fₒx = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_px, nothing)
    fₒy = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_py, nothing)
    fₒz = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_pz, nothing)
    b_mom_x = data.Vx * fₒx
    b_mom_y = data.Vy * fₒy
    b_mom_z = data.Vz * fₒz

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_px, nothing)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_py, nothing)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_pz, nothing)
    b_con = zeros(np)
    b = vcat(b_mom_x, g_cut_x, b_mom_y, g_cut_y, b_mom_z, g_cut_z, b_con)

    apply_velocity_dirichlet_3D!(A, b,
                                 s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 uωz_off=off_uωz, uγz_off=off_uγz,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 row_uωz_off=row_uωz, row_uγz_off=row_uγz)

    #apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
    #                      p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end



@inline function scheme_to_theta(scheme::Symbol)
    s = lowercase(String(scheme))
    if s in ("cn", "crank_nicolson", "cranknicolson")
        return 0.5
    elseif s in ("be", "backward_euler", "implicit_euler")
        return 1.0
    else
        error("Unsupported time scheme $(scheme). Use :CN or :BE.")
    end
end

function assemble_stokes1D_unsteady!(s::StokesMono, data, Δt::Float64,
                                     x_prev::AbstractVector{<:Real},
                                     p_half_prev::AbstractVector{<:Real},
                                     t_prev::Float64, t_next::Float64,
                                     θ::Float64)
    nu = data.nu
    np = data.np

    rows = 3 * nu
    cols = 2 * nu + np
    A = spzeros(Float64, rows, cols)

    mass_dt = (1.0 / Δt) * data.mass
    θc = 1.0 - θ

    # Momentum block
    A[1:nu, 1:nu]         = mass_dt - θ * data.visc_uω
    A[1:nu, nu+1:2nu]     = -θ * data.visc_uγ
    A[1:nu, 2nu+1:2nu+np] = data.grad

    # Tie and continuity blocks
    A[nu+1:2nu, 1:nu]       .= 0.0
    A[nu+1:2nu, nu+1:2nu]   = data.tie
    A[2nu+1:3nu, 1:nu]      = data.div_uω
    A[2nu+1:3nu, nu+1:2nu]  = data.div_uγ

    u_prev_ω = view(x_prev, 1:nu)
    u_prev_γ = view(x_prev, nu+1:2nu)

    f_prev = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_prev)
    f_next = safe_build_source(data.op_u, s.fluid.fᵤ, data.cap_u, t_next)
    load = data.V * (θ .* f_next .+ θc .* f_prev)

    rhs_mom = mass_dt * u_prev_ω
    rhs_mom .+= θc * (data.visc_uω * u_prev_ω + data.visc_uγ * u_prev_γ)

    grad_prev_coeff = θ == 1.0 ? 0.0 : (1.0 - θ) / θ
    if grad_prev_coeff != 0.0
        rhs_mom .+= grad_prev_coeff * (data.grad * p_half_prev)
    end

    rhs_mom .+= load
    g_cut_next = safe_build_g(data.op_u, s.bc_cut, data.cap_u, t_next)
    b = vcat(rhs_mom, g_cut_next, zeros(np))

    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=0, uγ_offset=nu, t=t_next)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=2nu, np=np, row_start=2nu+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes2D_unsteady!(s::StokesMono, data, Δt::Float64,
                                     x_prev::AbstractVector{<:Real},
                                     p_half_prev::AbstractVector{<:Real},
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

    # Momentum x-component
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt - θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = -θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    # Tie x rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    # Momentum y-component
    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt - θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = -θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie y rows
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

    rhs_mom_x = mass_x_dt * uωx_prev
    rhs_mom_x .+= θc * (data.visc_x_ω * uωx_prev + data.visc_x_γ * uγx_prev)

    grad_prev_coeff = θ == 1.0 ? 0.0 : (1.0 - θ) / θ
    if grad_prev_coeff != 0.0
        rhs_mom_x .+= grad_prev_coeff * (data.grad_x * p_half_prev)
    end

    rhs_mom_x .+= load_x

    rhs_mom_y = mass_y_dt * uωy_prev
    rhs_mom_y .+= θc * (data.visc_y_ω * uωy_prev + data.visc_y_γ * uγy_prev)
    if grad_prev_coeff != 0.0
        rhs_mom_y .+= grad_prev_coeff * (data.grad_y * p_half_prev)
    end
    rhs_mom_y .+= load_y

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
    return nothing
end

function assemble_stokes3D_unsteady!(s::StokesMono, data, Δt::Float64,
                                     x_prev::AbstractVector{<:Real},
                                     p_half_prev::AbstractVector{<:Real},
                                     t_prev::Float64, t_next::Float64,
                                     θ::Float64)
    nu_x = data.nu_x
    nu_y = data.nu_y
    nu_z = data.nu_z
    sum_nu = nu_x + nu_y + nu_z
    np = data.np

    rows = 2 * sum_nu + np
    cols = 2 * sum_nu + np
    A = spzeros(Float64, rows, cols)

    mass_x_dt = (1.0 / Δt) * data.mass_x
    mass_y_dt = (1.0 / Δt) * data.mass_y
    mass_z_dt = (1.0 / Δt) * data.mass_z
    θc = 1.0 - θ

    off_uωx = 0
    off_uγx = nu_x
    off_uωy = 2 * nu_x
    off_uγy = 2 * nu_x + nu_y
    off_uωz = 2 * nu_x + 2 * nu_y
    off_uγz = 2 * nu_x + 2 * nu_y + nu_z
    off_p   = 2 * sum_nu

    row_uωx = 0
    row_uγx = nu_x
    row_uωy = 2 * nu_x
    row_uγy = 2 * nu_x + nu_y
    row_uωz = 2 * nu_x + 2 * nu_y
    row_uγz = 2 * nu_x + 2 * nu_y + nu_z
    row_con = 2 * sum_nu

    # Momentum x-component
    A[row_uωx+1:row_uωx+nu_x, off_uωx+1:off_uωx+nu_x] = mass_x_dt - θ * data.visc_x_ω
    A[row_uωx+1:row_uωx+nu_x, off_uγx+1:off_uγx+nu_x] = -θ * data.visc_x_γ
    A[row_uωx+1:row_uωx+nu_x, off_p+1:off_p+np]       = data.grad_x

    # Tie x rows
    A[row_uγx+1:row_uγx+nu_x, off_uγx+1:off_uγx+nu_x] = data.tie_x

    # Momentum y-component
    A[row_uωy+1:row_uωy+nu_y, off_uωy+1:off_uωy+nu_y] = mass_y_dt - θ * data.visc_y_ω
    A[row_uωy+1:row_uωy+nu_y, off_uγy+1:off_uγy+nu_y] = -θ * data.visc_y_γ
    A[row_uωy+1:row_uωy+nu_y, off_p+1:off_p+np]       = data.grad_y

    # Tie y rows
    A[row_uγy+1:row_uγy+nu_y, off_uγy+1:off_uγy+nu_y] = data.tie_y

    # Momentum z-component
    A[row_uωz+1:row_uωz+nu_z, off_uωz+1:off_uωz+nu_z] = mass_z_dt - θ * data.visc_z_ω
    A[row_uωz+1:row_uωz+nu_z, off_uγz+1:off_uγz+nu_z] = -θ * data.visc_z_γ
    A[row_uωz+1:row_uωz+nu_z, off_p+1:off_p+np]       = data.grad_z

    # Tie z rows
    A[row_uγz+1:row_uγz+nu_z, off_uγz+1:off_uγz+nu_z] = data.tie_z

    # Continuity rows
    con_rows = row_con+1:row_con+np
    A[con_rows, off_uωx+1:off_uωx+nu_x] = data.div_x_ω
    A[con_rows, off_uγx+1:off_uγx+nu_x] = data.div_x_γ
    A[con_rows, off_uωy+1:off_uωy+nu_y] = data.div_y_ω
    A[con_rows, off_uγy+1:off_uγy+nu_y] = data.div_y_γ
    A[con_rows, off_uωz+1:off_uωz+nu_z] = data.div_z_ω
    A[con_rows, off_uγz+1:off_uγz+nu_z] = data.div_z_γ

    uωx_prev = view(x_prev, off_uωx+1:off_uωx+nu_x)
    uγx_prev = view(x_prev, off_uγx+1:off_uγx+nu_x)
    uωy_prev = view(x_prev, off_uωy+1:off_uωy+nu_y)
    uγy_prev = view(x_prev, off_uγy+1:off_uγy+nu_y)
    uωz_prev = view(x_prev, off_uωz+1:off_uωz+nu_z)
    uγz_prev = view(x_prev, off_uγz+1:off_uγz+nu_z)

    f_prev_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_ux, t_prev)
    f_next_x = safe_build_source(data.op_ux, s.fluid.fᵤ, data.cap_ux, t_next)
    load_x = data.Vx * (θ .* f_next_x .+ θc .* f_prev_x)

    f_prev_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_uy, t_prev)
    f_next_y = safe_build_source(data.op_uy, s.fluid.fᵤ, data.cap_uy, t_next)
    load_y = data.Vy * (θ .* f_next_y .+ θc .* f_prev_y)

    f_prev_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_uz, t_prev)
    f_next_z = safe_build_source(data.op_uz, s.fluid.fᵤ, data.cap_uz, t_next)
    load_z = data.Vz * (θ .* f_next_z .+ θc .* f_prev_z)

    rhs_mom_x = mass_x_dt * uωx_prev
    rhs_mom_x .+= θc * (data.visc_x_ω * uωx_prev + data.visc_x_γ * uγx_prev)

    rhs_mom_y = mass_y_dt * uωy_prev
    rhs_mom_y .+= θc * (data.visc_y_ω * uωy_prev + data.visc_y_γ * uγy_prev)

    rhs_mom_z = mass_z_dt * uωz_prev
    rhs_mom_z .+= θc * (data.visc_z_ω * uωz_prev + data.visc_z_γ * uγz_prev)

    grad_prev_coeff = θ == 1.0 ? 0.0 : (1.0 - θ) / θ
    if grad_prev_coeff != 0.0
        rhs_mom_x .+= grad_prev_coeff * (data.grad_x * p_half_prev)
        rhs_mom_y .+= grad_prev_coeff * (data.grad_y * p_half_prev)
        rhs_mom_z .+= grad_prev_coeff * (data.grad_z * p_half_prev)
    end

    rhs_mom_x .+= load_x
    rhs_mom_y .+= load_y
    rhs_mom_z .+= load_z

    g_cut_x = safe_build_g(data.op_ux, s.bc_cut, data.cap_ux, t_next)
    g_cut_y = safe_build_g(data.op_uy, s.bc_cut, data.cap_uy, t_next)
    g_cut_z = safe_build_g(data.op_uz, s.bc_cut, data.cap_uz, t_next)

    b = vcat(rhs_mom_x, g_cut_x, rhs_mom_y, g_cut_y, rhs_mom_z, g_cut_z, zeros(np))

    apply_velocity_dirichlet_3D!(A, b,
                                 s.bc_u[1], s.bc_u[2], s.bc_u[3], s.fluid.mesh_u;
                                 nu_x=nu_x, nu_y=nu_y, nu_z=nu_z,
                                 uωx_off=off_uωx, uγx_off=off_uγx,
                                 uωy_off=off_uωy, uγy_off=off_uγy,
                                 uωz_off=off_uωz, uγz_off=off_uγz,
                                 row_uωx_off=row_uωx, row_uγx_off=row_uγx,
                                 row_uωy_off=row_uωy, row_uγy_off=row_uγy,
                                 row_uωz_off=row_uωz, row_uγz_off=row_uγz,
                                 t=t_next)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p, s.fluid.capacity_p;
                          p_offset=off_p, np=np, row_start=row_con+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes_unsteady!(s::StokesMono, blocks, Δt::Float64,
                                   x_prev::AbstractVector{<:Real},
                                   p_half_prev::AbstractVector{<:Real},
                                   t_prev::Float64, t_next::Float64,
                                   θ::Float64)
    N = length(s.fluid.operator_u)
    if N == 1
        assemble_stokes1D_unsteady!(s, blocks, Δt, x_prev, p_half_prev, t_prev, t_next, θ)
    elseif N == 2
        assemble_stokes2D_unsteady!(s, blocks, Δt, x_prev, p_half_prev, t_prev, t_next, θ)
    elseif N == 3
        assemble_stokes3D_unsteady!(s, blocks, Δt, x_prev, p_half_prev, t_prev, t_next, θ)
    else
        error("Unsteady Stokes assembly not implemented for N=$(N)")
    end
    return nothing
end


"""
    apply_velocity_dirichlet_2D!(A, b, bc_ux, bc_uy, mesh_u;
                                 nu_x, nu_y,
                                 uωx_off, uγx_off,
                                 uωy_off, uγy_off,
                                 row_uωx_off, row_uγx_off,
                                 row_uωy_off, row_uγy_off)

Apply Dirichlet BC for 2D velocity components on their respective meshes.
Enforces values on both uω and uγ rows for each component and boundary node.
"""
function apply_velocity_dirichlet_2D!(A::SparseMatrixCSC{Float64, Int}, b,
                                      bc_ux::BorderConditions,
                                      bc_uy::BorderConditions,
                                      mesh_u::NTuple{2,AbstractMesh};
                                      nu_x::Int, nu_y::Int,
                                      uωx_off::Int, uγx_off::Int,
                                      uωy_off::Int, uγy_off::Int,
                                      row_uωx_off::Int, row_uγx_off::Int,
                                      row_uωy_off::Int, row_uγy_off::Int,
                                      t::Union{Nothing,Float64}=nothing)
    mesh_ux, mesh_uy = mesh_u
    nx = length(mesh_ux.nodes[1]); ny = length(mesh_ux.nodes[2])
    nx_y = length(mesh_uy.nodes[1]); ny_y = length(mesh_uy.nodes[2])
    @assert nx == nx_y && ny == ny_y "Velocity component meshes must share grid dimensions"

    LIx = LinearIndices((nx, ny))
    LIy = LinearIndices((nx_y, ny_y))

    # Apply at last interior velocity node (nx, ny) consistent with BC_border_mono!
    iright = max(nx - 1, 1)
    jtop   = max(ny - 1, 1)

    xs_x = mesh_ux.nodes[1]; ys_x = mesh_ux.nodes[2]
    xs_y = mesh_uy.nodes[1]; ys_y = mesh_uy.nodes[2]

    # Helper: evaluate Dirichlet value
    eval_val(bc, x, y) = (bc isa Dirichlet) ? (bc.value isa Function ? eval_boundary_func(bc.value, x, y) : bc.value) : nothing
    eval_val(bc, x, y, t) = (bc isa Dirichlet) ? (bc.value isa Function ? bc.value(x, y, t) : bc.value) : nothing

    # Helper to handle both time-dependent and time-independent boundary functions
    function eval_boundary_func(f, x, y)
        try
            return f(x, y)  # Try 2-argument form first
        catch MethodError
            return f(x, y, 0.0)  # Fall back to 3-argument form with t=0
        end
    end

    # Gather BCs
    bcx_bottom = get(bc_ux.borders, :bottom, nothing)
    bcy_bottom = get(bc_uy.borders, :bottom, nothing)
    bcx_top    = get(bc_ux.borders, :top, nothing)
    bcy_top    = get(bc_uy.borders, :top, nothing)
    bcx_left   = get(bc_ux.borders, :left, nothing)
    bcy_left   = get(bc_uy.borders, :left, nothing)
    bcx_right  = get(bc_ux.borders, :right, nothing)
    bcy_right  = get(bc_uy.borders, :right, nothing)

    # Apply along each side for x and y components using their respective meshes
    # Bottom/top (vary along x)
    for jside in ((1, bcx_bottom, bcy_bottom), (jtop, bcx_top, bcy_top))
        jx, bcx, bcy = jside
        isnothing(bcx) && isnothing(bcy) && continue
        jy = jx  # meshes share sizes (asserted above)
        for i in 1:nx
            vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[jx]) : eval_val(bcx, xs_x[i], ys_x[jx], t)
            vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[jy]) : eval_val(bcy, xs_y[i], ys_y[jy], t)
            if vx !== nothing
                vx = Float64(vx)
                lix = LIx[i, jx]
                r = row_uωx_off + lix
                enforce_dirichlet!(A, b, r, uωx_off + lix, vx)
                rt = row_uγx_off + lix
                enforce_dirichlet!(A, b, rt, uγx_off + lix, vx)
            end
            if vy !== nothing
                vy = Float64(vy)
                liy = LIy[i, jy]
                r = row_uωy_off + liy
                enforce_dirichlet!(A, b, r, uωy_off + liy, vy)
                rt = row_uγy_off + liy
                enforce_dirichlet!(A, b, rt, uγy_off + liy, vy)
            end
        end
    end

    # Left/right (vary along y)
    for iside in ((1, bcx_left, bcy_left), (iright, bcx_right, bcy_right))
        ix, bcx, bcy = iside
        isnothing(bcx) && isnothing(bcy) && continue
        iy = ix
        for j in 1:ny
            vx = t === nothing ? eval_val(bcx, xs_x[ix], ys_x[j]) : eval_val(bcx, xs_x[ix], ys_x[j], t)
            vy = t === nothing ? eval_val(bcy, xs_y[iy], ys_y[j]) : eval_val(bcy, xs_y[iy], ys_y[j], t)
            if vx !== nothing
                vx = Float64(vx)
                lix = LIx[ix, j]
                r = row_uωx_off + lix
                enforce_dirichlet!(A, b, r, uωx_off + lix, vx)
                rt = row_uγx_off + lix
                enforce_dirichlet!(A, b, rt, uγx_off + lix, vx)
            end
            if vy !== nothing
                vy = Float64(vy)
                liy = LIy[iy, j]
                r = row_uωy_off + liy
                enforce_dirichlet!(A, b, r, uωy_off + liy, vy)
                rt = row_uγy_off + liy
                enforce_dirichlet!(A, b, rt, uγy_off + liy, vy)
            end
        end
    end
    return nothing
end

"""
Apply Dirichlet BC for 3D velocity components on their respective meshes.
Enforces values on both uω and uγ rows for each component and boundary node.
"""
function apply_velocity_dirichlet_3D!(A::SparseMatrixCSC{Float64, Int}, b,
                                      bc_ux::BorderConditions,
                                      bc_uy::BorderConditions,
                                      bc_uz::BorderConditions,
                                      mesh_u::NTuple{3,AbstractMesh};
                                      nu_x::Int, nu_y::Int, nu_z::Int,
                                      uωx_off::Int, uγx_off::Int,
                                      uωy_off::Int, uγy_off::Int,
                                      uωz_off::Int, uγz_off::Int,
                                      row_uωx_off::Int, row_uγx_off::Int,
                                      row_uωy_off::Int, row_uγy_off::Int,
                                      row_uωz_off::Int, row_uγz_off::Int,
                                      t::Union{Nothing,Float64}=nothing)
    mesh_ux, mesh_uy, mesh_uz = mesh_u
    nx = length(mesh_ux.nodes[1]); ny = length(mesh_ux.nodes[2]); nz = length(mesh_ux.nodes[3])
    nx_y = length(mesh_uy.nodes[1]); ny_y = length(mesh_uy.nodes[2]); nz_y = length(mesh_uy.nodes[3])
    nx_z = length(mesh_uz.nodes[1]); ny_z = length(mesh_uz.nodes[2]); nz_z = length(mesh_uz.nodes[3])
    @assert nx == nx_y == nx_z && ny == ny_y == ny_z && nz == nz_y == nz_z "Velocity component meshes must share grid dimensions"

    LIx = LinearIndices((nx, ny, nz))
    LIy = LinearIndices((nx_y, ny_y, nz_y))
    LIz = LinearIndices((nx_z, ny_z, nz_z))

    # Apply at last interior velocity node consistent with BC patterns
    iright = max(nx - 1, 1)
    jtop   = max(ny - 1, 1)
    kfront = max(nz - 1, 1)

    xs_x = mesh_ux.nodes[1]; ys_x = mesh_ux.nodes[2]; zs_x = mesh_ux.nodes[3]
    xs_y = mesh_uy.nodes[1]; ys_y = mesh_uy.nodes[2]; zs_y = mesh_uy.nodes[3]
    xs_z = mesh_uz.nodes[1]; ys_z = mesh_uz.nodes[2]; zs_z = mesh_uz.nodes[3]

    # Helper: evaluate Dirichlet value
    eval_val(bc, x, y, z) = (bc isa Dirichlet) ? (bc.value isa Function ? eval_boundary_func(bc.value, x, y, z) : bc.value) : nothing
    eval_val(bc, x, y, z, t) = (bc isa Dirichlet) ? (bc.value isa Function ? bc.value(x, y, z, t) : bc.value) : nothing

    # Helper to handle both time-dependent and time-independent boundary functions
    function eval_boundary_func(f, x, y, z)
        try
            return f(x, y, z)  # Try 3-argument form first
        catch MethodError
            return f(x, y, z, 0.0)  # Fall back to 4-argument form with t=0
        end
    end

    # Gather BCs - 3D has 6 faces
    bcx_bottom = get(bc_ux.borders, :bottom, nothing); bcy_bottom = get(bc_uy.borders, :bottom, nothing); bcz_bottom = get(bc_uz.borders, :bottom, nothing)
    bcx_top    = get(bc_ux.borders, :top, nothing);    bcy_top    = get(bc_uy.borders, :top, nothing);    bcz_top    = get(bc_uz.borders, :top, nothing)
    bcx_left   = get(bc_ux.borders, :left, nothing);   bcy_left   = get(bc_uy.borders, :left, nothing);   bcz_left   = get(bc_uz.borders, :left, nothing)
    bcx_right  = get(bc_ux.borders, :right, nothing);  bcy_right  = get(bc_uy.borders, :right, nothing);  bcz_right  = get(bc_uz.borders, :right, nothing)
    bcx_back   = get(bc_ux.borders, :back, nothing);   bcy_back   = get(bc_uy.borders, :back, nothing);   bcz_back   = get(bc_uz.borders, :back, nothing)
    bcx_front  = get(bc_ux.borders, :front, nothing);  bcy_front  = get(bc_uy.borders, :front, nothing);  bcz_front  = get(bc_uz.borders, :front, nothing)

    # Apply along each face for x, y, z components using their respective meshes
    # Bottom/top faces (vary along x, z)
    for jside in ((1, bcx_bottom, bcy_bottom, bcz_bottom), (jtop, bcx_top, bcy_top, bcz_top))
        jx, bcx, bcy, bcz = jside
        isnothing(bcx) && isnothing(bcy) && isnothing(bcz) && continue
        jy = jx; jz = jx  # meshes share sizes
        for i in 1:nx, k in 1:nz
            vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[jx], zs_x[k]) : eval_val(bcx, xs_x[i], ys_x[jx], zs_x[k], t)
            vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[jy], zs_y[k]) : eval_val(bcy, xs_y[i], ys_y[jy], zs_y[k], t)
            vz = t === nothing ? eval_val(bcz, xs_z[i], ys_z[jz], zs_z[k]) : eval_val(bcz, xs_z[i], ys_z[jz], zs_z[k], t)
            if vx !== nothing
                vx = Float64(vx)
                lix = LIx[i, jx, k]
                r = row_uωx_off + lix
                enforce_dirichlet!(A, b, r, uωx_off + lix, vx)
                rt = row_uγx_off + lix
                enforce_dirichlet!(A, b, rt, uγx_off + lix, vx)
            end
            if vy !== nothing
                vy = Float64(vy)
                liy = LIy[i, jy, k]
                r = row_uωy_off + liy
                enforce_dirichlet!(A, b, r, uωy_off + liy, vy)
                rt = row_uγy_off + liy
                enforce_dirichlet!(A, b, rt, uγy_off + liy, vy)
            end
            if vz !== nothing
                vz = Float64(vz)
                liz = LIz[i, jz, k]
                r = row_uωz_off + liz
                enforce_dirichlet!(A, b, r, uωz_off + liz, vz)
                rt = row_uγz_off + liz
                enforce_dirichlet!(A, b, rt, uγz_off + liz, vz)
            end
        end
    end

    # Left/right faces (vary along y, z)
    for iside in ((1, bcx_left, bcy_left, bcz_left), (iright, bcx_right, bcy_right, bcz_right))
        ix, bcx, bcy, bcz = iside
        isnothing(bcx) && isnothing(bcy) && isnothing(bcz) && continue
        iy = ix; iz = ix
        for j in 1:ny, k in 1:nz
            vx = t === nothing ? eval_val(bcx, xs_x[ix], ys_x[j], zs_x[k]) : eval_val(bcx, xs_x[ix], ys_x[j], zs_x[k], t)
            vy = t === nothing ? eval_val(bcy, xs_y[iy], ys_y[j], zs_y[k]) : eval_val(bcy, xs_y[iy], ys_y[j], zs_y[k], t)
            vz = t === nothing ? eval_val(bcz, xs_z[iz], ys_z[j], zs_z[k]) : eval_val(bcz, xs_z[iz], ys_z[j], zs_z[k], t)
            if vx !== nothing
                vx = Float64(vx)
                lix = LIx[ix, j, k]
                r = row_uωx_off + lix
                enforce_dirichlet!(A, b, r, uωx_off + lix, vx)
                rt = row_uγx_off + lix
                enforce_dirichlet!(A, b, rt, uγx_off + lix, vx)
            end
            if vy !== nothing
                vy = Float64(vy)
                liy = LIy[iy, j, k]
                r = row_uωy_off + liy
                enforce_dirichlet!(A, b, r, uωy_off + liy, vy)
                rt = row_uγy_off + liy
                enforce_dirichlet!(A, b, rt, uγy_off + liy, vy)
            end
            if vz !== nothing
                vz = Float64(vz)
                liz = LIz[iz, j, k]
                r = row_uωz_off + liz
                enforce_dirichlet!(A, b, r, uωz_off + liz, vz)
                rt = row_uγz_off + liz
                enforce_dirichlet!(A, b, rt, uγz_off + liz, vz)
            end
        end
    end

    # Back/front faces (vary along x, y)
    for kside in ((1, bcx_back, bcy_back, bcz_back), (kfront, bcx_front, bcy_front, bcz_front))
        kx, bcx, bcy, bcz = kside
        isnothing(bcx) && isnothing(bcy) && isnothing(bcz) && continue
        ky = kx; kz = kx
        for i in 1:nx, j in 1:ny
            vx = t === nothing ? eval_val(bcx, xs_x[i], ys_x[j], zs_x[kx]) : eval_val(bcx, xs_x[i], ys_x[j], zs_x[kx], t)
            vy = t === nothing ? eval_val(bcy, xs_y[i], ys_y[j], zs_y[ky]) : eval_val(bcy, xs_y[i], ys_y[j], zs_y[ky], t)
            vz = t === nothing ? eval_val(bcz, xs_z[i], ys_z[j], zs_z[kz]) : eval_val(bcz, xs_z[i], ys_z[j], zs_z[kz], t)
            if vx !== nothing
                vx = Float64(vx)
                lix = LIx[i, j, kx]
                r = row_uωx_off + lix
                enforce_dirichlet!(A, b, r, uωx_off + lix, vx)
                rt = row_uγx_off + lix
                enforce_dirichlet!(A, b, rt, uγx_off + lix, vx)
            end
            if vy !== nothing
                vy = Float64(vy)
                liy = LIy[i, j, ky]
                r = row_uωy_off + liy
                enforce_dirichlet!(A, b, r, uωy_off + liy, vy)
                rt = row_uγy_off + liy
                enforce_dirichlet!(A, b, rt, uγy_off + liy, vy)
            end
            if vz !== nothing
                vz = Float64(vz)
                liz = LIz[i, j, kz]
                r = row_uωz_off + liz
                enforce_dirichlet!(A, b, r, uωz_off + liz, vz)
                rt = row_uγz_off + liz
                enforce_dirichlet!(A, b, rt, uγz_off + liz, vz)
            end
        end
    end
    return nothing
end



"""
    apply_velocity_dirichlet!(A, b, bc_u, mesh_u; nu, uω_offset, uγ_offset)

Apply Dirichlet BC to velocity at the two domain boundary nodes for both uω and uγ
by replacing corresponding momentum and tie rows.
"""
function apply_velocity_dirichlet!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64},
                                   bc_u::BorderConditions, mesh_u::AbstractMesh;
                                   nu::Int, uω_offset::Int, uγ_offset::Int,
                                   t::Union{Nothing,Float64}=nothing)
    # Determine boundary values
    left_bc  = get(bc_u.borders, :bottom, nothing)
    right_bc = get(bc_u.borders, :top, nothing)

    # Node coordinates
    xnodes = mesh_u.nodes[1]
    iL, iR = 1, max(length(xnodes) - 1, 1)

    # Helper to evaluate value at position
    function eval_value(bc, x)
        isnothing(bc) && return nothing
        bc isa Dirichlet || return nothing
        v = bc.value
        if v isa Function
            return t === nothing ? v(x) : v(x, 0.0, t)  # 1D: y=0, but pass t if available
        else
            return v
        end
    end

    vL = eval_value(left_bc,  xnodes[1])
    vR = eval_value(right_bc, xnodes[end-1])

    # Row indices: momentum rows are 1:nu, tie rows are nu+1:2nu
    if vL !== nothing
        vL = Float64(vL)
        # Enforce uω[iL] = vL via momentum row iL (use last interior row index convention: iL = 1)
        r = iL
        enforce_dirichlet!(A, b, r, uω_offset + iL, vL)
        rt = nu + iL
        enforce_dirichlet!(A, b, rt, uγ_offset + iL, vL)
    end
    if vR !== nothing
        vR = Float64(vR)
        # Enforce uω[iR] = vR via momentum row iR (rightmost velocity index = nx+1)
        r = iR
        enforce_dirichlet!(A, b, r, uω_offset + iR, vR)
        rt = nu + iR
        enforce_dirichlet!(A, b, rt, uγ_offset + iR, vR)
    end
    return nothing
end

"""
    apply_pressure_gauge!(A, b, bc_p, mesh_p, capacity_p; p_offset, np, row_start)

Fix one pressure dof: if left/right pressure Dirichlet exist, enforce them; otherwise
set p=0 in the first fluid cell (based on capacity). Rows
`row_start : row_start+np-1` belong to the continuity block.
"""
function apply_pressure_gauge!(A::SparseMatrixCSC{Float64, Int}, b,
                               bc_p::BorderConditions,
                               mesh_p::AbstractMesh,
                               capacity_p::AbstractCapacity;
                               p_offset::Int, np::Int, row_start::Int)
    nodes = mesh_p.nodes
    nd = length(nodes)
    dims = ntuple(i -> length(nodes[i]), nd)
    LI = LinearIndices(Tuple(dims))

    side_specs = Dict{Symbol,Tuple{Int,Int}}(
        :left  => (1, 1),
        :right => (1, dims[1]),
    )
    if nd >= 2
        side_specs[:bottom] = (2, 1)
        side_specs[:top]    = (2, dims[2])
    else
        side_specs[:bottom] = (1, 1)
        side_specs[:top]    = (1, dims[1])
    end
    if nd >= 3
        side_specs[:front] = (3, 1)
        side_specs[:back]  = (3, dims[3])
    end

    firstn(coords::NTuple{N,Float64}, m::Int) where {N} = ntuple(i -> coords[i], m)
    function call_boundary_value(f::Function, coords::NTuple{N,Float64}) where {N}
        for m in N:-1:0
            args = m == 0 ? () : firstn(coords, m)
            try
                return f(args...)
            catch err
                err isa MethodError || rethrow(err)
            end
        end
        return nothing
    end

    function eval_dirichlet(bc::AbstractBoundary, coords::NTuple{N,Float64}) where {N}
        bc isa Dirichlet || return nothing
        val = bc.value
        if val isa Function
            return call_boundary_value(val, coords)
        else
            return val
        end
    end

    applied = false
    ranges_full = ntuple(i -> 1:dims[i], nd)

    function apply_face!(dim::Int, fixed_idx::Int, bc::AbstractBoundary)
        dim <= nd || return false
        bc isa Dirichlet || return false
        ranges = collect(ranges_full)
        ranges[dim] = fixed_idx:fixed_idx
        did_apply = false
        for idx in Iterators.product(ranges...)
            coords = ntuple(i -> nodes[i][idx[i]], nd)
            val = eval_dirichlet(bc, coords)
            val === nothing && continue
            lin = LI[idx...]
            row = row_start + lin - 1
            col = p_offset + lin
            enforce_dirichlet!(A, b, row, col, Float64(val))
            did_apply = true
        end
        return did_apply
    end

    for (side, bc) in bc_p.borders
        spec = get(side_specs, side, nothing)
        isnothing(spec) && continue
        applied |= apply_face!(spec[1], spec[2], bc)
    end

    if !applied
        diagV = diag(capacity_p.V)
        tol = 1e-12
        idx = findfirst(x -> x > tol, diagV)
        gauge_idx = idx === nothing ? 1 : idx
        row = row_start + gauge_idx - 1
        col = p_offset + gauge_idx
        enforce_dirichlet!(A, b, row, col, 0.0)
    end
    return nothing
end

function solve_stokes_linear_system!(s::StokesMono; method=Base.:\, algorithm=nothing, kwargs...)
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

function solve_StokesMono!(s::StokesMono; method=Base.:\, algorithm=nothing, kwargs...)
    println("[StokesMono] Assembling steady Stokes and solving (fully coupled)")
    # Re-assemble in case anything changed
    assemble_stokes!(s)
    solve_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)
    return s
end

"""
    solve_StokesMono_unsteady!(s::StokesMono; Δt, T_end, scheme=:CN, method=Base, algorithm=nothing, store_states=true, kwargs...)

Solve the unsteady Stokes problem using an implicit θ-scheme (Backward Euler or
Crank–Nicolson). Returns the sampled times and solution snapshots when
`store_states=true`.
"""
function solve_StokesMono_unsteady!(s::StokesMono; Δt::Float64, T_end::Float64,
                                    scheme::Symbol=:CN, method=Base.:\,
                                    algorithm=nothing, store_states::Bool=true,
                                    kwargs...)
    θ = scheme_to_theta(scheme)
    N = length(s.fluid.operator_u)
    blocks = if N == 1
        stokes1D_blocks(s)
    elseif N == 2
        stokes2D_blocks(s)
    elseif N == 3
        stokes3D_blocks(s)
    else
        error("StokesMono unsteady solver not implemented for N=$(N)")
    end

    if N == 1
        p_offset = 2 * blocks.nu
        np = blocks.np
        Ntot = p_offset + np
    elseif N == 2
        p_offset = 2 * (blocks.nu_x + blocks.nu_y)
        np = blocks.np
        Ntot = p_offset + np
    else
        p_offset = 2 * (blocks.nu_x + blocks.nu_y + blocks.nu_z)
        np = blocks.np
        Ntot = p_offset + np
    end

    x_prev = if length(s.x) == Ntot
        copy(s.x)
    else
        zeros(Ntot)
    end

    p_half_prev = zeros(np)
    if length(s.x) == Ntot
        p_half_prev .= s.x[p_offset+1:p_offset+np]
    end

    histories = store_states ? Vector{Vector{Float64}}() : Vector{Vector{Float64}}()
    if store_states
        push!(histories, copy(x_prev))
    end
    times = Float64[0.0]

    t = 0.0
    println("[StokesMono] Starting unsteady solve up to T=$(T_end) with Δt=$(Δt) and θ=$(θ)")
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        assemble_stokes_unsteady!(s, blocks, dt_step, x_prev, p_half_prev, t, t_next, θ)
        solve_stokes_linear_system!(s; method=method, algorithm=algorithm, kwargs...)

        x_prev = copy(s.x)
        p_half_prev .= s.x[p_offset+1:p_offset+np]
        push!(times, t_next)
        if store_states
            push!(histories, x_prev)
        end
        println("[StokesMono] t=$(round(t_next; digits=6)) max|state|=$(maximum(abs, x_prev))")

        t = t_next
    end

    return times, histories
end
