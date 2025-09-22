"""
    StokesDiph

Diphasic Stokes solver container holding phase A/B data, interface conditions, and
assembled system matrices. Assembly is not yet implemented.
"""
mutable struct StokesDiph{N}
    fluid_a::Fluid{N}
    fluid_b::Fluid{N}
    bc_u_a::NTuple{N, BorderConditions}
    bc_u_b::NTuple{N, BorderConditions}
    bc_p::BorderConditions
    interface::InterfaceConditions
    bc_cut::AbstractBoundary

    A::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}
    x::Vector{Float64}
    ch::Vector{Any}
end

struct StokesPhaseBlocks1D
    nu::Int
    np::Int
    op_u::DiffusionOps
    cap_u::Capacity
    visc_uω::SparseMatrixCSC{Float64,Int}
    visc_uγ::SparseMatrixCSC{Float64,Int}
    grad::SparseMatrixCSC{Float64,Int}
    div_uω::SparseMatrixCSC{Float64,Int}
    div_uγ::SparseMatrixCSC{Float64,Int}
    mass::SparseMatrixCSC{Float64,Int}
    V::SparseMatrixCSC{Float64,Int}
end

function build_phase_blocks_1D(fluid::Fluid{1})
    op_u = fluid.operator_u[1]
    op_p = fluid.operator_p
    cap_u = fluid.capacity_u[1]
    cap_p = fluid.capacity_p

    nu = prod(op_u.size)
    np = prod(op_p.size)

    μ = fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹ = build_I_D(op_u, μinv, cap_u)

    WG_uG = op_u.Wꜝ * op_u.G
    WG_uH = op_u.Wꜝ * op_u.H

    visc_uω = -(Iμ⁻¹ * op_u.G' * WG_uG)
    visc_uγ = -(Iμ⁻¹ * op_u.G' * WG_uH)
    grad = -((op_p.G + op_p.H))

    div_uω = - (op_p.G' + op_p.H')
    div_uγ =   (op_p.H')

    ρ = fluid.ρ
    Iρ = build_I_D(op_u, ρ, cap_u)
    mass = Iρ * op_u.V

    return StokesPhaseBlocks1D(nu, np, op_u, cap_u,
                               visc_uω, visc_uγ,
                               grad, div_uω, div_uγ,
                               mass, op_u.V)
end

struct StokesPhaseBlocks2D
    nu_x::Int
    nu_y::Int
    np::Int
    op_ux::DiffusionOps
    op_uy::DiffusionOps
    op_p::DiffusionOps
    cap_ux::Capacity
    cap_uy::Capacity
    visc_x_ω::SparseMatrixCSC{Float64,Int}
    visc_x_γ::SparseMatrixCSC{Float64,Int}
    visc_y_ω::SparseMatrixCSC{Float64,Int}
    visc_y_γ::SparseMatrixCSC{Float64,Int}
    grad_x::SparseMatrixCSC{Float64,Int}
    grad_y::SparseMatrixCSC{Float64,Int}
    div_x_ω::SparseMatrixCSC{Float64,Int}
    div_x_γ::SparseMatrixCSC{Float64,Int}
    div_y_ω::SparseMatrixCSC{Float64,Int}
    div_y_γ::SparseMatrixCSC{Float64,Int}
    mass_x::SparseMatrixCSC{Float64,Int}
    mass_y::SparseMatrixCSC{Float64,Int}
    Vx::SparseMatrixCSC{Float64,Int}
    Vy::SparseMatrixCSC{Float64,Int}
end

function build_phase_blocks_2D(fluid::Fluid{2})
    opx, opy = fluid.operator_u
    capx, capy = fluid.capacity_u
    opp = fluid.operator_p
    capp = fluid.capacity_p

    nu_x = prod(opx.size)
    nu_y = prod(opy.size)
    np = prod(opp.size)

    μ = fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμx⁻¹ = build_I_D(opx, μinv, capx)
    Iμy⁻¹ = build_I_D(opy, μinv, capy)

    WGx_Gx = opx.Wꜝ * opx.G
    WGx_Hx = opx.Wꜝ * opx.H
    Vx_x = -(Iμx⁻¹ * opx.G' * WGx_Gx)
    Vx_y = -(Iμx⁻¹ * opx.G' * WGx_Hx)

    WGy_Gy = opy.Wꜝ * opy.G
    WGy_Hy = opy.Wꜝ * opy.H
    Vy_x = -(Iμy⁻¹ * opy.G' * WGy_Gy)
    Vy_y = -(Iμy⁻¹ * opy.G' * WGy_Hy)

    grad_full = (opp.G + opp.H)
    total_rows = size(grad_full, 1)
    @assert total_rows == nu_x + nu_y
    x_rows = 1:nu_x
    y_rows = nu_x+1:nu_x+nu_y
    grad_x = -grad_full[x_rows, :]
    grad_y = -grad_full[y_rows, :]

    Gp = opp.G; Hp = opp.H
    Gp_x = Gp[x_rows, :]; Hp_x = Hp[x_rows, :]
    Gp_y = Gp[y_rows, :]; Hp_y = Hp[y_rows, :]
    div_x_ω = - (Gp_x' + Hp_x')
    div_x_γ =   (Hp_x')
    div_y_ω = - (Gp_y' + Hp_y')
    div_y_γ =   (Hp_y')

    ρ = fluid.ρ
    mass_x = build_I_D(opx, ρ, capx) * opx.V
    mass_y = build_I_D(opy, ρ, capy) * opy.V

    return StokesPhaseBlocks2D(nu_x, nu_y, np,
                               opx, opy, opp, capx, capy,
                               Vx_x, Vx_y, Vy_x, Vy_y,
                               grad_x, grad_y,
                               div_x_ω, div_x_γ, div_y_ω, div_y_γ,
                               mass_x, mass_y,
                               opx.V, opy.V)
end

function StokesDiph(fluid_a::Fluid{N}, fluid_b::Fluid{N},
                    bc_u_a::NTuple{N,BorderConditions},
                    bc_u_b::NTuple{N,BorderConditions},
                    bc_p::BorderConditions,
                    interface::InterfaceConditions,
                    bc_cut::AbstractBoundary;
                    x0=zeros(0)) where {N}
    nu_components_a = ntuple(i -> prod(fluid_a.operator_u[i].size), N)
    nu_components_b = ntuple(i -> prod(fluid_b.operator_u[i].size), N)
    np = prod(fluid_a.operator_p.size)
    Ntot = 2 * (sum(nu_components_a) + sum(nu_components_b)) + np
    x_init = length(x0) == Ntot ? x0 : zeros(Ntot)

    A = spzeros(Float64, Ntot, Ntot)
    b = zeros(Ntot)

    return StokesDiph{N}(fluid_a, fluid_b,
                         bc_u_a, bc_u_b,
                         bc_p, interface, bc_cut,
                         A, b, x_init, Any[])
end

StokesDiph(fluid_a::Fluid{1}, fluid_b::Fluid{1},
           bc_u_a::BorderConditions,
           bc_u_b::BorderConditions,
           bc_p::BorderConditions,
           interface::InterfaceConditions,
           bc_cut::AbstractBoundary;
           x0=zeros(0)) = StokesDiph(fluid_a, fluid_b,
                                     (bc_u_a,), (bc_u_b,),
                                     bc_p, interface, bc_cut;
                                     x0=x0)

function assemble_stokes!(s::StokesDiph)
    N = length(s.fluid_a.operator_u)
    if N == 1
        return assemble_stokes1D!(s)
    elseif N == 2
        return assemble_stokes2D!(s)
    else
        error("StokesDiph assembly not implemented for N=$(N)")
    end
end

function solve_StokesDiph!(s::StokesDiph; method=Base.:\, algorithm=nothing, kwargs...)
    assemble_stokes!(s)
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

function assemble_stokes2D!(s::StokesDiph)
    A1 = build_phase_blocks_2D(s.fluid_a)
    A2 = build_phase_blocks_2D(s.fluid_b)

    @assert A1.nu_x == A2.nu_x && A1.nu_y == A2.nu_y "Velocity DOFs mismatch between phases"
    @assert A1.np == A2.np "Pressure DOFs mismatch between phases"

    nx = A1.nu_x; ny = A1.nu_y; np = A1.np
    sum_nu = nx + ny

    rows = 4 * sum_nu + 2 * np
    cols = 4 * sum_nu + 2 * np
    A = spzeros(Float64, rows, cols)

    # Offsets for unknowns (phase 1 then phase 2)
    off_u1ωx = 0
    off_u1γx = nx
    off_u1ωy = 2 * nx
    off_u1γy = 2 * nx + ny
    off_p1   = 2 * sum_nu

    off_u2ωx = off_p1 + np
    off_u2γx = off_u2ωx + nx
    off_u2ωy = off_u2γx + nx
    off_u2γy = off_u2ωy + ny
    off_p2   = off_u2γy + ny

    # Row offsets
    row_m1x = 0
    row_m1y = nx
    row_m2x = 2 * nx + ny
    row_m2y = 2 * (nx + ny)
    row_jumpx = 2 * nx + 2 * ny
    row_jumpy = row_jumpx + nx
    row_fluxx = row_jumpy + ny
    row_fluxy = row_fluxx + nx
    row_div1 = row_fluxy + ny
    row_div2 = row_div1 + np

    # Momentum phase 1
    A[row_m1x+1:row_m1x+nx, off_u1ωx+1:off_u1ωx+nx] = A1.visc_x_ω
    A[row_m1x+1:row_m1x+nx, off_u1γx+1:off_u1γx+nx] = A1.visc_x_γ
    A[row_m1x+1:row_m1x+nx, off_p1+1:off_p1+np]     = A1.grad_x

    A[row_m1y+1:row_m1y+ny, off_u1ωy+1:off_u1ωy+ny] = A1.visc_y_ω
    A[row_m1y+1:row_m1y+ny, off_u1γy+1:off_u1γy+ny] = A1.visc_y_γ
    A[row_m1y+1:row_m1y+ny, off_p1+1:off_p1+np]     = A1.grad_y

    # Momentum phase 2
    A[row_m2x+1:row_m2x+nx, off_u2ωx+1:off_u2ωx+nx] = A2.visc_x_ω
    A[row_m2x+1:row_m2x+nx, off_u2γx+1:off_u2γx+nx] = A2.visc_x_γ
    A[row_m2x+1:row_m2x+nx, off_p2+1:off_p2+np]     = A2.grad_x

    A[row_m2y+1:row_m2y+ny, off_u2ωy+1:off_u2ωy+ny] = A2.visc_y_ω
    A[row_m2y+1:row_m2y+ny, off_u2γy+1:off_u2γy+ny] = A2.visc_y_γ
    A[row_m2y+1:row_m2y+ny, off_p2+1:off_p2+np]     = A2.grad_y

    # Interface velocity continuity u1γ = u2γ (per component)
    jump = s.interface.scalar
    α1 = jump === nothing ? 1.0 : jump.α₁
    α2 = jump === nothing ? 1.0 : jump.α₂
    g_jump_x = jump === nothing ? zeros(nx) : build_g_g(A1.op_ux, jump, A1.cap_ux)
    g_jump_y = jump === nothing ? zeros(ny) : build_g_g(A1.op_uy, jump, A1.cap_uy)

    A[row_jumpx+1:row_jumpx+nx, off_u1γx+1:off_u1γx+nx] = α1 * I(nx)
    A[row_jumpx+1:row_jumpx+nx, off_u2γx+1:off_u2γx+nx] = -α2 * I(nx)

    A[row_jumpy+1:row_jumpy+ny, off_u1γy+1:off_u1γy+ny] = α1 * I(ny)
    A[row_jumpy+1:row_jumpy+ny, off_u2γy+1:off_u2γy+ny] = -α2 * I(ny)

    # Flux continuity μ(H'W†G uω + H'W†H uγ) equal across phases per component
    flux = s.interface.flux
    β1 = flux === nothing ? 1.0 : flux.β₁
    β2 = flux === nothing ? 1.0 : flux.β₂
    g_flux_x = flux === nothing ? zeros(nx) : build_g_g(A1.op_ux, flux, A1.cap_ux)
    g_flux_y = flux === nothing ? zeros(ny) : build_g_g(A1.op_uy, flux, A1.cap_uy)

    WG1x = A1.op_ux.Wꜝ * A1.op_ux.G
    WH1x = A1.op_ux.Wꜝ * A1.op_ux.H
    WG2x = A2.op_ux.Wꜝ * A2.op_ux.G
    WH2x = A2.op_ux.Wꜝ * A2.op_ux.H
    WG1y = A1.op_uy.Wꜝ * A1.op_uy.G
    WH1y = A1.op_uy.Wꜝ * A1.op_uy.H
    WG2y = A2.op_uy.Wꜝ * A2.op_uy.G
    WH2y = A2.op_uy.Wꜝ * A2.op_uy.H

    H1xT = A1.op_ux.H'
    H2xT = A2.op_ux.H'
    H1yT = A1.op_uy.H'
    H2yT = A2.op_uy.H'

    μ1x = build_I_D(A1.op_ux, s.fluid_a.μ, A1.cap_ux)
    μ2x = build_I_D(A2.op_ux, s.fluid_b.μ, A2.cap_ux)
    μ1y = build_I_D(A1.op_uy, s.fluid_a.μ, A1.cap_uy)
    μ2y = build_I_D(A2.op_uy, s.fluid_b.μ, A2.cap_uy)

    A[row_fluxx+1:row_fluxx+nx, off_u1ωx+1:off_u1ωx+nx] = -β1 * (μ1x * (H1xT * WG1x))
    A[row_fluxx+1:row_fluxx+nx, off_u1γx+1:off_u1γx+nx] = -β1 * (μ1x * (H1xT * WH1x))
    A[row_fluxx+1:row_fluxx+nx, off_u2ωx+1:off_u2ωx+nx] =  β2 * (μ2x * (H2xT * WG2x))
    A[row_fluxx+1:row_fluxx+nx, off_u2γx+1:off_u2γx+nx] =  β2 * (μ2x * (H2xT * WH2x))

    A[row_fluxy+1:row_fluxy+ny, off_u1ωy+1:off_u1ωy+ny] = -β1 * (μ1y * (H1yT * WG1y))
    A[row_fluxy+1:row_fluxy+ny, off_u1γy+1:off_u1γy+ny] = -β1 * (μ1y * (H1yT * WH1y))
    A[row_fluxy+1:row_fluxy+ny, off_u2ωy+1:off_u2ωy+ny] =  β2 * (μ2y * (H2yT * WG2y))
    A[row_fluxy+1:row_fluxy+ny, off_u2γy+1:off_u2γy+ny] =  β2 * (μ2y * (H2yT * WH2y))

    # Divergence per phase
    A[row_div1+1:row_div1+np, off_u1ωx+1:off_u1ωx+nx] = A1.div_x_ω
    A[row_div1+1:row_div1+np, off_u1γx+1:off_u1γx+nx] = A1.div_x_γ
    A[row_div1+1:row_div1+np, off_u1ωy+1:off_u1ωy+ny] = A1.div_y_ω
    A[row_div1+1:row_div1+np, off_u1γy+1:off_u1γy+ny] = A1.div_y_γ

    A[row_div2+1:row_div2+np, off_u2ωx+1:off_u2ωx+nx] = A2.div_x_ω
    A[row_div2+1:row_div2+np, off_u2γx+1:off_u2γx+nx] = A2.div_x_γ
    A[row_div2+1:row_div2+np, off_u2ωy+1:off_u2ωy+ny] = A2.div_y_ω
    A[row_div2+1:row_div2+np, off_u2γy+1:off_u2γy+ny] = A2.div_y_γ

    # RHS
    f1x = safe_build_source(A1.op_ux, s.fluid_a.fᵤ, A1.cap_ux, nothing)
    f1y = safe_build_source(A1.op_uy, s.fluid_a.fᵤ, A1.cap_uy, nothing)
    f2x = safe_build_source(A2.op_ux, s.fluid_b.fᵤ, A2.cap_ux, nothing)
    f2y = safe_build_source(A2.op_uy, s.fluid_b.fᵤ, A2.cap_uy, nothing)
    b_m1x = A1.Vx * f1x
    b_m1y = A1.Vy * f1y
    b_m2x = A2.Vx * f2x
    b_m2y = A2.Vy * f2y
    b_jumpx = g_jump_x
    b_jumpy = g_jump_y
    b_fluxx = g_flux_x
    b_fluxy = g_flux_y
    b_div1 = zeros(np)
    b_div2 = zeros(np)

    b = vcat(b_m1x, b_m1y, b_m2x, b_m2y, b_jumpx, b_jumpy, b_fluxx, b_fluxy, b_div1, b_div2)

    # Apply Dirichlet BCs for velocities of both phases
    apply_velocity_dirichlet_2D!(A, b, s.bc_u_a[1], s.bc_u_a[2], s.fluid_a.mesh_u;
                                 nu_x=nx, nu_y=ny,
                                 uωx_off=off_u1ωx, uγx_off=off_u1γx,
                                 uωy_off=off_u1ωy, uγy_off=off_u1γy,
                                 row_uωx_off=row_m1x, row_uγx_off=row_m1x+nx,
                                 row_uωy_off=row_m1y, row_uγy_off=row_m1y+ny)

    apply_velocity_dirichlet_2D!(A, b, s.bc_u_b[1], s.bc_u_b[2], s.fluid_b.mesh_u;
                                 nu_x=nx, nu_y=ny,
                                 uωx_off=off_u2ωx, uγx_off=off_u2γx,
                                 uωy_off=off_u2ωy, uγy_off=off_u2γy,
                                 row_uωx_off=row_m2x, row_uγx_off=row_m2x+nx,
                                 row_uωy_off=row_m2y, row_uγy_off=row_m2y+ny)

    apply_pressure_gauge!(A, b, s.bc_p, s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_p1, np=np, row_start=row_div1+1)
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_p2, np=np, row_start=row_div2+1)

    s.A = A
    s.b = b
    return nothing
end

function assemble_stokes1D!(s::StokesDiph)
    blocks_a = build_phase_blocks_1D(s.fluid_a)
    blocks_b = build_phase_blocks_1D(s.fluid_b)

    @assert blocks_a.nu == blocks_b.nu "Interface requires matching velocity DOFs per phase"
    @assert blocks_a.np == blocks_b.np "Pressure grids must have matching DOFs"

    nu = blocks_a.nu
    np = blocks_a.np

    rows = 4 * nu + 2 * np
    cols = 4 * nu + 2 * np
    A = spzeros(Float64, rows, cols)

    off_u1ω = 0
    off_u1γ = nu
    off_p1  = 2 * nu
    off_u2ω = off_p1 + np
    off_u2γ = off_u2ω + nu
    off_p2  = off_u2γ + nu

    row_mom1 = 0
    row_jump = nu
    row_mom2 = 2 * nu
    row_flux = 3 * nu
    row_div1 = 4 * nu
    row_div2 = row_div1 + np

    # Phase A momentum
    A[row_mom1+1:row_mom1+nu, off_u1ω+1:off_u1ω+nu] = blocks_a.visc_uω
    A[row_mom1+1:row_mom1+nu, off_u1γ+1:off_u1γ+nu] = blocks_a.visc_uγ
    A[row_mom1+1:row_mom1+nu, off_p1+1:off_p1+np]   = blocks_a.grad

    # Phase B momentum
    A[row_mom2+1:row_mom2+nu, off_u2ω+1:off_u2ω+nu] = blocks_b.visc_uω
    A[row_mom2+1:row_mom2+nu, off_u2γ+1:off_u2γ+nu] = blocks_b.visc_uγ
    A[row_mom2+1:row_mom2+nu, off_p2+1:off_p2+np]   = blocks_b.grad

    jump = s.interface.scalar
    α1 = jump === nothing ? 1.0 : jump.α₁
    α2 = jump === nothing ? 1.0 : jump.α₂
    g_jump = jump === nothing ? zeros(nu) : build_g_g(blocks_a.op_u, jump, blocks_a.cap_u)

    A[row_jump+1:row_jump+nu, off_u1γ+1:off_u1γ+nu] = α1 * I(nu)
    A[row_jump+1:row_jump+nu, off_u2γ+1:off_u2γ+nu] = -α2 * I(nu)

    flux = s.interface.flux
    β1 = flux === nothing ? 1.0 : flux.β₁
    β2 = flux === nothing ? 1.0 : flux.β₂
    g_flux = flux === nothing ? zeros(nu) : build_g_g(blocks_a.op_u, flux, blocks_a.cap_u)

    WG_a = blocks_a.op_u.Wꜝ * blocks_a.op_u.G
    WH_a = blocks_a.op_u.Wꜝ * blocks_a.op_u.H
    WG_b = blocks_b.op_u.Wꜝ * blocks_b.op_u.G
    WH_b = blocks_b.op_u.Wꜝ * blocks_b.op_u.H

    traction_a_ω = blocks_a.op_u.H' * WG_a
    traction_a_γ = blocks_a.op_u.H' * WH_a
    traction_b_ω = blocks_b.op_u.H' * WG_b
    traction_b_γ = blocks_b.op_u.H' * WH_b

    μa = build_I_D(blocks_a.op_u, s.fluid_a.μ, blocks_a.cap_u)
    μb = build_I_D(blocks_b.op_u, s.fluid_b.μ, blocks_b.cap_u)

    A[row_flux+1:row_flux+nu, off_u1ω+1:off_u1ω+nu] = -β1 * (μa * traction_a_ω)
    A[row_flux+1:row_flux+nu, off_u1γ+1:off_u1γ+nu] = -β1 * (μa * traction_a_γ)
    A[row_flux+1:row_flux+nu, off_u2ω+1:off_u2ω+nu] =  β2 * (μb * traction_b_ω)
    A[row_flux+1:row_flux+nu, off_u2γ+1:off_u2γ+nu] =  β2 * (μb * traction_b_γ)

    # Divergence (phase A)
    A[row_div1+1:row_div1+np, off_u1ω+1:off_u1ω+nu] = blocks_a.div_uω
    A[row_div1+1:row_div1+np, off_u1γ+1:off_u1γ+nu] = blocks_a.div_uγ

    # Divergence (phase B)
    A[row_div2+1:row_div2+np, off_u2ω+1:off_u2ω+nu] = blocks_b.div_uω
    A[row_div2+1:row_div2+np, off_u2γ+1:off_u2γ+nu] = blocks_b.div_uγ

    f_a = safe_build_source(blocks_a.op_u, s.fluid_a.fᵤ, blocks_a.cap_u, nothing)
    f_b = safe_build_source(blocks_b.op_u, s.fluid_b.fᵤ, blocks_b.cap_u, nothing)
    b_mom1 = blocks_a.V * f_a
    b_mom2 = blocks_b.V * f_b
    b_div1 = zeros(np)
    b_div2 = zeros(np)

    b = vcat(b_mom1, g_jump, b_mom2, g_flux, b_div1, b_div2)

    # Apply velocity Dirichlet BCs per phase
    apply_velocity_dirichlet!(A, b, s.bc_u_a[1], s.fluid_a.mesh_u[1];
                              nu=nu, uω_offset=off_u1ω, uγ_offset=off_u1γ)
    apply_velocity_dirichlet!(A, b, s.bc_u_b[1], s.fluid_b.mesh_u[1];
                              nu=nu, uω_offset=off_u2ω, uγ_offset=off_u2γ)

    # Pressure gauge/Dirichlet for each phase
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid_a.mesh_p, s.fluid_a.capacity_p;
                          p_offset=off_p1, np=np, row_start=row_div1+1)
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid_b.mesh_p, s.fluid_b.capacity_p;
                          p_offset=off_p2, np=np, row_start=row_div2+1)

    s.A = A
    s.b = b
    return nothing
end
