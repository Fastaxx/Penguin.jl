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

function StokesMono(fluid::Fluid{N},
                    bc_u::NTuple{N,BorderConditions},
                    bc_p::BorderConditions,
                    bc_cut::AbstractBoundary;
                    x0=zeros(0)) where {N}
    # Number of velocity dofs per component (assumes identical grids per component)
    nu = prod(fluid.operator_u[1].size)
    np = prod(fluid.operator_p.size)
    # Unknowns: [uω¹, uγ¹, ..., uωᴺ, uγᴺ, pω]
    Ntot = 2 * N * nu + np
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
Dispatches to 1D or 2D assembly based on operator dimensionality.
"""
function assemble_stokes!(s::StokesMono)
    # Number of velocity components (1D:1, 2D:2)
    N = length(s.fluid.operator_u)
    if N == 1
        return assemble_stokes1D!(s)
    elseif N == 2
        return assemble_stokes2D!(s)
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
    op_u = s.fluid.operator_u[1]
    op_p = s.fluid.operator_p
    cap_u = s.fluid.capacity_u[1]
    cap_p = s.fluid.capacity_p

    nu = prod(op_u.size)
    np = prod(op_p.size)

    # Build 1/μ diagonal
    μ = s.fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹ = build_I_D(op_u, μinv, cap_u)

    # Convenience products
    WG_uG = op_u.Wꜝ * op_u.G
    WG_uH = op_u.Wꜝ * op_u.H

    # Blocks for momentum (first n rows)
    Auu_ω = - (Iμ⁻¹ * op_u.G' * WG_uG)
    Auu_γ = - (Iμ⁻¹ * op_u.G' * WG_uH)
    # Pressure gradient as adjoint of discrete divergence under W/V weights
    Aup_ω = - ((op_p.G + op_p.H))

    # Blocks for cut-cell/interface BC on uγ (second n rows): I * uγ = g_cut
    Atu_ω =   spzeros(nu, nu)
    Atu_γ =   I(nu)
    Atp_ω =   spzeros(nu, np)

    # Blocks for continuity (third n rows)
    Acu_ω = - (op_p.G' + op_p.H')
    Acu_γ =   (op_p.H')
    Acp_ω =   spzeros(np, np)

    # Assemble full A with rows = 3*nu, cols = 2*nu + np
    N = nu + nu + np
    rows = 3*nu
    A = spzeros(Float64, rows, N)
    # Momentum rows
    A[1:nu, 1:nu]           = Auu_ω
    A[1:nu, nu+1:2nu]       = Auu_γ
    A[1:nu, 2nu+1:2nu+np]   = Aup_ω
    # Tie rows
    A[nu+1:2nu, 1:nu]       = Atu_ω
    A[nu+1:2nu, nu+1:2nu]   = Atu_γ
    A[nu+1:2nu, 2nu+1:2nu+np] = Atp_ω
    # Continuity rows
    A[2nu+1:3nu, 1:nu]      = Acu_ω
    A[2nu+1:3nu, nu+1:2nu]  = Acu_γ
    # Note: continuity has no direct pressure term in this formulation

    # Assemble RHS b
    fᵤ = s.fluid.fᵤ
    fₒ = build_source(op_u, fᵤ, cap_u)
    b_mom = op_u.V * fₒ
    # Build cut-cell RHS from provided boundary condition (Dirichlet for now)
    g_cut = build_g_g(op_u, s.bc_cut, cap_u)
    b_con = zeros(nu)
    b = vcat(b_mom, g_cut, b_con)

    # Apply Dirichlet velocity BC on uω at domain boundaries (1D)
    apply_velocity_dirichlet!(A, b, s.bc_u[1], s.fluid.mesh_u[1];
                              nu=nu, uω_offset=0, uγ_offset=nu)

    # Fix pressure gauge using left pressure Dirichlet if provided, otherwise p[1]=0
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p;
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
    # Per-component velocity operators and capacities
    ops_u = s.fluid.operator_u
    caps_u = s.fluid.capacity_u
    op_p = s.fluid.operator_p
    cap_p = s.fluid.capacity_p

    @assert length(ops_u) == 2 "assemble_stokes2D! expects Fluid with 2 velocity components"

    nu = prod(ops_u[1].size)
    np = prod(op_p.size)

    # Build 1/μ diagonals per component
    μ = s.fluid.μ
    μinv = μ isa Function ? (args...)->1.0/μ(args...) : 1.0/μ
    Iμ⁻¹_x = build_I_D(ops_u[1], μinv, caps_u[1])
    Iμ⁻¹_y = build_I_D(ops_u[2], μinv, caps_u[2])

    # Viscous/tie blocks using per-component operators
    WGx_Gx = ops_u[1].Wꜝ * ops_u[1].G
    WGx_Hx = ops_u[1].Wꜝ * ops_u[1].H
    Auu_ωx = - (Iμ⁻¹_x * ops_u[1].G' * WGx_Gx)
    Auu_γx = - (Iμ⁻¹_x * ops_u[1].G' * WGx_Hx)

    WGy_Gy = ops_u[2].Wꜝ * ops_u[2].G
    WGy_Hy = ops_u[2].Wꜝ * ops_u[2].H
    Auu_ωy = - (Iμ⁻¹_y * ops_u[2].G' * WGy_Gy)
    Auu_γy = - (Iμ⁻¹_y * ops_u[2].G' * WGy_Hy)

    # Pressure gradient coupling blocks (directional picks from p-operators)
    # Use the same discrete form as 1D: -(G + H) acting on pω, then select directional parts.
    # Here we approximate by splitting rows for x and y equally.
    Gp = op_p.G; Hp = op_p.H
    # Heuristic split of rows (first nu rows -> x, next nu rows -> y)
    Aupx_ω = - Gp[1:nu, :] - Hp[1:nu, :]
    Aupy_ω = - Gp[nu+1:2nu, :] - Hp[nu+1:2nu, :]

    # Tie/interface identity
    Mγ = I(nu)

    # Continuity rows: use divergence form similar to 1D with p-operator as test space
    # Map component u onto p-grid via p-operators; simple surrogate using (Gp'+Hp') and H parts
    Acx_ω = - (Gp[1:nu, :]' + Hp[1:nu, :]')
    Acx_γ =   (Hp[1:nu, :]') 
    Acy_ω = - (Gp[nu+1:2nu, :]' + Hp[nu+1:2nu, :]')
    Acy_γ =   (Hp[nu+1:2nu, :]') 

    # Assemble full matrix A with rows = 4*nu + np, cols = 4*nu + np
    rows = 4*nu + np
    cols = 4*nu + np
    A = spzeros(Float64, rows, cols)

    # Offsets
    off_uωx = 0
    off_uγx = nu
    off_uωy = 2nu
    off_uγy = 3nu
    off_p   = 4nu

    # Momentum x rows [1:nu]
    A[1:nu, off_uωx+1:off_uωx+nu] = Auu_ωx
    A[1:nu, off_uγx+1:off_uγx+nu] = Auu_γx
    A[1:nu, off_p+1:off_p+np]     = Aupx_ω

    # Tie x rows [nu+1:2nu]
    A[nu+1:2nu, off_uγx+1:off_uγx+nu] = Mγ

    # Momentum y rows [2nu+1:3nu]
    A[2nu+1:3nu, off_uωy+1:off_uωy+nu] = Auu_ωy
    A[2nu+1:3nu, off_uγy+1:off_uγy+nu] = Auu_γy
    A[2nu+1:3nu, off_p+1:off_p+np]     = Aupy_ω

    # Tie y rows [3nu+1:4nu]
    A[3nu+1:4nu, off_uγy+1:off_uγy+nu] = Mγ

    # Continuity rows [4nu+1:4nu+np]
    A[4nu+1:4nu+np, off_uωx+1:off_uωx+nu] = Acx_ω
    A[4nu+1:4nu+np, off_uγx+1:off_uγx+nu] = Acx_γ
    A[4nu+1:4nu+np, off_uωy+1:off_uωy+nu] = Acy_ω
    A[4nu+1:4nu+np, off_uγy+1:off_uγy+nu] = Acy_γ

    # RHS b: per-component body force, tie from cut-cell BC, continuity zeros
    fᵤ = s.fluid.fᵤ
    fₒx = build_source(ops_u[1], fᵤ, caps_u[1])
    fₒy = build_source(ops_u[2], fᵤ, caps_u[2])
    b_mom_x = ops_u[1].V * fₒx
    b_mom_y = ops_u[2].V * fₒy
    g_cut_x = build_g_g(ops_u[1], s.bc_cut, caps_u[1])
    g_cut_y = build_g_g(ops_u[2], s.bc_cut, caps_u[2])
    b_con = zeros(np)
    b = vcat(b_mom_x, g_cut_x, b_mom_y, g_cut_y, b_con)

    # Apply Dirichlet velocity BCs at domain boundaries for both components
    apply_velocity_dirichlet_2D!(A, b, s.bc_u[1], s.bc_u[2], s.fluid.mesh_u;
                                 nu=nu,
                                  uωx_off=off_uωx, uγx_off=off_uγx,
                                  uωy_off=off_uωy, uγy_off=off_uγy)

    # Fix pressure gauge or apply pressure Dirichlet at boundaries if provided
    apply_pressure_gauge!(A, b, s.bc_p, s.fluid.mesh_p;
                          p_offset=off_p, np=np, row_start=4nu+1)

    s.A = A
    s.b = b
    return nothing
end

"""
    apply_velocity_dirichlet_2D!(A, b, bc_ux, bc_uy, mesh_u; nu, uωx_off, uγx_off, uωy_off, uγy_off)

Apply Dirichlet BC for 2D velocity components on their respective meshes.
Enforces values on both uω and uγ rows for each component and boundary node.
"""
function apply_velocity_dirichlet_2D!(A::SparseMatrixCSC{Float64, Int}, b,
                                      bc_ux::BorderConditions,
                                      bc_uy::BorderConditions,
                                      mesh_u::NTuple{2,AbstractMesh};
                                      nu::Int, uωx_off::Int, uγx_off::Int, uωy_off::Int, uγy_off::Int)
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
    eval_val(bc, x, y) = (bc isa Dirichlet) ? (bc.value isa Function ? bc.value(x, y) : bc.value) : nothing

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
            vx = eval_val(bcx, xs_x[i], ys_x[jx])
            vy = eval_val(bcy, xs_y[i], ys_y[jy])
            if vx !== nothing
                lix = LIx[i, jx]
                r = lix
                A[r, :] .= 0.0; A[r, uωx_off + lix] = 1.0; b[r] = vx
                rt = nu + lix
                A[rt, :] .= 0.0; A[rt, uγx_off + lix] = 1.0; b[rt] = vx
            end
            if vy !== nothing
                liy = LIy[i, jy]
                r = 2nu + liy
                A[r, :] .= 0.0; A[r, uωy_off + liy] = 1.0; b[r] = vy
                rt = 3nu + liy
                A[rt, :] .= 0.0; A[rt, uγy_off + liy] = 1.0; b[rt] = vy
            end
        end
    end

    # Left/right (vary along y)
    for iside in ((1, bcx_left, bcy_left), (iright, bcx_right, bcy_right))
        ix, bcx, bcy = iside
        isnothing(bcx) && isnothing(bcy) && continue
        iy = ix
        for j in 1:ny
            vx = eval_val(bcx, xs_x[ix], ys_x[j])
            vy = eval_val(bcy, xs_y[iy], ys_y[j])
            if vx !== nothing
                lix = LIx[ix, j]
                r = lix
                A[r, :] .= 0.0; A[r, uωx_off + lix] = 1.0; b[r] = vx
                rt = nu + lix
                A[rt, :] .= 0.0; A[rt, uγx_off + lix] = 1.0; b[rt] = vx
            end
            if vy !== nothing
                liy = LIy[iy, j]
                r = 2nu + liy
                A[r, :] .= 0.0; A[r, uωy_off + liy] = 1.0; b[r] = vy
                rt = 3nu + liy
                A[rt, :] .= 0.0; A[rt, uγy_off + liy] = 1.0; b[rt] = vy
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
                                   nu::Int, uω_offset::Int, uγ_offset::Int)
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
        return v isa Function ? v(x) : v
    end

    vL = eval_value(left_bc,  xnodes[1])
    vR = eval_value(right_bc, xnodes[end-1])

    # Row indices: momentum rows are 1:nu, tie rows are nu+1:2nu
    if vL !== nothing
        # Enforce uω[iL] = vL via momentum row iL (use last interior row index convention: iL = 1)
        r = iL
        A[r, :] .= 0.0
        A[r, uω_offset + iL] = 1.0
        b[r] = vL
        # Also enforce on tie row for uγ
        rt = nu + iL
        A[rt, :] .= 0.0
        A[rt, uγ_offset + iL] = 1.0
        b[rt] = vL
    end
    if vR !== nothing
        # Enforce uω[iR] = vR via momentum row iR (rightmost velocity index = nx+1)
        r = iR
        A[r, :] .= 0.0
        A[r, uω_offset + iR] = 1.0
        b[r] = vR
        # Also enforce on tie row for uγ
        rt = nu + iR
        A[rt, :] .= 0.0
        A[rt, uγ_offset + iR] = 1.0
        b[rt] = vR
    end
    return nothing
end

"""
    apply_pressure_gauge!(A, b, bc_p, mesh_p; p_offset, np, row_start)

Fix one pressure dof: if left/right pressure Dirichlet exist, enforce them; otherwise set p[1]=0.
Rows `row_start : row_start+np-1` are the continuity block.
"""
function apply_pressure_gauge!(A::SparseMatrixCSC{Float64, Int}, b,
                               bc_p::BorderConditions, mesh_p::AbstractMesh;
                               p_offset::Int, np::Int, row_start::Int)
    # Determine gauge/Dirichlet values
    left_bc = get(bc_p.borders, :bottom, nothing)
    right_bc = get(bc_p.borders, :top, nothing)
    xnodes = mesh_p.nodes[1]
    vL = if (left_bc isa Dirichlet)
        v = left_bc.value
        v isa Function ? v(xnodes[1]) : v
    else
        nothing
    end
    vR = if (right_bc isa Dirichlet)
        v = right_bc.value
        v isa Function ? v(xnodes[end]) : v
    else
        nothing
    end

    # Row indices for continuity block
    rL = row_start
    rR = row_start + np - 1

    if vL !== nothing && vR !== nothing && np >= 2
        A[rL, :] .= 0.0; A[rL, p_offset + 1]  = 1.0; b[rL] = vL
        A[rR, :] .= 0.0; A[rR, p_offset + np] = 1.0; b[rR] = vR
    elseif vL !== nothing
        A[rL, :] .= 0.0; A[rL, p_offset + 1] = 1.0; b[rL] = vL
    elseif vR !== nothing
        A[rR, :] .= 0.0; A[rR, p_offset + np] = 1.0; b[rR] = vR
    else
        # No pressure Dirichlet; fix gauge p[1] = 0
        A[rL, :] .= 0.0; A[rL, p_offset + 1] = 1.0; b[rL] = 0.0
    end
    return nothing
end

function solve_StokesMono!(s::StokesMono; method=Base.:\, algorithm=nothing, kwargs...)
    println("[StokesMono] Assembling steady Stokes and solving (fully coupled)")
    # Re-assemble in case anything changed
    assemble_stokes!(s)

    # Remove zero rows and columns using a common index set to keep A square
    Ared, bred, keep_idx_rows, keep_idx_cols = remove_zero_rows_cols!(s.A, s.b)

    # Choose solver path
    xred = nothing
    if algorithm !== nothing
        # Use LinearSolve.jl algorithm if provided
        prob = LinearSolve.LinearProblem(Ared, bred)
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        xred = sol.u
    elseif method === Base.:\
        # Direct factorization/backsolve with singular fallback
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
        # IterativeSolvers.jl method (e.g., gmres, bicgstab, bicgstabl)
        kwargs_nt = (; kwargs...)
        log = get(kwargs_nt, :log, false)
        if log
            xred, ch = method(Ared, bred; kwargs...)
            push!(s.ch, ch)
        else
            xred = method(Ared, bred; kwargs...)
        end
    end

    # Reconstruct full solution in original column space
    N = size(s.A, 2)
    s.x = zeros(N)
    s.x[keep_idx_cols] = xred
    return s
end
