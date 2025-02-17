# AdvectionDiffusion - Steady - Monophasic
"""
    AdvectionDiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)

Creates a solver for a steady-state monophasic advection-diffusion problem.

# Arguments
- `phase::Phase`: The phase object representing the physical properties of the system.
- `bc_b::BorderConditions`: The border conditions object representing the boundary conditions at the outer border.
- `bc_i::AbstractBoundary`: The boundary conditions object representing the boundary conditions at the inner border.
"""
function AdvectionDiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)
    println("Solver Creation:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Steady, Monophasic, DiffusionAdvection, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    s.A = A_mono_stead_advdiff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_i)
    s.b = b_mono_stead_advdiff(phase.operator, phase.source, phase.capacity, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

function A_mono_stead_advdiff(operator::ConvectionOps, capacite::Capacity, D, bc::AbstractBoundary)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ = capacite.Γ 
    Id = build_I_D(operator, D, capacite)

    C = operator.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K = operator.K # NTuple{N, SparseMatrixCSC{Float64, Int}}

    A11 = (sum(C) + 0.5 * sum(K)) + Id * operator.G' * operator.Wꜝ * operator.G
    A12 = 0.5 * sum(K) + Id * operator.G' * operator.Wꜝ * operator.H
    A21 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    A22 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    A = vcat(hcat(A11, A12), hcat(A21, A22))
    return A
end

function b_mono_stead_advdiff(operator::ConvectionOps, f, capacite::Capacity, bc::AbstractBoundary)
    N = prod(operator.size)
    b = zeros(2N)

    Iᵧ = capacite.Γ
    fₒ = build_source(operator, f, capacite)
    gᵧ = build_g_g(operator, bc, capacite)

    # Build the right-hand side
    b = vcat(operator.V*fₒ, Iᵧ * gᵧ)

    return b
end

function solve_AdvectionDiffusionSteadyMono!(s::Solver; method::Function = gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    solve_system!(s; method, kwargs...)
end


# AdvectionDiffusion - Steady - Diphasic
"""
    AdvectionDiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)

Creates a solver for a steady-state two-phase advection-diffusion problem.

# Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
"""
function AdvectionDiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)
    println("Création du solveur:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Advection-Diffusion problem")
    
    s = Solver(Steady, Diphasic, DiffusionAdvection, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    s.A = A_diph_stead_advdiff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, ic)
    s.b = b_diph_stead_advdiff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, ic)

    BC_border_diph!(s.A, s.b, bc_b, phase2.capacity.mesh)

    return s
end

function A_diph_stead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, ic::InterfaceConditions)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁*I(n), jump.α₂*I(n)
    Iᵦ1, Iᵦ2 = flux.β₁*I(n), flux.β₂*I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    C1 = operator1.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K1 = operator1.K # NTuple{N, SparseMatrixCSC{Float64, Int}}
    C2 = operator2.C # NTuple{N, SparseMatrixCSC{Float64, Int}}
    K2 = operator2.K # NTuple{N, SparseMatrixCSC{Float64, Int}

    block1 = Id1 * operator1.G' * operator1.Wꜝ * operator1.G + (sum(C1) + 0.5 * sum(K1))
    block2 = Id1 * operator1.G' * operator1.Wꜝ * operator1.H + 0.5 * sum(K1)
    block3 = Id2 * operator2.G' * operator2.Wꜝ * operator2.G + (sum(C2) + 0.5 * sum(K2))
    block4 = Id2 * operator2.G' * operator2.Wꜝ * operator2.H + 0.5 * sum(K2)
    block5 = operator1.H' * operator1.Wꜝ * operator1.G
    block6 = operator1.H' * operator1.Wꜝ * operator1.H 
    block7 = operator2.H' * operator2.Wꜝ * operator2.G
    block8 = operator2.H' * operator2.Wꜝ * operator2.H

    A = vcat(hcat(block1, block2, zeros(n, n), zeros(n, n)),
             hcat(zeros(n, n), Iₐ1, zeros(n, n), -Iₐ2),
             hcat(zeros(n, n), zeros(n, n), block3, block4),
             hcat(Iᵦ1*block5, Iᵦ1*block6, Iᵦ2*block7, Iᵦ2*block8))
    return A
end

function b_diph_stead_advdiff(operator1::ConvectionOps, operator2::ConvectionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, ic::InterfaceConditions)
    N = prod(operator1.size)
    b = zeros(4N)

    jump, flux = ic.scalar, ic.flux
    Iᵧ1, Iᵧ2 = capacite1.Γ, capacite2.Γ
    gᵧ, hᵧ = build_g_g(operator1, jump, capacite1), build_g_g(operator2, flux, capacite2)

    fₒ1 = build_source(operator1, f1, capacite1)
    fₒ2 = build_source(operator2, f2, capacite2)

    # Build the right-hand side
    b = vcat(operator1.V*fₒ1, gᵧ, operator2.V*fₒ2, Iᵧ2*hᵧ)

    return b
end

function solve_AdvectionDiffusionSteadyDiph!(s::Solver; method::Function = gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    solve_system!(s; method, kwargs...)
end
