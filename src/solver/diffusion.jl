"""
    DiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)

Create a solver for a steady-state monophasic diffusion problem.

Arguments:
- `phase` : Phase object representing the phase of the problem.
- `bc_b` : BorderConditions object representing the boundary conditions of the problem at the boundary.
- `bc_i` : AbstractBoundary object representing the internal boundary conditions of the problem.

Returns:
- `s` : Solver object representing the created solver.
"""
function DiffusionSteadyMono(phase::Phase, bc_b::BorderConditions, bc_i::AbstractBoundary)
    println("Solver creation:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    
    s = Solver(Steady, Monophasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    s.A = A_mono_stead_diff(phase.operator, phase.capacity, phase.Diffusion_coeff, bc_b, bc_i)
    s.b = b_mono_stead_diff(phase.operator, phase.source, phase.capacity, bc_b, bc_i)

    BC_border_mono!(s.A, s.b, bc_b, phase.capacity.mesh)

    return s
end

function A_mono_stead_diff(operator::DiffusionOps, capacity::Capacity, D, bc_b::BorderConditions, bc::AbstractBoundary)
    n = prod(operator.size)
    Iₐ, Iᵦ = build_I_bc(operator, bc)
    Iᵧ =  capacity.Γ
    Id = build_I_D(operator, D, capacity)

    A1 = Id * operator.G' * operator.Wꜝ * operator.G
    A2 = Id * operator.G' * operator.Wꜝ * operator.H
    A3 = Iᵦ * operator.H' * operator.Wꜝ * operator.G
    A4 = Iᵦ * operator.H' * operator.Wꜝ * operator.H + Iₐ * Iᵧ

    A = vcat(hcat(A1, A2), hcat(A3, A4))
    return A
end

function b_mono_stead_diff(operator::DiffusionOps, f::Function, capacite::Capacity, bc_b::BorderConditions, bc::AbstractBoundary)
    N = prod(operator.size)
    b = zeros(2N)

    Iᵧ = capacite.Γ 
    fₒ = build_source(operator, f, capacite)
    gᵧ = build_g_g(operator, bc, capacite)

    # Build the right-hand side
    b1 = operator.V * fₒ
    b2 = Iᵧ * gᵧ
    b = vcat(b1, b2)
    return b
end

function solve_DiffusionSteadyMono!(s::Solver; method::Function = gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the system:")
    println("- Monophasic problem")
    println("- Steady problem")
    println("- Diffusion problem")

    # Solve the system
    solve_system!(s; method, kwargs...)
end



# Diffusion - Steady - Diphasic
"""
    DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)

Creates a solver for a steady-state two-phase diffusion problem.

# Arguments
- `phase1::Phase`: The first phase of the problem.
- `phase2::Phase`: The second phase of the problem.
- `bc_b::BorderConditions`: The boundary conditions of the problem.
- `ic::InterfaceConditions`: The conditions at the interface between the two phases.
"""
function DiffusionSteadyDiph(phase1::Phase, phase2::Phase, bc_b::BorderConditions, ic::InterfaceConditions)
    println("Solver creation:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Diffusion problem")
    
    s = Solver(Steady, Diphasic, Diffusion, nothing, nothing, nothing, ConvergenceHistory(), [])
    
    s.A = A_diph_stead_diff(phase1.operator, phase2.operator, phase1.capacity, phase2.capacity, phase1.Diffusion_coeff, phase2.Diffusion_coeff, bc_b, ic)
    s.b = b_diph_stead_diff(phase1.operator, phase2.operator, phase1.source, phase2.source, phase1.capacity, phase2.capacity, bc_b, ic)

    BC_border_diph!(s.A, s.b, bc_b, phase2.capacity.mesh)

    return s
end

function A_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, capacite1::Capacity, capacite2::Capacity, D1, D2, bc_b::BorderConditions, ic::InterfaceConditions)
    n = prod(operator1.size)

    jump, flux = ic.scalar, ic.flux
    Iₐ1, Iₐ2 = jump.α₁ * I(n), jump.α₂ * I(n)
    Iᵦ1, Iᵦ2 = flux.β₁ * I(n), flux.β₂ * I(n)
    Id1, Id2 = build_I_D(operator1, D1, capacite1), build_I_D(operator2, D2, capacite2)

    block1 = Id1 * operator1.G' * operator1.Wꜝ * operator1.G
    block2 = Id1 * operator1.G' * operator1.Wꜝ * operator1.H
    block3 = Id2 * operator2.G' * operator2.Wꜝ * operator2.G
    block4 = Id2 * operator2.G' * operator2.Wꜝ * operator2.H
    block5 = operator1.H' * operator1.Wꜝ * operator1.G
    block6 = operator1.H' * operator1.Wꜝ * operator1.H 
    block7 = operator2.H' * operator2.Wꜝ * operator2.G
    block8 = operator2.H' * operator2.Wꜝ * operator2.H

    A = spzeros(Float64, 4n, 4n)

    @inbounds begin
        # Top-left blocks
        A[1:n, 1:n] += block1
        A[1:n, n+1:2n] += block2

        # Middle blocks
        A[n+1:2n, n+1:2n] += Iₐ1
        A[n+1:2n, 3n+1:4n] -= Iₐ2

        # Bottom-left blocks
        A[2n+1:3n, 2n+1:3n] += block3
        A[2n+1:3n, 3n+1:4n] += block4

        # Bottom blocks with Iᵦ
        A[3n+1:4n, 1:n] += Iᵦ1 * block5
        A[3n+1:4n, n+1:2n] += Iᵦ1 * block6
        A[3n+1:4n, 2n+1:3n] += Iᵦ2 * block7
        A[3n+1:4n, 3n+1:4n] += Iᵦ2 * block8
    end

    return A
end

function b_diph_stead_diff(operator1::DiffusionOps, operator2::DiffusionOps, f1, f2, capacite1::Capacity, capacite2::Capacity, bc_b::BorderConditions, ic::InterfaceConditions)
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

function solve_DiffusionSteadyDiph!(s::Solver; method::Function = gmres, kwargs...)
    if s.A === nothing
        error("Solver is not initialized. Call a solver constructor first.")
    end

    println("Solving the system:")
    println("- Diphasic problem")
    println("- Steady problem")
    println("- Diffusion problem")

    # Solve the system
    solve_system!(s; method, kwargs...)
end