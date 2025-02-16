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
