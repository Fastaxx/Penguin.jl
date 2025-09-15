"""
    struct Phase

The `Phase` struct represents a phase in a system.

# Fields
- `capacity::AbstractCapacity`: The capacity of the phase.
- `operator::AbstractOperators`: The operators associated with the phase.
- `source::Function`: The source function.
- `Diffusion_coeff::Function`: The diffusion coefficient function.
"""
struct Phase
    capacity::AbstractCapacity
    operator::AbstractOperators
    source::Function
    Diffusion_coeff::Function
end

"""
    struct Fluid{N}

A fluid phase grouping velocity and pressure discretizations and material data
for (incompressible) Stokes/Navier–Stokes formulations.

# Fields
- `capacity_u::NTuple{N,AbstractCapacity}`: Per-component capacities for the velocity field (e.g., (ux,) in 1D, (ux,uy) in 2D)
- `operator_u::NTuple{N,AbstractOperators}`: Per-component operators for the velocity field
- `capacity_p::AbstractCapacity`: Capacity for the pressure field
- `operator_p::AbstractOperators`: Operators for the pressure field
- `μ::Union{Float64, Function}`: Dynamic viscosity (constant or function of space)
- `ρ::Union{Float64, Function}`: Density (constant or function of space)
- `fᵤ::Function`: Body force source term in momentum equation
- `fₚ::Function`: Mass source in continuity equation
"""
struct Fluid{N}
    capacity_u::NTuple{N,AbstractCapacity}
    operator_u::NTuple{N,AbstractOperators}
    capacity_p::AbstractCapacity
    operator_p::AbstractOperators
    μ::Union{Float64, Function}
    ρ::Union{Float64, Function}
    fᵤ::Function
    fₚ::Function
end

# Convenience constructors
Fluid(cap_u::AbstractCapacity,
      op_u::AbstractOperators,
      cap_p::AbstractCapacity,
      op_p::AbstractOperators,
      μ, ρ, fᵤ, fₚ) = Fluid{1}((cap_u,), (op_u,), cap_p, op_p, μ, ρ, fᵤ, fₚ)

function Fluid(cap_u::NTuple{N,AbstractCapacity},
               op_u::NTuple{N,AbstractOperators},
               cap_p::AbstractCapacity,
               op_p::AbstractOperators,
               μ, ρ, fᵤ, fₚ) where N
    return Fluid{N}(cap_u, op_u, cap_p, op_p, μ, ρ, fᵤ, fₚ)
end
