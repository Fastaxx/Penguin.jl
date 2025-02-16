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