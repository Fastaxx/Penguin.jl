module Penguin

using SparseArrays
using StaticArrays
using CartesianGeometry
using ImplicitIntegration
using LinearAlgebra
# Write your package code here.

include("mesh.jl")
export Mesh, get_border_cells, nC

include("capacity.jl")
export Capacity

include("operators.jl")
export ẟ_m, δ_p, Σ_m, Σ_p, I
export AbstractOperators, DiffusionOps

end
