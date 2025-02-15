"""
    abstract type AbstractOperators

An abstract type representing a collection of operators.
"""
abstract type AbstractOperators end

# Elementary operators
function ẟ_m(n::Int, periodicity::Bool=false) D = spdiagm(0 => ones(n), -1 => -ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = -1.0; D[n, 1] = 1.0; end; D end
function δ_p(n::Int, periodicity::Bool=false) D = spdiagm(0 => -ones(n), 1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = -1.0; D[n, 1] = 1.0; end; D end
function Σ_m(n::Int, periodicity::Bool=false) D = 0.5 * spdiagm(0 => ones(n), -1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = 0.5; D[n, 1] = 0.5; end; D end
function Σ_p(n::Int, periodicity::Bool=false) D = 0.5 * spdiagm(0 => ones(n), 1 => ones(n-1)); D[n, n] = 0.0; if periodicity; D[1, n-1] = 0.5; D[n, 1] = 0.5; end; D end
function I(n::Int) spdiagm(0 => ones(n)) end

"""
    struct DiffusionOps{N} <: AbstractOperators where N

Struct representing diffusion operators.

# Fields
- `G::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator G.
- `H::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator H.
- `Wꜝ::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator Wꜝ.
- `V::SparseMatrixCSC{Float64, Int}`: Matrix representing the diffusion operator V.
- `size::NTuple{N, Int}`: Tuple representing the size of the diffusion operators.

"""
struct DiffusionOps{N} <: AbstractOperators where N
    G::SparseMatrixCSC{Float64, Int}
    H::SparseMatrixCSC{Float64, Int}
    Wꜝ::SparseMatrixCSC{Float64, Int}
    V::SparseMatrixCSC{Float64, Int}
    size::NTuple{N, Int}
end

"""
    function DiffusionOps(Capacity::AbstractCapacity)

Compute the diffusion operators from a given capacity.

# Arguments
- `Capacity`: Capacity of the system.  

# Returns
- `DiffusionOps`: Diffusion operators for the system.
"""
function DiffusionOps(Capacity::AbstractCapacity)
    mesh = Capacity.mesh
    N = length(mesh.nodes)
    if N == 1
        nx = length(mesh.nodes[1])
        G = ẟ_m(nx) * Capacity.B[1] # Gérer le periodicity
        H = Capacity.A[1]*ẟ_m(nx) - ẟ_m(nx)*Capacity.B[1]
        diagW = diag(blockdiag(Capacity.W[1]))
        new_diagW = [val != 0 ? 1.0 / val : 1.0 for val in diagW]
        Wꜝ = spdiagm(0 => new_diagW)
        sizes = (nx,)
    elseif N == 2
        nx, ny = length(mesh.nodes[1]), length(mesh.nodes[2])
        Dx_m = kron(I(ny), ẟ_m(nx))
        Dy_m = kron(ẟ_m(ny), I(nx))
        G = [Dx_m * Capacity.B[1]; Dy_m * Capacity.B[2]]
        H = [Capacity.A[1]*Dx_m - Dx_m*Capacity.B[1]; Capacity.A[2]*Dy_m - Dy_m*Capacity.B[2]]
        diagW = diag(blockdiag(Capacity.W[1], Capacity.W[2]))
        new_diagW = [val != 0 ? 1.0 / val : 1.0 for val in diagW]
        Wꜝ = spdiagm(0 => new_diagW)
        sizes = (nx, ny)
    elseif N == 3
        nx, ny, nz = length(mesh.nodes[1]), length(mesh.nodes[2]), length(mesh.nodes[3])
        Dx_m = kron(I(nz), kron(I(ny), ẟ_m(nx)))
        Dy_m = kron(I(nz), kron(ẟ_m(ny), I(nx)))
        Dz_m = kron(ẟ_m(nz), kron(I(ny), I(nx)))
        G = [Dx_m * Capacity.B[1]; Dy_m * Capacity.B[2]; Dz_m * Capacity.B[3]]
        H = [Capacity.A[1]*Dx_m - Dx_m*Capacity.B[1]; Capacity.A[2]*Dy_m - Dy_m*Capacity.B[2]; Capacity.A[3]*Dz_m - Dz_m*Capacity.B[3]]
        diagW = diag(blockdiag(Capacity.W[1], Capacity.W[2], Capacity.W[3]))
        new_diagW = [val != 0 ? 1.0 / val : 1.0 for val in diagW]
        Wꜝ = spdiagm(0 => new_diagW)
        sizes = (nx, ny, nz)
    end
    return DiffusionOps{N}(G, H, Wꜝ, Capacity.V, sizes)
end