mutable struct MeshTag{N}
    border_cells::Vector{Tuple{CartesianIndex, NTuple{N, Float64}}}
end

abstract type AbstractMesh end

"""
    Mesh{N}(n::NTuple{N, Int}, domain_size::NTuple{N, Float64}, x0::NTuple{N, Float64}=ntuple(_ -> 0.0, N))

Create a mesh object with `N` dimensions, `n` cells in each dimension, and a domain size of `domain_size`.

# Arguments
- `n::NTuple{N, Int}`: A tuple of integers specifying the number of cells in each dimension.
- `domain_size::NTuple{N, Float64}`: A tuple of floats specifying the size of the domain in each dimension.
- `x0::NTuple{N, Float64}`: A tuple of floats specifying the origin of the domain in each dimension. Default is the origin.

# Returns
- A `Mesh{N}` object with `N` dimensions, `n` cells in each dimension, and a domain size of `domain_size`.
"""
mutable struct Mesh{N} <: AbstractMesh
    nodes::NTuple{N, Vector{Float64}}
    centers::NTuple{N, Vector{Float64}}
    tag::MeshTag

    function Mesh(n::NTuple{N, Int}, domain_size::NTuple{N, Float64}, x0::NTuple{N, Float64}=ntuple(_ -> 0.0, N)) where N
        h_uniform = ntuple(i -> fill(domain_size[i] / n[i], n[i]), N)
        centers_uniform = ntuple(i -> [x0[i] + j * (domain_size[i] / n[i]) for j in 0:n[i]-1], N)
        nodes_uniform  = ntuple(i -> [x0[i] + (j + 0.5) * (domain_size[i] / n[i]) for j in 0:(n[i])], N) 
        temp_mesh = new{N}(nodes_uniform, centers_uniform, MeshTag{N}([]))

        # Get the border cells
        bc = get_border_cells(temp_mesh)
        temp_mesh.tag = MeshTag{N}(bc)
        return temp_mesh
    end
end

"""
    get_border_cells(mesh::Mesh{N})

Return a collection of tuples identifying the cells at the boundary of `mesh`. Each
tuple contains a `CartesianIndex` referencing the cell’s location in the mesh and
an `NTuple{N, Float64}` specifying the physical center coordinates of that cell.

A cell is considered a border cell if it lies at the first or last index along
any dimension of the mesh. This procedure determines the set of boundary cells
by iterating over all cell indices in the mesh and checking each one.

# Parameters
- `mesh::Mesh{N}`: The mesh for which border cells are to be identified.

# Returns
- `Vector{Tuple{CartesianIndex, NTuple{N, Float64}}}`: A list of tuples representing
  each border cell by its index and center coordinates.
"""
function get_border_cells(mesh::Mesh{N}) where N
    # Number of cells in each dimension
    dims = ntuple(i -> length(mesh.centers[i]), N)
    border_cells = Vector{Tuple{CartesianIndex, NTuple{N, Float64}}}()
    
    # Iterate over all cell indices using Iterators.product
    for idx in Iterators.product((1:d for d in dims)...)
        # A cell is at the border if any index equals 1 or the maximum in that dimension.
        if any(d -> idx[d] == 1 || idx[d] == dims[d], 1:N)
            # Get the physical cell center: tuple (mesh.centers[1][i₁], mesh.centers[2][i₂], ...)
            pos = ntuple(d -> mesh.centers[d][idx[d]], N)
            push!(border_cells, (CartesianIndex(idx), pos))
        end
    end
    return border_cells
end

"""
    nC(mesh::Mesh{N}) where N

Calculate the total number of cells in a mesh.

# Arguments
- `mesh::Mesh{N}`: A mesh object of type `Mesh` with `N` dimensions.

# Returns
- An integer representing the total number of cells in the mesh, calculated as the product of the lengths of the mesh centers.

# Example
x = range(0.0, stop=1.0, length=5)
mesh1D = Mesh((x,))
nC(mesh1D) == 5
"""
nC(mesh::Mesh{N}) where N = prod(length.(mesh.centers))

mutable struct SpaceTimeMesh{M} <: AbstractMesh
    nodes::NTuple{M, Vector{Float64}}
    centers::NTuple{M, Vector{Float64}}
    tag::MeshTag

    function SpaceTimeMesh(spaceMesh::Mesh{N}, time::Vector{Float64}; tag::MeshTag=MeshTag{N}([])) where {N}
        local M = N + 1

        Δt = diff(time)
        centers_time = [(time[i+1] + time[i]) / 2 for i in 1:length(time)-1]
        nodes = ntuple(i -> i<=N ? spaceMesh.nodes[i] : time, M)
        centers = ntuple(i -> i<=N ? spaceMesh.centers[i] : centers_time, M)
        return new{M}(nodes, centers, tag)
    end
end

nC(mesh::SpaceTimeMesh{M}) where M = prod(length.(mesh.centers))