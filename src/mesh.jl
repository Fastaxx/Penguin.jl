abstract type AbstractMesh end

"""
    Mesh{N}(nodes::NTuple{N, AbstractVector{<:Real}})

Construct a Mesh from a given tuple of node coordinates. For each dimension, this mesh 
is defined by its cell centers, boundaries, and the resulting cell sizes. 

# Arguments
- `nodes`: A tuple of coordinate vectors, where each vector represents 
  the positions of cell centers along that dimension.

# Fields
- `nodes::NTuple{N, Vector{Float64}}`: Computed positions of cell boundaries.
- `centers::NTuple{N, Vector{Float64}}`: Positions of the cell centers 
  (same as provided node coordinates).
- `sizes::NTuple{N, Vector{Float64}}`: Dimensions of each cell, with half-size 
  adjustments for the first and last cells in each dimension.

# Example
# Mesh struct 
# Type of mesh : x--!--x--!--x with x: cell center and !: cell boundary
# Start by a cell center then a cell boundary and so on  ... and finish by a cell center
"""
struct Mesh{N} <: AbstractMesh
    nodes::NTuple{N, Vector{Float64}}
    centers::NTuple{N, Vector{Float64}}
    sizes::NTuple{N, Vector{Float64}}

    function Mesh(nodes::NTuple{N, AbstractVector{<:Real}}) where N
        centers = ntuple(i -> collect(Float64.(nodes[i])), N)
        nodes = ntuple(i -> (centers[i][1:end-1] .+ centers[i][2:end]) ./ 2.0, N)
        # Compute the sizes of the cells : The first and last cells have a size that is half of the others
        sizes = ntuple(i -> diff(nodes[i]), N)
        sizes = ntuple(i -> [sizes[i][1] / 2; sizes[i][1:end]; sizes[i][end] / 2], N)
        return new{N}(nodes, centers, sizes)
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
x = collect(range(0.0, stop=1.0, length=5))
mesh1D = Mesh((x,))
nC(mesh1D) == 5
"""
# Function to get the total number of cells in a mesh
nC(mesh::Mesh{N}) where N = prod(length.(mesh.centers))