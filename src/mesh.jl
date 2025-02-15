abstract type AbstractMesh end

# Mesh struct 
# Type of mesh : x--!--x--!--x with x: cell center and !: cell boundary
# Start by a cell center then a cell boundary and so on  ... and finish by a cell center
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

# Function to extract border cells from a Mesh
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

# Function to get the total number of cells in a mesh
nC(mesh::Mesh{N}) where N = prod(length.(mesh.centers))