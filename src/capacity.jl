"""
    abstract type AbstractCapacity

Abstract type representing a capacity.
"""
abstract type AbstractCapacity end

"""
    mutable struct Capacity{N} <: AbstractCapacity

The `Capacity` struct represents the capacity of a system in `N` dimensions.

# Fields
- `A`: A capacity represented by `N` sparse matrices (`Ax`, `Ay`).
- `B`: B capacity represented by `N` sparse matrices (`Bx`, `By`).
- `V`: Volume capacity represented by a sparse matrix.
- `W`: Staggered volume capacity represented by `N` sparse matrices.
- `C_ω`: Cell centroid represented by a vector of `N`-dimensional static vectors.
- `C_γ`: Interface centroid represented by a vector of `N`-dimensional static vectors.
- `Γ`: Interface norm represented by a sparse matrix.
- `cell_types`: Cell types.
- `mesh`: Mesh of `N` dimensions.

"""
mutable struct Capacity{N} <: AbstractCapacity
    A :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # A capacity : Ax, Ay
    B :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # B capacity : Bx, By
    V :: SparseMatrixCSC{Float64, Int}              # Volume
    W :: NTuple{N, SparseMatrixCSC{Float64, Int}}   # Staggered Volume
    C_ω :: Vector{SVector{N,Float64}}               # Cell Centroid
    C_γ :: Vector{SVector{N,Float64}}               # Interface Centroid
    Γ :: SparseMatrixCSC{Float64, Int}              # Interface Norm
    cell_types :: Vector{Float64}                   # Cell Types
    mesh :: AbstractMesh                            # Mesh
    body :: Function                                # Body function (signed distance function)
end

"""
    Capacity(body::Function, mesh::CartesianMesh; method::String = "VOFI")

Compute the capacity of a body in a given mesh using a specified method.

# Arguments
- `body::Function`: The body for which to compute the capacity.
- `mesh::CartesianMesh`: The mesh in which the body is located.
- `method::String`: The method to use for computing the capacity. Default is "VOFI".

# Returns
- `Capacity{N}`: The capacity of the body.
"""
function Capacity(body::Function, mesh::AbstractMesh; method::String = "VOFI")

    if method == "VOFI"
        A, B, V, W, C_ω, C_γ, Γ, cell_types = VOFI(body, mesh)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    elseif method == "ImplicitIntegration"
        println("Not implemented yet")
        #A, B, V, W, C_ω, C_γ, Γ, cell_types = ImplicitIntegration(body, mesh)
        #N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    end    
end

"""
    VOFI(body::Function, mesh::CartesianMesh)

Compute the Capacity quantities based on VOFI for a given body and mesh.

# Arguments
- `body::Function`: The body for which to compute the VOFI quantities.
- `mesh::CartesianMesh`: The mesh on which to compute the VOFI quantities.

# Returns
- `A::Tuple`: The A matrices for each dimension of the mesh.
- `B::Tuple`: The B matrices for each dimension of the mesh.
- `V::SparseMatrixCSC`: The V matrix.
- `W::Tuple`: The W matrices for each dimension of the mesh.
- `C_ω::Vector`: The C_ω vector : Cell centroid.
- `C_γ::Vector`: The C_γ vector : Interface centroid.
- `Γ::SparseMatrixCSC`: The Γ matrix : Interface Norm.

"""
function VOFI(body::Function, mesh::AbstractMesh)
    N = length(mesh.nodes)
    nc = nC(mesh)

    Vs, bary, interface_length, cell_types = spzeros(nc), zeros(N), spzeros(nc), spzeros(nc)
    As, Bs, Ws = (spzeros(nc), spzeros(nc), spzeros(nc)), (spzeros(nc), spzeros(nc), spzeros(nc)), (spzeros(nc), spzeros(nc), spzeros(nc))

    Vs, bary, interface_length, cell_types = CartesianGeometry.integrate(Tuple{0}, body, mesh.nodes, Float64, zero)
    As = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero)
    Ws = CartesianGeometry.integrate(Tuple{0}, body, mesh.nodes, Float64, zero, bary)
    Bs = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero, bary)

    C_ω = bary
    if N == 1
        V = spdiagm(0 => Vs)
        A = (spdiagm(0 => As[1]),)
        B = (spdiagm(0 => Bs[1]),)
        W = (spdiagm(0 => Ws[1]),)
        Γ = spdiagm(0 => interface_length)
        C_γ = computeInterfaceCentroids(mesh, body)
    elseif N == 2
        V = spdiagm(0 => Vs)
        A = (spdiagm(0 => As[1]), spdiagm(0 => As[2]))
        B = (spdiagm(0 => Bs[1]), spdiagm(0 => Bs[2]))
        W = (spdiagm(0 => Ws[1]), spdiagm(0 => Ws[2]))
        Γ = spdiagm(0 => interface_length)
        C_γ = computeInterfaceCentroids(mesh, body)
    elseif N == 3
        V = spdiagm(0 => Vs)
        A = (spdiagm(0 => As[1]), spdiagm(0 => As[2]), spdiagm(0 => As[3]))
        B = (spdiagm(0 => Bs[1]), spdiagm(0 => Bs[2]), spdiagm(0 => Bs[3]))
        W = (spdiagm(0 => Ws[1]), spdiagm(0 => Ws[2]), spdiagm(0 => Ws[3]))
        Γ = spdiagm(0 => interface_length)
        C_γ = computeInterfaceCentroids(mesh, body)
    end

    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

"""
    computeInterfaceCentroids(mesh, body)

Compute the interface centroids for a 1D mesh and body.

# Arguments
- `mesh::AbstractMesh`: The mesh on which to compute the interface centroids.
- `body::Function`: The body for which to compute the interface centroids.

# Returns
- `C_γ::Vector`: The interface centroids.
"""
function computeInterfaceCentroids(mesh::Mesh{1}, body)
    x_coords = mesh.nodes[1]
    nx = length(x_coords) - 1
    Φ = (r) -> body(r[1])

    # Create a *vector* of SVector{1,Float64}
    C_γ = Vector{SVector{1,Float64}}(undef, nx+1)

    for i in 1:nx
        a = (x_coords[i],)   # 1D point
        b = (x_coords[i+1],) # 1D point

        measure_val = ImplicitIntegration.integrate(_->1, Φ, a, b; surface=true).val
        if measure_val > 0
            x_c = ImplicitIntegration.integrate(p->p[1], Φ, a, b; surface=true).val / measure_val
            C_γ[i] = SVector{1,Float64}(x_c)
        else
            C_γ[i] = SVector{1,Float64}(0.0)
        end
    end

    return C_γ
end

"""
    computeInterfaceCentroids(mesh, body) 

Compute the interface centroids for a 2D mesh and body.

# Arguments
- `mesh::AbstractMesh`: The mesh on which to compute the interface centroids.
- `body::Function`: The body for which to compute the interface centroids.

# Returns
- `C_γ::Vector`: The interface centroids.
"""
function computeInterfaceCentroids(mesh::Mesh{2}, body)
    x_coords, y_coords = mesh.nodes
    nx, ny = length(x_coords) - 1, length(y_coords) - 1
    Φ = (r) -> body(r[1], r[2], 0.0)

    # Build a single vector of length nx * ny
    C_γ = Vector{SVector{2,Float64}}(undef, (nx+1) * (ny+1))

    # Fill each entry by flattening i,j -> idx
    for j in 1:ny
        for i in 1:nx
            idx = (nx+1) * (j-1) + i
            a = (x_coords[i],   y_coords[j])
            b = (x_coords[i+1], y_coords[j+1])

            measure_val = ImplicitIntegration.integrate(_->1, Φ, a, b; surface=true).val
            if measure_val > 0
                x_c = ImplicitIntegration.integrate(p->p[1], Φ, a, b; surface=true).val / measure_val
                y_c = ImplicitIntegration.integrate(p->p[2], Φ, a, b; surface=true).val / measure_val
                C_γ[idx] = SVector{2,Float64}(x_c, y_c)
            else
                C_γ[idx] = SVector{2,Float64}(0.0, 0.0)
            end
        end
    end

    return C_γ
end

"""
    computeInterfaceCentroids(mesh, body)

Compute the interface centroids for a 3D mesh and body.

# Arguments
- `mesh::AbstractMesh`: The mesh on which to compute the interface centroids.
- `body::Function`: The body for which to compute the interface centroids.

# Returns
- `C_γ::Vector`: The interface centroids.
"""
function computeInterfaceCentroids(mesh::Mesh{3}, body)
    x_coords, y_coords, z_coords = mesh.nodes
    nx, ny, nz = length(x_coords)-1, length(y_coords)-1, length(z_coords)-1
    Φ = (r) -> body(r[1], r[2], r[3])

    # Single vector with nx * ny * nz entries
    C_γ = Vector{SVector{3,Float64}}(undef, (nx+1)* (ny+1)* (nz+1))

    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                idx = (nx+1) * (ny+1) * (k-1) + (nx+1) * (j-1) + i
                a = (x_coords[i],   y_coords[j],   z_coords[k])
                b = (x_coords[i+1], y_coords[j+1], z_coords[k+1])

                measure_val = ImplicitIntegration.integrate(_->1, Φ, a, b; surface=true).val
                if measure_val > 0
                    x_c = ImplicitIntegration.integrate(p->p[1], Φ, a, b; surface=true).val / measure_val
                    y_c = ImplicitIntegration.integrate(p->p[2], Φ, a, b; surface=true).val / measure_val
                    z_c = ImplicitIntegration.integrate(p->p[3], Φ, a, b; surface=true).val / measure_val
                    C_γ[idx] = SVector{3,Float64}(x_c, y_c, z_c)
                else
                    C_γ[idx] = SVector{3,Float64}(0.0, 0.0, 0.0)
                end
            end
        end
    end

    return C_γ
end