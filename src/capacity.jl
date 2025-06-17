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
function Capacity(body::Function, mesh::AbstractMesh; method::String = "VOFI", compute_centroids::Bool = true)

    if method == "VOFI"
        println("When using VOFI, the body must be a scalar function.")
        A, B, V, W, C_ω, C_γ, Γ, cell_types = VOFI(body, mesh; compute_centroids=compute_centroids)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    elseif method == "ImplicitIntegration"
        println("When using ImplicitIntegration, the body must be a vectorized function.")
        V, cell_types, C_ω, C_γ, Γ, W, A, B = IINT(mesh, body)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    end    
end

"""
    IINT(mesh::AbstractMesh, body::Function; compute_centroids::Bool = true)

Compute capacity quantities using Implicit Integration method from CartesianGeometry.

# Arguments
- `mesh::AbstractMesh`: The mesh on which to compute the capacity
- `body::Function`: The level set function defining the domain
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
Tuple of capacity components:
- `V`: Volume capacity represented by a sparse matrix
- `cell_types`: Cell types
- `C_ω`: Cell centroids
- `C_γ`: Interface centroids
- `Γ`: Interface norms
- `W`: Staggered volumes
- `A`: Face capacity matrices
- `B`: Center line capacity matrices
"""
function IINT(mesh::AbstractMesh, body::Function; compute_centroids::Bool = true)
    # Use the implicit integration method to compute capacities
    V, cell_types, C_ω, C_γ, Γ, W, A, B = CartesianGeometry.implicit_integration(mesh.nodes,body)
    
    # Convert V to sparse matrix if not already
    V_sparse = issparse(V) ? V : sparse(Diagonal(V))
    
    # Convert Γ to sparse matrix if not already
    Γ_sparse = issparse(Γ) ? Γ : sparse(Diagonal(Γ))
    
    # Process A, B, W tuples to ensure they're sparse matrices
    function ensure_sparse_tuple(tuple_matrices)
        return ntuple(i -> begin
            issparse(tuple_matrices[i]) ? tuple_matrices[i] : sparse(Diagonal(tuple_matrices[i]))
        end, length(tuple_matrices))
    end
    
    A_sparse = ensure_sparse_tuple(A)
    B_sparse = ensure_sparse_tuple(B)
    W_sparse = ensure_sparse_tuple(W)
    
    # Handle interface centroids based on compute_centroids flag
    if !compute_centroids
        N = length(mesh.nodes)
        if N == 1
            C_γ = Vector{SVector{1,Float64}}(undef, 0)
        elseif N == 2
            C_γ = Vector{SVector{2,Float64}}(undef, 0)
        elseif N == 3
            C_γ = Vector{SVector{3,Float64}}(undef, 0)
        end
    end
    
    return V_sparse, cell_types, C_ω, C_γ, Γ_sparse, W_sparse, A_sparse, B_sparse
end

"""
    Capacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute the capacity directly from a front tracker without using a level set function.

# Arguments
- `front::FrontTracker`: The front tracker object defining the fluid domain
- `mesh::AbstractMesh`: The mesh on which to compute the capacity 
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- `Capacity{N}`: The capacity of the domain defined by the front tracker
"""
function Capacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)
    # Convert front tracking capacities to Capacity format
    A, B, V, W, C_ω, C_γ, Γ, cell_types = FrontTrackingToCapacity(front, mesh; compute_centroids=compute_centroids)
    N = 2  # Only 2D is supported
    
    # Create a dummy level set function based on the front tracker's SDF
    dummy_body(x, y, z=0.0) = sdf(front, x, y)
    
    return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, dummy_body)
end

"""
    FrontTrackingToCapacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)

Convert front tracking capacities to the format expected by the Capacity struct.

# Arguments
- `front::FrontTracker`: The front tracker object defining the fluid domain
- `mesh::AbstractMesh`: The mesh on which to compute the capacity
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components in Capacity struct format
"""
function FrontTrackingToCapacity(front::FrontTracker, mesh::AbstractMesh; compute_centroids::Bool = true)
    if isa(mesh, Mesh{2}) || isa(mesh, SpaceTimeMesh{2})
        # Compute all capacities using front tracking
        ft_capacities = compute_capacities(mesh, front)
        
        # Extract dimensions
        x_nodes = mesh.nodes[1]
        y_nodes = mesh.nodes[2]
        nx = length(x_nodes) 
        ny = length(y_nodes) 
        nc = nx * ny  # Total number of cells
        
        # Convert volumes to the required sparse format
        volumes = ft_capacities[:volumes][1:nx, 1:ny]
        V = spdiagm(0 => reshape(volumes, :))
        
        # Get interface information
        interface_lengths_vec = zeros(nc)
        for ((i, j), length) in ft_capacities[:interface_lengths]
            if 1 <= i <= nx && 1 <= j <= ny
                idx = (j-1)*nx + i
                interface_lengths_vec[idx] = length
            end
        end
        Γ = spdiagm(0 => interface_lengths_vec)
        
        # Extract and convert Ax, Ay to sparse format
        Ax_dense = ft_capacities[:Ax][1:nx, 1:ny]
        Ay_dense = ft_capacities[:Ay][1:nx, 1:ny]
        
        # Create vectors for the sparse matrices
        Ax_vec = reshape(ft_capacities[:Ax][1:nx, 1:ny], :)
        Ay_vec = reshape(ft_capacities[:Ay][1:nx, 1:ny], :)
        
        A = (spdiagm(0 => Ax_vec), spdiagm(0 => Ay_vec))
        
        # Extract and convert Bx, By
        Bx_dense = ft_capacities[:Bx][1:nx, 1:ny]
        By_dense = ft_capacities[:By][1:nx, 1:ny]
        
        Bx_vec = reshape(Bx_dense, :)
        By_vec = reshape(By_dense, :)
        
        B = (spdiagm(0 => Bx_vec), spdiagm(0 => By_vec))
        
        # Extract and convert Wx, Wy
        # Note: Need to adjust indices to match VOFI convention
        Wx_dense = ft_capacities[:Wx][1:nx, 1:ny]  # Wx[i+1,j] in front tracking
        Wy_dense = ft_capacities[:Wy][1:nx, 1:ny]  # Wy[i,j+1] in front tracking
        
        Wx_vec = reshape(Wx_dense, :)
        Wy_vec = reshape(Wy_dense, :)
        
        W = (spdiagm(0 => Wx_vec), spdiagm(0 => Wy_vec))
        
        # Create cell centroids in required format
        centroids_x = ft_capacities[:centroids_x][1:nx, 1:ny]
        centroids_y = ft_capacities[:centroids_y][1:nx, 1:ny]
        
        C_ω = [SVector{2, Float64}(centroids_x[i, j], centroids_y[i, j]) 
               for j in 1:ny for i in 1:nx]
        
        # Get cell fractions (cell types)
        cell_types = ft_capacities[:cell_types][1:nx, 1:ny]
        cell_types = reshape(cell_types, :)
        
        # Create interface centroids if requested
        if compute_centroids
            C_γ = Vector{SVector{2, Float64}}(undef, nc)
            
            # Initialize with zeros
            for i in 1:nc
                C_γ[i] = SVector{2, Float64}(0.0, 0.0)
            end
            
            # Fill in known interface points
            for ((i, j), point) in ft_capacities[:interface_points]
                if 1 <= i <= nx && 1 <= j <= ny
                    idx = (j-1)*nx + i
                    C_γ[idx] = SVector{2, Float64}(point[1], point[2])
                end
            end
        else
            C_γ = Vector{SVector{2, Float64}}(undef, 0)
        end
        
        return A, B, V, W, C_ω, C_γ, Γ, cell_types
    else
        error("Front Tracking capacity computation is only supported for 2D meshes")
    end
end

"""
    FrontTracker1DToCapacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)

Convert 1D front tracking capacities to the format expected by the Capacity struct.
"""
function FrontTracker1DToCapacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)
    if isa(mesh, Mesh{1})
        # Compute all capacities using front tracking
        ft_capacities = compute_capacities_1d(mesh, front)
        
        # Extract dimensions
        x_nodes = mesh.nodes[1]
        nx = length(x_nodes)
        
        # Convert volumes to the required sparse format
        volumes = ft_capacities[:volumes][1:nx]
        V = spdiagm(0 => volumes)
        
        # Create interface information
        interface_lengths = zeros(nx)
        for (i, pos) in ft_capacities[:interface_positions]
            interface_lengths[i] = 1.0  # In 1D, interface "length" is 1
        end
        Γ = spdiagm(0 => interface_lengths)
        
        # Extract face capacities
        Ax_vec = ft_capacities[:Ax][1:nx]
        A = (spdiagm(0 => Ax_vec),)
        
        # Extract center line capacities
        Bx_vec = ft_capacities[:Bx][1:nx]
        B = (spdiagm(0 => Bx_vec),)
        
        # Extract staggered volumes
        Wx_vec = ft_capacities[:Wx][1:nx]
        W = (spdiagm(0 => Wx_vec),)
        
        # Create cell centroids
        centroids_x = ft_capacities[:centroids_x][1:nx]
        C_ω = [SVector{1, Float64}(centroids_x[i]) for i in 1:nx]
        
        # Get cell types
        cell_types = ft_capacities[:cell_types][1:nx]
        
        # Create interface centroids if requested
        if compute_centroids
            C_γ = Vector{SVector{1, Float64}}(undef, nx)
            
            # Initialize with zeros
            for i in 1:nx
                C_γ[i] = SVector{1, Float64}(0.0)
            end
            
            # Fill in known interface points
            for (i, pos) in ft_capacities[:interface_positions]
                if 1 <= i <= nx
                    C_γ[i] = SVector{1, Float64}(pos)
                end
            end
        else
            C_γ = Vector{SVector{1, Float64}}(undef, 0)
        end
        
        return (A, B, V, W, C_ω, C_γ, Γ, cell_types)
    else
        error("1D Front Tracking capacity computation is only supported for 1D meshes")
    end
end

"""
    Capacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute the capacity directly from a 1D front tracker.
"""
function Capacity(front::FrontTracker1D, mesh::AbstractMesh; compute_centroids::Bool = true)
    # Convert front tracking capacities to Capacity format
    A, B, V, W, C_ω, C_γ, Γ, cell_types = FrontTracker1DToCapacity(front, mesh; compute_centroids=compute_centroids)
    
    # Create a dummy level set function based on the front tracker's SDF
    dummy_body(x, y=0.0, z=0.0) = sdf(front, x)
    
    return Capacity{1}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, dummy_body)
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
function VOFI(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true)
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
        if compute_centroids
            C_γ = computeInterfaceCentroids(mesh, body)
        else
            C_γ = Vector{SVector{1,Float64}}(undef, 0)
        end
    elseif N == 2
        V = spdiagm(0 => Vs)
        A = (spdiagm(0 => As[1]), spdiagm(0 => As[2]))
        B = (spdiagm(0 => Bs[1]), spdiagm(0 => Bs[2]))
        W = (spdiagm(0 => Ws[1]), spdiagm(0 => Ws[2]))
        Γ = spdiagm(0 => interface_length)
        if compute_centroids
            C_γ = computeInterfaceCentroids(mesh, body)
        else
            C_γ = Vector{SVector{2,Float64}}(undef, 0)
        end
    elseif N == 3
        V = spdiagm(0 => Vs)
        A = (spdiagm(0 => As[1]), spdiagm(0 => As[2]), spdiagm(0 => As[3]))
        B = (spdiagm(0 => Bs[1]), spdiagm(0 => Bs[2]), spdiagm(0 => Bs[3]))
        W = (spdiagm(0 => Ws[1]), spdiagm(0 => Ws[2]), spdiagm(0 => Ws[3]))
        Γ = spdiagm(0 => interface_length)
        if compute_centroids
            C_γ = computeInterfaceCentroids(mesh, body)
        else
            C_γ = Vector{SVector{3,Float64}}(undef, 0)
        end
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
function computeInterfaceCentroids(mesh::Union{Mesh{1}, SpaceTimeMesh{1}}, body)
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
function computeInterfaceCentroids(mesh::Union{Mesh{2}, SpaceTimeMesh{2}}, body)
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
function computeInterfaceCentroids(mesh::Union{Mesh{3}, SpaceTimeMesh{3}}, body)
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