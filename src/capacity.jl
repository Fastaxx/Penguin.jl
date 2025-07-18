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
        println("Computing capacity using geometric moments integration.")
        A, B, V, W, C_ω, C_γ, Γ, cell_types = GeometricMoments(body, mesh; compute_centroids=compute_centroids)
        N = length(A)
        return Capacity{N}(A, B, V, W, C_ω, C_γ, Γ, cell_types, mesh, body)
    end    
end

# VOFI implementation

"""
    VOFI(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute capacity quantities based on VOFI for a given body and mesh.

# Arguments
- `body::Function`: The level set function defining the domain
- `mesh::AbstractMesh`: The mesh on which to compute the VOFI quantities
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components (A, B, V, W, C_ω, C_γ, Γ, cell_types)
"""
function VOFI(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true)
    N = length(mesh.nodes)
    nc = nC(mesh)
    
    # Only initialize variables we actually need
    local V, A, B, W, C_ω, C_γ, Γ, cell_types
    
    # Get volume capacity, barycenters, interface length and cell types in a single call
    # This avoids redundant computations and memory allocations
    Vs, bary, interface_length, cell_types = CartesianGeometry.integrate(
        Tuple{0}, body, mesh.nodes, Float64, zero
    )
    
    # Store cell centroids
    C_ω = bary
    
    # Create shared sparse matrices (identical for all dimensions)
    V = spdiagm(0 => Vs)
    Γ = spdiagm(0 => interface_length)
    
    # Do dimension-specific calculations in one shot
    # Compute face capacities, center line capacities, and staggered volumes
    As = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero)
    Ws = CartesianGeometry.integrate(Tuple{0}, body, mesh.nodes, Float64, zero, bary)
    Bs = CartesianGeometry.integrate(Tuple{1}, body, mesh.nodes, Float64, zero, bary)
    
    # Create appropriate-sized tuples based on dimension
    # This avoids the dimension-specific if/elseif blocks
    A = ntuple(i -> i <= length(As) ? spdiagm(0 => As[i]) : spzeros(0), N)
    B = ntuple(i -> i <= length(Bs) ? spdiagm(0 => Bs[i]) : spzeros(0), N)
    W = ntuple(i -> i <= length(Ws) ? spdiagm(0 => Ws[i]) : spzeros(0), N)
    
    # Compute interface centroids if requested
    # Use a cache-friendly approach by checking for previously computed values
    if compute_centroids
        C_γ = computeInterfaceCentroids(mesh, body)
    else
        # Create empty vector of appropriate type based on dimension
        C_γ = Vector{SVector{N,Float64}}(undef, 0)
    end
    
    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

"""
    computeInterfaceCentroids(mesh::Union{Mesh{N}, SpaceTimeMesh{N}}, body) where N

Compute the interface centroids for an N-dimensional mesh and body.

# Arguments
- `mesh::Union{Mesh{N}, SpaceTimeMesh{N}}`: The mesh on which to compute the interface centroids
- `body::Function`: The body level set function

# Returns
- `C_γ::Vector{SVector{N,Float64}}`: Vector of interface centroids
"""
function computeInterfaceCentroids(mesh::Union{Mesh{N}, SpaceTimeMesh{N}}, body) where N
    # Extract node coordinates
    coords = mesh.nodes
    
    # Calculate dimensions of the grid
    dims = mesh.dims
    
    # Create dimension-appropriate level set function
    Φ = if N == 1
        (r) -> body(r[1])
    elseif N == 2
        (r) -> body(r[1], r[2])
    elseif N == 3
        (r) -> body(r[1], r[2], r[3])
    elseif N == 4
        (r) -> body(r[1], r[2], r[3], r[4])
    else
        error("Unsupported dimension: $N")
    end
    
    # Calculate total cells and preallocate result vector with appropriate size
    total_cells = prod(ntuple(i -> dims[i]+1, N))
    C_γ = Vector{SVector{N,Float64}}(undef, total_cells)
    
    # Generate all cell indices
    indices = CartesianIndices(ntuple(i -> 1:dims[i], N))
    
    # Process each cell
    for idx in indices
        # Calculate linear index based on dimension
        if N == 1
            linear_idx = idx[1]
        elseif N == 2
            i, j = idx[1], idx[2]
            linear_idx = (dims[1]+1) * (j-1) + i
        elseif N == 3
            i, j, k = idx[1], idx[2], idx[3]
            linear_idx = (dims[1]+1) * (dims[2]+1) * (k-1) + (dims[1]+1) * (j-1) + i
        end
        
        # Get cell bounds
        a = ntuple(i -> coords[i][idx[i]], N)
        b = ntuple(i -> coords[i][idx[i]+1], N)
        
        # Compute measure
        measure_val = ImplicitIntegration.integrate(_->1, Φ, a, b; surface=true).val
        
        if measure_val > 0
            # Compute centroid coordinates
            centroid_coords = ntuple(N) do d
                ImplicitIntegration.integrate(p->p[d], Φ, a, b; surface=true).val / measure_val
            end
            
            C_γ[linear_idx] = SVector{N,Float64}(centroid_coords)
        else
            C_γ[linear_idx] = SVector{N,Float64}(ntuple(_ -> 0.0, N))
        end
    end
    
    return C_γ
end

# Implicit Integration Direct implementation
"""
    GeometricMoments(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true)

Compute geometric moments (volumes, centroids, face capacities) using direct integration.
This method uses ImplicitIntegration to compute various capacity quantities for a level set function.

# Arguments
- `body::Function`: The level set function defining the domain
- `mesh::AbstractMesh`: The mesh on which to compute geometric quantities
- `compute_centroids::Bool`: Whether to compute interface centroids

# Returns
- Tuple of capacity components (A, B, V, W, C_ω, C_γ, Γ, cell_types)
"""
function GeometricMoments(body::Function, mesh::AbstractMesh; compute_centroids::Bool = true, tol=1e-6)
    # Extract mesh dimensions
    N = length(mesh.nodes)
    dims = mesh.dims
    coords = mesh.nodes
    
    # Create dimension-appropriate level set function wrapper once
    Φ = if N == 1
        (r) -> body(r[1])
    elseif N == 2
        (r) -> body(r[1], r[2])
    elseif N == 3
        (r) -> body(r[1], r[2], r[3])
    elseif N == 4
        (r) -> body(r[1], r[2], r[3], r[4])
    else
        error("Unsupported dimension: $N")
    end
    
    # Get cell size for reference
    cell_sizes = ntuple(i -> (coords[i][2] - coords[i][1]), N)
    full_volume = prod(cell_sizes)
    
    # Pre-allocate result arrays with extended dimensions
    dims_extended = ntuple(i -> dims[i] + 1, N)
    total_cells_extended = prod(dims_extended)
    
    # Initialize all matrices with extended dimensions
    V_dense = zeros(dims_extended)
    cell_types_array = zeros(Int, dims_extended)
    C_ω = Vector{SVector{N,Float64}}(undef, total_cells_extended)
    for i in 1:total_cells_extended
        C_ω[i] = SVector{N,Float64}(zeros(N))
    end
    
    Γ_dense = zeros(dims_extended)
    
    # Initialize A, B, W with consistent extended dimensions
    A_dense = ntuple(_ -> zeros(dims_extended), N)
    B_dense = ntuple(_ -> zeros(dims_extended), N)
    W_dense = ntuple(_ -> zeros(dims_extended), N)
    
    # Pre-allocate arrays for repeated calculations
    centroid_coords = zeros(N)
    
    
    # Pre-create all fixed-coordinate level set functions
    # Only needed for A calculation
    face_funcs = Dict{Tuple{Int, Float64}, Function}()
    
    # First pass: Compute volumes, centroids, cell types, interface measures
    for I in CartesianIndices(dims)
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Get cell bounds (only compute once per cell)
        a = ntuple(d -> coords[d][I[d]], N)
        b = ntuple(d -> coords[d][I[d] + 1], N)
        
        # Compute volume fraction
        vol = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
        V_dense[I] = vol
        
        # Classify cell
        if vol < 1e-14
            cell_types_array[I] = 0  # Solid
            # Use geometric center for solid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        elseif abs(vol - full_volume) < 1e-14
            cell_types_array[I] = 1  # Fluid
            # Use geometric center for fluid cells
            for d in 1:N
                centroid_coords[d] = 0.5 * (a[d] + b[d])
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
        else
            cell_types_array[I] = -1  # Cut
            
            # Only compute detailed centroid for cut cells
            for d in 1:N
                coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; tol=tol).val
                centroid_coords[d] = isnan(coord_integral/vol) ? 0.5*(a[d] + b[d]) : coord_integral/vol
            end
            C_ω[linear_idx] = SVector{N,Float64}(centroid_coords)
            
            # Only compute interface measure for cut cells
            Γ_dense[I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; surface=true, tol=tol).val
        end
        
        # Cache face functions for each face coordinate (for A calculation)
        for face_dim in 1:N
            face_coord = coords[face_dim][I[face_dim]]
            key = (face_dim, face_coord)
            if !haskey(face_funcs, key)
                face_funcs[key] = create_fixed_coordinate_function(body, face_dim, face_coord, N)
            end
        end
    end
    
    # Compute interface centroids if requested (only for cut cells)
    C_γ = if compute_centroids
        interface_centroids = Vector{SVector{N,Float64}}(undef, total_cells_extended)
        
        # Initialize all to zero
        for i in 1:total_cells_extended
            interface_centroids[i] = SVector{N,Float64}(zeros(N))
        end
        
        # Only process cut cells with non-zero interface measure
        for I in CartesianIndices(dims)
            if cell_types_array[I] == -1 && Γ_dense[I] > 1e-14
                linear_idx = LinearIndices(dims_extended)[I]
                
                a = ntuple(d -> coords[d][I[d]], N)
                b = ntuple(d -> coords[d][I[d] + 1], N)
                
                interface_measure = Γ_dense[I]
                
                # Compute interface centroid for cut cells
                for d in 1:N
                    coord_integral = ImplicitIntegration.integrate(x -> x[d], Φ, a, b; surface=true, tol=tol).val
                    centroid_coords[d] = isnan(coord_integral/interface_measure) ? 
                                        0.5*(a[d] + b[d]) : coord_integral/interface_measure
                end
                interface_centroids[linear_idx] = SVector{N,Float64}(centroid_coords)
            end
        end
        
        interface_centroids
    else
        Vector{SVector{N,Float64}}(undef, 0)
    end
    
    # Second pass: Compute A, B
    for I in CartesianIndices(dims)
        # Only process cells that exist in the mesh (efficiency)
        linear_idx = LinearIndices(dims_extended)[I]
        
        # Compute A (face capacities)
        for face_dim in 1:N
            face_coord = coords[face_dim][I[face_dim]]
            key = (face_dim, face_coord)
            Φ_face = face_funcs[key]
            
            # Get integration bounds for remaining dimensions
            a_reduced = [coords[d][I[d]] for d in 1:N if d != face_dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != face_dim]
            
            # Integrate to get face capacity
            if N == 1
                A_dense[face_dim][I] = Φ_face() ≤ 0.0 ? 1.0 : 0.0
            else
                A_dense[face_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_face, 
                                                   tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
        
        # Compute B (center line capacities)
        centroid = C_ω[linear_idx]
        
        for dim in 1:N
            # Create centroid-fixed function
            Φ_center = create_fixed_coordinate_function(body, dim, centroid[dim], N)
            
            # Get integration bounds
            a_reduced = [coords[d][I[d]] for d in 1:N if d != dim]
            b_reduced = [coords[d][I[d] + 1] for d in 1:N if d != dim]
            
            # Integrate to get center line capacity
            if N == 1
                B_dense[dim][I] = Φ_center() ≤ 0.0 ? 1.0 : 0.0
            else
                B_dense[dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ_center, 
                                                 tuple(a_reduced...), tuple(b_reduced...); tol=tol).val
            end
        end
    end
    
    # Third pass: Compute W (staggered volumes)
    # This is separate because it uses different indices
    for stagger_dim in 1:N
        for I in CartesianIndices(dims_extended)
            # Skip boundary cells to avoid out of bounds
            if all(1 <= I[d] <= (d == stagger_dim ? dims[d]+1 : dims[d]) for d in 1:N)
                # Find neighboring cells
                prev_idx = max(I[stagger_dim] - 1, 1)
                next_idx = min(I[stagger_dim], dims[stagger_dim])
                
                # Get cell indices
                prev_I = CartesianIndex(ntuple(d -> d == stagger_dim ? prev_idx : I[d], N))
                next_I = CartesianIndex(ntuple(d -> d == stagger_dim ? next_idx : I[d], N))
                
                # Get centroids
                prev_centroid = C_ω[LinearIndices(dims_extended)[prev_I]]
                next_centroid = C_ω[LinearIndices(dims_extended)[next_I]]
                
                # Build integration domain
                a = ntuple(d -> d == stagger_dim ? prev_centroid[d] : coords[d][I[d]], N)
                b = ntuple(d -> d == stagger_dim ? next_centroid[d] : coords[d][I[d] + 1], N)
                
                # Compute staggered volume - only if cells differ in type
                prev_type = cell_types_array[prev_I]
                next_type = cell_types_array[next_I] 
                
                if prev_type != next_type || prev_type == -1 || next_type == -1
                    W_dense[stagger_dim][I] = ImplicitIntegration.integrate(x -> 1.0, Φ, a, b; tol=tol).val
                else
                    # For consistent cell types (both fluid or both solid), we know the result
                    W_dense[stagger_dim][I] = (prev_type == 1) ? prod(d -> d == stagger_dim ? 
                                              (next_centroid[d] - prev_centroid[d]) : 
                                              (coords[d][I[d]+1] - coords[d][I[d]]), 1:N) : 0.0
                end
            end
        end
    end
    
    # Convert arrays to format required by Capacity struct (all at once)
    V = spdiagm(0 => reshape(V_dense, :))
    Γ = spdiagm(0 => reshape(Γ_dense, :))
    A = ntuple(i -> spdiagm(0 => reshape(A_dense[i], :)), N)
    B = ntuple(i -> spdiagm(0 => reshape(B_dense[i], :)), N)
    W = ntuple(i -> spdiagm(0 => reshape(W_dense[i], :)), N)
    cell_types = reshape(cell_types_array, :)
    
    return A, B, V, W, C_ω, C_γ, Γ, cell_types
end

"""
    create_fixed_coordinate_function(body, fixed_dim, fixed_value, N)

Create a level set function with one coordinate fixed at a specific value.
This generalizes both face functions and centroid-fixed functions.

# Arguments
- `body`: The original level set function
- `fixed_dim`: The dimension to fix (1 for x, 2 for y, etc.)
- `fixed_value`: The value to fix the coordinate at
- `N`: Total number of dimensions

# Returns
- A new function with the specified dimension fixed
"""
function create_fixed_coordinate_function(body, fixed_dim, fixed_value, N)
    # Special case for 1D - no input needed
    if N == 1
        return () -> body(fixed_value)
    else
        # For higher dimensions, create a function that inserts the fixed value
        # at the correct position and calls the body function
        return y -> body(ntuple(i -> i == fixed_dim ? fixed_value : y[i - (i > fixed_dim ? 1 : 0)], N)...)
    end
end

# Capacity from Front Tracker

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