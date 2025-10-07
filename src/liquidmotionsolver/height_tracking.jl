# Shared utilities for tracking interface heights across liquid motion solvers.

"""
    spatial_shape_from_dims(dims)

Return a tuple describing the spatial resolution encoded in an operator size
`dims`. The last entry of `dims` corresponds to the temporal stencil and is
removed from the result.
"""
function spatial_shape_from_dims(dims)
    nd = length(dims)
    nd > 1 || error("Operator size must contain at least one spatial dimension.")
    return Tuple(dims[1:nd-1])
end

"""
    extract_height_fields(capacity, dims)

Return the diagonal capacity contributions `Vₙ` (time level `n`) and `Vₙ₊₁`
(time level `n + 1`) reshaped to the spatial grid described by `dims`.
"""
function extract_height_fields(capacity::Capacity, dims)
    cap_index = length(dims)
    height_block = capacity.A[cap_index]
    half = size(height_block, 1) ÷ 2
    spatial_shape = spatial_shape_from_dims(dims)

    @views Vₙ₊₁ = reshape(diag(height_block[1:half, 1:half]), spatial_shape...)
    @views Vₙ   = reshape(diag(height_block[half+1:end, half+1:end]), spatial_shape...)

    return Vₙ, Vₙ₊₁
end

"""
    column_height_profile(V)

Collapse the height field `V` along the streamwise direction to obtain the
column-wise interface height vector.
"""
function column_height_profile(V::AbstractArray)
    nd = ndims(V)
    if nd == 1
        return collect(V)
    else
        return vec(sum(V, dims=1))
    end
end

"""
    extract_height_profiles(capacity, dims)

Convenience wrapper that returns the column-wise height vectors at times `n`
and `n + 1`.
"""
function extract_height_profiles(capacity::Capacity, dims)
    Vₙ, Vₙ₊₁ = extract_height_fields(capacity, dims)
    return column_height_profile(Vₙ), column_height_profile(Vₙ₊₁)
end

"""
    interface_positions_from_heights(heights, mesh)

Map a height profile back to physical interface positions using the transverse
mesh spacing. In 1D the result is a scalar position; in 2D (and higher) a
vector of positions is returned.
"""
function interface_positions_from_heights(heights, mesh::AbstractMesh)
    n_dims = length(mesh.nodes)
    @assert n_dims in (1, 2) "Interface reconstruction only implemented for 1D or 2D meshes."

    if n_dims == 1
        x_nodes = mesh.nodes[1]
        length(x_nodes) > 1 || error("Mesh requires at least two nodes to compute spacing.")
        Δx = x_nodes[2] - x_nodes[1]
        x0 = x_nodes[1]
        return x0 + sum(heights) / Δx
    else
        x_nodes = mesh.nodes[1]
        y_nodes = mesh.nodes[2]
        length(y_nodes) > 1 || error("Mesh requires at least two nodes in the transverse direction.")
        Δy = y_nodes[2] - y_nodes[1]
        x0 = x_nodes[1]
        return x0 .+ heights ./ Δy
    end
end

"""
    ensure_periodic!(positions)

For vector-valued interface positions ensure periodicity by matching the last
entry to the first one. Returns the modified vector for convenience.
"""
function ensure_periodic!(positions)
    if !isempty(positions)
        positions[end] = positions[1]
    end
    return positions
end
