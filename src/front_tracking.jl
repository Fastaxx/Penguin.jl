"""
A Julia implementation of front tracking for fluid-solid interfaces.
Replaces the Python implementation to avoid segmentation faults.
"""
mutable struct FrontTracker
    markers::Vector{Tuple{Float64, Float64}}
    is_closed::Bool
    interface::Union{Nothing, LibGEOS.LineString}
    interface_poly::Union{Nothing, LibGEOS.Polygon}
    
    # Constructor for empty front
    function FrontTracker()
        return new([], true, nothing, nothing)
    end
    
    # Constructor with markers
    function FrontTracker(markers::Vector{Tuple{Float64, Float64}}, is_closed::Bool=true)
        ft = new(markers, is_closed, nothing, nothing)
        update_geometry!(ft)
        return ft
    end
end

"""
Updates the interface geometry based on marker positions.
"""
function update_geometry!(ft::FrontTracker)
    if length(ft.markers) < 3
        # Not enough points to create a valid geometry
        ft.interface = nothing
        ft.interface_poly = nothing
        return
    end
    
    # Ensure interface is closed for polygon creation
    markers_to_use = ft.markers
    if ft.is_closed && length(markers_to_use) > 0 && markers_to_use[1] != markers_to_use[end]
        # Add closing point
        push!(markers_to_use, markers_to_use[1])
    end
    
    # For LibGEOS, we need to create the geometry directly from coordinates
    coords = [collect(point) for point in markers_to_use]
    
    # Create LineString
    ft.interface = LibGEOS.LineString(coords)
    
    # Create Polygon (if closed and valid)
    if ft.is_closed && length(ft.markers) >= 3
        ft.interface_poly = LibGEOS.Polygon([coords])
    else
        ft.interface_poly = nothing
    end
end

"""
Gets all markers of the interface.
"""
function get_markers(ft::FrontTracker)
    return ft.markers
end

"""
Adds a marker to the interface.
"""
function add_marker!(ft::FrontTracker, x::Float64, y::Float64)
    push!(ft.markers, (x, y))
    if length(ft.markers) >= 3  # Only update geometry when we have enough points
        update_geometry!(ft)
    end
    return ft
end

"""
Sets all markers of the interface.
"""
function set_markers!(ft::FrontTracker, markers::AbstractVector, is_closed=nothing)
    # Convert to proper type if needed
    typed_markers = convert(Vector{Tuple{Float64, Float64}}, markers)
    ft.markers = typed_markers
    if is_closed !== nothing
        ft.is_closed = is_closed
    end
    update_geometry!(ft)
    return ft
end

"""
Creates a circular interface.
"""
function create_circle!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius::Float64, n_markers::Int=100)
    # Create properly typed vector
    markers = Vector{Tuple{Float64, Float64}}(undef, n_markers)
    
    for i in 0:n_markers-1
        angle = 2.0 * π * (i / n_markers)
        x = center_x + radius * cos(angle)
        y = center_y + radius * sin(angle)
        markers[i+1] = (x, y)
    end
    
    set_markers!(ft, markers, true)
    return ft
end

"""
    create_rectangle!(ft::FrontTracker, min_x::Float64, min_y::Float64, max_x::Float64, max_y::Float64)

Creates a rectangular interface.
"""
function create_rectangle!(ft::FrontTracker, min_x::Float64, min_y::Float64, max_x::Float64, max_y::Float64)
    markers = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    set_markers!(ft, markers, true)
    return ft
end

"""
    create_ellipse!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius_x::Float64, radius_y::Float64, n_markers::Int=100)

Creates an elliptical interface.
"""
function create_ellipse!(ft::FrontTracker, center_x::Float64, center_y::Float64, radius_x::Float64, radius_y::Float64, n_markers::Int=100)
    theta = range(0, 2π, length=n_markers+1)[1:end-1] # Avoid duplicating the first point
    markers = [(center_x + radius_x*cos(t), center_y + radius_y*sin(t)) for t in theta]
    set_markers!(ft, markers, true)
    return ft
end

"""
    get_fluid_polygon(ft::FrontTracker)

Returns a polygon representing the fluid domain bounded by the interface.
"""
function get_fluid_polygon(ft::FrontTracker)
    if length(ft.markers) < 3
        return nothing
    end
    
    if ft.is_closed
        # For a closed interface, directly return the polygon
        if !LibGEOS.isValid(ft.interface_poly)
            # Try to fix invalid polygon
            return LibGEOS.buffer(ft.interface_poly, 0.0)
        end
        return ft.interface_poly
    else
        # For an open interface, we need to define domain closure
        # Using convex hull as an approximation
        return LibGEOS.convexhull(ft.interface_poly)
    end
end

"""
    is_point_inside(ft::FrontTracker, x::Float64, y::Float64)

Checks if a point is inside the interface.
"""
function is_point_inside(ft::FrontTracker, x::Float64, y::Float64)
    if isnothing(ft.interface_poly)
        return false
    end
    
    point = LibGEOS.Point(x, y)
    return LibGEOS.contains(ft.interface_poly, point)
end

"""
    get_intersection(ft::FrontTracker, other_geometry)

Calculates the intersection with another geometry.
"""
function get_intersection(ft::FrontTracker, other_geometry)
    if isnothing(ft.interface_poly)
        return nothing
    end
    
    return LibGEOS.intersection(other_geometry, ft.interface_poly)
end


"""
    sdf(ft::FrontTracker, x::Float64, y::Float64)

Calculates the signed distance function for a given point.
"""
function sdf(ft::FrontTracker, x::Float64, y::Float64)
    if isnothing(ft.interface)
        return Inf
    end
    
    # Create a point using LibGEOS
    point = LibGEOS.Point(x, y)
    
    # Calculate the distance to the interface
    distance = LibGEOS.distance(point, ft.interface)
    
    # Determine the sign (negative inside, positive outside)
    is_inside_val = is_point_inside(ft, x, y)
    
    return is_inside_val ? -distance : distance
end

"""
    compute_marker_normals(ft::FrontTracker, markers=nothing)

Calculates the normal vectors for each marker of the interface.
"""
function compute_marker_normals(ft::FrontTracker, markers=nothing)
    if isnothing(markers)
        markers = ft.markers
    end
    
    if length(markers) < 3
        return [(0.0, 1.0) for _ in markers]
    end
    
    normals = []
    n_markers = length(markers)
    is_closed = ft.is_closed
    
    # For maintaining orientation consistency
    prev_normal = nothing
    
    for i in 1:n_markers
        # Handle indices for previous and next points with boundary conditions
        prev_idx = is_closed ? mod1(i-1, n_markers) : max(1, i-1)
        next_idx = is_closed ? mod1(i+1, n_markers) : min(n_markers, i+1)
        
        # MODIFICATION IMPORTANTE: Utiliser la méthode des tangentes pour tous les points du premier marqueur
        # Pour les contours fermés, traiter le premier marqueur comme un point spécial
        if is_closed && (i == 1 || i == n_markers && markers[1] == markers[end])
            # Utiliser les points adjacents pour calculer les tangentes
            prev_idx = n_markers > 2 ? n_markers - 1 : 1
            next_idx = 2
            
            t1_x = markers[i][1] - markers[prev_idx][1]
            t1_y = markers[i][2] - markers[prev_idx][2]
            
            t2_x = markers[next_idx][1] - markers[i][1]
            t2_y = markers[next_idx][2] - markers[i][2]
            
            # Normaliser les vecteurs tangents
            t1_len = sqrt(t1_x^2 + t1_y^2)
            t2_len = sqrt(t2_x^2 + t2_y^2)
            
            if t1_len > 0 && t2_len > 0
                t1_x, t1_y = t1_x/t1_len, t1_y/t1_len
                t2_x, t2_y = t2_x/t2_len, t2_y/t2_len
                
                # Moyenner les tangentes
                tx = (t1_x + t2_x)
                ty = (t1_y + t2_y)
                
                # Normaliser
                t_len = sqrt(tx^2 + ty^2)
                if t_len > 0
                    tx, ty = tx/t_len, ty/t_len
                    
                    # La normale est perpendiculaire à la tangente
                    n_x, n_y = -ty, tx
                    
                    # Vérifier l'orientation
                    test_x = markers[i][1] + 1e-3 * n_x
                    test_y = markers[i][2] + 1e-3 * n_y
                    if is_point_inside(ft, test_x, test_y)
                        n_x, n_y = -n_x, -n_y
                    end
                    
                    push!(normals, (n_x, n_y))
                    prev_normal = [n_x, n_y]
                    
                    # Si c'est le dernier point qui est dupliqué, assurez-vous de sauter l'itération
                    if i == n_markers && markers[1] == markers[end]
                        continue
                    end
                else
                    push!(normals, (0.0, 1.0))
                    prev_normal = [0.0, 1.0]
                end
                
                continue
            end
        end
        # Special handling for endpoints of open curves
        if !is_closed && (i == 1 || i == n_markers)
            # For first point in open curve, use forward difference
            if i == 1
                # Use the vector from first to second marker as tangent
                t_x = markers[2][1] - markers[1][1]
                t_y = markers[2][2] - markers[1][2]
                
                # Normalize tangent
                t_len = sqrt(t_x^2 + t_y^2)
                if t_len > 0
                    t_x /= t_len
                    t_y /= t_len
                    
                    # Rotate 90° to get normal (outward pointing)
                    n_x, n_y = -t_y, t_x
                else
                    n_x, n_y = 0.0, 1.0
                end
                
                push!(normals, (n_x, n_y))
                prev_normal = [n_x, n_y]
                continue
            end
            
            # For last point in open curve, use backward difference
            if i == n_markers
                # Use the vector from second-to-last to last marker as tangent
                t_x = markers[n_markers][1] - markers[n_markers-1][1]
                t_y = markers[n_markers][2] - markers[n_markers-1][2]
                
                # Normalize tangent
                t_len = sqrt(t_x^2 + t_y^2)
                if t_len > 0
                    t_x /= t_len
                    t_y /= t_len
                    
                    # Rotate 90° to get normal (outward pointing)
                    n_x, n_y = -t_y, t_x
                    
                    # Ensure consistency with previous normal
                    if !isnothing(prev_normal)
                        dot_product = n_x * prev_normal[1] + n_y * prev_normal[2]
                        if dot_product < 0
                            n_x, n_y = -n_x, -n_y
                        end
                    end
                else
                    n_x, n_y = prev_normal[1], prev_normal[2]
                end
                
                push!(normals, (n_x, n_y))
                continue
            end
        end
        
        # Regular points use the osculating circle method
        # Points P1, P2, P3 for calculating the osculating circle
        p1 = collect(markers[prev_idx])
        p2 = collect(markers[i])
        p3 = collect(markers[next_idx])
        
        # Rest of the function remains the same...
        
        try
            # Check if points are distinct
            if (norm(p1 - p2) < 1e-10 || 
                norm(p2 - p3) < 1e-10 || 
                norm(p3 - p1) < 1e-10)
                error("Points too close")
            end
            
            # Equations to find the center of the circle through three points
            # Linear system: Ax = b where x = [center_x, center_y, -r²]
            A = [2*p1[1] 2*p1[2] 1;
                 2*p2[1] 2*p2[2] 1;
                 2*p3[1] 2*p3[2] 1]
            
            b = [p1[1]^2 + p1[2]^2,
                 p2[1]^2 + p2[2]^2,
                 p3[1]^2 + p3[2]^2]
            
            # Solve for the circle center
            x = A \ b
            center = x[1:2]
            
            # Calculate the normal vector (from point to center)
            normal_vector = center - p2
            
            # Normalize the vector
            norm_val = norm(normal_vector)
            if norm_val < 1e-10
                error("Norm too small")
            end
            normal_vector = normal_vector / norm_val
            
            # Check orientation
            if is_closed
                # For closed interface, check if normal points outward
                test_point = p2 + 1e-3 * normal_vector
                if is_point_inside(ft, test_point[1], test_point[2])
                    normal_vector = -normal_vector  # Invert normal
                end
            else
                # For open interface, ensure consistency with previous normal
                if !isnothing(prev_normal) && i > 1
                    # Calculate dot product to check if normals point in similar directions
                    dot_product = normal_vector[1] * prev_normal[1] + normal_vector[2] * prev_normal[2]
                    if dot_product < 0
                        normal_vector = -normal_vector
                    end
                elseif i == 1
                    # For first point, prefer upward-pointing normal
                    if normal_vector[2] < 0
                        normal_vector = -normal_vector
                    end
                end
            end
            
            prev_normal = normal_vector
            push!(normals, (normal_vector[1], normal_vector[2]))
            
        catch e
            # Fallback: use tangent method if osculating circle fails
            if i > 1 || is_closed
                t1_x = markers[i][1] - markers[prev_idx][1]
                t1_y = markers[i][2] - markers[prev_idx][2]
            else
                t1_x = markers[next_idx][1] - markers[i][1]
                t1_y = markers[next_idx][2] - markers[i][2]
            end
            
            if i < n_markers || is_closed
                t2_x = markers[next_idx][1] - markers[i][1]
                t2_y = markers[next_idx][2] - markers[i][2]
            else
                t2_x = markers[i][1] - markers[prev_idx][1]
                t2_y = markers[i][2] - markers[prev_idx][2]
            end
            
            # Normalize tangent vectors
            t1_len = sqrt(t1_x^2 + t1_y^2)
            t2_len = sqrt(t2_x^2 + t2_y^2)
            
            if t1_len > 0 && t2_len > 0
                t1_x, t1_y = t1_x/t1_len, t1_y/t1_len
                t2_x, t2_y = t2_x/t2_len, t2_y/t2_len
                
                # Average tangents
                tx = (t1_x + t2_x)
                ty = (t1_y + t2_y)
                
                # Normalize
                t_len = sqrt(tx^2 + ty^2)
                if t_len > 0
                    tx, ty = tx/t_len, ty/t_len
                    
                    # Normal is perpendicular to tangent
                    nx, ny = -ty, tx
                    
                    # Check orientation
                    if is_closed
                        test_x = markers[i][1] + 1e-3 * nx
                        test_y = markers[i][2] + 1e-3 * ny
                        if is_point_inside(ft, test_x, test_y)
                            nx, ny = -nx, -ny
                        end
                    else
                        # Ensure consistency
                        normal_vector = [nx, ny]
                        if !isnothing(prev_normal) && i > 1
                            dot_product = normal_vector[1] * prev_normal[1] + normal_vector[2] * prev_normal[2]
                            if dot_product < 0
                                nx, ny = -nx, -ny
                            end
                        elseif i == 1
                            if ny < 0
                                nx, ny = -nx, -ny
                            end
                        end
                    end
                    
                    prev_normal = [nx, ny]
                    push!(normals, (nx, ny))
                else
                    # Degenerate case
                    if !isnothing(prev_normal)
                        push!(normals, (prev_normal[1], prev_normal[2]))
                    else
                        push!(normals, (0.0, 1.0))
                    end
                end
            else
                # Degenerate case
                if !isnothing(prev_normal)
                    push!(normals, (prev_normal[1], prev_normal[2]))
                else
                    push!(normals, (0.0, 1.0))
                end
            end
        end
    end
    
    return normals
end

"""
    compute_volume_jacobian(ft::FrontTracker, x_faces::Vector{Float64}, y_faces::Vector{Float64}, epsilon::Float64=1e-6)

Calculates the volume Jacobian matrix for a given mesh and interface.
This is a more reliable Julia implementation that avoids Python interop issues.
"""
function compute_volume_jacobian(ft::FrontTracker, x_faces::AbstractVector{<:Real}, y_faces::AbstractVector{<:Real}, epsilon::Float64=1e-8)
    # Convert ranges to vectors if needed
    x_faces_vec = collect(x_faces)
    y_faces_vec = collect(y_faces)
    
    # Get mesh dimensions
    nx = length(x_faces_vec) - 1
    ny = length(y_faces_vec) - 1
    
    # Get markers and compute their normals
    markers = get_markers(ft)
    normals = compute_marker_normals(ft, markers)
    
    # Calculate original cell volumes
    fluid_poly = get_fluid_polygon(ft)
    original_volumes = Dict{Tuple{Int, Int}, Float64}()
    
    # For each cell in the mesh
    for i in 1:nx
        for j in 1:ny
            # Create cell coordinates properly for LibGEOS
            cell_coords = [
                [x_faces_vec[i], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j]]  # Close the polygon
            ]
            
            # Create cell polygon
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calculate intersection with fluid polygon
            intersection = LibGEOS.intersection(cell_poly, fluid_poly)
            
            # Store the fluid volume for this cell
            if LibGEOS.isEmpty(intersection)
                original_volumes[(i, j)] = 0.0
            else
                original_volumes[(i, j)] = LibGEOS.area(intersection)
            end
        end
    end
    
    # Initialize dictionary for storing the Jacobian
    volume_jacobian = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    for key in keys(original_volumes)
        volume_jacobian[key] = []
    end
    
    # Track which markers have entries
    markers_with_entries = Set{Int}()
    
    # For closed interfaces, only process unique markers
    n_unique_markers = ft.is_closed ? length(markers) - 1 : length(markers)
    
    for marker_idx in 1:n_unique_markers
        # Original marker position
        original_marker = markers[marker_idx]
        
        # Calculate perturbed positions (both positive and negative)
        normal = normals[marker_idx]
        
        # Positive perturbation
        pos_perturbed_marker = (
            original_marker[1] + epsilon * normal[1],
            original_marker[2] + epsilon * normal[2]
        )
        
        # Negative perturbation
        neg_perturbed_marker = (
            original_marker[1] - epsilon * normal[1],
            original_marker[2] - epsilon * normal[2]
        )
        
        # Create copies of markers with positive perturbation
        pos_perturbed_markers = copy(markers)
        pos_perturbed_markers[marker_idx] = pos_perturbed_marker
        
        # Create copies of markers with negative perturbation
        neg_perturbed_markers = copy(markers)
        neg_perturbed_markers[marker_idx] = neg_perturbed_marker
        
        # Update last marker if interface is closed and first marker is perturbed
        if ft.is_closed && marker_idx == 1
            pos_perturbed_markers[end] = pos_perturbed_marker
            neg_perturbed_markers[end] = neg_perturbed_marker
        end
        
        # Create new front trackers with perturbed markers
        pos_perturbed_tracker = FrontTracker(pos_perturbed_markers, ft.is_closed)
        neg_perturbed_tracker = FrontTracker(neg_perturbed_markers, ft.is_closed)
        
        pos_fluid_poly = get_fluid_polygon(pos_perturbed_tracker)
        neg_fluid_poly = get_fluid_polygon(neg_perturbed_tracker)
        
        # Track the max Jacobian value for this marker
        max_jac_value = 0.0
        max_jac_cell = nothing
        
        # Calculate perturbed volumes using central differencing
        for ((i, j), _) in original_volumes
            # Create cell coordinates
            cell_coords = [
                [x_faces_vec[i], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j]]  # Close the polygon
            ]
            
            # Create cell polygon
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calculate intersection with positive perturbed fluid polygon
            pos_intersection = LibGEOS.intersection(cell_poly, pos_fluid_poly)
            pos_volume = LibGEOS.isEmpty(pos_intersection) ? 0.0 : LibGEOS.area(pos_intersection)
            
            # Calculate intersection with negative perturbed fluid polygon
            neg_intersection = LibGEOS.intersection(cell_poly, neg_fluid_poly)
            neg_volume = LibGEOS.isEmpty(neg_intersection) ? 0.0 : LibGEOS.area(neg_intersection)
            
            # Calculate Jacobian value using central differencing
            jacobian_value = (pos_volume - neg_volume) / (2.0 * epsilon)
            
            # Store significant changes and track maximum value
            if abs(jacobian_value) > 1e-10
                push!(volume_jacobian[(i, j)], (marker_idx, jacobian_value))
                push!(markers_with_entries, marker_idx)
                
                if abs(jacobian_value) > abs(max_jac_value)
                    max_jac_value = jacobian_value
                    max_jac_cell = (i, j)
                end
            elseif abs(jacobian_value) > abs(max_jac_value)
                max_jac_value = jacobian_value
                max_jac_cell = (i, j)
            end
        end
        
        # If this marker has no entries, add its maximum value entry
        if marker_idx ∉ markers_with_entries && max_jac_cell !== nothing
            push!(volume_jacobian[max_jac_cell], (marker_idx, max_jac_value))
            push!(markers_with_entries, marker_idx)
        end
    end

        # For closed interfaces, copy the entries of the first marker to the last marker
    if ft.is_closed && length(markers) > 1
        # Get the index of the last marker
        last_marker_idx = length(markers)
        
        # For each cell with entries
        for ((i, j), entries) in volume_jacobian
            # Check if the first marker has an entry for this cell
            for (marker_idx, jacobian_value) in entries
                if marker_idx == 1
                    # Copy the first marker's entry to the last marker
                    push!(volume_jacobian[(i, j)], (last_marker_idx, jacobian_value))
                    push!(markers_with_entries, last_marker_idx)
                    break
                end
            end
        end
    end
    
    return volume_jacobian
end

"""
    fluid_cell_properties(mesh::Mesh{2}, front::FrontTracker)

Calcule pour chaque cellule (i,j):
- la fraction fluide α_{ij}
- la capacité volumique V_{ij}
- les coordonnées du centroïde fluide (X_{i,j}, Y_{i,j})

Retourne (fractions, volumes, centroids_x, centroids_y)
"""
function fluid_cell_properties(mesh::Mesh{2}, front::FrontTracker)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    fractions = zeros(nx+1, ny+1)  # Fraction fluide
    volumes = zeros(nx+1, ny+1)    # Volume fluide
    centroids_x = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en x
    centroids_y = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en y
    cell_types = zeros(Int, nx+1, ny+1)  # Type de cellule (0: solide/empty, 1: fluide/full, -1: cut)
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)
    
    # Parcours de toutes les cellules
    for i in 1:nx
        for j in 1:ny
            # Création du polygone représentant la cellule
            cell_coords = [
                [x_nodes[i], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j]]  # Fermer le polygone
            ]
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calcul de l'aire de la cellule
            cell_area = LibGEOS.area(cell_poly)
            
            # Calcul de l'intersection avec le domaine fluide
            if LibGEOS.intersects(cell_poly, fluid_poly)
                intersection = LibGEOS.intersection(cell_poly, fluid_poly)
                
                if !LibGEOS.isEmpty(intersection)
                    # Calcul de la fraction fluide
                    fluid_area = LibGEOS.area(intersection)
                    fractions[i, j] = fluid_area / cell_area
                    volumes[i, j] = fluid_area
                    
                    # Calcul du centroïde de la partie fluide
                    centroid = LibGEOS.centroid(intersection)
                    centroids_x[i, j] = GeoInterface.x(centroid)
                    centroids_y[i, j] = GeoInterface.y(centroid)
                    
                    # Détermination du type de cellule
                    if isapprox(fractions[i, j], 1.0, atol=1e-10)
                        cell_types[i, j] = 1   # Cellule complètement fluide
                    elseif isapprox(fractions[i, j], 0.0, atol=1e-10)
                        cell_types[i, j] = 0   # Cellule vide
                    else
                        cell_types[i, j] = -1  # Cellule coupée
                    end
                else
                    # Cellule entièrement solide
                    centroids_x[i, j] = (x_nodes[i] + x_nodes[i+1]) / 2
                    centroids_y[i, j] = (y_nodes[j] + y_nodes[j+1]) / 2
                    cell_types[i, j] = 0  # Cellule vide
                end
            else
                # Vérification si le centre est dans le fluide
                center_x = (x_nodes[i] + x_nodes[i+1]) / 2
                center_y = (y_nodes[j] + y_nodes[j+1]) / 2
                
                if is_point_inside(front, center_x, center_y)
                    fractions[i, j] = 1.0
                    volumes[i, j] = cell_area
                    centroids_x[i, j] = center_x
                    centroids_y[i, j] = center_y
                    cell_types[i, j] = 1  # Cellule complètement fluide
                else
                    centroids_x[i, j] = center_x
                    centroids_y[i, j] = center_y
                    cell_types[i, j] = 0  # Cellule vide
                end
            end
        end
    end

    return fractions, volumes, centroids_x, centroids_y, cell_types
end

"""
    compute_surface_capacities(mesh::Mesh{2}, front::FrontTracker)

Calcule les capacités de surface:
- A^x_{i,j}: longueur mouillée des faces verticales
- A^y_{i,j}: longueur mouillée des faces horizontales

Retourne (Ax, Ay)
"""
function compute_surface_capacities(mesh::Mesh{2}, front::FrontTracker)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Ax = zeros(nx+1, ny+1)    # Faces verticales
    Ay = zeros(nx+1, ny+1)    # Faces horizontales
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)
    
    # Calculer les fractions fluides pour toutes les cellules d'abord
    fractions, _, _, _,_ = fluid_cell_properties(mesh, front)
    
    # Calcul pour les faces verticales (Ax)
    for i in 1:nx+1
        for j in 1:ny
            x = x_nodes[i]
            y_min, y_max = y_nodes[j], y_nodes[j+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x, y_min], [x, y_max]])
            
            # Si la face est entre deux cellules (comme dans le code Python)
            if 1 < i <= nx
                # Identifier les cellules gauche et droite
                left_cell_fluid = fractions[i-1, j] > 0
                right_cell_fluid = fractions[i, j] > 0
                
                if left_cell_fluid && right_cell_fluid && fractions[i-1, j] == 1.0 && fractions[i, j] == 1.0
                    # Face entre deux cellules entièrement fluides
                    Ax[i, j] = y_max - y_min
                elseif !left_cell_fluid && !right_cell_fluid
                    # Face entre deux cellules entièrement solides
                    Ax[i, j] = 0.0
                else
                    # Face à la frontière fluide/solide ou impliquant une cut cell
                    if isnothing(front.interface)
                        # Sans interface définie
                        if left_cell_fluid && right_cell_fluid
                            Ax[i, j] = y_max - y_min
                        end
                    else
                        # Vérifier si la face est intersectée par l'interface
                        if LibGEOS.intersects(face_line, fluid_poly)
                            intersection = LibGEOS.intersection(face_line, fluid_poly)
                            
                            # Calculer la longueur correctement selon le type de géométrie
                            if isa(intersection, LibGEOS.LineString)
                                Ax[i, j] = LibGEOS.geomLength(intersection)
                            elseif isa(intersection, LibGEOS.MultiLineString)
                                # Somme des longueurs pour géométries multiples
                                total_length = 0.0
                                for k in 1:LibGEOS.getNumGeometries(intersection)
                                    line = LibGEOS.getGeometry(intersection, k-1)
                                    total_length += LibGEOS.geomLength(line)
                                end
                                Ax[i, j] = total_length
                            elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
                                # Un point n'a pas de longueur
                                Ax[i, j] = 0.0
                            else
                                # Type de géométrie inconnu - utiliser le test du point milieu
                                mid_y = (y_min + y_max) / 2
                                if is_point_inside(front, x, mid_y)
                                    Ax[i, j] = y_max - y_min
                                end
                            end
                        else
                            # Face non intersectée par l'interface
                            mid_y = (y_min + y_max) / 2
                            if is_point_inside(front, x, mid_y)
                                Ax[i, j] = y_max - y_min
                            else
                                Ax[i, j] = 0.0
                            end
                        end
                    end
                end
            else
                # Faces aux bords du domaine (i=1 ou i=nx+1)
                # Vérifier si la face intersecte le domaine fluide
                if LibGEOS.intersects(face_line, fluid_poly)
                    intersection = LibGEOS.intersection(face_line, fluid_poly)
                    if isa(intersection, LibGEOS.LineString)
                        Ax[i, j] = LibGEOS.geomLength(intersection)
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        Ax[i, j] = total_length
                    end
                else
                    # Vérifier si le milieu est dans le fluide
                    mid_y = (y_min + y_max) / 2
                    if is_point_inside(front, x, mid_y)
                        Ax[i, j] = y_max - y_min
                    end
                end
            end
            
            # Valider la valeur calculée (ne devrait pas dépasser la hauteur de la cellule)
            if Ax[i, j] > (y_max - y_min) * (1 + 1e-10)
                Ax[i, j] = y_max - y_min
            end
        end
    end
    
    # Même logique pour les faces horizontales (Ay)
    for i in 1:nx
        for j in 1:ny+1
            y = y_nodes[j]
            x_min, x_max = x_nodes[i], x_nodes[i+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x_min, y], [x_max, y]])
            
            # Si la face est entre deux cellules
            if 1 < j <= ny
                bottom_cell_fluid = fractions[i, j-1] > 0
                top_cell_fluid = fractions[i, j] > 0
                
                if bottom_cell_fluid && top_cell_fluid && fractions[i, j-1] == 1.0 && fractions[i, j] == 1.0
                    # Face entre deux cellules entièrement fluides
                    Ay[i, j] = x_max - x_min
                elseif !bottom_cell_fluid && !top_cell_fluid
                    # Face entre deux cellules entièrement solides
                    Ay[i, j] = 0.0
                else
                    # Face à la frontière fluide/solide ou impliquant une cut cell
                    if isnothing(front.interface)
                        if bottom_cell_fluid && top_cell_fluid
                            Ay[i, j] = x_max - x_min
                        end
                    else
                        # Vérifier si la face est intersectée par l'interface
                        if LibGEOS.intersects(face_line, fluid_poly)
                            intersection = LibGEOS.intersection(face_line, fluid_poly)
                            
                            # Calculer la longueur selon le type de géométrie
                            if isa(intersection, LibGEOS.LineString)
                                Ay[i, j] = LibGEOS.geomLength(intersection)
                            elseif isa(intersection, LibGEOS.MultiLineString)
                                total_length = 0.0
                                for k in 1:LibGEOS.getNumGeometries(intersection)
                                    line = LibGEOS.getGeometry(intersection, k-1)
                                    total_length += LibGEOS.geomLength(line)
                                end
                                Ay[i, j] = total_length
                            elseif isa(intersection, LibGEOS.Point) || isa(intersection, LibGEOS.MultiPoint)
                                Ay[i, j] = 0.0
                            else
                                # Type de géométrie inconnu - utiliser le test du point milieu
                                mid_x = (x_min + x_max) / 2
                                if is_point_inside(front, mid_x, y)
                                    Ay[i, j] = x_max - x_min
                                end
                            end
                        else
                            # Face non intersectée
                            mid_x = (x_min + x_max) / 2
                            if is_point_inside(front, mid_x, y)
                                Ay[i, j] = x_max - x_min
                            else
                                Ay[i, j] = 0.0
                            end
                        end
                    end
                end
            else
                # Faces aux bords du domaine (j=1 ou j=ny+1)
                if LibGEOS.intersects(face_line, fluid_poly)
                    intersection = LibGEOS.intersection(face_line, fluid_poly)
                    if isa(intersection, LibGEOS.LineString)
                        Ay[i, j] = LibGEOS.geomLength(intersection)
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        Ay[i, j] = total_length
                    end
                else
                    mid_x = (x_min + x_max) / 2
                    if is_point_inside(front, mid_x, y)
                        Ay[i, j] = x_max - x_min
                    end
                end
            end
            
            # Valider la valeur calculée
            if Ay[i, j] > (x_max - x_min) * (1 + 1e-10)
                Ay[i, j] = x_max - x_min
            end
        end
    end
    
    return Ax, Ay
end

"""
    compute_second_type_capacities(mesh::Mesh{2}, front::FrontTracker, centroids_x, centroids_y)

Calcule les capacités de second type:
- W^x: volume fluide entre les centres de masse horizontalement adjacents
- W^y: volume fluide entre les centres de masse verticalement adjacents
- B^x: longueur mouillée des lignes verticales passant par les centroïdes
- B^y: longueur mouillée des lignes horizontales passant par les centroïdes

Retourne (Wx, Wy, Bx, By)
"""
function compute_second_type_capacities(mesh::Mesh{2}, front::FrontTracker, 
                                       centroids_x, centroids_y)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Wx = zeros(nx+1, ny+1)    # Capacités horizontales entre centroïdes
    Wy = zeros(nx+1, ny+1)    # Capacités verticales entre centroïdes
    Bx = zeros(nx+1, ny+1)      # Longueur mouillée des lignes verticales
    By = zeros(nx+1, ny+1)      # Longueur mouillée des lignes horizontales
    
    # Récupération du domaine fluide
    fluid_poly = get_fluid_polygon(front)

    # Récupération des volumes fluides pour chaque cellule
    fractions, volumes, _, _,_ = fluid_cell_properties(mesh, front)
    
     # Calcul des capacités de volume horizontales Wx
    for i in 1:nx
        for j in 1:ny
            # Uniquement calculer si au moins une des cellules contient du fluide
            if volumes[i, j] > 0 || volumes[i+1, j] > 0
                # Définition du domaine d'intégration entre les centroïdes
                x_left = centroids_x[i, j]
                x_right = centroids_x[i+1, j]
                y_min, y_max = y_nodes[j], y_nodes[j+1]
                
                # Création du polygone pour le domaine d'intégration
                poly_coords = [
                    [x_left, y_min],
                    [x_right, y_min],
                    [x_right, y_max],
                    [x_left, y_max],
                    [x_left, y_min]
                ]
                poly = LibGEOS.Polygon([poly_coords])
                
                # Calcul de l'intersection avec le domaine fluide
                if LibGEOS.intersects(poly, fluid_poly)
                    intersection = LibGEOS.intersection(poly, fluid_poly)
                    Wx[i+1, j] = LibGEOS.area(intersection)
                else
                    # Vérification si le milieu est dans le fluide
                    mid_x = (x_left + x_right) / 2
                    mid_y = (y_min + y_max) / 2
                    if is_point_inside(front, mid_x, mid_y)
                        Wx[i+1, j] = LibGEOS.area(poly)
                    else
                        # Si complètement dans le solide
                        Wx[i+1, j] = 0.0
                    end
                end
            else
                # Si les deux cellules sont solides, Wx est 0
                Wx[i+1, j] = 0.0
            end
        end
    end
    
    # Calcul des capacités de volume verticales Wy
    for i in 1:nx
        for j in 1:ny
            if volumes[i, j] > 0 || volumes[i, j+1] > 0
                # Définition du domaine d'intégration entre les centroïdes
                y_bottom = centroids_y[i, j]
                y_top = centroids_y[i, j+1]
                x_min, x_max = x_nodes[i], x_nodes[i+1]
                
                # Création du polygone pour le domaine d'intégration
                poly_coords = [
                    [x_min, y_bottom],
                    [x_max, y_bottom],
                    [x_max, y_top],
                    [x_min, y_top],
                    [x_min, y_bottom]
                ]
                poly = LibGEOS.Polygon([poly_coords])
                
                # Calcul de l'intersection avec le domaine fluide
                if LibGEOS.intersects(poly, fluid_poly)
                    intersection = LibGEOS.intersection(poly, fluid_poly)
                    Wy[i, j+1] = LibGEOS.area(intersection)
                else
                    # Vérification si le milieu est dans le fluide
                    mid_x = (x_min + x_max) / 2
                    mid_y = (y_bottom + y_top) / 2
                    if is_point_inside(front, mid_x, mid_y)
                        Wy[i, j+1] = LibGEOS.area(poly)
                    else
                        # Si complètement dans le solide
                        Wy[i, j+1] = 0.0
                    end
                end
            else
                # Si les deux cellules sont solides, Wy est 0
                Wy[i, j+1] = 0.0
            end
        end
    end
    
       # Calcul des longueurs B^x et B^y
    for i in 1:nx
        for j in 1:ny
            # Récupérer le volume et la fraction fluide pour cette cellule
            cell_volume = volumes[i, j]
            cell_fraction = fractions[i, j]
            x_cm = centroids_x[i, j]
            y_cm = centroids_y[i, j]
            
            # Calculer uniquement pour les cellules avec du fluide
            if cell_volume > 0
                # Dimensions de la cellule
                cell_height = y_nodes[j+1] - y_nodes[j]
                cell_width = x_nodes[i+1] - x_nodes[i]
                
                # Cellule complètement fluide vs. cellule coupée
                if cell_fraction == 1.0
                    # Pour une cellule complètement fluide
                    Bx[i, j] = cell_height
                    By[i, j] = cell_width
                else
                    # Pour une cellule coupée - calculer les intersections
                    
                    # Ligne verticale passant par le centroïde fluide
                    vertical_line = LibGEOS.LineString([
                        [x_cm, y_nodes[j]],
                        [x_cm, y_nodes[j+1]]
                    ])
                    
                    # Ligne horizontale passant par le centroïde fluide
                    horizontal_line = LibGEOS.LineString([
                        [x_nodes[i], y_cm],
                        [x_nodes[i+1], y_cm]
                    ])
                    
                    # Calcul de Bx - longueur mouillée verticale
                    if LibGEOS.intersects(vertical_line, fluid_poly)
                        intersection = LibGEOS.intersection(vertical_line, fluid_poly)
                        
                        if isa(intersection, LibGEOS.LineString)
                            Bx[i, j] = LibGEOS.geomLength(intersection)
                        elseif isa(intersection, LibGEOS.MultiLineString)
                            # Somme des longueurs pour géométries multiples
                            total_length = 0.0
                            for k in 1:LibGEOS.getNumGeometries(intersection)
                                line = LibGEOS.getGeometry(intersection, k-1)
                                total_length += LibGEOS.geomLength(line)
                            end
                            Bx[i, j] = total_length
                        else
                            # Point ou autre géométrie - pas de longueur
                            Bx[i, j] = 0.0
                        end
                    else
                        # Si la ligne ne coupe pas l'interface, vérifier si elle est du côté fluide
                        # Utiliser le centre de la cellule comme point de test
                        y_center = (y_nodes[j] + y_nodes[j+1]) / 2
                        if is_point_inside(front, x_cm, y_center)
                            Bx[i, j] = cell_height
                        else
                            Bx[i, j] = 0.0
                        end
                    end
                    
                    # Calcul de By - longueur mouillée horizontale
                    if LibGEOS.intersects(horizontal_line, fluid_poly)
                        intersection = LibGEOS.intersection(horizontal_line, fluid_poly)
                        
                        if isa(intersection, LibGEOS.LineString)
                            By[i, j] = LibGEOS.geomLength(intersection)
                        elseif isa(intersection, LibGEOS.MultiLineString)
                            # Somme des longueurs pour géométries multiples
                            total_length = 0.0
                            for k in 1:LibGEOS.getNumGeometries(intersection)
                                line = LibGEOS.getGeometry(intersection, k-1)
                                total_length += LibGEOS.geomLength(line)
                            end
                            By[i, j] = total_length
                        else
                            # Point ou autre géométrie - pas de longueur
                            By[i, j] = 0.0
                        end
                    else
                        # Si la ligne ne coupe pas l'interface, vérifier si elle est du côté fluide
                        x_center = (x_nodes[i] + x_nodes[i+1]) / 2
                        if is_point_inside(front, x_center, y_cm)
                            By[i, j] = cell_width
                        else
                            By[i, j] = 0.0
                        end
                    end
                end
            else
                # Cellule sans fluide
                Bx[i, j] = 0.0
                By[i, j] = 0.0
            end
        end
    end
    return Wx, Wy, Bx, By
end

"""
    compute_interface_info(mesh::Mesh{2}, front::FrontTracker)

Calcule les longueurs d'interface et les points représentatifs dans chaque cellule coupée.
Retourne (interface_lengths, interface_points)
"""
function compute_interface_info(mesh::Mesh{2}, front::FrontTracker)
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    interface_lengths = Dict{Tuple{Int, Int}, Float64}()
    interface_points = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()
    
    # Récupération de l'interface
    interface_line = front.interface
    
    for i in 1:nx
        for j in 1:ny
            # Création du polygone représentant la cellule
            cell_coords = [
                [x_nodes[i], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j]],
                [x_nodes[i+1], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j+1]],
                [x_nodes[i], y_nodes[j]]
            ]
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Vérification si la cellule intersecte l'interface
            if LibGEOS.intersects(cell_poly, interface_line)
                intersection = LibGEOS.intersection(cell_poly, interface_line)
                total_length = 0.0
                weighted_x = 0.0
                weighted_y = 0.0
                for line in LibGEOS.getGeometries(intersection)
                    length = LibGEOS.geomLength(line)
                    total_length += length
                    
                    # Point milieu de ce segment
                    mid_point = LibGEOS.interpolate(line, 0.5)
                    
                    # Ajout au barycentre pondéré
                    weighted_x += GeoInterface.x(mid_point) * length
                    weighted_y += GeoInterface.y(mid_point) * length
                end
                # Stockage de la longueur d'interface
                interface_lengths[(i, j)] = total_length

                if total_length > 0
                    interface_points[(i, j)] = (weighted_x / total_length, 
                                               weighted_y / total_length)
                else
                    interface_points[(i, j)] = (0.0, 0.0)  # Cas où il n'y a pas d'interface
                end
                # MODIFICATION IMPORTANTE: Commenté pour éviter les erreurs de type
                       
                """
                if LibGEOS.getGeometryType(intersection) == :LineString
                    # Stockage de la longueur d'interface
                    interface_lengths[(i, j)] = LibGEOS.length(intersection)
                    
                    # Point représentatif (milieu de la ligne)
                    mid_point = LibGEOS.interpolate(intersection, 0.5)
                    interface_points[(i, j)] = (mid_point[1], mid_point[2])
                    
                elseif LibGEOS.getGeometryType(intersection) == :MultiLineString
                    # Calcul de la longueur totale et du barycentre pondéré
                    total_length = 0.0
                    weighted_x = 0.0
                    weighted_y = 0.0
                    
                    for line in LibGEOS.getGeometries(intersection)
                        length = LibGEOS.length(line)
                        total_length += length
                        
                        # Point milieu de ce segment
                        mid_point = LibGEOS.interpolate(line, 0.5)
                       
                        
                        # Ajout au barycentre pondéré
                        weighted_x += mid_point[1] * length
                        weighted_y += mid_point[2] * length
                    end
                    
                    interface_lengths[(i, j)] = total_length
                    
                    if total_length > 0
                        interface_points[(i, j)] = (weighted_x / total_length, 
                                                   weighted_y / total_length)
                    end
                end
                """
            end
        end
    end
    
    return interface_lengths, interface_points
end

"""
    compute_capacities(mesh::Mesh{2}, front::FrontTracker)

Calcule toutes les capacités géométriques pour un maillage et une interface donnés.
Retourne un dictionnaire avec tous les résultats.
"""
function compute_capacities(mesh::Mesh{2}, front::FrontTracker)
    # Calcul des propriétés des cellules
    fractions, volumes, centroids_x, centroids_y, cell_types = fluid_cell_properties(mesh, front)
    
    # Calcul des capacités de surface
    Ax, Ay = compute_surface_capacities(mesh, front)
    
    # Calcul des capacités de second type
    Wx, Wy, Bx, By = compute_second_type_capacities(mesh, front, centroids_x, centroids_y)
    
    # Calcul des informations d'interface
    interface_lengths, interface_points = compute_interface_info(mesh, front)
    
    # Retourne toutes les capacités dans un dictionnaire
    return Dict(
        :fractions => fractions,          # Fractions fluides α_{ij}
        :volumes => volumes,              # Capacités volumiques V_{ij}
        :centroids_x => centroids_x,      # Coordonnées x des centroïdes X_{i,j}
        :centroids_y => centroids_y,      # Coordonnées y des centroïdes Y_{i,j}
        :cell_types => cell_types,        # Types de cellules (0: empty, 1: full, -1: cut)
        :Ax => Ax,                        # Capacités des faces verticales A^x_{i,j}
        :Ay => Ay,                        # Capacités des faces horizontales A^y_{i,j}
        :Wx => Wx,                        # Capacités de volume horizontales W^x_{i+1/2,j}
        :Wy => Wy,                        # Capacités de volume verticales W^y_{i,j+1/2}
        :Bx => Bx,                        # Longueurs mouillées verticales B^x_{i,j}
        :By => By,                        # Longueurs mouillées horizontales B^y_{i,j}
        :interface_lengths => interface_lengths,  # Longueurs d'interface par cellule
        :interface_points => interface_points     # Points représentatifs sur l'interface
    )
end


# Space-Time Capacities
"""
    compute_spacetime_volumes(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calcule les capacités de volume spatio-temporelles en intégrant les capacités 2D dans le temps.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `dt::Float64`: Le pas de temps Δt

# Retourne
- `V_st::Matrix{Float64}`: Les volumes spatio-temporels
"""
function compute_spacetime_volumes(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    V_st = zeros(nx+1, ny+1)  # Volumes spatio-temporels
    
    # Calculer les propriétés des cellules aux deux instants
    fractions_n, volumes_n, _, _, cell_types_n = fluid_cell_properties(mesh, front_n)
    fractions_np1, volumes_np1, _, _, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Pour chaque cellule
    for i in 1:nx
        for j in 1:ny
            # Récupérer les types de cellules et volumes aux deux instants
            cell_type_n = cell_types_n[i, j]
            cell_type_np1 = cell_types_np1[i, j]
            vol_n = volumes_n[i, j]
            vol_np1 = volumes_np1[i, j]
            
            # Utiliser le tableau pour déterminer la méthode de calcul
            if cell_type_n == 0 && cell_type_np1 == 0
                # empty → empty
                V_st[i, j] = 0.0
            elseif cell_type_n == 1 && cell_type_np1 == 1
                # full → full
                dx = x_nodes[i+1] - x_nodes[i]
                dy = y_nodes[j+1] - y_nodes[j]
                V_st[i, j] = dt * dx * dy
            elseif (cell_type_n == -1 && cell_type_np1 == -1) || 
                  (cell_type_n == -1 && cell_type_np1 == 1) || 
                  (cell_type_n == 1 && cell_type_np1 == -1)
                # cut → cut, cut → full, full → cut
                V_st[i, j] = (dt / 2.0) * (vol_n + vol_np1)
            elseif cell_type_n == 0 && cell_type_np1 == -1
                # empty → cut : V_{ec}
                V_st[i, j] = compute_special_spacetime_volume(mesh, front_n, front_np1, i, j, dt, "ec")
            elseif cell_type_n == -1 && cell_type_np1 == 0
                # cut → empty : V_{ce}
                V_st[i, j] = compute_special_spacetime_volume(mesh, front_n, front_np1, i, j, dt, "ce")
            elseif cell_type_n == 0 && cell_type_np1 == 1
                # empty → full : V_{ef}
                V_st[i, j] = compute_special_spacetime_volume(mesh, front_n, front_np1, i, j, dt, "ef")
            elseif cell_type_n == 1 && cell_type_np1 == 0
                # full → empty : V_{fe}
                V_st[i, j] = compute_special_spacetime_volume(mesh, front_n, front_np1, i, j, dt, "fe")
            end
        end
    end
    
    return V_st
end

"""
    compute_special_spacetime_volume(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, 
                                   i::Int, j::Int, dt::Float64, transition_type::String)

Calcule le volume spatio-temporel pour les cas spéciaux où la cellule change de type.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la cellule
- `dt::Float64`: Le pas de temps
- `transition_type::String`: Type de transition ("ec", "ce", "ef", "fe")

# Retourne
- `volume::Float64`: Le volume spatio-temporel
"""
function compute_special_spacetime_volume(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, 
                                         i::Int, j::Int, dt::Float64, transition_type::String)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Coordonnées des sommets de la cellule
    x_min, x_max = x_nodes[i], x_nodes[i+1]
    y_min, y_max = y_nodes[j], y_nodes[j+1]
    
    # Créer une représentation temporelle interpolée de l'interface
    # Déterminer les instants τₖ où l'interface traverse les sommets de la cellule
    tau = find_crossing_times(mesh, front_n, front_np1, i, j, dt)
    
    # Si aucun temps de croisement n'est trouvé, utiliser la méthode trapézoïdale standard
    if length(tau) <= 2  # Seulement t^n et t^{n+1}
        vol_n = compute_volume_at_time(mesh, front_n, front_np1, i, j, 0.0, dt)
        vol_np1 = compute_volume_at_time(mesh, front_n, front_np1, i, j, dt, dt)
        return (dt / 2.0) * (vol_n + vol_np1)
    end
    
    # Sinon, intégrer sur chaque sous-intervalle [τₖ, τₖ₊₁]
    volume = 0.0
    for k in 1:(length(tau)-1)
        t_k = tau[k]
        t_kp1 = tau[k+1]
        
        # Calculer le volume aux extrémités de l'intervalle
        vol_k = compute_volume_at_time(mesh, front_n, front_np1, i, j, t_k, dt)
        vol_kp1 = compute_volume_at_time(mesh, front_n, front_np1, i, j, t_kp1, dt)
        
        # Utiliser la règle trapézoïdale pour ce sous-intervalle
        volume += ((t_kp1 - t_k) / 2.0) * (vol_k + vol_kp1)
    end
    
    return volume
end

"""
    find_crossing_times(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, 
                      i::Int, j::Int, dt::Float64)

Détermine les instants τₖ où l'interface traverse les sommets de la cellule.

# Retourne
- `tau::Vector{Float64}`: Temps de croisement, incluant t^n et t^{n+1}
"""
function find_crossing_times(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, 
                           i::Int, j::Int, dt::Float64)
    # Commencer avec les temps aux extrémités de l'intervalle
    tau = [0.0, dt]
    
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Coordonnées des sommets de la cellule
    vertices = [
        (x_nodes[i], y_nodes[j]),      # Coin inférieur gauche
        (x_nodes[i+1], y_nodes[j]),    # Coin inférieur droit
        (x_nodes[i+1], y_nodes[j+1]),  # Coin supérieur droit
        (x_nodes[i], y_nodes[j+1])     # Coin supérieur gauche
    ]
    
    # Pour chaque sommet, trouver s'il y a un changement de statut (dedans/dehors)
    for vertex in vertices
        x, y = vertex
        # Vérifier le statut au temps initial
        inside_n = is_point_inside(front_n, x, y)
        # Vérifier le statut au temps final
        inside_np1 = is_point_inside(front_np1, x, y)
        
        # Si le statut change, trouver le temps de croisement
        if inside_n != inside_np1
            # On recherche par dichotomie le temps où le point traverse l'interface
            t_low, t_high = 0.0, dt
            crossing_time = dt / 2.0  # Valeur initiale
            
            # Précision souhaitée
            tolerance = 1e-8 * dt
            
            # Recherche dichotomique
            while t_high - t_low > tolerance
                crossing_time = (t_low + t_high) / 2.0
                
                # Créer une interface interpolée à cet instant
                front_t = interpolate_front(front_n, front_np1, crossing_time / dt)
                
                # Vérifier le statut à cet instant
                is_inside = is_point_inside(front_t, x, y)
                
                # Ajuster les bornes de recherche
                if is_inside == inside_n
                    t_low = crossing_time
                else
                    t_high = crossing_time
                end
            end
            
            # Ajouter ce temps de croisement
            push!(tau, crossing_time)
        end
    end
    
    # Trier les temps de croisement
    sort!(tau)
    
    return tau
end

"""
    interpolate_front(front_n::FrontTracker, front_np1::FrontTracker, t_ratio::Float64)

Interpolation linéaire de l'interface à un instant t = t_n + t_ratio * (t_{n+1} - t_n).

# Arguments
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `t_ratio::Float64`: Ratio temporel entre 0 et 1 (0 = t^n, 1 = t^{n+1})

# Retourne
- `front_t::FrontTracker`: L'interface interpolée au temps t
"""
function interpolate_front(front_n::FrontTracker, front_np1::FrontTracker, t_ratio::Float64)
    # Recupérer les marqueurs aux deux instants
    markers_n = get_markers(front_n)
    markers_np1 = get_markers(front_np1)
    
    # Vérifier que les deux interfaces ont le même nombre de marqueurs
    if length(markers_n) != length(markers_np1)
        error("Les deux interfaces doivent avoir le même nombre de marqueurs pour l'interpolation.")
    end
    
    # Interpoler chaque marqueur
    markers_t = Vector{Tuple{Float64, Float64}}(undef, length(markers_n))
    for i in 1:length(markers_n)
        x_n, y_n = markers_n[i]
        x_np1, y_np1 = markers_np1[i]
        
        # Interpolation linéaire
        x_t = x_n + t_ratio * (x_np1 - x_n)
        y_t = y_n + t_ratio * (y_np1 - y_n)
        
        markers_t[i] = (x_t, y_t)
    end
    
    # Créer une nouvelle interface avec les marqueurs interpolés
    front_t = FrontTracker(markers_t, front_n.is_closed)
    
    return front_t
end

"""
    compute_volume_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                         i::Int, j::Int, t::Float64, dt::Float64)

Calcule le volume fluide dans la cellule (i,j) à l'instant t = t_n + t.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la cellule
- `t::Float64`: L'instant auquel calculer le volume (relatif à t^n)
- `dt::Float64`: Le pas de temps total

# Retourne
- `volume::Float64`: Le volume fluide à l'instant t
"""
function compute_volume_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                              i::Int, j::Int, t::Float64, dt::Float64)
    # Si t est exactement t^n ou t^{n+1}, utiliser directement les volumes calculés
    if isapprox(t, 0.0, atol=1e-10)
        fractions_n, volumes_n, _, _, _ = fluid_cell_properties(mesh, front_n)
        return volumes_n[i, j]
    elseif isapprox(t, dt, atol=1e-10)
        fractions_np1, volumes_np1, _, _, _ = fluid_cell_properties(mesh, front_np1)
        return volumes_np1[i, j]
    end
    
    # Sinon, interpoler l'interface et calculer le volume
    t_ratio = t / dt
    front_t = interpolate_front(front_n, front_np1, t_ratio)
    
    # Calculer les propriétés de la cellule à cet instant
    fractions_t, volumes_t, _, _, _ = fluid_cell_properties(mesh, front_t)
    
    return volumes_t[i, j]
end

"""
    compute_spacetime_centroid(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calcule les centroïdes des volumes spatio-temporels pour chaque cellule.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `dt::Float64`: Le pas de temps Δt

# Retourne
- `centroids_st::Vector{Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Float64}}}`: 
    Les centroïdes spatio-temporels sous forme (x,y,t) pour chaque cellule
"""
function compute_spacetime_centroid(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Calculer les volumes spatio-temporels d'abord
    V_st = compute_spacetime_volumes(mesh, front_n, front_np1, dt)
    
    # Initialisation des matrices pour les moments
    xV_st = zeros(nx+1, ny+1)  # Moment en x
    yV_st = zeros(nx+1, ny+1)  # Moment en y
    tV_st = zeros(nx+1, ny+1)  # Moment en t
    
    # Calculer les propriétés des cellules aux deux instants
    _, volumes_n, centroids_x_n, centroids_y_n, cell_types_n = fluid_cell_properties(mesh, front_n)
    _, volumes_np1, centroids_x_np1, centroids_y_np1, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Pour chaque cellule
    for i in 1:nx
        for j in 1:ny
            # Récupérer le volume spatio-temporel calculé précédemment
            volume_st = V_st[i, j]
            
            # Si le volume est négligeable, passer à la cellule suivante
            if volume_st < 1e-10
                continue
            end
            
            # Récupérer les types de cellules et volumes aux deux instants
            cell_type_n = cell_types_n[i, j]
            cell_type_np1 = cell_types_np1[i, j]
            vol_n = volumes_n[i, j]
            vol_np1 = volumes_np1[i, j]
            
            # Cas simple: cellule complètement fluide aux deux instants
            if cell_type_n == 1 && cell_type_np1 == 1
                # Le centroïde spatial est au centre de la cellule
                x_center = (x_nodes[i] + x_nodes[i+1]) / 2
                y_center = (y_nodes[j] + y_nodes[j+1]) / 2
                # Le centroïde temporel est au milieu du pas de temps
                t_center = dt / 2
                
                xV_st[i, j] = volume_st * x_center
                yV_st[i, j] = volume_st * y_center
                tV_st[i, j] = volume_st * t_center
                
            # Cas avec changements de type ou cellules coupées
            else
                # Déterminer les temps de croisement pour une intégration précise
                tau = find_crossing_times(mesh, front_n, front_np1, i, j, dt)
                
                # Si aucun temps de croisement intermédiaire n'est trouvé,
                # utiliser simplement la règle trapézoïdale
                if length(tau) <= 2  # Seulement t^n et t^{n+1}
                    # Calculer les centroïdes aux instants extrêmes
                    cx_n = centroids_x_n[i, j]
                    cy_n = centroids_y_n[i, j]
                    cx_np1 = centroids_x_np1[i, j]
                    cy_np1 = centroids_y_np1[i, j]
                    
                    # Le centroïde est la moyenne pondérée par le volume à chaque instant
                    # Pour x et y, pondérée par les deux volumes aux extrémités
                    if vol_n + vol_np1 > 0
                        xV_st[i, j] = volume_st * (vol_n * cx_n + vol_np1 * cx_np1) / (vol_n + vol_np1)
                        yV_st[i, j] = volume_st * (vol_n * cy_n + vol_np1 * cy_np1) / (vol_n + vol_np1)
                    end
                    
                    # Pour t, la pondération dépend de la variation de volume
                    if vol_n == 0 && vol_np1 > 0
                        # Remplissage progressif, centroïde temporel plus proche de t^{n+1}
                        tV_st[i, j] = volume_st * (2*dt/3)
                    elseif vol_n > 0 && vol_np1 == 0
                        # Vidage progressif, centroïde temporel plus proche de t^n
                        tV_st[i, j] = volume_st * (dt/3)
                    else
                        # Cas général, centroïde temporel au milieu
                        tV_st[i, j] = volume_st * (dt/2)
                    end
                    
                # Avec des temps de croisement intermédiaires, intégrer sur chaque sous-intervalle
                else
                    xV_interval = 0.0
                    yV_interval = 0.0
                    tV_interval = 0.0
                    
                    for k in 1:(length(tau)-1)
                        t_k = tau[k]
                        t_kp1 = tau[k+1]
                        
                        # Calculer le volume et les centroïdes aux extrémités de l'intervalle
                        t_ratio_k = t_k / dt
                        t_ratio_kp1 = t_kp1 / dt
                        
                        # Interpoler les interfaces à ces instants
                        front_k = interpolate_front(front_n, front_np1, t_ratio_k)
                        front_kp1 = interpolate_front(front_n, front_np1, t_ratio_kp1)
                        
                        # Calculer les propriétés à ces instants
                        _, vol_k, cx_k, cy_k, _ = fluid_cell_properties(mesh, front_k)
                        _, vol_kp1, cx_kp1, cy_kp1, _ = fluid_cell_properties(mesh, front_kp1)
                        
                        # Volume de ce sous-intervalle (méthode trapézoïdale)
                        sub_volume = ((t_kp1 - t_k) / 2.0) * (vol_k[i, j] + vol_kp1[i, j])
                        
                        # Contribution au moment x
                        if vol_k[i, j] + vol_kp1[i, j] > 0
                            xV_sub = sub_volume * (vol_k[i, j] * cx_k[i, j] + vol_kp1[i, j] * cx_kp1[i, j]) / (vol_k[i, j] + vol_kp1[i, j])
                            yV_sub = sub_volume * (vol_k[i, j] * cy_k[i, j] + vol_kp1[i, j] * cy_kp1[i, j]) / (vol_k[i, j] + vol_kp1[i, j])
                        else
                            xV_sub = 0.0
                            yV_sub = 0.0
                        end
                        
                        # Contribution au moment temporel (milieu de l'intervalle)
                        t_mid = (t_k + t_kp1) / 2.0
                        tV_sub = sub_volume * t_mid
                        
                        # Accumuler les contributions
                        xV_interval += xV_sub
                        yV_interval += yV_sub
                        tV_interval += tV_sub
                    end
                    
                    # Stocker les moments pour cette cellule
                    xV_st[i, j] = xV_interval
                    yV_st[i, j] = yV_interval
                    tV_st[i, j] = tV_interval
                end
            end
        end
    end
    
    # Construire le dictionnaire des centroïdes
    centroids_st = Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Float64}}()
    
    for i in 1:nx
        for j in 1:ny
            if V_st[i, j] > 1e-10
                # Calculer le centroïde en divisant les moments par le volume
                cx = xV_st[i, j] / V_st[i, j]
                cy = yV_st[i, j] / V_st[i, j]
                ct = tV_st[i, j] / V_st[i, j]
                
                centroids_st[(i, j)] = (cx, cy, ct)
            else
                # Cellule vide, centroïde au centre géométrique par défaut
                cx = (x_nodes[i] + x_nodes[i+1]) / 2
                cy = (y_nodes[j] + y_nodes[j+1]) / 2
                ct = dt / 2
                
                centroids_st[(i, j)] = (cx, cy, ct)
            end
        end
    end
    
    return centroids_st
end

"""
    compute_spacetime_surfaces(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calcule les capacités de surface spatio-temporelles en intégrant les capacités de surface (Ax, Ay) dans le temps.
Classification selon les types de cellules adjacentes suivant le tableau de transitions.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `dt::Float64`: Le pas de temps Δt

# Retourne
- `Ax_st::Matrix{Float64}`: Les longueurs mouillées spatio-temporelles des faces verticales
- `Ay_st::Matrix{Float64}`: Les longueurs mouillées spatio-temporelles des faces horizontales
"""
function compute_spacetime_surfaces(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Ax_st = zeros(nx+1, ny+1)  # Capacités de surface spatio-temporelles verticales
    Ay_st = zeros(nx+1, ny+1)  # Capacités de surface spatio-temporelles horizontales
    
    # Calculer les capacités de surface aux deux instants
    Ax_n, Ay_n = compute_surface_capacities(mesh, front_n)
    Ax_np1, Ay_np1 = compute_surface_capacities(mesh, front_np1)
    
    # Calculer les types de cellules aux deux instants
    _, _, _, _, cell_types_n = fluid_cell_properties(mesh, front_n)
    _, _, _, _, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Pour chaque face verticale (Ax)
    for i in 2:nx  # Faces verticales intérieures
        for j in 1:ny
            # Récupérer les types des cellules adjacentes à la face
            left_type_n = cell_types_n[i-1, j]
            right_type_n = cell_types_n[i, j]
            left_type_np1 = cell_types_np1[i-1, j]
            right_type_np1 = cell_types_np1[i, j]
            
            # Déterminer le type de face aux deux instants
            # 0: face sèche (empty/empty), 1: face mouillée (full/full), -1: face coupée (autres cas)
            face_type_n = (left_type_n == 1 && right_type_n == 1) ? 1 : 
                         ((left_type_n == 0 && right_type_n == 0) ? 0 : -1)
            face_type_np1 = (left_type_np1 == 1 && right_type_np1 == 1) ? 1 : 
                           ((left_type_np1 == 0 && right_type_np1 == 0) ? 0 : -1)
            
            # Récupérer les valeurs des capacités aux deux instants
            ax_n = Ax_n[i, j]
            ax_np1 = Ax_np1[i, j]
            
            # Appliquer la logique du tableau
            if face_type_n == 0 && face_type_np1 == 0
                # empty → empty : surface toujours sèche
                Ax_st[i, j] = 0.0
            elseif face_type_n == 1 && face_type_np1 == 1
                # full → full : surface toujours mouillée
                Ax_st[i, j] = dt * (y_nodes[j+1] - y_nodes[j])
            elseif face_type_n == 0 && face_type_np1 == -1
                # empty → cut : A_ε,ec
                Ax_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ec")
            elseif face_type_n == -1 && face_type_np1 == 0
                # cut → empty : A_ε,ce
                Ax_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ce")
            elseif face_type_n == 0 && face_type_np1 == 1
                # empty → full : A_ε,ef
                Ax_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ef")
            elseif face_type_n == 1 && face_type_np1 == 0
                # full → empty : A_ε,fe
                Ax_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "fe")
            else
                # Autres cas (cut → cut, cut → full, full → cut) : règle trapézoïdale
                Ax_st[i, j] = (dt / 2.0) * (ax_n + ax_np1)
            end
        end
    end
    
    # Traiter les faces verticales aux bords du domaine (i=1 ou i=nx+1)
    for i in [1, nx+1]
        for j in 1:ny
            # Pour les faces aux bords, faire une détection simple basée sur les valeurs
            ax_n = Ax_n[i, j]
            ax_np1 = Ax_np1[i, j]
            
            if isapprox(ax_n, 0.0, atol=1e-10) && isapprox(ax_np1, 0.0, atol=1e-10)
                # Face sèche aux deux instants
                Ax_st[i, j] = 0.0
            elseif isapprox(ax_n, y_nodes[j+1] - y_nodes[j], atol=1e-10) && 
                   isapprox(ax_np1, y_nodes[j+1] - y_nodes[j], atol=1e-10)
                # Face complètement mouillée aux deux instants
                Ax_st[i, j] = dt * (y_nodes[j+1] - y_nodes[j])
            elseif abs(ax_n - ax_np1) < 0.1 * max(ax_n, ax_np1)
                # Variation faible, utiliser méthode trapézoïdale standard
                Ax_st[i, j] = (dt / 2.0) * (ax_n + ax_np1)
            else
                # Variation importante, utiliser l'approche des temps de croisement
                Ax_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "general")
            end
        end
    end
    
    # Pour chaque face horizontale (Ay)
    for i in 1:nx
        for j in 2:ny  # Faces horizontales intérieures
            # Récupérer les types des cellules adjacentes à la face
            bottom_type_n = cell_types_n[i, j-1]
            top_type_n = cell_types_n[i, j]
            bottom_type_np1 = cell_types_np1[i, j-1]
            top_type_np1 = cell_types_np1[i, j]
            
            # Déterminer le type de face aux deux instants
            face_type_n = (bottom_type_n == 1 && top_type_n == 1) ? 1 : 
                         ((bottom_type_n == 0 && top_type_n == 0) ? 0 : -1)
            face_type_np1 = (bottom_type_np1 == 1 && top_type_np1 == 1) ? 1 : 
                           ((bottom_type_np1 == 0 && top_type_np1 == 0) ? 0 : -1)
            
            # Récupérer les valeurs des capacités aux deux instants
            ay_n = Ay_n[i, j]
            ay_np1 = Ay_np1[i, j]
            
            # Appliquer la logique du tableau
            if face_type_n == 0 && face_type_np1 == 0
                # empty → empty : surface toujours sèche
                Ay_st[i, j] = 0.0
            elseif face_type_n == 1 && face_type_np1 == 1
                # full → full : surface toujours mouillée
                Ay_st[i, j] = dt * (x_nodes[i+1] - x_nodes[i])
            elseif face_type_n == 0 && face_type_np1 == -1
                # empty → cut : A_ε,ec
                Ay_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ec")
            elseif face_type_n == -1 && face_type_np1 == 0
                # cut → empty : A_ε,ce
                Ay_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ce")
            elseif face_type_n == 0 && face_type_np1 == 1
                # empty → full : A_ε,ef
                Ay_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ef")
            elseif face_type_n == 1 && face_type_np1 == 0
                # full → empty : A_ε,fe
                Ay_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "fe")
            else
                # Autres cas (cut → cut, cut → full, full → cut) : règle trapézoïdale
                Ay_st[i, j] = (dt / 2.0) * (ay_n + ay_np1)
            end
        end
    end
    
    # Traiter les faces horizontales aux bords du domaine (j=1 ou j=ny+1)
    for i in 1:nx
        for j in [1, ny+1]
            # Pour les faces aux bords, faire une détection simple basée sur les valeurs
            ay_n = Ay_n[i, j]
            ay_np1 = Ay_np1[i, j]
            
            if isapprox(ay_n, 0.0, atol=1e-10) && isapprox(ay_np1, 0.0, atol=1e-10)
                # Face sèche aux deux instants
                Ay_st[i, j] = 0.0
            elseif isapprox(ay_n, x_nodes[i+1] - x_nodes[i], atol=1e-10) && 
                   isapprox(ay_np1, x_nodes[i+1] - x_nodes[i], atol=1e-10)
                # Face complètement mouillée aux deux instants
                Ay_st[i, j] = dt * (x_nodes[i+1] - x_nodes[i])
            elseif abs(ay_n - ay_np1) < 0.1 * max(ay_n, ay_np1)
                # Variation faible, utiliser méthode trapézoïdale standard
                Ay_st[i, j] = (dt / 2.0) * (ay_n + ay_np1)
            else
                # Variation importante, utiliser l'approche des temps de croisement
                Ay_st[i, j] = compute_special_face_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "general")
            end
        end
    end
    
    return Ax_st, Ay_st
end

"""
    compute_special_face_spacetime(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                  i::Int, j::Int, dt::Float64, direction::String, transition_type::String)

Calcule la capacité de surface spatio-temporelle pour les cas spéciaux de transition.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la face
- `dt::Float64`: Le pas de temps
- `direction::String`: "x" pour les faces verticales (Ax), "y" pour les faces horizontales (Ay)
- `transition_type::String`: Type de transition ("ec", "ce", "ef", "fe", "general")

# Retourne
- `surface::Float64`: La capacité de surface spatio-temporelle
"""
function compute_special_face_spacetime(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                       i::Int, j::Int, dt::Float64, direction::String, transition_type::String)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Déterminer les points qui définissent la face
    if direction == "x"
        # Face verticale: points haut et bas
        p1 = (x_nodes[i], y_nodes[j])     # Bas
        p2 = (x_nodes[i], y_nodes[j+1])   # Haut
    else  # direction == "y"
        # Face horizontale: points gauche et droite
        p1 = (x_nodes[i], y_nodes[j])     # Gauche
        p2 = (x_nodes[i+1], y_nodes[j])   # Droite
    end
    
    # Déterminer les temps où l'interface traverse la face
    tau = find_face_crossing_times(mesh, front_n, front_np1, i, j, dt, direction)
    
    # Si aucun temps de croisement intermédiaire n'est trouvé, 
    # utiliser la méthode trapézoïdale standard
    if length(tau) <= 2  # Seulement t^n et t^{n+1}
        surf_n = compute_surface_at_time(mesh, front_n, front_np1, i, j, 0.0, dt, direction)
        surf_np1 = compute_surface_at_time(mesh, front_n, front_np1, i, j, dt, dt, direction)
        
        # Ajuster pour les transitions spécifiques
        if transition_type == "ec" || transition_type == "ef"  # empty → cut/full
            # La valeur initiale est 0, donc l'intégrale devrait donner plus de poids au temps final
            return (dt / 3.0) * surf_np1
        elseif transition_type == "ce" || transition_type == "fe"  # cut/full → empty
            # La valeur finale est 0, donc l'intégrale devrait donner plus de poids au temps initial
            return (dt / 3.0) * surf_n
        else
            # Cas général
            return (dt / 2.0) * (surf_n + surf_np1)
        end
    end
    
    # Sinon, intégrer sur chaque sous-intervalle [τₖ, τₖ₊₁]
    surface = 0.0
    for k in 1:(length(tau)-1)
        t_k = tau[k]
        t_kp1 = tau[k+1]
        
        # Calculer la capacité de surface aux extrémités de l'intervalle
        surf_k = compute_surface_at_time(mesh, front_n, front_np1, i, j, t_k, dt, direction)
        surf_kp1 = compute_surface_at_time(mesh, front_n, front_np1, i, j, t_kp1, dt, direction)
        
        # Utiliser la règle trapézoïdale pour ce sous-intervalle
        surface += ((t_kp1 - t_k) / 2.0) * (surf_k + surf_kp1)
    end
    
    return surface
end

"""
    find_face_crossing_times(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                           i::Int, j::Int, dt::Float64, direction::String)

Détermine les instants où l'interface traverse une face donnée.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la face
- `dt::Float64`: Le pas de temps
- `direction::String`: "x" pour les faces verticales (Ax), "y" pour les faces horizontales (Ay)

# Retourne
- `tau::Vector{Float64}`: Temps de croisement, incluant t^n et t^{n+1}
"""
function find_face_crossing_times(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                i::Int, j::Int, dt::Float64, direction::String)
    # Commencer avec les temps aux extrémités de l'intervalle
    tau = [0.0, dt]
    
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Déterminer les points qui définissent la face
    if direction == "x"
        # Face verticale: points haut et bas
        p1 = (x_nodes[i], y_nodes[j])     # Bas
        p2 = (x_nodes[i], y_nodes[j+1])   # Haut
        # Points additionnels le long de la face pour mieux détecter les croisements
        additional_points = [(x_nodes[i], y_nodes[j] + k*(y_nodes[j+1]-y_nodes[j])/6) 
                             for k in 1:5]
    else  # direction == "y"
        # Face horizontale: points gauche et droite
        p1 = (x_nodes[i], y_nodes[j])     # Gauche
        p2 = (x_nodes[i+1], y_nodes[j])   # Droite
        # Points additionnels le long de la face pour mieux détecter les croisements
        additional_points = [(x_nodes[i] + k*(x_nodes[i+1]-x_nodes[i])/6, y_nodes[j]) 
                             for k in 1:5]
    end
    
    # Points à tester, incluant les extrémités et les points additionnels
    test_points = [p1, p2, additional_points...]
    
    # Pour chaque point, trouver le temps de croisement
    for p in test_points
        x, y = p
        
        # Vérifier le statut au temps initial et final
        inside_n = is_point_inside(front_n, x, y)
        inside_np1 = is_point_inside(front_np1, x, y)
        
        # Si le statut change, trouver le temps de croisement
        if inside_n != inside_np1
            # Recherche par dichotomie
            t_low, t_high = 0.0, dt
            crossing_time = dt / 2.0  # Valeur initiale
            tolerance = 1e-8 * dt
            
            while t_high - t_low > tolerance
                crossing_time = (t_low + t_high) / 2.0
                
                # Créer une interface interpolée à cet instant
                front_t = interpolate_front(front_n, front_np1, crossing_time / dt)
                
                # Vérifier le statut à cet instant
                is_inside = is_point_inside(front_t, x, y)
                
                # Ajuster les bornes de recherche
                if is_inside == inside_n
                    t_low = crossing_time
                else
                    t_high = crossing_time
                end
            end
            
            # Ajouter ce temps de croisement
            push!(tau, crossing_time)
        end
    end
    
    # Éliminer les doublons et trier
    tau = sort(unique(tau))
    
    return tau
end

"""
    compute_surface_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                          i::Int, j::Int, t::Float64, dt::Float64, direction::String)

Calcule la capacité de surface dans la face (i,j) à l'instant t = t_n + t.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la face
- `t::Float64`: L'instant auquel calculer la capacité (relatif à t^n)
- `dt::Float64`: Le pas de temps total
- `direction::String`: "x" pour les faces verticales (Ax), "y" pour les faces horizontales (Ay)

# Retourne
- `surface::Float64`: La capacité de surface à l'instant t
"""
function compute_surface_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                               i::Int, j::Int, t::Float64, dt::Float64, direction::String)
    # Si t est exactement t^n ou t^{n+1}, utiliser directement les capacités calculées
    if isapprox(t, 0.0, atol=1e-10)
        Ax_n, Ay_n = compute_surface_capacities(mesh, front_n)
        return direction == "x" ? Ax_n[i, j] : Ay_n[i, j]
    elseif isapprox(t, dt, atol=1e-10)
        Ax_np1, Ay_np1 = compute_surface_capacities(mesh, front_np1)
        return direction == "x" ? Ax_np1[i, j] : Ay_np1[i, j]
    end
    
    # Sinon, interpoler l'interface et calculer les capacités
    t_ratio = t / dt
    front_t = interpolate_front(front_n, front_np1, t_ratio)
    
    # Calculer les capacités à cet instant
    Ax_t, Ay_t = compute_surface_capacities(mesh, front_t)
    
    return direction == "x" ? Ax_t[i, j] : Ay_t[i, j]
end

"""
    compute_spacetime_centerline_lengths(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calcule les capacités de lignes centrales spatio-temporelles en intégrant les capacités Bx et By dans le temps.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `dt::Float64`: Le pas de temps Δt

# Retourne
- `Bx_st::Matrix{Float64}`: Les longueurs mouillées spatio-temporelles des lignes verticales passant par le centroïde
- `By_st::Matrix{Float64}`: Les longueurs mouillées spatio-temporelles des lignes horizontales passant par le centroïde
"""
function compute_spacetime_centerline_lengths(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialisation des matrices de résultats
    Bx_st = zeros(nx+1, ny+1)  # Longueurs verticales spatio-temporelles
    By_st = zeros(nx+1, ny+1)  # Longueurs horizontales spatio-temporelles
    
    # Calculer les propriétés des cellules aux deux instants
    _, _, centroids_x_n, centroids_y_n, cell_types_n = fluid_cell_properties(mesh, front_n)
    _, _, centroids_x_np1, centroids_y_np1, cell_types_np1 = fluid_cell_properties(mesh, front_np1)
    
    # Calculer les capacités Bx et By aux deux instants
    Wx_n, Wy_n, Bx_n, By_n = compute_second_type_capacities(mesh, front_n, centroids_x_n, centroids_y_n)
    Wx_np1, Wy_np1, Bx_np1, By_np1 = compute_second_type_capacities(mesh, front_np1, centroids_x_np1, centroids_y_np1)
    
    # Pour chaque cellule, calculer les capacités spatio-temporelles
    for i in 1:nx
        for j in 1:ny
            # Récupérer le type de cellule aux deux instants
            cell_type_n = cell_types_n[i, j]
            cell_type_np1 = cell_types_np1[i, j]
            
            # Récupérer les valeurs aux deux instants
            bx_n = Bx_n[i, j]
            bx_np1 = Bx_np1[i, j]
            by_n = By_n[i, j]
            by_np1 = By_np1[i, j]
            
            # Appliquer une logique similaire à celle utilisée pour les surfaces
            # Pour Bx (longueurs verticales)
            if cell_type_n == 0 && cell_type_np1 == 0
                # empty → empty : aucun fluide
                Bx_st[i, j] = 0.0
            elseif cell_type_n == 1 && cell_type_np1 == 1
                # full → full : cellule toujours fluide
                Bx_st[i, j] = dt * (y_nodes[j+1] - y_nodes[j])
            elseif cell_type_n == 0 && cell_type_np1 == -1
                # empty → cut : cas spécial
                Bx_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ec")
            elseif cell_type_n == -1 && cell_type_np1 == 0
                # cut → empty : cas spécial
                Bx_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ce")
            elseif cell_type_n == 0 && cell_type_np1 == 1
                # empty → full : cas spécial
                Bx_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "ef")
            elseif cell_type_n == 1 && cell_type_np1 == 0
                # full → empty : cas spécial
                Bx_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "x", "fe")
            else
                # Autres cas (cut → cut, cut → full, full → cut) : règle trapézoïdale
                Bx_st[i, j] = (dt / 2.0) * (bx_n + bx_np1)
            end
            
            # Pour By (longueurs horizontales)
            if cell_type_n == 0 && cell_type_np1 == 0
                # empty → empty : aucun fluide
                By_st[i, j] = 0.0
            elseif cell_type_n == 1 && cell_type_np1 == 1
                # full → full : cellule toujours fluide
                By_st[i, j] = dt * (x_nodes[i+1] - x_nodes[i])
            elseif cell_type_n == 0 && cell_type_np1 == -1
                # empty → cut : cas spécial
                By_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ec")
            elseif cell_type_n == -1 && cell_type_np1 == 0
                # cut → empty : cas spécial
                By_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ce")
            elseif cell_type_n == 0 && cell_type_np1 == 1
                # empty → full : cas spécial
                By_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "ef")
            elseif cell_type_n == 1 && cell_type_np1 == 0
                # full → empty : cas spécial
                By_st[i, j] = compute_special_centerline_spacetime(mesh, front_n, front_np1, i, j, dt, "y", "fe")
            else
                # Autres cas (cut → cut, cut → full, full → cut) : règle trapézoïdale
                By_st[i, j] = (dt / 2.0) * (by_n + by_np1)
            end
        end
    end
    
    return Bx_st, By_st
end

"""
    compute_special_centerline_spacetime(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                        i::Int, j::Int, dt::Float64, direction::String, transition_type::String)

Calcule la capacité de ligne centrale spatio-temporelle pour les cas spéciaux de transition.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la cellule
- `dt::Float64`: Le pas de temps
- `direction::String`: "x" pour les lignes verticales (Bx), "y" pour les lignes horizontales (By)
- `transition_type::String`: Type de transition ("ec", "ce", "ef", "fe")

# Retourne
- `length::Float64`: La capacité de ligne centrale spatio-temporelle
"""
function compute_special_centerline_spacetime(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                             i::Int, j::Int, dt::Float64, direction::String, transition_type::String)
    # Extraction des coordonnées des noeuds
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    
    # Déterminer les temps caractéristiques de changement
    # Utiliser les mêmes temps de croisement que pour les volumes
    tau = find_crossing_times(mesh, front_n, front_np1, i, j, dt)
    
    # Si aucun temps de croisement intermédiaire n'est trouvé, 
    # utiliser la méthode trapézoïdale standard
    if length(tau) <= 2  # Seulement t^n et t^{n+1}
        centerline_n = compute_centerline_at_time(mesh, front_n, front_np1, i, j, 0.0, dt, direction)
        centerline_np1 = compute_centerline_at_time(mesh, front_n, front_np1, i, j, dt, dt, direction)
        
        # Ajuster pour les transitions spécifiques
        if transition_type == "ec" || transition_type == "ef"  # empty → cut/full
            # La valeur initiale est 0, favoriser la valeur finale
            return (dt / 3.0) * centerline_np1
        elseif transition_type == "ce" || transition_type == "fe"  # cut/full → empty
            # La valeur finale est 0, favoriser la valeur initiale
            return (dt / 3.0) * centerline_n
        else
            # Cas général
            return (dt / 2.0) * (centerline_n + centerline_np1)
        end
    end
    
    # Sinon, intégrer sur chaque sous-intervalle [τₖ, τₖ₊₁]
    centerline_length = 0.0
    for k in 1:(length(tau)-1)
        t_k = tau[k]
        t_kp1 = tau[k+1]
        
        # Calculer les longueurs aux extrémités de l'intervalle
        length_k = compute_centerline_at_time(mesh, front_n, front_np1, i, j, t_k, dt, direction)
        length_kp1 = compute_centerline_at_time(mesh, front_n, front_np1, i, j, t_kp1, dt, direction)
        
        # Utiliser la règle trapézoïdale pour ce sous-intervalle
        centerline_length += ((t_kp1 - t_k) / 2.0) * (length_k + length_kp1)
    end
    
    return centerline_length
end

"""
    compute_centerline_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                              i::Int, j::Int, t::Float64, dt::Float64, direction::String)

Calcule la longueur de ligne centrale pour la cellule (i,j) à l'instant t = t_n + t.

# Arguments
- `mesh::Mesh{2}`: Le maillage spatial
- `front_n::FrontTracker`: La position de l'interface au temps t^n
- `front_np1::FrontTracker`: La position de l'interface au temps t^{n+1}
- `i::Int`, `j::Int`: Les indices de la cellule
- `t::Float64`: L'instant auquel calculer la longueur (relatif à t^n)
- `dt::Float64`: Le pas de temps total
- `direction::String`: "x" pour les lignes verticales (Bx), "y" pour les lignes horizontales (By)

# Retourne
- `length::Float64`: La longueur de ligne centrale à l'instant t
"""
function compute_centerline_at_time(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker,
                                   i::Int, j::Int, t::Float64, dt::Float64, direction::String)
    # Si t est exactement t^n ou t^{n+1}, utiliser directement les valeurs calculées
    if isapprox(t, 0.0, atol=1e-10)
        _, _, centroids_x_n, centroids_y_n, _ = fluid_cell_properties(mesh, front_n)
        _, _, Bx_n, By_n = compute_second_type_capacities(mesh, front_n, centroids_x_n, centroids_y_n)
        return direction == "x" ? Bx_n[i, j] : By_n[i, j]
    elseif isapprox(t, dt, atol=1e-10)
        _, _, centroids_x_np1, centroids_y_np1, _ = fluid_cell_properties(mesh, front_np1)
        _, _, Bx_np1, By_np1 = compute_second_type_capacities(mesh, front_np1, centroids_x_np1, centroids_y_np1)
        return direction == "x" ? Bx_np1[i, j] : By_np1[i, j]
    end
    
    # Sinon, interpoler l'interface et calculer les capacités
    t_ratio = t / dt
    front_t = interpolate_front(front_n, front_np1, t_ratio)
    
    # Calculer les propriétés à cet instant t
    _, _, centroids_x_t, centroids_y_t, _ = fluid_cell_properties(mesh, front_t)
    _, _, Bx_t, By_t = compute_second_type_capacities(mesh, front_t, centroids_x_t, centroids_y_t)
    
    return direction == "x" ? Bx_t[i, j] : By_t[i, j]
end