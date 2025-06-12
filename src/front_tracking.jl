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
    create_crystal!(ft::FrontTracker, center_x::Float64, center_y::Float64, 
                   base_radius::Float64, n_lobes::Int=6, amplitude::Float64=0.2, 
                   n_markers::Int=100)

Crée une interface en forme de cristal (un cercle avec perturbation angulaire).
- center_x, center_y: coordonnées du centre
- base_radius: rayon de base
- n_lobes: nombre de lobes du cristal (6 pour symétrie hexagonale)
- amplitude: amplitude de la perturbation (0-1)
- n_markers: nombre de points sur l'interface
"""
function create_crystal!(ft::FrontTracker, center_x::Float64, center_y::Float64, 
                        base_radius::Float64, n_lobes::Int=6, amplitude::Float64=0.2, 
                        n_markers::Int=100)
    # Créer un vecteur avec le type approprié
    markers = Vector{Tuple{Float64, Float64}}(undef, n_markers+1)
    
    for i in 1:n_markers
        # Angle pour ce marqueur
        θ = 2.0 * π * (i-1) / n_markers
        
        # Rayon perturbé pour un effet cristallin
        r = base_radius * (1.0 + amplitude * cos(n_lobes * θ))
        
        # Coordonnées cartésiennes
        x = center_x + r * cos(θ)
        y = center_y + r * sin(θ)
        
        markers[i] = (x, y)
    end
    
    # Fermer la courbe en répétant le premier point
    markers[n_markers+1] = markers[1]
    
    # Définir les marqueurs dans le FrontTracker
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
                push!(volume_jacobian[(i, j)], (marker_idx-1, jacobian_value))
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
            push!(volume_jacobian[max_jac_cell], (marker_idx-1, max_jac_value))
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
                if marker_idx == 0
                    # Copy the first marker's entry to the last marker
                    push!(volume_jacobian[(i, j)], (last_marker_idx, jacobian_value))
                    push!(markers_with_entries, last_marker_idx+1)
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

# Compute Space-Time Capacities
"""
    compute_spacetime_capacities(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)

Calculates the space-time surface capacities for a 2D mesh between two time steps
using a polygon-based approach similar to the 1D method.
"""
function compute_spacetime_capacities(mesh::Mesh{2}, front_n::FrontTracker, front_np1::FrontTracker, dt::Float64)
    # Extract mesh information
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Initialize capacity arrays
    Ax_spacetime = zeros(nx+1, ny+1)     # Space-time vertical face capacities
    Ay_spacetime = zeros(nx+1, ny+1)     # Space-time horizontal face capacities
    
    # Arrays to store face classifications
    face_types_x = fill(:unknown, (nx+1, ny))
    face_types_y = fill(:unknown, (nx, ny+1))
    t_crosses_x = zeros(nx+1, ny)
    t_crosses_y = zeros(nx, ny+1)
    
    # 1. Calculate Ax capacities (vertical faces)
    for i in 1:nx+1
        for j in 1:ny
            x_face = x_nodes[i]
            y_min, y_max = y_nodes[j], y_nodes[j+1]
            
            # Check if vertices are inside fluid at time n and n+1
            bottom_n = is_point_inside(front_n, x_face, y_min)
            bottom_np1 = is_point_inside(front_np1, x_face, y_min)
            top_n = is_point_inside(front_n, x_face, y_max)
            top_np1 = is_point_inside(front_np1, x_face, y_max)
            
            # Classify the bottom and top edges
            bottom_edge_type = classify_edge_type(bottom_n, bottom_np1)
            top_edge_type = classify_edge_type(top_n, top_np1)
            
            # Calculate crossing times
            t_cross_bottom = calculate_crossing_time(bottom_edge_type, front_n, front_np1, x_face, y_min, dt)
            t_cross_top = calculate_crossing_time(top_edge_type, front_n, front_np1, x_face, y_max, dt)
            
            t_crosses_x[i, j] = (bottom_n != bottom_np1) ? t_cross_bottom : 
                               (top_n != top_np1 ? t_cross_top : 0.5*dt)
            
            # Calculate case ID based on edge types (similar to 1D approach)
            case_id = 0
            
            # Bottom edge contribution
            if bottom_edge_type == :empty
                case_id += 0      # 0
            elseif bottom_edge_type == :dead
                case_id += 1      # 1
            elseif bottom_edge_type == :fresh
                case_id += 2      # 2
            elseif bottom_edge_type == :full
                case_id += 3      # 3
            end
            
            # Top edge contribution
            if top_edge_type == :empty
                case_id += 0      # +0
            elseif top_edge_type == :dead
                case_id += 4      # +4
            elseif top_edge_type == :fresh
                case_id += 8      # +8
            elseif top_edge_type == :full
                case_id += 12     # +12
            end
            
            # Calculate surface capacity using LibGEOS polygon approach
            try
                area, _ = calculate_spacetime_geometry_libgeos2(
                    case_id, y_min, y_max, dt, t_cross_bottom, t_cross_top
                )
                Ax_spacetime[i, j] = area
                face_types_x[i, j] = get_face_type(case_id)
            catch e
                # Fallback to a simple calculation if the polygon approach fails
                println("Warning: Error calculating Ax at ($i,$j): ", e)
                wet_fraction = (Int(bottom_n) + Int(bottom_np1) + Int(top_n) + Int(top_np1)) / 4.0
                Ax_spacetime[i, j] = wet_fraction * (y_max - y_min) * dt
            end
            
            # Safety bounds check
            Ax_spacetime[i, j] = clamp(Ax_spacetime[i, j], 0.0, (y_max - y_min) * dt)
        end
    end
    
    # 2. Calculate Ay capacities (horizontal faces)
    for i in 1:nx
        for j in 1:ny+1
            y_face = y_nodes[j]
            x_min, x_max = x_nodes[i], x_nodes[i+1]
            
            # Check if vertices are inside fluid at time n and n+1
            left_n = is_point_inside(front_n, x_min, y_face)
            left_np1 = is_point_inside(front_np1, x_min, y_face)
            right_n = is_point_inside(front_n, x_max, y_face)
            right_np1 = is_point_inside(front_np1, x_max, y_face)
            
            # Classify the left and right edges
            left_edge_type = classify_edge_type(left_n, left_np1)
            right_edge_type = classify_edge_type(right_n, right_np1)
            
            # Calculate crossing times
            t_cross_left = calculate_crossing_time(left_edge_type, front_n, front_np1, x_min, y_face, dt)
            t_cross_right = calculate_crossing_time(right_edge_type, front_n, front_np1, x_max, y_face, dt)
            
            t_crosses_y[i, j] = (left_n != left_np1) ? t_cross_left : 
                               (right_n != right_np1 ? t_cross_right : 0.5*dt)
            
            # Calculate case ID based on edge types
            case_id = 0
            
            # Left edge contribution
            if left_edge_type == :empty
                case_id += 0      # 0
            elseif left_edge_type == :dead
                case_id += 1      # 1
            elseif left_edge_type == :fresh
                case_id += 2      # 2
            elseif left_edge_type == :full
                case_id += 3      # 3
            end

            # Right edge contribution
            if right_edge_type == :empty
                case_id += 0      # +0
            elseif right_edge_type == :dead
                case_id += 4      # +4
            elseif right_edge_type == :fresh
                case_id += 8      # +8
            elseif right_edge_type == :full
                case_id += 12     # +12
            end
            
            # Calculate surface capacity using LibGEOS polygon approach
            try
                area, _ = calculate_spacetime_geometry_libgeos2(
                    case_id, x_min, x_max, dt, t_cross_left, t_cross_right
                )
                Ay_spacetime[i, j] = area
                face_types_y[i, j] = get_face_type(case_id)
            catch e
                # Fallback to a simple calculation if the polygon approach fails
                println("Warning: Error calculating Ay at ($i,$j): ", e)
                wet_fraction = (Int(left_n) + Int(left_np1) + Int(right_n) + Int(right_np1)) / 4.0
                Ay_spacetime[i, j] = wet_fraction * (x_max - x_min) * dt
            end
            
            # Safety bounds check
            Ay_spacetime[i, j] = clamp(Ay_spacetime[i, j], 0.0, (x_max - x_min) * dt)
        end
    end
    
    return Dict(
        :Ax_spacetime => Ax_spacetime,
        :Ay_spacetime => Ay_spacetime,
        :face_types_x => face_types_x,
        :face_types_y => face_types_y,
        :t_crosses_x => t_crosses_x,
        :t_crosses_y => t_crosses_y
    )
end

"""
    calculate_spacetime_geometry_libgeos(case_id::Int, p_min::Float64, p_max::Float64, 
                                       dt::Float64, t_left::Float64, t_right::Float64)

Calculate the space-time volume and centroid using LibGEOS library for 2D faces.
Works for both Ax and Ay. This is adapted from the 1D implementation.
"""
function calculate_spacetime_geometry_libgeos2(case_id::Int, p_min::Float64, p_max::Float64, 
                                           dt::Float64, t_left::Float64, t_right::Float64)
    # Full dimensions
    dp = p_max - p_min
    full_area = dp * dt
    
    # Cases with 0% fluid
    if case_id == 0
        # Default centroid for empty cells is cell center
        return 0.0, [p_min + dp/2, dt/2]
    end
    
    # Cases with 100% fluid
    if case_id == 15
        # For full cells, centroid is cell center
        return full_area, [p_min + dp/2, dt/2]
    end
    
    # Get polygon coordinates for this case
    coords = get_spacetime_polygon_coords2(case_id, p_min, p_max, dt, t_left, t_right)
    
    # Create LibGEOS polygon
    polygon = LibGEOS.Polygon([coords])
    
    # Calculate area
    area = LibGEOS.area(polygon)
    
    # Calculate centroid
    centroid_geom = LibGEOS.centroid(polygon)
    centroid_coords = [LibGEOS.getcoord(centroid_geom, i) for i in 1:2]
    
    # Return both results
    return area, centroid_coords
end

"""
    classify_edge_type(is_wet_n::Bool, is_wet_np1::Bool)

Classify an edge based on its state at time n and n+1.
Returns :empty, :full, :fresh, or :dead.
"""
function classify_edge_type(is_wet_n::Bool, is_wet_np1::Bool)
    if !is_wet_n && !is_wet_np1
        return :empty      # Dry at both times
    elseif is_wet_n && is_wet_np1
        return :full       # Wet at both times
    elseif !is_wet_n && is_wet_np1
        return :fresh      # Becomes wet (dry -> wet)
    else  # is_wet_n && !is_wet_np1
        return :dead       # Becomes dry (wet -> dry)
    end
end

"""
    calculate_crossing_time(edge_type::Symbol, front_n::FrontTracker, front_np1::FrontTracker, 
                          x::Float64, y::Float64, dt::Float64)

Calculate the crossing time for an edge based on its type.
"""
function calculate_crossing_time(edge_type::Symbol, front_n::FrontTracker, front_np1::FrontTracker, 
                               x::Float64, y::Float64, dt::Float64)
    if edge_type == :empty || edge_type == :full
        return dt/2  # No crossing, placeholder value
    else
        # For fresh or dead edges, calculate the crossing time
        return find_crossing_time(front_n, front_np1, x, y, dt)
    end
end

"""
    get_face_type(case_id::Int)

Convert a numerical case ID to a symbolic face type.
"""
function get_face_type(case_id::Int)
    if case_id == 0
        return :empty
    elseif case_id == 15
        return :full
    elseif case_id in [2, 6, 8, 10]
        return :fresh
    elseif case_id in [1, 4, 5, 9]
        return :dead
    else
        return :complex
    end
end

"""
    get_spacetime_polygon_coords(case_id::Int, p_min::Float64, p_max::Float64, 
                               dt::Float64, t_left::Float64, t_right::Float64)

Creates polygon coordinates for each space-time case.
This function is adapted from the 1D implementation.
"""
function get_spacetime_polygon_coords2(case_id::Int, p_min::Float64, p_max::Float64, 
                                    dt::Float64, t_left::Float64, t_right::Float64)
    # Case 0: Empty-Empty (completely empty)
    if case_id == 0
        # Better degenerate polygon for empty case
        return [[p_min, 0.0], [p_min+1e-6, 0.0], [p_min+1e-6, 1e-6], [p_min, 1e-6], [p_min, 0.0]]
    end
    
    # Case 15: Full-Full (completely full)
    if case_id == 15
        # Full rectangle in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, dt], [p_min, dt], [p_min, 0.0]]
    end
    
    # Case 1: Dead-Empty (bottom-left only)
    if case_id == 1
        # Triangle in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_min, t_left], [p_min, 0.0]]
    end
    
    # Case 2: Fresh-Empty (top-left only)
    if case_id == 2
        # Triangle in counter-clockwise direction
        return [[p_min, t_left], [p_min, dt], [p_max, dt], [p_min, t_left]]
    end
    
    # Case 3: Full-Empty (left edge is full)
    if case_id == 3
        # Full rectangle in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, dt], [p_min, dt], [p_min, 0.0]]
    end
    
    # Case 4: Empty-Dead (bottom-right only)
    if case_id == 4
        # Triangle in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, t_right], [p_min, 0.0]]
    end
    
    # Case 5: Dead-Dead (bottom edge is full)
    if case_id == 5
        # Quadrilateral in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, t_right], [p_min, t_left], [p_min, 0.0]]
    end
    
    # Case 6: Fresh-Dead (complex case: bottom-right and top-left)
    if case_id == 6
        # This is a challenging shape - ensure it's correctly ordered
        # It should be a non-convex quadrilateral
        return [[p_min, t_left], [p_min, dt], [p_max, t_right], [p_max, 0.0], [p_min, t_left]]
    end
    
    # Case 7: Full-Dead (all except top-right)
    if case_id == 7
        # Quadrilateral in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, t_right], [p_min, dt], [p_min, 0.0]]
    end
    
    # Case 8: Empty-Fresh (top-right only)
    if case_id == 8
        # Triangle in counter-clockwise direction
        return [[p_min, dt], [p_max, dt], [p_max, t_right], [p_min, dt]]
    end
    
    # Case 9: Dead-Fresh (complex case: bottom-left and top-right)
    if case_id == 9
        # Non-convex quadrilateral - ensure correct ordering
        return [[p_min, 0.0], [p_min, t_left], [p_max, dt], [p_max, t_right], [p_min, 0.0]]
    end
    
    # Case 10: Fresh-Fresh (top edge is full)
    if case_id == 10
        # Quadrilateral in counter-clockwise direction
        return [[p_min, t_left], [p_max, t_right], [p_max, dt], [p_min, dt], [p_min, t_left]]
    end
    
    # Case 11: Full-Fresh (all except bottom-right)
    if case_id == 11
        # Quadrilateral in counter-clockwise direction
        return [[p_min, 0.0], [p_max, t_right], [p_max, dt], [p_min, dt], [p_min, 0.0]]
    end
    
    # Case 12: Empty-Full (right edge is full)
    if case_id == 12
        # Fixed: Ensure counter-clockwise orientation
        return [[p_min, 0.0], [p_min, dt], [p_max, dt], [p_max, 0.0], [p_min, 0.0]]
    end
    
    # Case 13: Dead-Full (all except top-left)
    if case_id == 13
        # Quadrilateral in counter-clockwise direction
        return [[p_min, 0.0], [p_max, 0.0], [p_max, dt], [p_min, t_left], [p_min, 0.0]]
    end
    
    # Case 14: Fresh-Full (all except bottom-left)
    if case_id == 14
        # Quadrilateral in counter-clockwise direction
        return [[p_min, t_left], [p_max, 0.0], [p_max, dt], [p_min, dt], [p_min, t_left]]
    end
    
    # Fallback for unexpected cases - full cell with warning
    println("Warning: Unexpected case_id: $case_id, using default full cell")
    return [[p_min, 0.0], [p_max, 0.0], [p_max, dt], [p_min, dt], [p_min, 0.0]]
end

"""
    find_crossing_time(front_n::FrontTracker, front_np1::FrontTracker, x::Float64, y::Float64, dt::Float64)

Estimates the time when the interface crosses a specific point (x,y) between two time steps.
Uses linear interpolation of the signed distance function between time steps.

Parameters:
- front_n: Front tracker at time t_n
- front_np1: Front tracker at time t_n+1
- x, y: Coordinates of the point to check
- dt: Time step size

Returns:
- t_cross: Estimated crossing time within [0, dt] interval
"""
function find_crossing_time(front_n::FrontTracker, front_np1::FrontTracker, x::Float64, y::Float64, dt::Float64)
    # Get the signed distance at both time steps
    sdf_n = sdf(front_n, x, y)
    sdf_np1 = sdf(front_np1, x, y)
    
    # Check if the interface actually crosses this point
    if (sdf_n * sdf_np1 > 0)
        # No crossing (same sign at both time steps)
        if sdf_n < 0
            # Inside fluid at both times
            return dt/2
        else
            # Outside fluid at both times
            return dt/2
        end
    end
    
    # If one SDF is zero, return that time
    if abs(sdf_n) < 1e-10
        return 0.0
    elseif abs(sdf_np1) < 1e-10
        return dt
    end
    
    # Linear interpolation to find crossing time:
    t_cross = dt * abs(sdf_n) / (abs(sdf_n) + abs(sdf_np1))
    
    # Ensure the result is within [0, dt]
    return clamp(t_cross, 0.0, dt)
end







"""
    compute_segment_parameters(ft::FrontTracker)

Calcule les paramètres de chaque segment de l'interface:
- n_I: vecteur normal unitaire du segment
- α_I: intercept du segment (distance signée à l'origine)
- length_I: longueur du segment
- midpoint_I: point milieu du segment

L'équation d'un segment est: n_I ⋅ x = α_I

Retourne (segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints)
"""
function compute_segment_parameters(ft::FrontTracker)
    markers = ft.markers
    n_markers = length(markers)
    
    if n_markers < 2
        return [], [], [], [], []
    end
    
    # Nombre de segments (n_markers pour un contour fermé, n_markers-1 pour un contour ouvert)
    n_segments = if ft.is_closed
        if n_markers > 0 && markers[1] == markers[end]
            n_markers - 1
        else
            n_markers
        end
    else
        n_markers - 1
    end
    
    # Initialiser les tableaux de résultats
    segments = Vector{Tuple{Int, Int}}(undef, n_segments)
    segment_normals = Vector{Tuple{Float64, Float64}}(undef, n_segments)
    segment_intercepts = Vector{Float64}(undef, n_segments)
    segment_lengths = Vector{Float64}(undef, n_segments)
    segment_midpoints = Vector{Tuple{Float64, Float64}}(undef, n_segments)
    
    # Parcourir tous les segments
    for i in 1:n_segments
        # Indice du marqueur suivant (avec gestion de la boucle pour contour fermé)
        next_i = i < n_markers ? i + 1 : 1
        
        # Points de début et de fin du segment
        p1 = markers[i]
        p2 = markers[next_i]
        
        # Vecteur du segment (de p1 à p2)
        segment_vector = (p2[1] - p1[1], p2[2] - p1[2])
        
        # Longueur du segment
        segment_length = sqrt(segment_vector[1]^2 + segment_vector[2]^2)
        
        if segment_length < 1e-15
            # Éviter la division par zéro pour les segments très courts
            segment_normals[i] = (0.0, 1.0)  # Normal arbitraire
            segment_intercepts[i] = p1[1] * 0.0 + p1[2] * 1.0  # α_I = n_I ⋅ p1
            segment_lengths[i] = 0.0
            segment_midpoints[i] = p1  # Point de milieu = point de début pour segments courts
        else
            # Normale unitaire (rotation de 90° dans le sens trigonométrique)
            normal = (-segment_vector[2] / segment_length, segment_vector[1] / segment_length)
            
            # Vérifier l'orientation de la normale (doit pointer à l'extérieur)
            if ft.is_closed
                # Pour un contour fermé, vérifier si la normale pointe vers l'extérieur
                test_point = (p1[1] + 1e-3 * normal[1], p1[2] + 1e-3 * normal[2])
                if is_point_inside(ft, test_point[1], test_point[2])
                    # Si le point test est à l'intérieur, inverser la normale
                    normal = (-normal[1], -normal[2])
                end
            end
            
            # Calcul de l'intercept α_I = n_I ⋅ p1
            intercept = normal[1] * p1[1] + normal[2] * p1[2]
            
            # Stockage des résultats
            segments[i] = (i, next_i)
            segment_normals[i] = normal
            segment_intercepts[i] = intercept
            segment_lengths[i] = segment_length
            segment_midpoints[i] = ((p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
        end
    end
    
    return segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints
end

"""
    create_segment_line(ft::FrontTracker, segment_idx::Int)

Crée une LineString représentant un segment de l'interface à partir des marqueurs.
"""
function create_segment_line(ft::FrontTracker, segment_idx::Int)
    markers = ft.markers
    n_markers = length(markers)
    
    if segment_idx < 1 || segment_idx > (ft.is_closed ? n_markers : n_markers - 1)
        error("Indice de segment invalide: $segment_idx")
    end
    
    # Récupérer les indices des marqueurs qui définissent le segment
    next_idx = segment_idx < n_markers ? segment_idx + 1 : 1
    
    # Récupérer les coordonnées des marqueurs
    start_point = markers[segment_idx]
    end_point = markers[next_idx]
    
    # Créer la LineString
    return LibGEOS.LineString([[start_point[1], start_point[2]], [end_point[1], end_point[2]]])
end

"""
    compute_segment_cell_intersections(mesh::Mesh{2}, ft::FrontTracker)

Calcule les intersections entre les segments de l'interface et les cellules du maillage.
Retourne un dictionnaire où les clés sont les indices de cellules (i,j) et les valeurs
sont des listes de tuples (segment_idx, intersection_length).
"""
function compute_segment_cell_intersections(mesh::Mesh{2}, ft::FrontTracker)
    # Calculer les paramètres des segments
    segments, segment_normals, segment_intercepts, segment_lengths, segment_midpoints = 
        compute_segment_parameters(ft)
    
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Dictionnaire pour stocker les intersections segment-cellule
    cell_segment_intersections = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    
    # Initialiser le dictionnaire pour toutes les cellules
    for i in 1:nx
        for j in 1:ny
            cell_segment_intersections[(i,j)] = []
        end
    end
    
    # Pour chaque segment, calculer les intersections avec les cellules
    n_segments = length(segments)
    for segment_idx in 1:n_segments
        # Créer une ligne représentant le segment
        segment_line = create_segment_line(ft, segment_idx)
        
        # Calculer les intersections avec toutes les cellules
        for i in 1:nx
            for j in 1:ny
                # Créer le polygone de la cellule
                cell_coords = [
                    [x_nodes[i], y_nodes[j]],
                    [x_nodes[i+1], y_nodes[j]],
                    [x_nodes[i+1], y_nodes[j+1]],
                    [x_nodes[i], y_nodes[j+1]],
                    [x_nodes[i], y_nodes[j]]
                ]
                cell_poly = LibGEOS.Polygon([cell_coords])
                
                # Vérifier l'intersection
                if LibGEOS.intersects(cell_poly, segment_line)
                    intersection = LibGEOS.intersection(cell_poly, segment_line)
                    
                    # Calculer la longueur d'intersection
                    if isa(intersection, LibGEOS.LineString)
                        intersection_length = LibGEOS.geomLength(intersection)
                        if intersection_length > 1e-10
                            push!(cell_segment_intersections[(i,j)], (segment_idx, intersection_length))
                        end
                    elseif isa(intersection, LibGEOS.MultiLineString)
                        total_length = 0.0
                        for k in 1:LibGEOS.getNumGeometries(intersection)
                            line = LibGEOS.getGeometry(intersection, k-1)
                            total_length += LibGEOS.geomLength(line)
                        end
                        if total_length > 1e-10
                            push!(cell_segment_intersections[(i,j)], (segment_idx, total_length))
                        end
                    end
                end
            end
        end
    end
    
    return cell_segment_intersections, segments, segment_normals, segment_intercepts, segment_lengths
end

"""
    compute_intercept_jacobian(mesh::Mesh{2}, ft::FrontTracker; density::Float64=1.0)

Calcule la jacobienne des volumes par rapport aux déplacements d'intercept.
Pour chaque cellule k=(i,j) et chaque segment I, J[k,I] = ∂V_k/∂δ_I = ρL × A_k,I,
où A_k,I est la longueur d'intersection du segment I avec la cellule k,
et ρL est un facteur physique (densité × latent heat)

Retourne:
- intercept_jacobian: Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}} - la jacobienne
- segments: vecteur des segments (i, j) où i, j sont les indices des marqueurs
- segment_normals: vecteur des normales unitaires pour chaque segment
- segment_intercepts: vecteur des intercepts initiaux pour chaque segment
- segment_lengths: vecteur des longueurs de chaque segment
"""
function compute_intercept_jacobian(mesh::Mesh{2}, ft::FrontTracker; density::Float64=1.0)
    # Calculer les intersections segment-cellule
    cell_segment_intersections, segments, segment_normals, segment_intercepts, segment_lengths = 
        compute_segment_cell_intersections(mesh, ft)
    
    # Récupérer les dimensions du maillage
    x_nodes = mesh.nodes[1]
    y_nodes = mesh.nodes[2]
    nx = length(x_nodes) - 1
    ny = length(y_nodes) - 1
    
    # Créer un dictionnaire pour stocker la jacobienne
    # Format: Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}
    # Clé: (i,j) = indice de la cellule, Valeur: liste de (segment_idx, jacobian_value)
    intercept_jacobian = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()
    
    # Pour chaque cellule, calculer sa contribution à la jacobienne
    for i in 1:nx
        for j in 1:ny
            intercept_jacobian[(i,j)] = []
            
            # Parcourir tous les segments qui intersectent cette cellule
            for (segment_idx, intersection_length) in cell_segment_intersections[(i,j)]
                # J[k,I] = ρL × A_k,I
                jacobian_value = density * intersection_length
                
                # Stocker la valeur dans la jacobienne
                push!(intercept_jacobian[(i,j)], (segment_idx, jacobian_value))
            end
        end
    end
    
    return intercept_jacobian, segments, segment_normals, segment_intercepts, segment_lengths
end

"""
    update_front_with_intercept_displacements!(ft::FrontTracker, displacements::AbstractVector{<:Real}, 
                                            segment_normals::Vector{Tuple{Float64, Float64}},
                                            segment_lengths::Vector{Float64})

Met à jour l'interface en déplaçant chaque segment selon ses déplacements d'intercept.
Applique une pondération basée sur la longueur des segments pour distribuer les déplacements
aux marqueurs partagés entre plusieurs segments.

Paramètres:
- ft: l'objet FrontTracker à mettre à jour
- displacements: vecteur des déplacements δ_I pour chaque segment
- segment_normals: vecteur des normales unitaires pour chaque segment
- segment_lengths: vecteur des longueurs de chaque segment

Retourne le FrontTracker mis à jour.
"""
function update_front_with_intercept_displacements!(ft::FrontTracker, displacements::AbstractVector{<:Real}, 
                                                  segment_normals::Vector{Tuple{Float64, Float64}},
                                                  segment_lengths::Vector{Float64})
    markers = copy(ft.markers)
    n_markers = length(markers)
    n_segments = length(displacements)
    
  
    
    # Structure pour stocker les contributions pondérées de chaque segment à chaque marqueur
    segment_contributions = Dict{Int, Vector{Tuple{Float64, Tuple{Float64, Float64}}}}()
    for i in 1:n_markers
        segment_contributions[i] = []
    end
    
    # Pour chaque segment, calculer sa contribution aux marqueurs
    for (s_idx, displacement) in enumerate(displacements)
        # Récupérer les indices des marqueurs aux extrémités du segment
        start_idx = s_idx
        end_idx = s_idx < n_markers ? s_idx + 1 : 1
        
        # Calculer le vecteur de déplacement
        normal = segment_normals[s_idx]
        vector_displacement = (displacement * normal[1], displacement * normal[2])
        
        # Utiliser la longueur du segment comme poids
        segment_weight = max(segment_lengths[s_idx], 1e-10)  # Éviter division par zéro
        
        # Enregistrer la contribution pondérée de ce segment pour chaque marqueur
        push!(segment_contributions[start_idx], (segment_weight, vector_displacement))
        push!(segment_contributions[end_idx], (segment_weight, vector_displacement))
    end
    
    # Calculer et appliquer le déplacement moyen pondéré pour chaque marqueur
    for i in 1:n_markers
        contributions = segment_contributions[i]
        if !isempty(contributions)
            # Calculer la somme des poids
            total_weight = sum(contrib[1] for contrib in contributions)
            
            if total_weight > 0
                # Calculer le déplacement moyen pondéré
                avg_dx = sum(contrib[1] * contrib[2][1] for contrib in contributions) / total_weight
                avg_dy = sum(contrib[1] * contrib[2][2] for contrib in contributions) / total_weight
                
                # Appliquer le déplacement
                markers[i] = (markers[i][1] + avg_dx, markers[i][2] + avg_dy)
            end
        end
    end
    
    # Mettre à jour l'interface avec les nouveaux marqueurs
    set_markers!(ft, markers, ft.is_closed)
    
    return ft
end
