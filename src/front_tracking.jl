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
function compute_volume_jacobian(ft::FrontTracker, x_faces::AbstractVector{<:Real}, y_faces::AbstractVector{<:Real}, epsilon::Float64=1e-6)
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
    
    # Process each marker except the duplicated closing point
    n_markers = ft.is_closed ? length(markers) - 1 : length(markers)
    
    for marker_idx in 1:n_markers
        # Original marker position
        original_marker = markers[marker_idx]
        
        # Calculate perturbed position
        normal = normals[marker_idx]
        perturbed_marker = (
            original_marker[1] + epsilon * normal[1],
            original_marker[2] + epsilon * normal[2]
        )
        
        # Create a copy of markers with this one perturbed
        perturbed_markers = copy(markers)
        perturbed_markers[marker_idx] = perturbed_marker
        
        # Update last marker if interface is closed and first marker is perturbed
        if ft.is_closed && marker_idx == 1 && markers[1] == markers[end]
            perturbed_markers[end] = perturbed_marker
        end
        
        # Create a new front tracker with perturbed markers
        perturbed_tracker = FrontTracker(perturbed_markers, ft.is_closed)
        perturbed_fluid_poly = get_fluid_polygon(perturbed_tracker)
        
        # Calculate perturbed volumes
        for ((i, j), original_volume) in original_volumes
            # Create cell coordinates
            cell_coords = [
                [x_faces_vec[i], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j]],
                [x_faces_vec[i+1], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j+1]],
                [x_faces_vec[i], y_faces_vec[j]]  # Close the polygon
            ]
            
            # Create cell polygon properly using a vector of coordinate vectors
            cell_poly = LibGEOS.Polygon([cell_coords])
            
            # Calculate intersection with perturbed fluid polygon
            intersection = LibGEOS.intersection(cell_poly, perturbed_fluid_poly)
            
            # Calculate perturbed volume
            perturbed_volume = LibGEOS.isEmpty(intersection) ? 0.0 : LibGEOS.area(intersection)
            
            # Calculate Jacobian value
            jacobian_value = (perturbed_volume - original_volume) / epsilon
            
            # Store only significant changes
            if abs(jacobian_value) > 1e-10
                push!(volume_jacobian[(i, j)], (marker_idx, jacobian_value))
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
    volumes = zeros(nx+1, ny+1)   # Volume fluide
    centroids_x = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en x
    centroids_y = zeros(nx+1, ny+1)  # Coordonnées du centroïde fluide en y
    
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
                else
                    # Cellule entièrement solide
                    centroids_x[i, j] = (x_nodes[i] + x_nodes[i+1]) / 2
                    centroids_y[i, j] = (y_nodes[j] + y_nodes[j+1]) / 2
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
                else
                    centroids_x[i, j] = center_x
                    centroids_y[i, j] = center_y
                end
            end
        end
    end

    return fractions, volumes, centroids_x, centroids_y
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
    
    # Calcul pour les faces verticales
    for i in 1:nx+1
        for j in 1:ny
            x = x_nodes[i]
            y_min, y_max = y_nodes[j], y_nodes[j+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x, y_min], [x, y_max]])
            
            if LibGEOS.intersects(face_line, fluid_poly)
                intersection = LibGEOS.intersection(face_line, fluid_poly)
                
                Ax[i,j] = LibGEOS.geomLength(intersection)

            else
                # Face entièrement dans le fluide ou dans le solide
                mid_y = (y_min + y_max) / 2
                if is_point_inside(front, x, mid_y)
                    Ax[i, j] = y_max - y_min
                end
            end
        end
    end
    
    # Calcul pour les faces horizontales
    for i in 1:nx
        for j in 1:ny+1
            y = y_nodes[j]
            x_min, x_max = x_nodes[i], x_nodes[i+1]
            
            # Création d'une ligne pour la face
            face_line = LibGEOS.LineString([[x_min, y], [x_max, y]])
            
            if LibGEOS.intersects(face_line, fluid_poly)
                intersection = LibGEOS.intersection(face_line, fluid_poly)
                Ay[i, j] = LibGEOS.geomLength(intersection)

            else
                # Face entièrement dans le fluide ou dans le solide
                mid_x = (x_min + x_max) / 2
                if is_point_inside(front, mid_x, y)
                    Ay[i, j] = x_max - x_min
                end
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
    
    # Calcul des capacités de volume horizontales Wx
    for i in 1:nx-1
        for j in 1:ny
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
                Wx[i, j] = LibGEOS.area(intersection)
            else
                # Vérification si le milieu est dans le fluide
                mid_x = (x_left + x_right) / 2
                mid_y = (y_min + y_max) / 2
                if is_point_inside(front, mid_x, mid_y)
                    Wx[i, j] = LibGEOS.area(poly)
                end
            end
        end
    end
    
    # Calcul des capacités de volume verticales Wy
    for i in 1:nx
        for j in 1:ny-1
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
                Wy[i, j] = LibGEOS.area(intersection)
            else
                # Vérification si le milieu est dans le fluide
                mid_x = (x_min + x_max) / 2
                mid_y = (y_bottom + y_top) / 2
                if is_point_inside(front, mid_x, mid_y)
                    Wy[i, j] = LibGEOS.area(poly)
                end
            end
        end
    end
    
    # Calcul des longueurs B^x et B^y
    for i in 1:nx
        for j in 1:ny
            x_cm = centroids_x[i, j]
            y_cm = centroids_y[i, j]
            
            if x_cm != 0.0 || y_cm != 0.0  # Si la cellule a du fluide
                # Ligne verticale passant par le centroïde
                vertical_line = LibGEOS.LineString([
                    [x_cm, y_nodes[j]],
                    [x_cm, y_nodes[j+1]]
                ])
                
                # Ligne horizontale passant par le centroïde
                horizontal_line = LibGEOS.LineString([
                    [x_nodes[i], y_cm],
                    [x_nodes[i+1], y_cm]
                ])
                
                # Calcul des longueurs mouillées
                if LibGEOS.intersects(vertical_line, fluid_poly)
                    intersection = LibGEOS.intersection(vertical_line, fluid_poly)
                    Bx[i, j] = LibGEOS.geomLength(intersection)
                else
                    mid_y = (y_nodes[j] + y_nodes[j+1]) / 2
                    if is_point_inside(front, x_cm, mid_y)
                        Bx[i, j] = y_nodes[j+1] - y_nodes[j]
                    end
                end
                
                if LibGEOS.intersects(horizontal_line, fluid_poly)
                    intersection = LibGEOS.intersection(horizontal_line, fluid_poly)
                    By[i, j] = LibGEOS.geomLength(intersection)
                else
                    mid_x = (x_nodes[i] + x_nodes[i+1]) / 2
                    if is_point_inside(front, mid_x, y_cm)
                        By[i, j] = x_nodes[i+1] - x_nodes[i]
                    end
                end
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
    fractions, volumes, centroids_x, centroids_y = fluid_cell_properties(mesh, front)
    
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

