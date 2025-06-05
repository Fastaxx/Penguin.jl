

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