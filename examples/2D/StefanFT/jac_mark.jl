using Penguin
using LinearAlgebra
using CairoMakie

"""
    visualize_jacobian_matrix(volume_jacobian, mesh, markers, front)

Visualise la matrice jacobienne des volumes par rapport aux déplacements des marqueurs.

# Arguments
- `volume_jacobian::Dict` : Dictionnaire contenant les jacobiennes de volume
- `mesh::Mesh` : Le maillage contenant les informations sur les cellules
- `markers::Vector` : Liste des marqueurs de l'interface
- `front::FrontTracker` : L'objet de suivi d'interface

# Retourne
- `Figure` : La figure contenant la visualisation de la matrice jacobienne
"""
function visualize_jacobian_matrix(volume_jacobian, mesh, markers, front)
    # Extraire les dimensions nécessaires
    n_markers = length(markers) - (front.is_closed ? 1 : 0)  # Ne pas compter le marqueur dupliqué à la fin
    
    # Créer un index pour chaque cellule active (celles ayant des dérivées non nulles)
    active_cells = sort([(i, j) for (i, j) in keys(volume_jacobian) if !isempty(volume_jacobian[(i, j)])])
    n_active_cells = length(active_cells)
    
    if n_active_cells == 0
        println("Aucune cellule active trouvée avec des dérivées non nulles.")
        return nothing
    end
    
    # Créer une matrice pleine et la remplir avec les valeurs de la jacobienne
    J = zeros(n_active_cells, n_markers)
    
    for (cell_idx, (i, j)) in enumerate(active_cells)
        for (marker_idx, jac_value) in volume_jacobian[(i, j)]
            if 0 <= marker_idx < n_markers  # Vérifier que l'indice du marqueur est valide
                J[cell_idx, marker_idx+1] = jac_value  # +1 car Julia est 1-indexé
            end
        end
    end
    
    # Créer la figure
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
              xlabel = "Indice du marqueur",
              ylabel = "Indice de la cellule",
              title = "Matrice jacobienne des volumes")
    
    # Déterminer l'échelle de couleur symétrique
    max_val = maximum(abs.(J))
    
    # Créer la heatmap
    heatmap!(ax, 1:n_markers, 1:n_active_cells, J, 
            colormap = :RdBu, colorrange = (-max_val, max_val))
    
    # Ajouter une colorbar
    Colorbar(fig[1, 2], colormap = :RdBu, limits = (-max_val, max_val),
            label = "∂V/∂n (direction normale)")
    
    # Ajouter des annotations pour les indices des cellules
    cell_labels = ["($(i),$(j))" for (i, j) in active_cells]
    ax.yticks = (1:n_active_cells, cell_labels)
    
    # Ajouter des graduations pour les marqueurs
    ax.xticks = (1:n_markers, string.(0:n_markers-1))
    
    return fig
end

"""
    visualize_marker_influence(volume_jacobian, mesh, markers, front, marker_idx)

Visualise l'influence d'un marqueur spécifique sur les volumes des cellules.

# Arguments
- `volume_jacobian::Dict` : Dictionnaire contenant les jacobiennes de volume
- `mesh::Mesh` : Le maillage contenant les informations sur les cellules
- `markers::Vector` : Liste des marqueurs de l'interface
- `front::FrontTracker` : L'objet de suivi d'interface
- `marker_idx::Int` : L'indice du marqueur dont on veut visualiser l'influence (0-indexé)

# Retourne
- `Figure` : La figure contenant la visualisation de l'influence du marqueur
"""
function visualize_marker_influence(volume_jacobian, mesh, markers, front, marker_idx)
    # Extraire les dimensions du maillage
    nx, ny = size(mesh.centers[1])[1], size(mesh.centers[2])[1]
    
    # Créer une matrice pour la visualisation
    marker_impact = zeros(nx, ny)
    
    # Remplir la matrice avec les valeurs d'impact du marqueur sélectionné
    for ((i, j), jac_values) in volume_jacobian
        for (m_idx, jac_value) in jac_values
            if m_idx == marker_idx
                marker_impact[i, j] = jac_value
            end
        end
    end
    
    # Calculer les normales aux marqueurs
    normals = compute_marker_normals(front, markers)
    
    # Créer la figure
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1],
              xlabel = "x",
              ylabel = "y",
              title = "Cellules affectées par le marqueur $marker_idx",
              aspect = DataAspect())
    
    # Déterminer l'échelle de couleur symétrique
    max_val = maximum(abs.(marker_impact))
    if max_val == 0
        max_val = 1.0  # Valeur par défaut si pas d'impact
    end
    
    # Extraire les coordonnées des faces pour l'affichage
    x_faces = vcat(mesh.nodes[1][1], mesh.nodes[1][2:end])
    y_faces = vcat(mesh.nodes[2][1], mesh.nodes[2][2:end])
    
    # Créer la heatmap
    heatmap!(ax, x_faces, y_faces, marker_impact, 
            colormap = :RdBu, colorrange = (-max_val, max_val))
    
    # Ajouter une colorbar
    Colorbar(fig[1, 2], colormap = :RdBu, limits = (-max_val, max_val),
            label = "Taux de variation du volume")
    
    # Dessiner l'interface
    interface_x = [m[1] for m in markers]
    interface_y = [m[2] for m in markers]
    lines!(ax, interface_x, interface_y, color = :black, linewidth = 1)
    
    # Mettre en évidence le marqueur sélectionné (0-indexé mais 1-indexé pour l'accès)
    mx, my = markers[marker_idx+1]
    scatter!(ax, [mx], [my], color = :red, markersize = 12)
    
    # Dessiner le vecteur normal au marqueur
    nx, ny = normals[marker_idx+1]
    arrow_scale = (maximum(x_faces) - minimum(x_faces)) * 0.05
    arrows!(ax, [mx], [my], [nx * arrow_scale], [ny * arrow_scale],
           color = :red, arrowsize = 15)
    
    return fig
end

"""
    visualization_combined(volume_jacobian, mesh, front)

Crée une visualisation combinée de la matrice jacobienne et de l'influence d'un marqueur sélectionné.

# Arguments
- `volume_jacobian::Dict` : Dictionnaire contenant les jacobiennes de volume
- `mesh::Mesh` : Le maillage contenant les informations sur les cellules
- `front::FrontTracker` : L'objet de suivi d'interface

# Retourne
- `Nothing` : Affiche les figures et offre une interface interactive
"""
function visualization_combined(volume_jacobian, mesh, front)
    markers = get_markers(front)
    n_markers = length(markers) - (front.is_closed ? 1 : 0)
    
    # Visualiser la matrice jacobienne
    fig1 = visualize_jacobian_matrix(volume_jacobian, mesh, markers, front)
    display(fig1)
    save("jacobian_matrix.png", fig1)
    
    # Interface interactive pour sélectionner un marqueur
    println("\nDisponible : $(n_markers) marqueurs (0-$(n_markers-1))")
    
    while true
        print("Entrez l'indice du marqueur à visualiser (ou 'q' pour quitter) : ")
        marker_input = readline()
        
        if lowercase(marker_input) == "q"
            break
        end
        
        try
            marker_idx = parse(Int, marker_input)
            if 0 <= marker_idx < n_markers
                # Visualiser l'influence du marqueur sélectionné
                fig2 = visualize_marker_influence(volume_jacobian, mesh, markers, front, marker_idx)
                display(fig2)
                save("marker_$(marker_idx)_influence.png", fig2)
            else
                println("Erreur: l'indice doit être entre 0 et $(n_markers-1)")
            end
        catch e
            println("Erreur: veuillez entrer un nombre entier valide ou 'q' pour quitter")
        end
    end
    
    return nothing
end

"""
    test_jacobian_accuracy(volume_jacobian, mesh, front, test_markers = 5, epsilon = 1e-5)

Teste la précision du jacobien en perturbant des marqueurs et en comparant les changements réels
avec ceux prédits par le jacobien.

# Arguments
- `volume_jacobian::Dict` : Dictionnaire contenant les jacobiennes de volume
- `mesh::Mesh` : Le maillage contenant les informations sur les cellules
- `front::FrontTracker` : L'objet de suivi d'interface
- `test_markers::Int` : Nombre de marqueurs à tester
- `epsilon::Float64` : Amplitude de la perturbation

# Retourne
- `Figure` : Une figure montrant les erreurs relatives
"""
function test_jacobian_accuracy(volume_jacobian, mesh, front, test_markers = 5, epsilon = 1e-5)
    markers = get_markers(front)
    n_markers = length(markers) - (front.is_closed ? 1 : 0)
    test_markers = min(test_markers, n_markers)
    
    # Calculer les normales aux marqueurs
    normals = compute_marker_normals(front, markers)
    
    # Initialiser la figure
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1],
              xlabel = "Marqueur",
              ylabel = "Erreur relative",
              yscale = log10,
              title = "Précision du jacobien pour différents marqueurs")
    
    # Calculer les volumes initiaux
    body_func = (x, y, t, _=0) -> sdf(front, x, y)
    STmesh = Penguin.SpaceTimeMesh(mesh, [0.0, 1.0], tag=mesh.tag)
    capacity = Capacity(body_func, STmesh; compute_centroids=false)
    V_matrices = capacity.A[end]
    V_matrix = V_matrices[1:end÷2, 1:end÷2]
    
    # Pour chaque marqueur de test
    all_errors = Float64[]
    for marker_idx in 0:test_markers-1
        # Créer une copie perturbée de l'interface
        markers_perturbed = copy(markers)
        normal = normals[marker_idx+1]
        
        markers_perturbed[marker_idx+1] = (
            markers[marker_idx+1][1] + epsilon * normal[1],
            markers[marker_idx+1][2] + epsilon * normal[2]
        )
        
        # Pour les interfaces fermées, mettre à jour le dernier marqueur si nécessaire
        if front.is_closed && marker_idx == 0
            markers_perturbed[end] = markers_perturbed[1]
        end
        
        # Recalculer les volumes avec l'interface perturbée
        front_perturbed = FrontTracker(markers_perturbed, front.is_closed)
        body_func_perturbed = (x, y, t, _=0) -> sdf(front_perturbed, x, y)
        capacity_perturbed = Capacity(body_func_perturbed, STmesh; compute_centroids=false)
        V_matrices_perturbed = capacity_perturbed.A[end]
        V_matrix_perturbed = V_matrices_perturbed[1:end÷2, 1:end÷2]
        
        # Calculer les erreurs relatives pour les cellules affectées
        marker_errors = Float64[]
        for (i, j) in keys(volume_jacobian)
            jac_value = 0.0
            for (m, v) in volume_jacobian[(i,j)]
                if m == marker_idx
                    jac_value = v
                    break
                end
            end
            
            # Calculer le changement réel et prédit
            actual_change = V_matrix_perturbed[i, j] - V_matrix[i, j]
            predicted_change = jac_value * epsilon
            
            if abs(actual_change) > 1e-10
                rel_error = abs(actual_change - predicted_change) / abs(actual_change)
                push!(marker_errors, rel_error)
            end
        end
        
        if !isempty(marker_errors)
            push!(all_errors, mean(marker_errors))
            scatter!(ax, [marker_idx], [mean(marker_errors)], 
                    color = :blue, markersize = 12)
        end
    end
    
    # Ajouter une référence pour l'erreur moyenne
    if !isempty(all_errors)
        avg_error = mean(all_errors)
        hlines!(ax, [avg_error], color = :red, linestyle = :dash,
               label = "Erreur moyenne: $(round(avg_error, digits=6))")
    end
    
    
    return fig
end

    # Définir un maillage et une interface
    nx, ny = 20, 20
    lx, ly = 4.0, 4.0
    x0, y0 = -2.0, -2.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Créer une interface circulaire
    R = 1.0
    nmarkers = 30
    front = FrontTracker()
    create_circle!(front, 0.0, 0.0, R, nmarkers)
    markers = get_markers(front)
    
    # Calculer le jacobien de volume
    volJ = compute_volume_jacobian(mesh, front, 1e-0)
    
    # Lancer la visualisation combinée
    visualization_combined(volJ, mesh, front)
    
    # Tester la précision du jacobien
    fig_accuracy = test_jacobian_accuracy(volJ, mesh, front)
    display(fig_accuracy)
    save("jacobian_accuracy.png", fig_accuracy)
