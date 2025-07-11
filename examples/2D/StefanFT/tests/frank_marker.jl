using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CSV
using DataFrames

### Analyse de convergence pour différents nombres de marqueurs
### dans un problème de Stefan avec front circulaire

# Fonction pour exécuter la simulation avec un nombre spécifique de marqueurs
function run_single_timestep(n_markers::Int)
    # Paramètres physiques
    L = 1.0      # Chaleur latente
    c = 1.0      # Capacité thermique
    TM = 0.0     # Température de fusion (intérieur)
    T∞ = -0.5    # Température lointaine (liquide sous-refroidi)

    # Calculer le nombre de Stefan
    Ste = (c * (TM - T∞)) / L
    
    # Définir le paramètre de similarité S
    S = 1.56
    
    # Conditions initiales
    R0 = 1.56    # Rayon initial
    t_init = 1.0  # Temps initial
    
    # Fonction de position de l'interface
    interface_position(t) = S * sqrt(t)
    
    # Fonction de température analytique
    function analytical_temperature(r, t)
        s = r / sqrt(t)
        if s < S
            return TM  # Dans le solide (glace)
        else
            # En région liquide, utiliser la solution de similarité
            return T∞ * (1.0 - expint(s^2/4)/expint(S^2/4))
        end
    end
    
    # Définir le maillage spatial
    nx, ny = 129, 129
    lx, ly = 16.0, 16.0
    x0, y0 = -8.0, -8.0
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    
    # Créer le front-tracking
    front = FrontTracker() 
    create_circle!(front, 0.0, 0.0, interface_position(t_init), n_markers)
    
    # Définir la position initiale du front
    body = (x, y, t, _=0) -> -sdf(front, x, y)
    
    # Définir le pas de temps
    Δt = 0.5*(lx / nx)^2
    t_final = t_init + Δt
    
    # Maillage espace-temps
    STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_final], tag=mesh.tag)
    
    # Définir la capacité
    capacity = Capacity(body, STmesh; compute_centroids=false)
    
    # Définir l'opérateur de diffusion
    operator = DiffusionOps(capacity)
    
    # Définir les conditions aux limites
    bc_b = Dirichlet(T∞)
    bc = Dirichlet(TM)
    bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
        :left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))
    
    # Condition de Stefan à l'interface
    stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))
    
    # Définir le terme source (pas de source)
    f = (x,y,z,t) -> 0.0
    K = (x,y,z) -> 1.0  # Conductivité thermique
    
    Fluide = Phase(capacity, operator, f, K)
    
    # Configurer la condition initiale
    u0ₒ = zeros((nx+1)*(ny+1))
    body_init = (x,y,_=0) -> -sdf(front, x, y)
    cap_init = Capacity(body_init, mesh; compute_centroids=false)
    centroids = cap_init.C_ω
    
    # Initialiser la température
    for idx in 1:length(centroids)
        centroid = centroids[idx]
        x, y = centroid[1], centroid[2]
        r = sqrt(x^2 + y^2)
        u0ₒ[idx] = analytical_temperature(r, t_init)
    end
    u0ᵧ = ones((nx+1)*(ny+1))*TM
    u0 = vcat(u0ₒ, u0ᵧ)
    
    # Paramètres Newton
    Newton_params = (20, 1e-8, 1e-8, 0.8)  # max_iter, tol, reltol, α
    
    # Exécuter la simulation
    solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Résoudre le problème pour un seul pas de temps
    t_final_single = t_init + Δt
    solver, residuals, xf_log, timestep_history, phase, position_increments = 
        solve_StefanMono2D!(solver, Fluide, front, Δt, t_init, t_final_single,
                            bc_b, bc, stef_cond, mesh, "BE";
                            Newton_params=Newton_params, adaptive_timestep=false, method=Base.:\)
    
    return residuals[1]  # Retourne les résidus du premier pas de temps uniquement
end

function run_marker_convergence_study()
    # Liste des nombres de marqueurs à tester
    markers_to_test = [20, 50, 100, 200]
    
    # Créer le répertoire pour les résultats
    results_dir = joinpath(pwd(), "marker_convergence_data")
    mkpath(results_dir)
    
    # Exécuter les simulations pour chaque nombre de marqueurs
    println("Exécution des simulations pour différents nombres de marqueurs...")
    
    # Dataframe pour stocker les résumés des résultats
    summary_df = DataFrame(
        n_markers = Int[],
        iterations = Int[],
        final_residual = Float64[],
        converged = Bool[]
    )
    
    # Convergence threshold
    convergence_threshold = 1e-6
    
    for n_markers in markers_to_test
        println("Test avec $n_markers marqueurs...")
        residuals = run_single_timestep(n_markers)
        
        # Créer un DataFrame pour les résidus de cette simulation
        residuals_df = DataFrame(
            iteration = 1:length(residuals),
            residual = residuals
        )
        
        # Sauvegarder les résidus dans un fichier CSV
        CSV.write(joinpath(results_dir, "residuals_$(n_markers)_markers.csv"), residuals_df)
        
        # Calculer les métriques de convergence
        converged_at = findfirst(r -> r < convergence_threshold, residuals)
        converged = converged_at !== nothing
        
        if converged
            iterations_to_converge = converged_at
        else
            iterations_to_converge = length(residuals)
        end
        
        # Ajouter au DataFrame résumé
        push!(summary_df, (
            n_markers = n_markers,
            iterations = iterations_to_converge,
            final_residual = residuals[end],
            converged = converged
        ))
    end
    
    # Sauvegarder le résumé dans un fichier CSV
    CSV.write(joinpath(results_dir, "marker_convergence_summary.csv"), summary_df)
    
    println("\nSimulations terminées. Données enregistrées dans: $results_dir")
    
    return results_dir
end

# Exécuter l'étude
#results_dir = run_marker_convergence_study()
println("Données enregistrées dans: $results_dir")

using CSV
using DataFrames
using CairoMakie
using Statistics

"""
    plot_marker_convergence(data_dir::String)

Charge les données CSV de l'étude de convergence des marqueurs et génère des graphiques.
"""
function plot_marker_convergence(data_dir::String)
    # Vérifier si le répertoire existe
    if !isdir(data_dir)
        error("Le répertoire $data_dir n'existe pas.")
    end
    
    # Charger le résumé des résultats
    summary_file = joinpath(data_dir, "marker_convergence_summary.csv")
    if !isfile(summary_file)
        error("Le fichier de résumé $summary_file n'existe pas.")
    end
    
    summary_df = CSV.read(summary_file, DataFrame)
    
    # Créer le répertoire pour les graphiques
    plots_dir = joinpath(dirname(data_dir), "marker_convergence_plots")
    mkpath(plots_dir)
    
    # Charger les données de résidus pour chaque simulation
    residuals_data = Dict{Int, Vector{Float64}}()
    
    for n_markers in summary_df.n_markers
        residual_file = joinpath(data_dir, "residuals_$(n_markers)_markers.csv")
        if isfile(residual_file)
            df = CSV.read(residual_file, DataFrame)
            residuals_data[n_markers] = df.residual
        else
            @warn "Fichier de résidus manquant pour $n_markers marqueurs."
        end
    end
    
    # 1. Graphique des résidus par itération pour chaque nombre de marqueurs
    fig_residuals = Figure(size=(900, 600))
    ax_residuals = Axis(fig_residuals[1, 1], 
          title="Convergence des résidus pour différents nombres de marqueurs", 
          xlabel="Itération", 
          ylabel="Résidu",
          yscale=log10)

    # Palette de couleurs distinctes
    distinct_colors = [:royalblue, :crimson, :darkgreen, :darkorange, :purple, :teal]
    
    # Tracer les résidus pour chaque nombre de marqueurs
    for (i, n_markers) in enumerate(summary_df.n_markers)
        if haskey(residuals_data, n_markers)
            residuals = residuals_data[n_markers]
            
            # Couleur à utiliser (avec cycle si nécessaire)
            color_idx = ((i-1) % length(distinct_colors)) + 1
            
            # Tracer la courbe de résidus
            lines!(ax_residuals, 1:length(residuals), residuals, 
                  label="$n_markers marqueurs",
                  linewidth=2,
                  color=distinct_colors[color_idx])
            
            # Marquer les points de données
            scatter!(ax_residuals, 1:length(residuals), residuals,
                    markersize=6,
                    color=distinct_colors[color_idx])
        end
    end
    
    # Ajouter une référence d'ordre 1 (décroissance linéaire)
    if !isempty(residuals_data)
        # Utiliser le premier ensemble de données comme référence
        first_key = first(keys(residuals_data))
        ref_residuals = residuals_data[first_key]
        
        if length(ref_residuals) >= 2
            # Calculer le taux de décroissance approximatif
            ref_factor = ref_residuals[2] / ref_residuals[1]
            
            # Générer la référence d'ordre 1 (décroissance linéaire)
            ref_x = 1:length(ref_residuals)
            ref_y = [ref_residuals[1] * ref_factor^(i-1) for i in ref_x]
            
            # Tracer la ligne de référence d'ordre 1 clairement visible
            lines!(ax_residuals, ref_x, ref_y,
                  color=:black, linewidth=3, linestyle=:dash,
                  label="Ordre 1 (r × $(round(ref_factor, digits=2)))")
            
            # Ajouter une annotation pour clarifier
            text_pos_x = length(ref_residuals) ÷ 2
            text_pos_y = ref_y[text_pos_x] * 0.5
            text!(ax_residuals, text_pos_x, text_pos_y, 
                 text="Décroissance d'ordre 1", 
                 fontsize=14)
        end
    end
    
    # Ajouter la légende
    axislegend(ax_residuals, position=:rt)
    
    # 2. Graphique du nombre d'itérations nécessaires pour converger
    fig_iterations = Figure(size=(900, 600))
    ax_iterations = Axis(fig_iterations[1, 1],
                        title="Nombre d'itérations pour atteindre la convergence", 
                        xlabel="Nombre de marqueurs", 
                        ylabel="Nombre d'itérations")

    barplot!(ax_iterations, summary_df.n_markers, summary_df.iterations, 
            width=8.0, 
            color=[distinct_colors[((i-1) % length(distinct_colors)) + 1] for i in 1:length(summary_df.n_markers)])
    
    # 3. Graphique des résidus finaux
    fig_final = Figure(size=(900, 600))
    ax_final = Axis(fig_final[1, 1],
                   title="Résidu final après 20 itérations", 
                   xlabel="Nombre de marqueurs", 
                   ylabel="Résidu final",
                   yscale=log10)

    scatter!(ax_final, summary_df.n_markers, summary_df.final_residual, 
            markersize=15, 
            color=[distinct_colors[((i-1) % length(distinct_colors)) + 1] for i in 1:length(summary_df.n_markers)])
    
    lines!(ax_final, summary_df.n_markers, summary_df.final_residual, 
          linewidth=2, color=:gray, linestyle=:dash)
    
    # Sauvegarder les graphiques
    save(joinpath(plots_dir, "residuals_comparison.png"), fig_residuals)
    save(joinpath(plots_dir, "iterations_to_converge.png"), fig_iterations)
    save(joinpath(plots_dir, "final_residuals.png"), fig_final)
    
    # Afficher les graphiques
    display(fig_residuals)
    display(fig_iterations)
    display(fig_final)
    
    println("Graphiques sauvegardés dans: $plots_dir")
    
    return plots_dir
end
"""
# Chercher le répertoire des données par défaut
    default_data_dir = joinpath(pwd(), "marker_convergence_data")
    
    if isdir(default_data_dir)
        plot_marker_convergence(default_data_dir)
    else
        println("Veuillez spécifier le chemin vers le répertoire contenant les données:")
        data_dir = readline()
        if isdir(data_dir)
            plot_marker_convergence(data_dir)
        else
            println("Répertoire invalide.")
        end
    end
"""

using CSV
using DataFrames
using CairoMakie
using Statistics
using ColorSchemes

"""
    plot_multi_mesh_marker_convergence(data_dir::String)

Charge les données CSV des études de convergence pour différentes tailles de maillage
et différents nombres de marqueurs, puis génère des graphiques comparatifs.
"""
function plot_multi_mesh_marker_convergence(data_dir::String)
    # Vérifier si le répertoire existe
    if !isdir(data_dir)
        error("Le répertoire $data_dir n'existe pas.")
    end
    
    # Détecter les sous-répertoires correspondant aux différentes tailles de maillage
    mesh_subdirs = filter(isdir, [joinpath(data_dir, d) for d in readdir(data_dir)])
    mesh_sizes = [parse(Int, basename(d)) for d in mesh_subdirs]
    
    if isempty(mesh_subdirs)
        error("Aucun sous-répertoire de taille de maillage trouvé dans $data_dir")
    end
    
    println("Tailles de maillage détectées: $mesh_sizes")
    
    # Créer le répertoire pour les graphiques
    plots_dir = joinpath(dirname(data_dir), "multi_mesh_marker_plots")
    mkpath(plots_dir)
    
    # Charger les résumés et données de résidus pour chaque maillage
    mesh_summaries = Dict{Int, DataFrame}()
    all_residuals = Dict{Tuple{Int, Int}, Vector{Float64}}() # (mesh_size, n_markers) => residuals
    
    for (mesh_size, subdir) in zip(mesh_sizes, mesh_subdirs)
        summary_file = joinpath(subdir, "marker_convergence_summary.csv")
        
        if isfile(summary_file)
            mesh_summaries[mesh_size] = CSV.read(summary_file, DataFrame)
            
            # Charger les fichiers de résidus pour ce maillage
            for n_markers in mesh_summaries[mesh_size].n_markers
                residual_file = joinpath(subdir, "residuals_$(n_markers)_markers.csv")
                if isfile(residual_file)
                    df = CSV.read(residual_file, DataFrame)
                    all_residuals[(mesh_size, n_markers)] = df.residual
                else
                    @warn "Fichier de résidus manquant pour maillage $mesh_size, $n_markers marqueurs"
                end
            end
        else
            @warn "Fichier de résumé manquant pour le maillage $mesh_size"
        end
    end
    
    # Créer un DataFrame combiné pour faciliter les visualisations
    combined_df = vcat([
        transform(df, :n_markers => ByRow(n -> (mesh_size, n)) => :mesh_markers, 
                 :n_markers => ByRow(n -> mesh_size) => :mesh_size)
        for (mesh_size, df) in mesh_summaries]...)
    
    # Définir les palettes de couleurs
    mesh_colors = ColorSchemes.thermal
    marker_colors = [:royalblue, :crimson, :darkgreen, :darkorange, :purple, :teal]
    marker_markers = [:circle, :rect, :utriangle, :diamond, :pentagon, :cross]
    
    # 1. Graphique: Résidus finaux vs Nombre de marqueurs vs Taille de maillage
    fig_final_residuals = Figure(size=(1000, 800), 
                               title="Résidus finaux par nombre de marqueurs et taille de maillage")
    ax_final = Axis(fig_final_residuals[1, 1], 
                  title="Résidus finaux", 
                  xlabel="Nombre de marqueurs", 
                  ylabel="Résidu final",
                  yscale=log10)
    
    # Tracer une ligne pour chaque taille de maillage
    for (i, mesh_size) in enumerate(sort(collect(keys(mesh_summaries))))
        df = mesh_summaries[mesh_size]
        
        # Normaliser l'index de couleur pour cette taille de maillage
        color_idx = (i-1) / (length(mesh_summaries)-1)
        color = get(mesh_colors, color_idx)
        
        lines!(ax_final, df.n_markers, df.final_residual,
              color=color, linewidth=2,
              label="Maillage $(mesh_size)×$(mesh_size)")
        
        scatter!(ax_final, df.n_markers, df.final_residual,
                color=color, markersize=10)
    end
    
    axislegend(ax_final, position=:rt, framevisible=true)
    
    # 2. Graphique: Nombre d'itérations pour converger vs Nombre de marqueurs vs Taille de maillage
    fig_iterations = Figure(size=(1000, 800))
    ax_iter = Axis(fig_iterations[1, 1], 
                 title="Nombre d'itérations pour atteindre la convergence", 
                 xlabel="Nombre de marqueurs", 
                 ylabel="Nombre d'itérations")
    
    # Tracer une ligne pour chaque taille de maillage
    for (i, mesh_size) in enumerate(sort(collect(keys(mesh_summaries))))
        df = mesh_summaries[mesh_size]
        
        # Normaliser l'index de couleur pour cette taille de maillage
        color_idx = (i-1) / (length(mesh_summaries)-1)
        color = get(mesh_colors, color_idx)
        
        lines!(ax_iter, df.n_markers, df.iterations,
              color=color, linewidth=2,
              label="Maillage $(mesh_size)×$(mesh_size)")
        
        scatter!(ax_iter, df.n_markers, df.iterations,
                color=color, markersize=10)
    end
    
    axislegend(ax_iter, position=:rt, framevisible=true)
    
    # 3. Graphique comparatif des courbes de résidus
    fig_residuals = Figure(size=(1200, 900))
    
    # Organiser en matrice: lignes = tailles de maillage, colonnes = nombres de marqueurs
    unique_markers = sort(unique(combined_df.n_markers))
    unique_meshes = sort(unique(combined_df.mesh_size))
    
    # Créer un tableau d'axes pour la matrice de visualisations
    n_rows = length(unique_meshes)
    n_cols = 1
    axes = Matrix{Axis}(undef, n_rows, n_cols)
    
    # Tracer toutes les courbes de résidus sur un seul graphique par taille de maillage
    for (i, mesh_size) in enumerate(unique_meshes)
        ax = Axis(fig_residuals[i, 1], 
                title="Maillage $(mesh_size)×$(mesh_size)", 
                xlabel="Itération", 
                ylabel="Résidu",
                yscale=log10)
        axes[i, 1] = ax
        
        # Tracer chaque courbe de marqueurs pour cette taille de maillage
        for (j, n_markers) in enumerate(unique_markers)
            key = (mesh_size, n_markers)
            if haskey(all_residuals, key)
                residuals = all_residuals[key]
                
                color_idx = ((j-1) % length(marker_colors)) + 1
                marker_idx = ((j-1) % length(marker_markers)) + 1
                
                lines!(ax, 1:length(residuals), residuals,
                      label="$n_markers marqueurs",
                      linewidth=2,
                      color=marker_colors[color_idx])
                
                scatter!(ax, 1:length(residuals), residuals,
                        markersize=5,
                        color=marker_colors[color_idx])
            end
        end
        
        # Ajouter la légende à la première rangée seulement
        if i == 1
            axislegend(ax, position=:rt, framevisible=true)
        end
    end
    
    # 4. Graphique 3D pour visualiser la relation entre les trois variables
    fig_3d = Figure(size=(1000, 800))
    ax_3d = Axis3(fig_3d[1, 1], 
                title="Relation entre taille de maillage, marqueurs, et résidu final",
                xlabel="Taille de maillage", 
                ylabel="Nombre de marqueurs", 
                zlabel="Résidu final")
    
    # Créer des points 3D pour chaque combinaison
    for (mesh_size, df) in mesh_summaries
        for row in eachrow(df)
            scatter!(ax_3d, [mesh_size], [row.n_markers], [row.final_residual],
                   markersize=15,
                   color=get(mesh_colors, (mesh_size-minimum(mesh_sizes)) / (maximum(mesh_sizes)-minimum(mesh_sizes))))
        end
    end
    
    # Sauvegarder les graphiques
    save(joinpath(plots_dir, "final_residuals_comparison.png"), fig_final_residuals)
    save(joinpath(plots_dir, "iterations_comparison.png"), fig_iterations)
    save(joinpath(plots_dir, "residuals_curves_by_mesh.png"), fig_residuals)
    save(joinpath(plots_dir, "3d_visualization.png"), fig_3d)
    
    # Créer un graphique de matrice pour comparer toutes les combinaisons
    fig_matrix = Figure(size=(1500, 1200))
    
    n_rows = length(unique_meshes)
    n_cols = length(unique_markers)
    matrix_axes = Matrix{Axis}(undef, n_rows, n_cols)
    
    for (i, mesh_size) in enumerate(unique_meshes)
        for (j, n_markers) in enumerate(unique_markers)
            ax = Axis(fig_matrix[i, j], 
                    title="Maillage $(mesh_size)×$(mesh_size), $(n_markers) marqueurs", 
                    xlabel="Itération", 
                    ylabel="Résidu",
                    yscale=log10)
            matrix_axes[i, j] = ax
            
            key = (mesh_size, n_markers)
            if haskey(all_residuals, key)
                residuals = all_residuals[key]
                
                # Couleur basée sur la taille de maillage
                color_idx = (i-1) / (n_rows-1)
                color = get(mesh_colors, color_idx)
                
                lines!(ax, 1:length(residuals), residuals,
                      linewidth=2,
                      color=color)
                
                scatter!(ax, 1:length(residuals), residuals,
                        markersize=4,
                        color=color)
                
                # Ajouter une référence d'ordre 1 pour chaque graphique
                if length(residuals) >= 2
                    # Calculer le taux de décroissance approximatif
                    ref_factor = residuals[2] / residuals[1]
                    
                    # Générer la référence d'ordre 1 (décroissance linéaire)
                    ref_x = 1:length(residuals)
                    ref_y = [residuals[1] * ref_factor^(k-1) for k in ref_x]
                    
                    # Tracer la ligne de référence d'ordre 1
                    lines!(ax, ref_x, ref_y,
                          color=:black, linewidth=2, linestyle=:dash)
                end
            end
        end
    end
    
    save(joinpath(plots_dir, "residuals_matrix.png"), fig_matrix)
    
    # Afficher les graphiques
    display(fig_final_residuals)
    display(fig_iterations)
    display(fig_residuals)
    display(fig_3d)
    display(fig_matrix)
    
    println("Graphiques sauvegardés dans: $plots_dir")
    
    return plots_dir
end

# Si exécuté directement, essayer de trouver les données et générer les graphiques
    # Chercher le répertoire des données par défaut
    default_data_dir = joinpath(pwd(), "marker_convergence_data")
    
    if isdir(default_data_dir)
        plot_multi_mesh_marker_convergence(default_data_dir)
    else
        println("Veuillez spécifier le chemin vers le répertoire contenant les données:")
        data_dir = readline()
        if isdir(data_dir)
            plot_multi_mesh_marker_convergence(data_dir)
        else
            println("Répertoire invalide.")
        end
    end

function plot_combined_residuals(data_dir::String)
    # Vérifier si le répertoire existe
    if !isdir(data_dir)
        error("Le répertoire $data_dir n'existe pas.")
    end
    
    # Détecter les sous-répertoires correspondant aux différentes tailles de maillage
    mesh_subdirs = filter(isdir, [joinpath(data_dir, d) for d in readdir(data_dir)])
    mesh_sizes = [parse(Int, basename(d)) for d in mesh_subdirs]
    
    if isempty(mesh_subdirs)
        error("Aucun sous-répertoire de taille de maillage trouvé dans $data_dir")
    end
    
    # Collecter toutes les données de résidus
    all_residuals = Dict{Tuple{Int, Int}, Vector{Float64}}() # (mesh_size, n_markers) => residuals
    all_markers = Set{Int}()
    
    for (mesh_size, subdir) in zip(mesh_sizes, mesh_subdirs)
        # Identifier tous les fichiers de résidus dans ce sous-répertoire
        residual_files = filter(f -> startswith(f, "residuals_") && endswith(f, "_markers.csv"), 
                               readdir(subdir))
        
        for file in residual_files
            # Extraire le nombre de marqueurs du nom de fichier
            m = match(r"residuals_(\d+)_markers\.csv", file)
            if m !== nothing
                n_markers = parse(Int, m.captures[1])
                push!(all_markers, n_markers)
                
                # Charger les données de résidus
                df = CSV.read(joinpath(subdir, file), DataFrame)
                all_residuals[(mesh_size, n_markers)] = df.residual
            end
        end
    end
    
    # Créer une figure combinée pour toutes les courbes de résidus
    fig_combined = Figure(size=(1200, 800))
    ax_combined = Axis(fig_combined[1, 1],
                     title="Convergence des résidus pour toutes les combinaisons",
                     xlabel="Itération",
                     ylabel="Résidu",
                     yscale=log10)
    
    # Définir des couleurs distinctes pour les tailles de maillage
    mesh_colors = cgrad(:thermal, length(mesh_sizes))
    
    # Utiliser différents styles de lignes et marqueurs pour les nombres de marqueurs
    marker_styles = [:circle, :rect, :utriangle, :diamond, :star5]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    # Tracer toutes les courbes sur le même graphique
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        for (j, n_markers) in enumerate(sort(collect(all_markers)))
            key = (mesh_size, n_markers)
            if haskey(all_residuals, key)
                residuals = all_residuals[key]
                
                color = mesh_colors[i]
                marker_style = marker_styles[mod1(j, length(marker_styles))]
                line_style = line_styles[mod1(j, length(line_styles))]
                
                # Tracer la courbe avec ligne et points
                lines!(ax_combined, 1:length(residuals), residuals,
                      linewidth=2,
                      color=color,
                      linestyle=line_style,
                      label="$(mesh_size)×$(mesh_size), $(n_markers) marqueurs")
                
                # Ajouter des points aux positions clés (début, milieu, fin)
                scatter_indices = [1, length(residuals)÷2, length(residuals)]
                scatter!(ax_combined, scatter_indices, residuals[scatter_indices],
                        markersize=8,
                        marker=marker_style,
                        color=color)
            end
        end
    end
    
    # Ajouter une référence d'ordre 1 basée sur la première courbe disponible
    first_key = first(keys(all_residuals))
    ref_residuals = all_residuals[first_key]
    
    if length(ref_residuals) >= 2
        ref_factor = ref_residuals[2] / ref_residuals[1]
        ref_x = 1:maximum(length(residuals) for residuals in values(all_residuals))
        ref_y = [ref_residuals[1] * ref_factor^(k-1) for k in ref_x]
        
        lines!(ax_combined, ref_x, ref_y,
              color=:black, linewidth=3, linestyle=:dash,
              label="Référence O($(round(ref_factor, digits=2)))")
        
        # Ajouter une annotation
        text!(ax_combined, ref_x[end]÷2, ref_y[ref_x[end]÷2] * 0.5,
             text="Décroissance d'ordre 1",
             fontsize=14)
    end
    
    # Ajouter la légende avec plusieurs colonnes pour la lisibilité
    axislegend(ax_combined, position=:rb, framevisible=true, nbanks=2, patchsize=(10, 10))
    
    # Créer le répertoire pour les graphiques si nécessaire
    plots_dir = joinpath(dirname(data_dir), "multi_mesh_marker_plots")
    mkpath(plots_dir)
    
    # Sauvegarder le graphique
    save(joinpath(plots_dir, "combined_residuals_all.png"), fig_combined)
    
    # Afficher le graphique
    display(fig_combined)
    
    # Créer un second graphique avec seulement les données des premières itérations
    # pour mieux voir le comportement initial
    fig_combined_initial = Figure(size=(1200, 800))
    ax_combined_initial = Axis(fig_combined_initial[1, 1],
                             title="Convergence initiale des résidus (10 premières itérations)",
                             xlabel="Itération",
                             ylabel="Résidu",
                             yscale=log10)
    
    # Limiter aux 10 premières itérations ou moins
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        for (j, n_markers) in enumerate(sort(collect(all_markers)))
            key = (mesh_size, n_markers)
            if haskey(all_residuals, key)
                residuals = all_residuals[key]
                n_iter = min(10, length(residuals))
                
                color = mesh_colors[i]
                marker_style = marker_styles[mod1(j, length(marker_styles))]
                line_style = line_styles[mod1(j, length(line_styles))]
                
                lines!(ax_combined_initial, 1:n_iter, residuals[1:n_iter],
                      linewidth=2,
                      color=color,
                      linestyle=line_style,
                      label="$(mesh_size)×$(mesh_size), $(n_markers) marqueurs")
                
                scatter!(ax_combined_initial, 1:n_iter, residuals[1:n_iter],
                        markersize=8,
                        marker=marker_style,
                        color=color)
            end
        end
    end
    
    # Ajouter la légende
    axislegend(ax_combined_initial, position=:rt, framevisible=true, nbanks=2, patchsize=(10, 10))
    
    save(joinpath(plots_dir, "combined_residuals_initial.png"), fig_combined_initial)
    display(fig_combined_initial)
    
    println("Graphiques sauvegardés dans: $plots_dir")
    
    return plots_dir
end

# Après avoir défini toutes vos autres fonctions
plot_combined_residuals(default_data_dir)