using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions, LsqFit
using CairoMakie
using Interpolations
using Colors
using Statistics
using FFTW
using DSP
using Roots
using Dates
using DataFrames
using CSV

"""
    run_parametric_study()

Exécute une étude paramétrique sur la méthode de résolution du problème de Stefan
avec front tracking, en analysant l'impact de différents paramètres sur la précision
et la convergence.

Paramètres étudiés:
- Nombre de marqueurs (nmarkers)
- Taille du maillage (nx, ny)
- Epsilon pour le Jacobien (jacobian_epsilon)
- Facteur de lissage (smooth_factor)
- Taille de fenêtre pour le lissage (window_size)
"""
function run_parametric_study()
    # Créer un répertoire pour les résultats avec horodatage
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_dir = joinpath(pwd(), "parametric_study_$(timestamp)")
    mkpath(results_dir)
    
    # Journal pour toutes les configurations et résultats
    results_df = DataFrame(
        config_id = Int[],
        nx = Int[],
        ny = Int[],
        nmarkers = Int[],
        jacobian_epsilon = Float64[],
        smooth_factor = Float64[],
        window_size = Int[],
        mean_iterations = Float64[],
        max_iterations = Int[],
        mean_residual = Float64[],
        mean_position_inc = Float64[],
        final_radius = Float64[],
        radius_std = Float64[],
        analytical_radius = Float64[],
        radius_error = Float64[],
        runtime = Float64[]
    )
    
    # Définir les valeurs à tester pour chaque paramètre
    mesh_sizes = [(16, 16), (24, 24), (32, 32), (48, 48)]
    n_markers_list = [50, 100, 200]
    jacobian_epsilons = [1e-7, 1e-6, 1e-5]
    smooth_factors = [0.3, 0.5, 0.7]
    window_sizes = [5, 10, 15]
    
    # Choix limités de paramètres (pour éviter trop de combinaisons)
    # Si on veut tester toutes les combinaisons, utiliser la ligne ci-dessous:
    # parameter_combinations = [(nx_ny, nm, je, sf, ws) for nx_ny in mesh_sizes for nm in n_markers_list for je in jacobian_epsilons for sf in smooth_factors for ws in window_sizes]
    
    # Pour réduire le nombre de simulations, choisir des combinaisons spécifiques:
    parameter_combinations = [
        # Étude de l'impact de la résolution du maillage (autres paramètres par défaut)
        ((24, 24), 100, 1e-6, 0.5, 10),
        ((32, 32), 100, 1e-6, 0.5, 10),
        ((48, 48), 100, 1e-6, 0.5, 10),
        
        # Étude de l'impact du nombre de marqueurs (maillage 32x32)
        ((32, 32), 50, 1e-6, 0.5, 10),
        ((32, 32), 100, 1e-6, 0.5, 10),  # Déjà inclus ci-dessus
        ((32, 32), 200, 1e-6, 0.5, 10),
        
        # Étude de l'impact de l'epsilon du Jacobien (maillage 32x32, 100 marqueurs)
        ((32, 32), 100, 1e-7, 0.5, 10),
        ((32, 32), 100, 1e-6, 0.5, 10),  # Déjà inclus
        ((32, 32), 100, 1e-5, 0.5, 10),
        
        # Étude de l'impact des paramètres de lissage (maillage 32x32, 100 marqueurs)
        ((32, 32), 100, 1e-6, 0.3, 10),
        ((32, 32), 100, 1e-6, 0.5, 10),  # Déjà inclus
        ((32, 32), 100, 1e-6, 0.7, 10),
        ((32, 32), 100, 1e-6, 0.5, 5),
        ((32, 32), 100, 1e-6, 0.5, 15)
    ]
    
    # Paramètres partagés pour toutes les simulations
    L = 1.0      # Chaleur latente
    c = 1.0      # Capacité thermique spécifique
    TM = 0.0     # Température de fusion
    T∞ = -0.5    # Température à l'infini
    
    # Calculer le nombre de Stefan
    Ste = (c * (TM - T∞)) / L
    
    # Paramètre de similarité (solution auto-similaire)
    S = 1.56
    
    # Condition initiale
    R0 = 1.56      # Rayon initial
    t_init = 1.0   # Temps initial
    
    # Paramètres de temps
    Δt_base = 0.02  # Pas de temps de base
    t_final = t_init + Δt_base  # Un seul pas de temps pour l'étude
    
    # Fonction pour la solution analytique
    function F(s)
        return expint(s^2/4)  # E₁(s²/4)
    end
    
    # Température analytique
    function analytical_temperature(r, t)
        s = r / sqrt(t)
        if s < S
            return TM
        else
            return T∞ * (1.0 - F(s)/F(S))
        end
    end
    
    # Position de l'interface
    function interface_position(t)
        return S * sqrt(t)
    end
    
    # Exécuter les simulations pour chaque combinaison de paramètres
    config_id = 0
    for ((nx, ny), nmarkers, jacobian_epsilon, smooth_factor, window_size) in parameter_combinations
        config_id += 1
        println("\n=========================================")
        println("Configuration #$config_id:")
        println("  - Maillage: $(nx)x$(ny)")
        println("  - Marqueurs: $nmarkers")
        println("  - Epsilon Jacobien: $jacobian_epsilon")
        println("  - Lissage: factor=$smooth_factor, window=$window_size")
        println("=========================================")
        
        # Mesurer le temps d'exécution
        start_time = time()
        
        # Créer le maillage
        lx, ly = 16.0, 16.0
        x0, y0 = -8.0, -8.0
        mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
        
        # Créer le front tracker avec le nombre spécifié de marqueurs
        front = FrontTracker()
        create_circle!(front, 0.0, 0.0, interface_position(t_init), nmarkers)
        
        # Définir la fonction de corps pour le front tracking
        body = (x, y, t, _=0) -> -sdf(front, x, y)
        
        # Pas de temps adapté à la résolution du maillage
        Δt = Δt_base
        
        # Créer le maillage spatio-temporel
        STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)
        
        # Définir la capacité et l'opérateur
        capacity = Capacity(body, STmesh; compute_centroids=false)
        operator = DiffusionOps(capacity)
        
        # Conditions aux limites
        bc_b = Dirichlet(T∞)
        bc = Dirichlet(TM)
        bc_b = BorderConditions(Dict{Symbol, AbstractBoundary}(
            :left => bc_b, :right => bc_b, :top => bc_b, :bottom => bc_b))
        
        # Condition de Stefan à l'interface
        stef_cond = InterfaceConditions(nothing, FluxJump(1.0, 1.0, L))
        
        # Définir la source (pas de source)
        f = (x,y,z,t) -> 0.0
        K = (x,y,z) -> 1.0  # Conductivité thermique
        
        Fluide = Phase(capacity, operator, f, K)
        
        # Initialiser la condition initiale
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
        
        # Paramètres de Newton
        Newton_params = (2, 1e-6, 1e-6, 1.0)  # max_iter, tol, reltol, α
        
        # Initialiser le solveur
        solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
        
        # Exécuter la simulation
        solver, residuals, xf_log, timestep_history, phase, position_increments = 
            solve_StefanMono2D!(solver, Fluide, front, Δt, t_init, t_final, bc_b, bc, stef_cond, mesh, "BE";
                                Newton_params=Newton_params,
                                jacobian_epsilon=jacobian_epsilon,
                                smooth_factor=smooth_factor,
                                window_size=window_size,
                                method=Base.:\)
        
        # Mesurer le temps total
        runtime = time() - start_time
        
        # Collecter les statistiques sur les résidus
        iterations_per_timestep = [length(res) for (_, res) in residuals]
        mean_iterations = mean(iterations_per_timestep)
        max_iterations = maximum(iterations_per_timestep)
        
        # Collecter les statistiques sur les résidus finaux
        final_residuals = [res[end] for (_, res) in residuals]
        mean_residual = mean(final_residuals)
        
        # Collecter les statistiques sur les incréments de position
        position_inc_per_timestep = [inc[end] for (_, inc) in position_increments]
        mean_position_inc = mean(position_inc_per_timestep)

        # Calculer le rayon final et la régularité
        last_timestep = maximum(keys(xf_log))
        final_markers = xf_log[last_timestep]
        center_x = sum(m[1] for m in final_markers) / length(final_markers)
        center_y = sum(m[2] for m in final_markers) / length(final_markers)
        final_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in final_markers]
        final_radius = mean(final_radii)
        radius_std = std(final_radii) / final_radius  # Normalized std dev

        # Calculer l'erreur par rapport à la solution analytique
        analytical_radius = interface_position(t_final)
        radius_error = abs(final_radius - analytical_radius) / analytical_radius
        
        # Ajouter les résultats au DataFrame
        push!(results_df, (
            config_id,
            nx, ny, nmarkers,
            jacobian_epsilon,
            smooth_factor,
            window_size,
            mean_iterations,
            max_iterations,
            mean_residual,
            mean_position_inc,
            final_radius,
            radius_std,
            analytical_radius,
            radius_error,
            runtime
        ))
        
        # Sauvegarder les résidus détaillés pour cette configuration
        config_dir = joinpath(results_dir, "config_$(config_id)")
        mkpath(config_dir)
        
        # Sauvegarder l'historique des résidus pour chaque pas de temps
        for (timestep, res) in residuals
            CSV.write(joinpath(config_dir, "residuals_timestep_$(timestep).csv"),
                     DataFrame(iteration=1:length(res), residual=res))
        end
        
        # Générer des visualisations pour cette configuration
        plot_config_results(solver, front, mesh, residuals, xf_log, position_increments, 
                           analytical_radius, config_id, config_dir)
        
        # Afficher un résumé pour cette configuration
        println("\nRésultats pour la configuration #$config_id:")
        println("  - Itérations moyennes: $(round(mean_iterations, digits=2))")
        println("  - Résidu moyen final: $(mean_residual)")
        println("  - Rayon final: $final_radius (analytique: $analytical_radius)")
        println("  - Erreur relative: $(100*radius_error)%")
        println("  - Irrégularité interface: $(100*radius_std)%")
        println("  - Temps d'exécution: $(round(runtime, digits=2)) secondes\n")
    end
    
    # Sauvegarder tous les résultats dans un fichier CSV
    CSV.write(joinpath(results_dir, "parametric_study_results.csv"), results_df)
    
    # Générer des graphiques comparatifs
    plot_comparative_results(results_df, results_dir)
    
    # Afficher un résumé
    println("\nÉtude paramétrique terminée!")
    println("Nombre total de configurations testées: $(config_id)")
    println("Résultats sauvegardés dans: $(results_dir)")
    
    return results_df, results_dir
end

"""
    plot_config_results(solver, front, mesh, residuals, xf_log, position_increments, 
                      analytical_radius, config_id, config_dir)

Générer des visualisations pour une configuration spécifique.
"""
function plot_config_results(solver, front, mesh, residuals, xf_log, position_increments, 
                           analytical_radius, config_id, config_dir)
    # 1. Visualiser la température finale et l'interface
    fig_temp = Figure(size=(600, 600))
    ax_temp = Axis(fig_temp[1, 1], 
                 title="Configuration #$config_id - Température Finale",
                 xlabel="x", ylabel="y",
                 aspect=DataAspect())
    
    # Extraire les dimensions du maillage
    xi = mesh.nodes[1]
    yi = mesh.nodes[2]
    nx1, ny1 = length(xi), length(yi)
    npts = nx1 * ny1
    
    # Extraire et reshaper la température finale
    final_temp = solver.states[end][1:npts]
    temp_2d = reshape(final_temp, (nx1, ny1))
    
    # Tracer la température
    hm = heatmap!(ax_temp, xi, yi, temp_2d, colormap=:thermal)
    Colorbar(fig_temp[1, 2], hm, label="Température")
    
    # Ajouter le contour de l'interface finale
    last_timestep = maximum(keys(xf_log))
    final_markers = xf_log[last_timestep]
    marker_x = [m[1] for m in final_markers]
    marker_y = [m[2] for m in final_markers]
    lines!(ax_temp, marker_x, marker_y, color=:black, linewidth=2)
    
    # Ajouter un cercle représentant la solution analytique
    theta = range(0, 2π, length=100)
    analytic_x = analytical_radius .* cos.(theta)
    analytic_y = analytical_radius .* sin.(theta)
    lines!(ax_temp, analytic_x, analytic_y, color=:red, linewidth=2, linestyle=:dash)
    
    # Sauvegarder la figure
    save(joinpath(config_dir, "temperature_interface.png"), fig_temp)
    
    # 2. Visualiser les résidus
    fig_res = Figure(size=(800, 400))
    ax_res = Axis(fig_res[1, 1],
                title="Configuration #$config_id - Historique des Résidus",
                xlabel="Itération", ylabel="Résidu (échelle log)",
                yscale=log10)
    
    for (timestep, res) in sort(collect(residuals))
        lines!(ax_res, 1:length(res), res, 
              label="Pas de temps $timestep", linewidth=2)
    end
    
    axislegend(ax_res)
    save(joinpath(config_dir, "residuals.png"), fig_res)
    
    # 3. Visualiser la comparaison résidus / incréments de position
    fig_comp = Figure(size=(800, 400))
    ax_comp = Axis(fig_comp[1, 1],
                  title="Configuration #$config_id - Résidus vs. Incréments de Position",
                  xlabel="Itération", ylabel="Valeur (échelle log)",
                  yscale=log10)
    
    # Supposons que nous examinions le premier pas de temps pour simplifier
    timestep = 1
    if haskey(residuals, timestep) && haskey(position_increments, timestep)
        res = residuals[timestep]
        pos_inc = position_increments[timestep]
        min_len = min(length(res), length(pos_inc))
        
        lines!(ax_comp, 1:min_len, res[1:min_len], 
              label="Résidu", linewidth=2, color=:blue)
        lines!(ax_comp, 1:min_len, pos_inc[1:min_len], 
              label="Incrément de position", linewidth=2, color=:red, linestyle=:dash)
    end
    
    axislegend(ax_comp)
    save(joinpath(config_dir, "residual_vs_increment.png"), fig_comp)
    
    return fig_temp, fig_res, fig_comp
end

"""
    plot_comparative_results(results_df, results_dir)

Générer des visualisations comparatives entre les différentes configurations.
"""
function plot_comparative_results(results_df, results_dir)
    # 1. Précision vs. Nombre de marqueurs
    fig_markers = Figure(size=(800, 500))
    ax_markers = Axis(fig_markers[1, 1],
                     title="Impact du Nombre de Marqueurs",
                     xlabel="Nombre de marqueurs",
                     ylabel="Erreur relative (%)")
    
    # Filtrer pour garder seulement les configurations où seul nmarkers varie
    base_nx = results_df.nx[1]
    base_ny = results_df.ny[1]
    base_eps = results_df.jacobian_epsilon[1]
    base_smooth = results_df.smooth_factor[1]
    base_window = results_df.window_size[1]
    base_nmarkers = results_df.nmarkers[1]  # Added this missing line
    
    marker_configs = results_df[(results_df.nx .== base_nx) .&
                               (results_df.ny .== base_ny) .&
                               (results_df.jacobian_epsilon .== base_eps) .&
                               (results_df.smooth_factor .== base_smooth) .&
                               (results_df.window_size .== base_window), :]
    
    if nrow(marker_configs) > 1
        marker_configs = sort(marker_configs, :nmarkers)
        scatter!(ax_markers, marker_configs.nmarkers, 100 .* marker_configs.radius_error,
                marker=:circle, color=:blue, markersize=12)
        lines!(ax_markers, marker_configs.nmarkers, 100 .* marker_configs.radius_error,
              color=:blue, linewidth=2)
        
        # Ajouter une ligne pour l'irrégularité de l'interface
        ax_irreg = Axis(fig_markers[1, 1], ylabel="Irrégularité (%)",
                       yaxisposition=:right,
                       yticklabelcolor=:red)
        scatter!(ax_irreg, marker_configs.nmarkers, 100 .* marker_configs.radius_std,
                marker=:square, color=:red, markersize=10)
        lines!(ax_irreg, marker_configs.nmarkers, 100 .* marker_configs.radius_std,
              color=:red, linewidth=2, linestyle=:dash)
        
        # Légende
        Legend(fig_markers[1, 2], [
            MarkerElement(marker=:circle, color=:blue),
            MarkerElement(marker=:square, color=:red)
        ], ["Erreur rayon", "Irrégularité"])
    end
    
    save(joinpath(results_dir, "marker_impact.png"), fig_markers)
    
    # 2. Précision vs. Résolution du maillage
    fig_mesh = Figure(size=(800, 500))
    ax_mesh = Axis(fig_mesh[1, 1],
                  title="Impact de la Résolution du Maillage",
                  xlabel="Taille du maillage (nx = ny)",
                  ylabel="Erreur relative (%)")
    
    # Filtrer pour garder les configurations où seul nx/ny varie
    mesh_configs = results_df[(results_df.nmarkers .== base_nmarkers) .&
                             (results_df.jacobian_epsilon .== base_eps) .&
                             (results_df.smooth_factor .== base_smooth) .&
                             (results_df.window_size .== base_window), :]
    
    if nrow(mesh_configs) > 1
        mesh_configs = sort(mesh_configs, :nx)  # Supposant nx = ny
        scatter!(ax_mesh, mesh_configs.nx, 100 .* mesh_configs.radius_error,
                marker=:circle, color=:blue, markersize=12)
        lines!(ax_mesh, mesh_configs.nx, 100 .* mesh_configs.radius_error,
              color=:blue, linewidth=2)
        
        # Ajouter une ligne pour le temps d'exécution
        ax_time = Axis(fig_mesh[1, 1], ylabel="Temps d'exécution (s)",
                      yaxisposition=:right,
                      yticklabelcolor=:red)
        scatter!(ax_time, mesh_configs.nx, mesh_configs.runtime,
                marker=:square, color=:red, markersize=10)
        lines!(ax_time, mesh_configs.nx, mesh_configs.runtime,
              color=:red, linewidth=2, linestyle=:dash)
        
        # Légende
        Legend(fig_mesh[1, 2], [
            MarkerElement(marker=:circle, color=:blue),
            MarkerElement(marker=:square, color=:red)
        ], ["Erreur rayon", "Temps d'exécution"])
    end
    
    save(joinpath(results_dir, "mesh_impact.png"), fig_mesh)
    
    # 3. Précision vs. Epsilon du Jacobien
    fig_eps = Figure(size=(800, 500))
    ax_eps = Axis(fig_eps[1, 1],
                 title="Impact de l'Epsilon du Jacobien",
                 xlabel="Epsilon",
                 ylabel="Erreur relative (%)",
                 xscale=log10)
    
    # Filtrer pour garder les configurations où seul epsilon varie
    eps_configs = results_df[(results_df.nx .== base_nx) .&
                            (results_df.ny .== base_ny) .&
                            (results_df.nmarkers .== base_nmarkers) .&
                            (results_df.smooth_factor .== base_smooth) .&
                            (results_df.window_size .== base_window), :]
    
    if nrow(eps_configs) > 1
        eps_configs = sort(eps_configs, :jacobian_epsilon)
        scatter!(ax_eps, eps_configs.jacobian_epsilon, 100 .* eps_configs.radius_error,
                marker=:circle, color=:blue, markersize=12)
        lines!(ax_eps, eps_configs.jacobian_epsilon, 100 .* eps_configs.radius_error,
              color=:blue, linewidth=2)
        
        # Ajouter une ligne pour le nombre moyen d'itérations
        ax_iter = Axis(fig_eps[1, 1], ylabel="Itérations moyennes",
                      yaxisposition=:right,
                      yticklabelcolor=:red)
        scatter!(ax_iter, eps_configs.jacobian_epsilon, eps_configs.mean_iterations,
                marker=:square, color=:red, markersize=10)
        lines!(ax_iter, eps_configs.jacobian_epsilon, eps_configs.mean_iterations,
              color=:red, linewidth=2, linestyle=:dash)
        
        # Légende
        Legend(fig_eps[1, 2], [
            MarkerElement(marker=:circle, color=:blue),
            MarkerElement(marker=:square, color=:red)
        ], ["Erreur rayon", "Itérations moyennes"])
    end
    
    save(joinpath(results_dir, "epsilon_impact.png"), fig_eps)
    
    # 4. Précision vs. Paramètres de Lissage
    fig_smooth = Figure(size=(800, 500))
    ax_smooth = Axis(fig_smooth[1, 1],
                    title="Impact des Paramètres de Lissage",
                    xlabel="Facteur de lissage",
                    ylabel="Erreur relative (%)")
    
    # Filtrer pour les configurations où seul le facteur de lissage varie
    smooth_configs = results_df[(results_df.nx .== base_nx) .&
                              (results_df.ny .== base_ny) .&
                              (results_df.nmarkers .== base_nmarkers) .&
                              (results_df.jacobian_epsilon .== base_eps) .&
                              (results_df.window_size .== base_window), :]
    
    if nrow(smooth_configs) > 1
        smooth_configs = sort(smooth_configs, :smooth_factor)
        scatter!(ax_smooth, smooth_configs.smooth_factor, 100 .* smooth_configs.radius_error,
                marker=:circle, color=:blue, markersize=12)
        lines!(ax_smooth, smooth_configs.smooth_factor, 100 .* smooth_configs.radius_error,
              color=:blue, linewidth=2)
        
        # Ajouter l'irrégularité
        ax_smooth_irreg = Axis(fig_smooth[1, 1], ylabel="Irrégularité (%)",
                              yaxisposition=:right,
                              yticklabelcolor=:red)
        scatter!(ax_smooth_irreg, smooth_configs.smooth_factor, 100 .* smooth_configs.radius_std,
                marker=:square, color=:red, markersize=10)
        lines!(ax_smooth_irreg, smooth_configs.smooth_factor, 100 .* smooth_configs.radius_std,
              color=:red, linewidth=2, linestyle=:dash)
        
        # Légende
        Legend(fig_smooth[1, 2], [
            MarkerElement(marker=:circle, color=:blue),
            MarkerElement(marker=:square, color=:red)
        ], ["Erreur rayon", "Irrégularité"])
    end
    
    save(joinpath(results_dir, "smooth_factor_impact.png"), fig_smooth)
    
    # 5. Précision vs. Taille de Fenêtre
    fig_window = Figure(size=(800, 500))
    ax_window = Axis(fig_window[1, 1],
                    title="Impact de la Taille de Fenêtre",
                    xlabel="Taille de fenêtre",
                    ylabel="Erreur relative (%)")
    
    # Filtrer pour les configurations où seule la taille de fenêtre varie
    window_configs = results_df[(results_df.nx .== base_nx) .&
                              (results_df.ny .== base_ny) .&
                              (results_df.nmarkers .== base_nmarkers) .&
                              (results_df.jacobian_epsilon .== base_eps) .&
                              (results_df.smooth_factor .== base_smooth), :]
    
    if nrow(window_configs) > 1
        window_configs = sort(window_configs, :window_size)
        scatter!(ax_window, window_configs.window_size, 100 .* window_configs.radius_error,
                marker=:circle, color=:blue, markersize=12)
        lines!(ax_window, window_configs.window_size, 100 .* window_configs.radius_error,
              color=:blue, linewidth=2)
        
        # Ajouter l'irrégularité
        ax_window_irreg = Axis(fig_window[1, 1], ylabel="Irrégularité (%)",
                              yaxisposition=:right,
                              yticklabelcolor=:red)
        scatter!(ax_window_irreg, window_configs.window_size, 100 .* window_configs.radius_std,
                marker=:square, color=:red, markersize=10)
        lines!(ax_window_irreg, window_configs.window_size, 100 .* window_configs.radius_std,
              color=:red, linewidth=2, linestyle=:dash)
        
        # Légende
        Legend(fig_window[1, 2], [
            MarkerElement(marker=:circle, color=:blue),
            MarkerElement(marker=:square, color=:red)
        ], ["Erreur rayon", "Irrégularité"])
    end
    
    save(joinpath(results_dir, "window_size_impact.png"), fig_window)
    
    # 6. Tableau récapitulatif des meilleures configurations
    fig_summary = Figure(size=(1000, 400))
    ax_summary = Axis(fig_summary[1, 1],
                     title="Comparaison des Configurations",
                     xlabel="Configuration",
                     ylabel="Erreur relative (%)")
    
    # Trier les configurations par erreur croissante
    top_configs = sort(results_df, :radius_error)[1:min(6, nrow(results_df)), :]
    
    # Create x positions for the bars
    x_positions = collect(1:nrow(top_configs))
    config_labels = string.(top_configs.config_id)
    
    # Use positions and heights format for barplot
    barplot!(ax_summary, x_positions, 100 .* top_configs.radius_error,
            color=:blue, label="Erreur rayon")
    
    # Set the x-tick labels to configuration IDs
    ax_summary.xticks = (x_positions, config_labels)
    
    # Axe pour le temps d'exécution
    ax_runtime = Axis(fig_summary[1, 1], ylabel="Temps d'exécution (s)",
                     yaxisposition=:right,
                     yticklabelcolor=:red)
    
    # Use positions and heights format for barplot on second axis
    barplot!(ax_runtime, x_positions .+ 0.2, top_configs.runtime,
            color=:red, label="Temps d'exécution", width=0.2)
    
    # Légende commune
    Legend(fig_summary[1, 2], [
        PolyElement(color=:blue),
        PolyElement(color=:red)
    ], ["Erreur rayon", "Temps d'exécution"])
    save(joinpath(results_dir, "best_configs.png"), fig_summary)
    
    return fig_markers, fig_mesh, fig_eps, fig_smooth, fig_window, fig_summary
end

# Exécuter l'étude paramétrique
results_df, results_dir = run_parametric_study()
println("Étude paramétrique terminée. Résultats sauvegardés dans: $results_dir")

