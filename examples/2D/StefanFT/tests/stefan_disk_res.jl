using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Statistics
using DataFrames
using CSV
using Printf
using ColorSchemes

"""
    run_stefan_simulation(nx::Int; n_timesteps::Int=10)

Exécute la simulation du problème de Stefan avec la taille de maillage spécifiée et
retourne les résidus sur plusieurs pas de temps.
"""
function run_stefan_simulation(nx::Int; n_timesteps::Int=10)
    ny = nx  # Maillage carré
    
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
    lx, ly = 16.0, 16.0
    x0, y0 = -8.0, -8.0
    Δx, Δy = lx/nx, ly/ny
    mesh = Penguin.Mesh((nx, ny), (lx, ly), (x0, y0))
    
    println("\nSimulation avec maillage $(nx)×$(ny), Δx=$(Δx), Δy=$(Δy)")
    
    # Créer le front-tracking
    nmarkers = 100
    front = FrontTracker() 
    create_circle!(front, 0.0, 0.0, interface_position(t_init), nmarkers)
    
    # Définir la position initiale du front
    body = (x, y, t, _=0) -> -sdf(front, x, y)
    
    # Définir le pas de temps adapté à la taille du maillage
    Δt = 0.1*(lx / nx)^2
    t_final = t_init + n_timesteps * Δt
    
    # Maillage espace-temps
    STmesh = Penguin.SpaceTimeMesh(mesh, [t_init, t_init + Δt], tag=mesh.tag)
    
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
    Newton_params = (20, 1e-7, 1e-7, 0.8)  # max_iter, tol, reltol, α
    
    # Mesurer le temps d'exécution
    start_time = time()
    
    # Exécuter la simulation
    solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Initialiser la structure pour suivre les résidus
    all_residuals = Dict{Int, Vector{Float64}}()
    all_position_increments = Dict{Int, Vector{Float64}}()
    all_xf_log = Dict{Int, Vector{Tuple{Float64, Float64}}}()
    timestep_history = Vector{Tuple{Float64, Float64}}()
    
    # Simulation pour n_timesteps pas de temps
    current_time = t_init
    
    for step in 1:n_timesteps
        next_time = current_time + Δt
        
        # Résoudre le problème pour un seul pas de temps
        solver, step_residuals, xf_log, step_history, phase, position_increments = 
            solve_StefanMono2D!(solver, Fluide, front, Δt, current_time, next_time,
                               bc_b, bc, stef_cond, mesh, "BE";
                               Newton_params=Newton_params, 
                               jacobian_epsilon=1e-6,
                               smooth_factor=0.7,
                               window_size=10,
                               method=Base.:\)
        
        # Stocker les résidus pour ce pas de temps
        all_residuals[step] = step_residuals[minimum(keys(step_residuals))]
        
        # Stocker les incréments de position pour ce pas de temps
        all_position_increments[step] = position_increments[minimum(keys(position_increments))]
        
        # Mettre à jour les marqueurs d'interface
        all_xf_log[step] = xf_log[maximum(keys(xf_log))]
        
        # Stocker l'historique des pas de temps
        push!(timestep_history, (next_time, Δt))
        
        # Mettre à jour le temps courant
        current_time = next_time
        
        println("Pas de temps $(step) complété, t = $(current_time)")
    end
    
    runtime = time() - start_time
    println("Simulation terminée en $(round(runtime, digits=2)) secondes")
    
    return Dict(
        "nx" => nx,
        "ny" => ny,
        "dx" => Δx,
        "residuals" => all_residuals,
        "position_increments" => all_position_increments,
        "xf_log" => all_xf_log,
        "timestep_history" => timestep_history,
        "runtime" => runtime
    )
end

"""
    save_results_to_csv(results, output_dir::String)

Sauvegarde les résultats de simulation dans des fichiers CSV.
"""
function save_results_to_csv(results, output_dir::String)
    # Créer le répertoire si nécessaire
    mkpath(output_dir)
    
    # Extraire les informations de la simulation
    nx = results["nx"]
    
    # Créer un sous-répertoire pour cette taille de maillage
    mesh_dir = joinpath(output_dir, string(nx))
    mkpath(mesh_dir)
    
    # Sauvegarder les résidus pour chaque pas de temps
    for (step, residuals) in results["residuals"]
        df = DataFrame(
            iteration = 1:length(residuals),
            residual = residuals
        )
        CSV.write(joinpath(mesh_dir, "residuals_step_$(step).csv"), df)
    end
    
    # Sauvegarder les incréments de position pour chaque pas de temps
    for (step, increments) in results["position_increments"]
        df = DataFrame(
            iteration = 1:length(increments),
            increment = increments
        )
        CSV.write(joinpath(mesh_dir, "increments_step_$(step).csv"), df)
    end
    
    # Créer un résumé des statistiques par pas de temps
    summary_rows = []
    
    for step in sort(collect(keys(results["residuals"])))
        residuals = results["residuals"][step]
        
        # Collecter les informations pour ce pas de temps
        if haskey(results["position_increments"], step)
            increments = results["position_increments"][step]
            final_increment = increments[end]
        else
            final_increment = NaN
        end
        
        push!(summary_rows, (
            step = step,
            iterations = length(residuals),
            initial_residual = residuals[1],
            final_residual = residuals[end],
            convergence_ratio = length(residuals) > 1 ? residuals[end] / residuals[1] : NaN,
            final_position_increment = final_increment
        ))
    end
    
    # Sauvegarder le résumé
    summary_df = DataFrame(summary_rows)
    CSV.write(joinpath(mesh_dir, "simulation_summary.csv"), summary_df)
    
    # Créer un fichier de métadonnées
    metadata_df = DataFrame(
        nx = nx,
        ny = results["ny"],
        dx = results["dx"],
        runtime = results["runtime"],
        total_iterations = sum(length(res) for res in values(results["residuals"])),
        avg_iterations_per_step = sum(length(res) for res in values(results["residuals"])) / length(results["residuals"])
    )
    CSV.write(joinpath(mesh_dir, "metadata.csv"), metadata_df)
    
    println("Résultats sauvegardés dans: $(mesh_dir)")
    
    return mesh_dir
end

"""
    compare_mesh_residuals(mesh_sizes::Vector{Int}, n_timesteps::Int=10)

Compare les résidus de simulation pour différentes tailles de maillage, sauvegarde
les données en CSV et génère des graphiques de comparaison.
"""
function compare_mesh_residuals(mesh_sizes::Vector{Int}, n_timesteps::Int=10)
    # Stocker les résultats
    results = Dict[]
    
    # Créer le répertoire des résultats
    results_dir = joinpath(pwd(), "mesh_residuals_data")
    mkpath(results_dir)
    
    # Exécuter les simulations pour chaque taille de maillage
    for nx in mesh_sizes
        result = run_stefan_simulation(nx, n_timesteps=n_timesteps)
        push!(results, result)
        
        # Sauvegarder les résultats en CSV
        save_results_to_csv(result, results_dir)
    end
    
    # Créer le répertoire des graphiques
    plots_dir = joinpath(pwd(), "mesh_residuals_plots")
    mkpath(plots_dir)
    
    # 1. Comparaison des résidus par taille de maillage pour chaque pas de temps
    for step in 1:n_timesteps
        fig = Figure(size=(1000, 600))
        ax = Axis(fig[1, 1],
                 title="Résidus de Newton au pas de temps $step",
                 xlabel="Itération",
                 ylabel="Norme du résidu",
                 yscale=log10)
        
        for (i, result) in enumerate(results)
            nx = result["nx"]
            if haskey(result["residuals"], step)
                residual_vec = result["residuals"][step]
                lines!(ax, 1:length(residual_vec), residual_vec,
                      linewidth=2,
                      label="Maillage $(nx)×$(nx)")
            end
        end
        
        axislegend(ax, position=:rt)
        save(joinpath(plots_dir, "residuals_timestep_$(step).png"), fig)
    end
    
    # 2. Graphique composite des résidus de tous les pas de temps pour chaque maillage
    fig_composite = Figure(size=(1200, 800))
    n_rows = ceil(Int, length(mesh_sizes)/2)
    
    for (i, result) in enumerate(results)
        row = (i-1) ÷ 2 + 1
        col = ((i-1) % 2) + 1
        
        nx = result["nx"]
        ax = Axis(fig_composite[row, col],
                 title="Résidus pour maillage $(nx)×$(nx)",
                 xlabel="Itération",
                 ylabel="Norme du résidu",
                 yscale=log10)
        
        for step in 1:n_timesteps
            if haskey(result["residuals"], step)
                residual_vec = result["residuals"][step]
                lines!(ax, 1:length(residual_vec), residual_vec,
                      linewidth=2,
                      label="Pas de temps $step")
            end
        end
        
        # Limiter le nombre de légendes pour éviter l'encombrement
        if n_timesteps <= 10
            axislegend(ax, position=:rt)
        else
            # Sélectionner quelques pas de temps représentatifs
            steps_to_show = [1, n_timesteps÷2, n_timesteps]
            legend_entries = [(lines, Label("Pas de temps $step")) 
                            for (i, step) in enumerate(steps_to_show) 
                            if haskey(result["residuals"], step)]
            Legend(fig_composite[row, col+1], legend_entries)
        end
    end
    
    save(joinpath(plots_dir, "residuals_all_meshes.png"), fig_composite)
    
    # 3. Visualisation des taux de convergence
    fig_rates = Figure(size=(900, 600))
    ax_rates = Axis(fig_rates[1, 1],
                   title="Taux de Convergence par Taille de Maillage",
                   xlabel="Itération",
                   ylabel="Ratio Résidu_k+1 / Résidu_k",
                   yscale=log10)
    
    for (i, result) in enumerate(results)
        nx = result["nx"]
        
        # Calculer les taux de convergence moyens sur tous les pas de temps
        all_rates = Float64[]
        
        for step in 1:n_timesteps
            if haskey(result["residuals"], step)
                residual_vec = result["residuals"][step]
                if length(residual_vec) > 1
                    rates = [residual_vec[i+1] / residual_vec[i] for i in 1:(length(residual_vec)-1)]
                    append!(all_rates, rates)
                end
            end
        end
        
        # Tracer l'histogramme des taux pour ce maillage
        if !isempty(all_rates)
            density!(ax_rates, all_rates, label="Maillage $(nx)×$(nx)")
        end
    end
    
    axislegend(ax_rates, position=:rt)
    save(joinpath(plots_dir, "convergence_rates.png"), fig_rates)
    
    # 4. Tableau comparatif des statistiques
    println("\nComparaison des Statistiques de Convergence:")
    println("=================================================================")
    println("Maillage | Temps Exec. | Itér. Moy. | Itér. Total | Résidu Final")
    println("-----------------------------------------------------------------")
    
    for result in results
        nx = result["nx"]
        runtime = result["runtime"]
        
        # Calculer le nombre moyen d'itérations par pas de temps
        total_iters = 0
        for step in 1:n_timesteps
            if haskey(result["residuals"], step)
                total_iters += length(result["residuals"][step])
            end
        end
        avg_iters = total_iters / n_timesteps
        
        # Récupérer le dernier résidu du dernier pas de temps
        final_residual = NaN
        for step in n_timesteps:-1:1
            if haskey(result["residuals"], step)
                res_vec = result["residuals"][step]
                if !isempty(res_vec)
                    final_residual = res_vec[end]
                    break
                end
            end
        end
        
        @printf("%3d×%-3d | %10.2f | %10.2f | %11d | %11.2e\n", 
                nx, nx, runtime, avg_iters, total_iters, final_residual)
    end
    println("=================================================================")
    
    println("\nVisualisations sauvegardées dans: $plots_dir")
    println("Données CSV sauvegardées dans: $results_dir")
    
    return results_dir, plots_dir
end

"""
    plot_combined_mesh_residuals(data_dir::String)

Charge les données CSV des résidus pour différentes tailles de maillage et
génère des graphiques comparatifs combinés.
"""
function plot_combined_mesh_residuals(data_dir::String)
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
    
    println("Analyse des données pour les maillages: $mesh_sizes")
    
    # Créer le répertoire pour les graphiques
    plots_dir = joinpath(dirname(data_dir), "combined_residuals_plots")
    mkpath(plots_dir)
    
    # Collecter les résidus et incréments pour chaque maillage et pas de temps
    all_residuals = Dict{Tuple{Int, Int}, Vector{Float64}}()  # (mesh_size, timestep) => residuals
    all_increments = Dict{Tuple{Int, Int}, Vector{Float64}}() # (mesh_size, timestep) => increments
    max_timestep = 0
    
    # Charger toutes les données
    for (mesh_size, subdir) in zip(mesh_sizes, mesh_subdirs)
        # Trouver tous les fichiers de résidus
        residual_files = filter(f -> startswith(f, "residuals_step_") && endswith(f, ".csv"), 
                               readdir(subdir))
        
        for file in residual_files
            # Extraire le numéro du pas de temps du nom de fichier
            m = match(r"residuals_step_(\d+)\.csv", file)
            if m !== nothing
                step = parse(Int, m.captures[1])
                max_timestep = max(max_timestep, step)
                
                # Charger les données de résidus
                df = CSV.read(joinpath(subdir, file), DataFrame)
                all_residuals[(mesh_size, step)] = df.residual
            end
        end
        
        # Trouver tous les fichiers d'incréments
        increment_files = filter(f -> startswith(f, "increments_step_") && endswith(f, ".csv"), 
                                readdir(subdir))
        
        for file in increment_files
            # Extraire le numéro du pas de temps du nom de fichier
            m = match(r"increments_step_(\d+)\.csv", file)
            if m !== nothing
                step = parse(Int, m.captures[1])
                
                # Charger les données d'incréments
                df = CSV.read(joinpath(subdir, file), DataFrame)
                all_increments[(mesh_size, step)] = df.increment
            end
        end
    end
    
    # Définir des couleurs distinctes pour les tailles de maillage
    mesh_colors = cgrad(:thermal, length(mesh_sizes))
    
    # Utiliser différents styles de lignes pour les pas de temps
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    # 1. Graphique des résidus combinés pour le premier pas de temps
    fig_first_step = Figure(size=(1200, 800))
    ax_first = Axis(fig_first_step[1, 1],
                  title="Résidus pour toutes les tailles de maillage - Pas de temps 1",
                  xlabel="Itération",
                  ylabel="Résidu",
                  yscale=log10)
    
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        key = (mesh_size, 1)  # Premier pas de temps
        if haskey(all_residuals, key)
            residuals = all_residuals[key]
            
            color = mesh_colors[i]
            
            lines!(ax_first, 1:length(residuals), residuals,
                  linewidth=2,
                  color=color,
                  label="Maillage $(mesh_size)×$(mesh_size)")
            
            scatter!(ax_first, 1:length(residuals), residuals,
                    markersize=6,
                    color=color)
        end
    end
    
    axislegend(ax_first, position=:rt, framevisible=true)
    save(joinpath(plots_dir, "residuals_first_timestep.png"), fig_first_step)
    
    # 2. Graphique des résidus combinés pour tous les pas de temps et maillages
    # (limité pour la lisibilité)
    fig_selected = Figure(size=(1200, 800))
    ax_selected = Axis(fig_selected[1, 1],
                      title="Résidus pour différentes tailles de maillage et pas de temps sélectionnés",
                      xlabel="Itération",
                      ylabel="Résidu",
                      yscale=log10)
    
    # Sélectionner quelques pas de temps représentatifs
    timesteps_to_show = [1, max(1, max_timestep ÷ 2), max_timestep]
    
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        for (j, step) in enumerate(timesteps_to_show)
            key = (mesh_size, step)
            if haskey(all_residuals, key)
                residuals = all_residuals[key]
                
                color = mesh_colors[i]
                line_style = line_styles[mod1(j, length(line_styles))]
                
                lines!(ax_selected, 1:length(residuals), residuals,
                      linewidth=2,
                      color=color,
                      linestyle=line_style,
                      label="$(mesh_size)×$(mesh_size), t=$step")
            end
        end
    end
    
    axislegend(ax_selected, position=:rt, framevisible=true, nbanks=2)
    save(joinpath(plots_dir, "residuals_selected_timesteps.png"), fig_selected)
    
    # 3. Graphique des incréments de position combinés pour le premier pas de temps
    fig_inc_first = Figure(size=(1200, 800))
    ax_inc_first = Axis(fig_inc_first[1, 1],
                       title="Incréments de position pour toutes les tailles de maillage - Pas de temps 1",
                       xlabel="Itération",
                       ylabel="Incrément de position",
                       yscale=log10)
    
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        key = (mesh_size, 1)  # Premier pas de temps
        if haskey(all_increments, key)
            increments = all_increments[key]
            
            color = mesh_colors[i]
            
            lines!(ax_inc_first, 1:length(increments), increments,
                  linewidth=2,
                  color=color,
                  label="Maillage $(mesh_size)×$(mesh_size)")
            
            scatter!(ax_inc_first, 1:length(increments), increments,
                    markersize=6,
                    color=color)
        end
    end
    
    axislegend(ax_inc_first, position=:rt, framevisible=true)
    save(joinpath(plots_dir, "increments_first_timestep.png"), fig_inc_first)
    
    # 4. Graphique comparatif de la dernière itération pour chaque pas de temps
    fig_final = Figure(size=(1200, 800))
    
    # 4a. Résidus finaux
    ax_final_res = Axis(fig_final[1, 1],
                       title="Résidus finaux par pas de temps",
                       xlabel="Pas de temps",
                       ylabel="Résidu final",
                       yscale=log10)
    
    # 4b. Incréments finaux
    ax_final_inc = Axis(fig_final[1, 2],
                       title="Incréments finaux par pas de temps",
                       xlabel="Pas de temps",
                       ylabel="Incrément final",
                       yscale=log10)
    
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        final_residuals = Float64[]
        final_increments = Float64[]
        timesteps = Int[]
        
        for step in 1:max_timestep
            res_key = (mesh_size, step)
            inc_key = (mesh_size, step)
            
            if haskey(all_residuals, res_key) && !isempty(all_residuals[res_key])
                push!(timesteps, step)
                push!(final_residuals, all_residuals[res_key][end])
                
                if haskey(all_increments, inc_key) && !isempty(all_increments[inc_key])
                    push!(final_increments, all_increments[inc_key][end])
                else
                    push!(final_increments, NaN)
                end
            end
        end
        
        if !isempty(timesteps)
            color = mesh_colors[i]
            
            # Tracer les résidus finaux
            lines!(ax_final_res, timesteps, final_residuals,
                  color=color, linewidth=2,
                  label="Maillage $(mesh_size)×$(mesh_size)")
            
            scatter!(ax_final_res, timesteps, final_residuals,
                    color=color, markersize=8)
            
            # Tracer les incréments finaux
            valid_idx = .!isnan.(final_increments)
            if any(valid_idx)
                lines!(ax_final_inc, timesteps[valid_idx], final_increments[valid_idx],
                      color=color, linewidth=2,
                      label="Maillage $(mesh_size)×$(mesh_size)")
                
                scatter!(ax_final_inc, timesteps[valid_idx], final_increments[valid_idx],
                        color=color, markersize=8)
            end
        end
    end
    
    axislegend(ax_final_res, position=:rt, framevisible=true)
    axislegend(ax_final_inc, position=:rt, framevisible=true)
    
    save(joinpath(plots_dir, "final_values_by_timestep.png"), fig_final)
    
    # 5. Graphique des résidus et incréments combinés (limité à quelques pas de temps)
    fig_combined = Figure(size=(1500, 1000))
    
    # 5a. Résidus
    ax_combined_res = Axis(fig_combined[1, 1],
                          title="Résidus pour différentes tailles de maillage",
                          xlabel="Itération",
                          ylabel="Résidu",
                          yscale=log10)
    
    # 5b. Incréments
    ax_combined_inc = Axis(fig_combined[2, 1],
                          title="Incréments pour différentes tailles de maillage",
                          xlabel="Itération",
                          ylabel="Incrément de position",
                          yscale=log10)
    
    # Sélectionner le pas de temps 1 pour la comparaison
    step = 1
    
    for (i, mesh_size) in enumerate(sort(mesh_sizes))
        res_key = (mesh_size, step)
        inc_key = (mesh_size, step)
        
        if haskey(all_residuals, res_key)
            residuals = all_residuals[res_key]
            color = mesh_colors[i]
            
            lines!(ax_combined_res, 1:length(residuals), residuals,
                  linewidth=2,
                  color=color,
                  label="Maillage $(mesh_size)×$(mesh_size)")
            
            scatter!(ax_combined_res, 1:length(residuals), residuals,
                    markersize=6,
                    color=color)
        end
        
        if haskey(all_increments, inc_key)
            increments = all_increments[inc_key]
            color = mesh_colors[i]
            
            lines!(ax_combined_inc, 1:length(increments), increments,
                  linewidth=2,
                  color=color,
                  label="Maillage $(mesh_size)×$(mesh_size)")
            
            scatter!(ax_combined_inc, 1:length(increments), increments,
                    markersize=6,
                    color=color)
        end
    end
    
    axislegend(ax_combined_res, position=:rt, framevisible=true)
    axislegend(ax_combined_inc, position=:rt, framevisible=true)
    
    save(joinpath(plots_dir, "combined_convergence_step1.png"), fig_combined)
    
    println("Graphiques combinés sauvegardés dans: $plots_dir")
    
    return plots_dir
end

# Exécuter les simulations, sauvegarder les CSV et générer les graphiques
mesh_sizes = [129]
n_timesteps = 5  # Réduire le nombre de pas de temps pour l'exemple

# Exécuter les simulations et sauvegarder les données
#data_dir, initial_plots_dir = compare_mesh_residuals(mesh_sizes, n_timesteps)

data_dir = joinpath(pwd(), "mesh_residuals_data")
# Générer des visualisations combinées à partir des CSV
plot_combined_mesh_residuals(data_dir)