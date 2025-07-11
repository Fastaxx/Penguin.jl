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
using Interpolations
using Dates
using LsqFit  # Pour curve_fit

# Weighted Lp or L∞ norm helper
function lp_norm(errors, indices, pval, capacity)
    if pval == Inf
        return maximum(abs.(errors[indices]))
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V[indices,indices]))^(1/pval)
    end
end

"""
Calcule la norme Lp relative pondérée entre l'erreur et une solution de référence.
"""
function relative_lp_norm(errors, indices, pval, capacity, ref_solution)
    if pval == Inf
        return maximum(abs.(errors[indices])) / maximum(abs.(ref_solution[indices]))
    else
        error_sum = 0.0
        ref_sum = 0.0
        
        for i in indices
            Vi = capacity.V[i,i]
            error_sum += (abs(errors[i])^pval) * Vi
            ref_sum += (abs(ref_solution[i])^pval) * Vi
        end
        
        if isempty(indices) || ref_sum ≈ 0.0
            return 0.0
        end
        
        return (error_sum / sum(capacity.V[indices,indices]))^(1/pval) / 
               (ref_sum / sum(capacity.V[indices,indices]))^(1/pval)
    end
end

"""
    calculate_convergence_rates(dx_vals, summary_df)

Calcule les taux de convergence en ajustant une droite dans un plan log-log.
"""
function calculate_convergence_rates(dx_vals, summary_df)
    # Modèle pour ajustement: log(err) = p*log(h) + c
    fit_model(x, p) = p[1]*x .+ p[2]
    
    # Fit on log scale
    log_h = log.(dx_vals)
    
    function do_fit(log_err)
        try
            fit_result = curve_fit(fit_model, log_h, log_err, [-1.0, 0.0])
            return fit_result.param[1], fit_result.param[2]
        catch
            return NaN, NaN
        end
    end
    
    # Get convergence rates
    p_global, _ = do_fit(log.(summary_df.l2_global_error))
    p_full, _ = do_fit(log.(summary_df.l2_full_error))
    p_cut, _ = do_fit(log.(summary_df.l2_cut_error))
    p_linf_global, _ = do_fit(log.(summary_df.linf_global_error))
    
    # Round for display
    p_global = round(p_global, digits=2)
    p_full = round(p_full, digits=2)
    p_cut = round(p_cut, digits=2)
    p_linf_global = round(p_linf_global, digits=2)
    
    return Dict(
        "l2_global" => p_global,
        "l2_full" => p_full,
        "l2_cut" => p_cut,
        "linf_global" => p_linf_global
    )
end

"""
    run_simulation_with_mesh_size(nx::Int, ny::Int, nmarkers::Int=100)

Exécute une simulation du problème de Stefan avec la taille de maillage spécifiée
et retourne les données de résultats incluant le champ de température.
"""
function run_simulation_with_mesh_size(nx::Int, ny::Int, nmarkers::Int=100)
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
    front = FrontTracker() 
    create_circle!(front, 0.0, 0.0, interface_position(t_init), nmarkers)
    
    # Définir la position initiale du front
    body = (x, y, t, _=0) -> -sdf(front, x, y)
    
    # Définir le pas de temps
    Δt = 0.1*(lx / nx)^2  # Adapter le pas de temps à la taille de maillage
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
    Newton_params = (10, 1e-7, 1e-7, 1)  # max_iter, tol, reltol, α
    
    # Mesurer le temps d'exécution
    start_time = time()
    
    # Exécuter la simulation
    solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Résoudre le problème pour un seul pas de temps
    solver, residuals, xf_log, timestep_history, phase, position_increments = 
        solve_StefanMono2D!(solver, Fluide, front, Δt, t_init, t_final,
                            bc_b, bc, stef_cond, mesh, "BE";
                            Newton_params=Newton_params, 
                            jacobian_epsilon=1e-6,
                            smooth_factor=0.7,
                            window_size=10,
                            method=Base.:\)
    
    runtime = time() - start_time
    
    # Extraire le champ de température final
    final_state = solver.states[end]
    npts = (nx+1) * (ny+1)
    temperature_field = final_state[1:npts]
    
    # Calculer la solution analytique aux mêmes points pour comparaison
    analytical_temps = zeros(npts)
    for idx in 1:length(centroids)
        centroid = centroids[idx]
        x, y = centroid[1], centroid[2]
        r = sqrt(x^2 + y^2)
        analytical_temps[idx] = analytical_temperature(r, t_final)
    end
    
    # Calculer l'erreur par rapport à la solution analytique
    error_vs_analytic = temperature_field - analytical_temps
    
    # Utiliser la norme LP pondérée
    indices = 1:npts
    l2_error = lp_norm(error_vs_analytic, indices, 2, phase.capacity)
    linf_error = lp_norm(error_vs_analytic, indices, Inf, phase.capacity)
    
    # Pour l'erreur relative, normaliser par la norme pondérée de la solution analytique
    l2_norm_analytical = sqrt(sum((analytical_temps.^2) .* diag(capacity.V)[1:npts]) / sum(diag(capacity.V)[1:npts]))
    l2_rel_error = l2_error / l2_norm_analytical

        # Séparer les erreurs par type de cellule
    cell_types = phase.capacity.cell_types[1:npts]
    idx_all = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full = findall(cell_types .== 1)
    idx_cut = findall(cell_types .== -1)
    idx_empty = findall(cell_types .== 0)
    
    # Calculer les erreurs pour chaque type de cellule
    l2_global_error = lp_norm(error_vs_analytic, idx_all, 2, phase.capacity)
    l2_full_error = isempty(idx_full) ? 0.0 : lp_norm(error_vs_analytic, idx_full, 2, phase.capacity)
    l2_cut_error = isempty(idx_cut) ? 0.0 : lp_norm(error_vs_analytic, idx_cut, 2, phase.capacity)
    
    # Erreurs relatives par type de cellule
    l2_rel_global_error = relative_lp_norm(error_vs_analytic, idx_all, 2, phase.capacity, analytical_temps)
    l2_rel_full_error = relative_lp_norm(error_vs_analytic, idx_full, 2, phase.capacity, analytical_temps)
    l2_rel_cut_error = relative_lp_norm(error_vs_analytic, idx_cut, 2, phase.capacity, analytical_temps)
    
    # Également pour la norme Linf
    linf_global_error = lp_norm(error_vs_analytic, idx_all, Inf, phase.capacity)
    linf_full_error = isempty(idx_full) ? 0.0 : lp_norm(error_vs_analytic, idx_full, Inf, phase.capacity)
    linf_cut_error = isempty(idx_cut) ? 0.0 : lp_norm(error_vs_analytic, idx_cut, Inf, phase.capacity)
    
    # Formater pour retourner le champ et les données de maillage
    temp_mesh_data = (
        temperatures = temperature_field,
        x_coords = mesh.nodes[1],
        y_coords = mesh.nodes[2],
        nx = nx,
        ny = ny,
        analytical = analytical_temps
    )
    
    return Dict(
        "nx" => nx,
        "ny" => ny, 
        "dx" => Δx,
        "temperature_field" => temp_mesh_data,
        "capacity" => phase.capacity,
        "l2_error" => l2_error,
        "linf_error" => linf_error,
        "l2_rel_error" => l2_rel_error,
        "l2_global_error" => l2_global_error,
        "l2_full_error" => l2_full_error,
        "l2_cut_error" => l2_cut_error,
        "l2_rel_global_error" => l2_rel_global_error,
        "l2_rel_full_error" => l2_rel_full_error,
        "l2_rel_cut_error" => l2_rel_cut_error,
        "linf_global_error" => linf_global_error,
        "linf_full_error" => linf_full_error,
        "linf_cut_error" => linf_cut_error,
        "runtime" => runtime,
        "centroids" => centroids,
        "n_full_cells" => length(idx_full),
        "n_cut_cells" => length(idx_cut)
    )
end

"""
    save_temperature_data_to_csv(result, results_dir, prefix="")

Sauvegarde les données de température d'une simulation dans des fichiers CSV.
"""
function save_temperature_data_to_csv(result, results_dir, prefix="")
    nx, ny = result["nx"], result["ny"]
    
    # Créer un nom de base pour les fichiers
    base_name = isempty(prefix) ? "mesh_$(nx)_$(ny)" : "$(prefix)_mesh_$(nx)_$(ny)"
    
    # 1. Sauvegarder les métriques d'erreur et d'information sur le maillage
    metrics_df = DataFrame(
        nx = nx,
        ny = ny,
        dx = result["dx"],
        l2_error = result["l2_error"],
        linf_error = result["linf_error"],
        l2_rel_error = result["l2_rel_error"],
        l2_global_error = result["l2_global_error"],
        l2_full_error = result["l2_full_error"],
        l2_cut_error = result["l2_cut_error"],
        l2_rel_global_error = result["l2_rel_global_error"],
        l2_rel_full_error = result["l2_rel_full_error"],
        l2_rel_cut_error = result["l2_rel_cut_error"],
        linf_global_error = result["linf_global_error"],
        linf_full_error = result["linf_full_error"],
        linf_cut_error = result["linf_cut_error"],
        n_full_cells = result["n_full_cells"],
        n_cut_cells = result["n_cut_cells"],
        runtime = result["runtime"]
    )
    CSV.write(joinpath(results_dir, "$(base_name)_metrics.csv"), metrics_df)
    
    # 2. Sauvegarder le champ de température
    temp_data = result["temperature_field"]
    temps = temp_data.temperatures
    x_coords = temp_data.x_coords
    y_coords = temp_data.y_coords
    analytical = temp_data.analytical
    
    # Créer un DataFrame pour les données de température
    # Pour chaque point (i, j), nous avons besoin des coordonnées x, y, temp et temp_analytical
    temp_df = DataFrame(
        index = 1:length(temps),
        temperature = temps,
        temperature_analytical = analytical
    )
    CSV.write(joinpath(results_dir, "$(base_name)_temperature.csv"), temp_df)
    
    # 3. Sauvegarder les informations sur le maillage
    mesh_df = DataFrame(
        x = x_coords,
        y = y_coords
    )
    CSV.write(joinpath(results_dir, "$(base_name)_mesh.csv"), mesh_df)
    
    return joinpath(results_dir, "$(base_name)_metrics.csv")
end

"""
    calculate_interpolation_error_and_save(coarse_result, fine_result, results_dir)

Calcule l'erreur d'interpolation entre deux résultats et sauvegarde les métriques.
"""
function calculate_interpolation_error_and_save(coarse_result, fine_result, results_dir)
    # Extraire les informations de maillage
    coarse_nx, coarse_ny = coarse_result["nx"], coarse_result["ny"]
    fine_nx, fine_ny = fine_result["nx"], fine_result["ny"]
    
    # Extraire les données du champ fin
    fine_data = fine_result["temperature_field"]
    fine_temps = fine_data.temperatures
    fine_capacity = fine_result["capacity"]
    
    # Extraire les données du champ grossier
    coarse_data = coarse_result["temperature_field"]
    coarse_temps = coarse_data.temperatures
    coarse_x = coarse_data.x_coords
    coarse_y = coarse_data.y_coords
    
    # Extraire les coordonnées du maillage fin
    fine_x = fine_data.x_coords
    fine_y = fine_data.y_coords
    
    # Reshaper pour l'interpolation
    coarse_temps_2d = reshape(coarse_temps, (coarse_nx+1, coarse_ny+1))
    
    # Créer l'interpolation
    itp = LinearInterpolation((coarse_x, coarse_y), coarse_temps_2d, extrapolation_bc=Line())
    
    # Interpoler sur le maillage fin
    interp_temps = zeros(length(fine_x), length(fine_y))
    for i in 1:length(fine_x)
        for j in 1:length(fine_y)
            interp_temps[i, j] = itp(fine_x[i], fine_y[j])
        end
    end
    
    # Calculer l'erreur
    fine_temps_2d = reshape(fine_temps, (fine_nx+1, fine_ny+1))
    error = vec(interp_temps - fine_temps_2d)
    
    # Calculer les normes d'erreur
    npts = length(fine_temps)
    indices = 1:npts
    l2_error = lp_norm(error, indices, 2, fine_capacity)
    linf_error = lp_norm(error, indices, Inf, fine_capacity)
    
    # Pour l'erreur relative, normaliser par la norme pondérée
    l2_norm_fine = sqrt(sum((fine_temps.^2) .* diag(fine_capacity.V)[1:npts]) / sum(diag(fine_capacity.V)[1:npts]))
    l2_rel_error = l2_error / l2_norm_fine
    
    # Sauvegarder les métriques d'erreur dans un CSV
    error_df = DataFrame(
        coarse_nx = coarse_nx, 
        coarse_ny = coarse_ny,
        fine_nx = fine_nx,
        fine_ny = fine_ny,
        mesh_l2_error = l2_error,
        mesh_linf_error = linf_error,
        mesh_l2_rel_error = l2_rel_error
    )
    
    # Nom du fichier
    filename = joinpath(results_dir, "interpolation_error_$(coarse_nx)_$(coarse_ny)_to_$(fine_nx)_$(fine_ny).csv")
    CSV.write(filename, error_df)
    
    return l2_error, linf_error, l2_rel_error, filename
end

"""
    run_temperature_convergence_study(mesh_sizes=[(24, 24), (32, 32), (48, 48), (64, 64)], nmarkers=100)

Exécute une étude de convergence du champ de température et sauvegarde les résultats en CSV.
"""
function run_temperature_convergence_study(mesh_sizes=[(24, 24), (32, 32), (48, 48), (64, 64), (96, 96)], nmarkers=100)
    # Créer un répertoire pour les résultats avec timestamp
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    results_dir = joinpath(pwd(), "temperature_study_$(timestamp)")
    mkpath(results_dir)
    
    println("Étude de convergence du champ de température pour le problème de Stefan")
    println("===================================================================")
    println("Tailles de maillage à tester : ", mesh_sizes)
    println("Résultats sauvegardés dans : ", results_dir)
    
    # Stocker les résultats
    results = Dict[]
    metrics_files = String[]
    
    # Exécuter les simulations et sauvegarder les résultats
    for (nx, ny) in mesh_sizes
        result = run_simulation_with_mesh_size(nx, ny, nmarkers)
        push!(results, result)
        
        # Sauvegarder les données en CSV
        metrics_file = save_temperature_data_to_csv(result, results_dir)
        push!(metrics_files, metrics_file)
        
        println("Mesh $(nx)×$(ny): L2 relative error = $(result["l2_rel_error"] * 100)%, L∞ error = $(result["linf_error"]))")
    end
    
    # Considérer le maillage le plus fin comme référence
    finest_result = results[end]
    
    # Calculer et sauvegarder les erreurs d'interpolation
    interpolation_files = String[]
    for (i, result) in enumerate(results[1:end-1])
        l2_error, linf_error, l2_rel_error, error_file = calculate_interpolation_error_and_save(
            result, finest_result, results_dir
        )
        push!(interpolation_files, error_file)
        
        # Ajouter les erreurs aux résultats pour faciliter l'accès ultérieur
        result["mesh_l2_error"] = l2_error
        result["mesh_linf_error"] = linf_error
        result["mesh_l2_rel_error"] = l2_rel_error
    end
    
    # Ajouter des entrées pour le maillage le plus fin (erreurs nulles par définition)
    finest_result["mesh_l2_error"] = 0.0
    finest_result["mesh_linf_error"] = 0.0
    finest_result["mesh_l2_rel_error"] = 0.0
    
    # Créer un fichier récapitulatif
    summary_df = DataFrame(
        nx = [r["nx"] for r in results],
        ny = [r["ny"] for r in results],
        dx = [r["dx"] for r in results],
        l2_error_vs_analytic = [r["l2_error"] for r in results],
        linf_error_vs_analytic = [r["linf_error"] for r in results],
        l2_rel_error_vs_analytic = [r["l2_rel_error"] for r in results],
        l2_global_error = [r["l2_global_error"] for r in results],
        l2_full_error = [r["l2_full_error"] for r in results],
        l2_cut_error = [r["l2_cut_error"] for r in results],
        l2_rel_global_error = [r["l2_rel_global_error"] for r in results],
        l2_rel_full_error = [r["l2_rel_full_error"] for r in results],
        l2_rel_cut_error = [r["l2_rel_cut_error"] for r in results],
        linf_global_error = [r["linf_global_error"] for r in results],
        linf_full_error = [r["linf_full_error"] for r in results],
        linf_cut_error = [r["linf_cut_error"] for r in results],
        n_full_cells = [r["n_full_cells"] for r in results],
        n_cut_cells = [r["n_cut_cells"] for r in results],
        runtime = [r["runtime"] for r in results]
    )
    
    # Ajouter les erreurs d'interpolation
    summary_df.mesh_l2_error = [haskey(r, "mesh_l2_error") ? r["mesh_l2_error"] : missing for r in results]
    summary_df.mesh_linf_error = [haskey(r, "mesh_linf_error") ? r["mesh_linf_error"] : missing for r in results]
    summary_df.mesh_l2_rel_error = [haskey(r, "mesh_l2_rel_error") ? r["mesh_l2_rel_error"] : missing for r in results]
    
    summary_file = joinpath(results_dir, "temperature_convergence_summary.csv")
    CSV.write(summary_file, summary_df)
    
    # Sauvegarder les informations de configuration pour pouvoir refaire les graphiques plus tard
    config_df = DataFrame(
        study_date = timestamp,
        results_dir = results_dir,
        mesh_sizes = [string(m) for m in mesh_sizes],
        nmarkers = nmarkers
    )
    CSV.write(joinpath(results_dir, "study_config.csv"), config_df)
    
    println("\nÉtude de convergence terminée.")
    println("Résumé des résultats sauvegardé dans : $(summary_file)")
    
    return results_dir, results
end

"""
    plot_temperature_convergence_results(results_dir)

Génère des graphiques d'analyse de convergence à partir des fichiers CSV d'une étude précédente.
"""
function plot_temperature_convergence_results(results_dir)
    # Vérifier l'existence du répertoire
    if !isdir(results_dir)
        error("Le répertoire $results_dir n'existe pas")
    end
    
    # Créer un sous-répertoire pour les graphiques
    plots_dir = joinpath(results_dir, "plots")
    mkpath(plots_dir)
    
    # Lire le fichier de résumé
    summary_file = joinpath(results_dir, "temperature_convergence_summary.csv")
    if !isfile(summary_file)
        error("Fichier de résumé non trouvé: $summary_file")
    end
    
    summary = CSV.read(summary_file, DataFrame)
    
    # 1. Graphique d'erreur L2 vs taille de maillage
    fig_error = Figure(size=(900, 600))
    ax_error = Axis(fig_error[1, 1], 
                   title="Convergence du Champ de Température - Erreur L2", 
                   xlabel="Pas d'espace (Δx)", 
                   ylabel="Erreur Relative L2",
                   xscale=log10, yscale=log10)
    
    dx_values = summary.dx
    analytic_l2_errors = summary.l2_rel_error_vs_analytic
    mesh_l2_errors = filter(x -> !ismissing(x), summary.mesh_l2_rel_error)
    
    """
    # Ajouter les points pour les erreurs par rapport au maillage fin
    if length(mesh_l2_errors) > 1
        # Exclure le dernier point (maillage fin) car son erreur est nulle par définition
        dx_for_mesh = filter(!ismissing, dx_values[1:end-1])
        
        scatter!(ax_error, dx_for_mesh, mesh_l2_errors[1:end-1], 
                label="Vs. Maillage Fin", 
                marker=:circle, 
                markersize=12, 
                color=:blue)
        
        lines!(ax_error, dx_for_mesh, mesh_l2_errors[1:end-1],
              color=:blue, linewidth=2)
    end
    """
    
    # Ajouter les points pour les erreurs par rapport à la solution analytique
    scatter!(ax_error, dx_values, analytic_l2_errors, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=12, 
            color=:red)
    
    lines!(ax_error, dx_values, analytic_l2_errors, 
          color=:red, linewidth=2, linestyle=:dash)

    # Référence d'ordre 1 (Δx)
    ref_y = [analytic_l2_errors[1] / dx_values[1] * x for x in dx_values]
    lines!(ax_error, dx_values, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 1 (Δx)")

    
    # Référence d'ordre 2 (Δx²)
    # Utiliser les premiers points pour établir la référence
    ref_x = [minimum(dx_values), maximum(dx_values)]
    ref_factor = analytic_l2_errors[1] / dx_values[1]^2
    ref_y = [ref_factor * x^2 for x in ref_x]
    
    lines!(ax_error, ref_x, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 2 (Δx²)")
    
    # Légende
    axislegend(ax_error, position=:rb)
    
    # 2. Graphique d'erreur L∞ vs taille de maillage
    fig_linf = Figure(size=(900, 600))
    ax_linf = Axis(fig_linf[1, 1], 
                  title="Convergence du Champ de Température - Erreur L∞", 
                  xlabel="Pas d'espace (Δx)", 
                  ylabel="Erreur L∞",
                  xscale=log10, yscale=log10)
    
    analytic_linf_errors = summary.linf_error_vs_analytic
    mesh_linf_errors = filter(x -> !ismissing(x), summary.mesh_linf_error)
    
    """
    if length(mesh_linf_errors) > 1
        # Exclure le dernier point (maillage fin) car son erreur est nulle par définition
        dx_for_mesh = filter(!ismissing, dx_values[1:end-1])
        
        scatter!(ax_linf, dx_for_mesh, mesh_linf_errors[1:end-1],
                label="Vs. Maillage Fin", 
                marker=:circle, 
                markersize=12, 
                color=:blue)
        
        lines!(ax_linf, dx_for_mesh, mesh_linf_errors[1:end-1],
              color=:blue, linewidth=2)
    end
    """

    scatter!(ax_linf, dx_values, analytic_linf_errors, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=12, 
            color=:red)
    
    lines!(ax_linf, dx_values, analytic_linf_errors, 
          color=:red, linewidth=2, linestyle=:dash)

    # Référence d'ordre 1 (Δx)
    ref_factor = analytic_linf_errors[1] / dx_values[1]
    ref_y = [ref_factor * x for x in ref_x]

    lines!(ax_linf, ref_x, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 1 (Δx)")
    
    # Référence d'ordre 2 (Δx²)
    ref_factor = analytic_linf_errors[1] / dx_values[1]^2
    ref_y = [ref_factor * x^2 for x in ref_x]
    
    lines!(ax_linf, ref_x, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 2 (Δx²)")
    
    # Légende
    axislegend(ax_linf, position=:rb)
    
    # 3. Tableau de comparaison sous forme de graphique
    fig_table = Figure(size=(1000, 600))
    
    # Créer le tableau comme un graphique
    mesh_labels = ["$(nx)×$(ny)" for (nx, ny) in zip(summary.nx, summary.ny)]
    
    # Formater les valeurs pour le tableau
    l2_rel_values = ["$(round(err*100, digits=4))%" for err in summary.l2_rel_error_vs_analytic]
    linf_values = ["$(round(err, digits=6))" for err in summary.linf_error_vs_analytic]
    runtime_values = ["$(round(rt, digits=2))s" for rt in summary.runtime]
    
    # Créer un graphique de tableau
    ax_table = Axis(fig_table[1, 1])
    hidedecorations!(ax_table)
    hidespines!(ax_table)
    
    # Créer les en-têtes du tableau
    text!(ax_table, 0.1, 0.9, text="Maillage", fontsize=18, align=(:left, :center))
    text!(ax_table, 0.3, 0.9, text="L2 Rel. (vs Analyt.)", fontsize=18, align=(:left, :center))
    text!(ax_table, 0.55, 0.9, text="L∞ (vs Analyt.)", fontsize=18, align=(:left, :center))
    text!(ax_table, 0.75, 0.9, text="Temps (s)", fontsize=18, align=(:left, :center))
    
    # Ajouter les valeurs
    n_rows = length(mesh_labels)
    for i in 1:n_rows
        row_y = 0.85 - (i * 0.05)
        text!(ax_table, 0.1, row_y, text=mesh_labels[i], fontsize=16, align=(:left, :center))
        text!(ax_table, 0.3, row_y, text=l2_rel_values[i], fontsize=16, align=(:left, :center))
        text!(ax_table, 0.55, row_y, text=linf_values[i], fontsize=16, align=(:left, :center))
        text!(ax_table, 0.75, row_y, text=runtime_values[i], fontsize=16, align=(:left, :center))
    end
    
    # 4. Evolution de l'erreur avec la taille du maillage
    fig_evolution = Figure(size=(1200, 500))
    
    ax_evol_l2 = Axis(fig_evolution[1, 1],
                     title="Évolution de l'erreur L2",
                     xlabel="Taille de maillage",
                     ylabel="Erreur L2 Rel.",
                     yscale=log10)
    
    ax_evol_linf = Axis(fig_evolution[1, 2],
                       title="Évolution de l'erreur L∞",
                       xlabel="Taille de maillage",
                       ylabel="Erreur L∞",
                       yscale=log10)
    
    # Convertir les tailles de maillage pour l'affichage
    mesh_sizes_for_x = [nx for nx in summary.nx]
    
    lines!(ax_evol_l2, mesh_sizes_for_x, summary.l2_rel_error_vs_analytic,
           color=:red, linewidth=2)
    
    lines!(ax_evol_linf, mesh_sizes_for_x, summary.linf_error_vs_analytic,
           color=:blue, linewidth=2)

    # 5. Graphique de convergence par type de cellule
    fig_cell_types = Figure(size=(1200, 600)) # Réduire la hauteur puisqu'on n'a plus besoin d'espace pour le tableau

    # Calculer les taux de convergence
    conv_rates = calculate_convergence_rates(dx_values, summary)

    # 5a. Erreurs L2 par type de cellule
    ax_cell_l2 = Axis(fig_cell_types[1, 1],
                    title="Relative L2 Error by Cell Type",
                    xlabel="Space Step (Δx)",
                    ylabel="Relative L2 Error",
                    xscale=log10, yscale=log10)

    # Tracer les erreurs par type de cellule
    scatter!(ax_cell_l2, dx_values, summary.l2_global_error,
            label="Global Cells (ooc=$(conv_rates["l2_global"]))",
            marker=:circle, markersize=10, color=:blue)
    lines!(ax_cell_l2, dx_values, summary.l2_global_error,
        color=:blue, linewidth=2)

    scatter!(ax_cell_l2, dx_values, summary.l2_full_error,
            label="Full Cells (ooc=$(conv_rates["l2_full"]))",
            marker=:rect, markersize=10, color=:green)
    lines!(ax_cell_l2, dx_values, summary.l2_full_error,
        color=:green, linewidth=2)

    scatter!(ax_cell_l2, dx_values, summary.l2_cut_error,
            label="Cut Cells (ooc=$(conv_rates["l2_cut"]))",
            marker=:utriangle, markersize=10, color=:red)
    lines!(ax_cell_l2, dx_values, summary.l2_cut_error,
        color=:red, linewidth=2)

    # Référence d'ordre 1 et 2
    ref_x = [minimum(dx_values), maximum(dx_values)]
    ref_y1 = [summary.l2_global_error[1] / dx_values[1] * x for x in ref_x]
    ref_y2 = [summary.l2_global_error[1] / dx_values[1]^2 * x^2 for x in ref_x]

    lines!(ax_cell_l2, ref_x, ref_y1,
        color=:black, linewidth=1, linestyle=:dash,
        label="Order 1")
    lines!(ax_cell_l2, ref_x, ref_y2,
        color=:black, linewidth=1, linestyle=:dot,
        label="Order 2")

    axislegend(ax_cell_l2, position=:rb)

    # 5b. Erreurs L∞ par type de cellule
    ax_cell_linf = Axis(fig_cell_types[1, 2],  # Changer pour [1, 2] au lieu de [2, 1]
                    title="Relative L∞ Error by Cell Type",
                    xlabel="Space Step (Δx)",
                    ylabel="Relative L∞ Error",
                    xscale=log10, yscale=log10)

    # Calculer l'ordre de convergence pour L∞ si ce n'est pas déjà fait
    p_linf_full = NaN
    p_linf_cut = NaN
    try
        log_h = log.(dx_values)
        fit_model(x, p) = p[1]*x .+ p[2]
        
        # Pour les cellules complètes
        fit_result = curve_fit(fit_model, log_h, log.(summary.linf_full_error), [-1.0, 0.0])
        p_linf_full = round(fit_result.param[1], digits=2)
        
        # Pour les cellules coupées
        fit_result = curve_fit(fit_model, log_h, log.(summary.linf_cut_error), [-1.0, 0.0])
        p_linf_cut = round(fit_result.param[1], digits=2)
    catch
        # En cas d'erreur lors de l'ajustement
    end

    scatter!(ax_cell_linf, dx_values, summary.linf_global_error,
            label="Global Cells (ooc=$(conv_rates["linf_global"]))",
            marker=:circle, markersize=10, color=:blue)
    lines!(ax_cell_linf, dx_values, summary.linf_global_error,
        color=:blue, linewidth=2)

    scatter!(ax_cell_linf, dx_values, summary.linf_full_error,
            label="Full Cells (ooc=$(isnan(p_linf_full) ? "N/A" : p_linf_full))",
            marker=:rect, markersize=10, color=:green)
    lines!(ax_cell_linf, dx_values, summary.linf_full_error,
        color=:green, linewidth=2)

    scatter!(ax_cell_linf, dx_values, summary.linf_cut_error,
            label="Cut Cells (ooc=$(isnan(p_linf_cut) ? "N/A" : p_linf_cut))",
            marker=:utriangle, markersize=10, color=:red)
    lines!(ax_cell_linf, dx_values, summary.linf_cut_error,
        color=:red, linewidth=2)

    # Référence d'ordre 1 et 2
    ref_y1_linf = [summary.linf_global_error[1] / dx_values[1] * x for x in ref_x]
    ref_y2_linf = [summary.linf_global_error[1] / dx_values[1]^2 * x^2 for x in ref_x]

    lines!(ax_cell_linf, ref_x, ref_y1_linf,
        color=:black, linewidth=1, linestyle=:dash,
        label="Order 1")
    lines!(ax_cell_linf, ref_x, ref_y2_linf,
        color=:black, linewidth=1, linestyle=:dot,
        label="Order 2")

    axislegend(ax_cell_linf, position=:rb)

    
    # Sauvegarder les figures
    save(joinpath(plots_dir, "l2_convergence.png"), fig_error)
    save(joinpath(plots_dir, "linf_convergence.png"), fig_linf)
    save(joinpath(plots_dir, "results_table.png"), fig_table)
    save(joinpath(plots_dir, "error_evolution.png"), fig_evolution)
    save(joinpath(plots_dir, "cell_types_convergence.png"), fig_cell_types)

    println("Graphiques sauvegardés dans: $plots_dir")
    
    # Renvoyer les figures
    return fig_error, fig_linf, fig_table, fig_evolution, fig_cell_types
end

"""
    analyze_temperature_convergence()

Fonction principale pour exécuter l'étude de convergence:
1. Exécuter les simulations et sauvegarder les données en CSV
2. Générer les graphiques d'analyse
"""
function analyze_temperature_convergence()
    # 1. Configurer et exécuter l'étude de convergence
    mesh_sizes = [(24, 24), (32, 32), (48, 48), (64, 64), (96, 96)]
    nmarkers = 100
    
    # Exécuter les simulations et sauvegarder les résultats en CSV
    results_dir, _ = run_temperature_convergence_study(mesh_sizes, nmarkers)
    
    # 2. Générer les graphiques à partir des fichiers CSV
    fig_error, fig_linf, fig_table, fig_evolution, fig_cell_types = plot_temperature_convergence_results(results_dir)
    
    # Afficher les principales figures
    display(fig_error)
    display(fig_linf)
    display(fig_table)
    display(fig_cell_types)  # Ajouter l'affichage du nouveau graphique
    
    return results_dir
end

results_dir = "temperature_study_2025-06-12_09-08-02"
fig_error, fig_linf, fig_table, fig_evolution, fig_cell_types = plot_temperature_convergence_results(results_dir)


# Point d'entrée pour le script
#results_dir = analyze_temperature_convergence()
println("\nÉtude terminée. Résultats dans: $results_dir")
