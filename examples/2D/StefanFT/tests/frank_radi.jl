using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using Statistics
using Printf
using CSV
using DataFrames

"""
    run_simulation_with_mesh_size(nx::Int, ny::Int, nmarkers::Int=100)

Exécute une simulation du problème de Stefan avec la taille de maillage spécifiée
et retourne les données de résultats.
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
    Newton_params = (30, 1e-7, 1e-7, 0.8)  # max_iter, tol, reltol, α
    
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
    
    # Calculer le rayon final
    last_timestep = maximum(keys(xf_log))
    final_markers = xf_log[last_timestep]
    center_x = sum(m[1] for m in final_markers) / length(final_markers)
    center_y = sum(m[2] for m in final_markers) / length(final_markers)
    final_radii = [sqrt((m[1] - center_x)^2 + (m[2] - center_y)^2) for m in final_markers]
    final_radius = mean(final_radii)
    radius_std = std(final_radii) / final_radius  # Écart-type normalisé
    
    # Calcul du rayon analytique
    analytical_radius = interface_position(t_final)
    
    # Calculer l'erreur par rapport à la solution analytique
    analytical_error = abs(final_radius - analytical_radius) / analytical_radius
    
    # Calculer le nombre d'itérations total
    total_iterations = sum(length(residuals[k]) for k in keys(residuals))
    
    # Récupérer le dernier résidu
    if haskey(residuals, last_timestep)
        last_residual = minimum(residuals[last_timestep])
    else
        # Trouver le dernier pas de temps disponible
        available_steps = collect(keys(residuals))
        if !isempty(available_steps)
            last_residual = minimum(residuals[maximum(available_steps)])
        else
            last_residual = NaN
        end
    end
    
    # Sauvegarder les marqueurs dans un format plus simple pour le CSV
    markers_x = [m[1] for m in final_markers]
    markers_y = [m[2] for m in final_markers]
    
    return Dict(
        "nx" => nx,
        "ny" => ny, 
        "dx" => Δx,
        "radius" => final_radius,
        "radius_std" => radius_std,
        "analytical_radius" => analytical_radius,
        "analytical_error" => analytical_error,
        "iterations" => total_iterations,
        "final_residual" => last_residual,
        "runtime" => runtime,
        "markers_x" => markers_x,
        "markers_y" => markers_y
    )
end

"""
    run_convergence_study()

Exécute l'étude de convergence et sauvegarde les résultats dans des fichiers CSV.
"""
function run_convergence_study()
    # Tailles de maillage à tester
    mesh_sizes = [(24, 24), (32, 32), (48, 48), (64, 64), (96, 96)]
    
    println("Étude de convergence en maillage pour le problème de Stefan")
    println("==========================================================")
    println("Tailles de maillage à tester : ", mesh_sizes)
    
    # Nombre de marqueurs constant
    nmarkers = 100
    
    # Créer le répertoire des résultats
    results_dir = joinpath(pwd(), "mesh_convergence_data")
    mkpath(results_dir)
    
    # Stocker les résultats principaux
    convergence_results = []
    
    # Exécuter les simulations
    for (nx, ny) in mesh_sizes
        result = run_simulation_with_mesh_size(nx, ny, nmarkers)
        push!(convergence_results, result)
        
        # Sauvegarder les marqueurs de l'interface pour cette simulation
        markers_df = DataFrame(
            x = result["markers_x"],
            y = result["markers_y"]
        )
        CSV.write(joinpath(results_dir, "markers_$(nx)x$(ny).csv"), markers_df)
        
        println("Mesh $(nx)×$(ny): Rayon = $(result["radius"]), Erreur = $(result["analytical_error"] * 100)%")
    end
    
    # Créer un DataFrame pour les résultats principaux
    results_df = DataFrame(
        nx = [r["nx"] for r in convergence_results],
        ny = [r["ny"] for r in convergence_results],
        dx = [r["dx"] for r in convergence_results],
        radius = [r["radius"] for r in convergence_results],
        radius_std = [r["radius_std"] for r in convergence_results],
        analytical_radius = [r["analytical_radius"] for r in convergence_results],
        analytical_error = [r["analytical_error"] for r in convergence_results],
        iterations = [r["iterations"] for r in convergence_results],
        final_residual = [r["final_residual"] for r in convergence_results],
        runtime = [r["runtime"] for r in convergence_results]
    )
    
    # Sauvegarder le DataFrame principal
    CSV.write(joinpath(results_dir, "convergence_results.csv"), results_df)
    
    println("\nRésultats sauvegardés dans: $results_dir")
    
    return results_dir
end

# Exécuter l'étude de convergence
#results_dir = run_convergence_study()
println("Données enregistrées dans: $results_dir")

using CSV
using DataFrames
using CairoMakie
using Statistics
using Printf

"""
    plot_convergence_results(data_dir::String)

Charge les données de l'étude de convergence et génère des graphiques comparatifs.
"""
function plot_convergence_results(data_dir::String)
    # Charger les résultats principaux
    results_file = joinpath(data_dir, "convergence_results.csv")
    if !isfile(results_file)
        error("Le fichier de résultats $results_file n'existe pas.")
    end
    
    df = CSV.read(results_file, DataFrame)
    
    # Créer le répertoire pour les figures
    plots_dir = joinpath(dirname(data_dir), "mesh_convergence_plots")
    mkpath(plots_dir)
    
    # Calculer l'erreur par rapport au maillage le plus fin
    finest_idx = findmax(df.nx)[2]
    reference_radius = df.radius[finest_idx]
    df.mesh_error = abs.(df.radius .- reference_radius) ./ reference_radius
    
    # 1. Graphique d'erreur de rayon vs taille de maillage avec références d'ordre
    fig_error = Figure(size=(900, 700))
    ax_error = Axis(fig_error[1, 1], 
                  title="Convergence en Maillage - Erreur de Rayon", 
                  xlabel="Pas d'espace (Δx)", 
                  ylabel="Erreur Relative",
                  xscale=log10, yscale=log10)
    
    # Ajouter les points pour les erreurs par rapport à la solution analytique
    scatter!(ax_error, df.dx, df.analytical_error, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=15, 
            color=:red)
    
    lines!(ax_error, df.dx, df.analytical_error, 
          color=:red, linewidth=2, linestyle=:dash)
    
    # Référence d'ordre 1 (Δx) - clairement visible
    ref_idx = findfirst(x -> x > 0, df.mesh_error)  # Premier point non-zéro
    ref_x = [minimum(df.dx)*0.9, maximum(df.dx)*1.1]  # Étendre légèrement la ligne
    ref_factor = df.mesh_error[ref_idx] / df.dx[ref_idx]
    ref_y = [ref_factor * x for x in ref_x]
    
    lines!(ax_error, ref_x, ref_y, 
          color=:darkorange, linewidth=2, linestyle=:dash,
          label="Ordre 1 (Δx)")
    
    # Référence d'ordre 2 (Δx²)
    ref_factor2 = df.analytical_error[ref_idx] / (df.dx[ref_idx]^2)
    ref_y2 = [ref_factor2 * x^2 for x in ref_x]
    
    lines!(ax_error, ref_x, ref_y2, 
          color=:purple, linewidth=2, linestyle=:dot,
          label="Ordre 2 (Δx²)")
    
    # Annotations pour clarifier
    text!(ax_error, ref_x[1]*1.2, ref_y[1]*1.2, text="O(Δx)", fontsize=14)
    text!(ax_error, ref_x[1]*1.2, ref_y2[1]*1.2, text="O(Δx²)", fontsize=14)
    
    # Légende
    axislegend(ax_error, position=:rt, framevisible=true)
    
    # 2. Graphique de rayon vs taille de maillage
    fig_radius = Figure(size=(900, 700))
    ax_radius = Axis(fig_radius[1, 1],
                    title="Convergence en Maillage - Valeur du Rayon",
                    xlabel="Pas d'espace (Δx)",
                    ylabel="Rayon Final")
    
    scatter!(ax_radius, df.dx, df.radius,
            marker=:circle,
            markersize=15,
            color=:blue)
    
    lines!(ax_radius, df.dx, df.radius,
          color=:blue, linewidth=2)
    
    # Ajouter la ligne de référence pour le maillage le plus fin
    hlines!(ax_radius, [reference_radius],
           color=:red, linewidth=2, linestyle=:dash,
           label="Maillage le plus fin ($(df.nx[finest_idx])×$(df.ny[finest_idx]))")
    
    # Ajouter la ligne de référence pour la solution analytique
    hlines!(ax_radius, [df.analytical_radius[1]],  # Ils sont tous identiques
           color=:green, linewidth=2, linestyle=:dot,
           label="Solution Analytique")
    
    # Légende
    axislegend(ax_radius, position=:rb, framevisible=true)
    
    # 3. Graphique de performance
    fig_perf = Figure(size=(900, 700))
    ax_perf = Axis(fig_perf[1, 1],
                  title="Performance et Coût de Calcul",
                  xlabel="Nombre de Cellules (nx×ny)",
                  ylabel="Temps d'Exécution (s)")
    
    n_cells = df.nx .* df.ny
    
    scatter!(ax_perf, n_cells, df.runtime,
            marker=:circle,
            markersize=15,
            color=:blue)
    
    lines!(ax_perf, n_cells, df.runtime,
          color=:blue, linewidth=2)
    
    # Ajouter les labels de taille de maillage
    for i in 1:nrow(df)
        text!(ax_perf, n_cells[i], df.runtime[i] * 1.05,
             text="$(df.nx[i])×$(df.ny[i])",
             fontsize=14)
    end
    
    # Référence d'ordre N²
    if nrow(df) >= 2
        ref_factor = df.runtime[end-1] / n_cells[end-1]^2
        ref_x_cells = range(minimum(n_cells), maximum(n_cells), length=10)
        ref_y_runtime = [ref_factor * x^2 for x in ref_x_cells]
        lines!(ax_perf, ref_x_cells, ref_y_runtime,
              color=:red, linestyle=:dash, linewidth=2,
              label="Ordre O(N²)")
    end
    
    # Légende
    axislegend(ax_perf, position=:lt, framevisible=true)
    
    # 4. Visualisation des interfaces pour différentes tailles de maillage
    fig_interfaces = Figure(size=(900, 900))
    ax_interfaces = Axis(fig_interfaces[1, 1],
                       title="Comparaison des Interfaces pour Différentes Tailles de Maillage",
                       xlabel="x", ylabel="y",
                       aspect=DataAspect())
    
    # Générer des couleurs pour chaque taille de maillage
    colors = cgrad(:viridis, nrow(df))
    
    # Tracer l'interface pour chaque maillage
    for i in 1:nrow(df)
        nx, ny = df.nx[i], df.ny[i]
        
        # Charger les marqueurs
        markers_file = joinpath(data_dir, "markers_$(nx)x$(ny).csv")
        if isfile(markers_file)
            markers_df = CSV.read(markers_file, DataFrame)
            
            # Tracer l'interface comme une ligne fermée
            lines!(ax_interfaces, 
                  vcat(markers_df.x, markers_df.x[1]), 
                  vcat(markers_df.y, markers_df.y[1]),
                  color=colors[i], 
                  linewidth=3,
                  label="$(nx)×$(ny)")
        end
    end
    
    # Ajouter la légende
    axislegend(ax_interfaces, position=:rt, framevisible=true)
    
    # 5. Tableau récapitulatif
    println("\nRésumé des Résultats:")
    println("=================================================================")
    println("Maillage | Rayon  |  Erreur vs. Fin | Erreur vs. Analytic | Runtime (s)")
    println("-----------------------------------------------------------------")
    
    for i in 1:nrow(df)
        @printf("%3d×%-3d | %6.4f | %10.6f%% | %10.6f%% | %7.2f\n", 
                df.nx[i], df.ny[i], df.radius[i], 
                df.mesh_error[i]*100, df.analytical_error[i]*100, df.runtime[i])
    end
    println("=================================================================")
    
    # Sauvegarder les figures
    save(joinpath(plots_dir, "mesh_error_convergence.png"), fig_error)
    save(joinpath(plots_dir, "mesh_radius_convergence.png"), fig_radius)
    save(joinpath(plots_dir, "mesh_performance.png"), fig_perf)
    save(joinpath(plots_dir, "mesh_interfaces.png"), fig_interfaces)
    
    # Afficher les figures
    display(fig_error)
    display(fig_radius)
    display(fig_perf)
    display(fig_interfaces)
    
    println("\nFigures sauvegardées dans: $plots_dir")
    
    return plots_dir
end

# Chercher le répertoire des données par défaut
default_data_dir = joinpath(pwd(), "mesh_convergence_data")
    
if isdir(default_data_dir)
    plot_convergence_results(default_data_dir)
else
    println("Veuillez spécifier le chemin vers le répertoire contenant les données:")
    data_dir = readline()
    if isdir(data_dir)
        plot_convergence_results(data_dir)
    else
        println("Répertoire invalide.")
    end
end
