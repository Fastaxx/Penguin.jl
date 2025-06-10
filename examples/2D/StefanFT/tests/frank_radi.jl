using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Statistics
using DataFrames
using Printf

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
    
       # Récupérer le dernier résidu - robustly handle missing keys
    if haskey(residuals, last_timestep)
        last_residual = minimum(residuals[last_timestep])
    else
        # If the specific timestep key doesn't exist, find the latest available
        available_steps = keys(residuals) |> collect |> sort
        if !isempty(available_steps)
            latest_step = available_steps[end]
            last_residual = minimum(residuals[latest_step])
        else
            # Fallback if no residuals are available
            last_residual = NaN
        end
    end
    
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
        "markers" => final_markers
    )
end

"""
    analyze_mesh_convergence()

Réalise une étude de convergence en maillage et génère des graphiques comparatifs.
"""
function analyze_mesh_convergence()
    # Tailles de maillage à tester
    mesh_sizes = [ (24, 24), (32, 32), (48, 48), (64, 64)]
    
    println("Étude de convergence en maillage pour le problème de Stefan")
    println("==========================================================")
    println("Tailles de maillage à tester : ", mesh_sizes)
    
    # Nombre de marqueurs constant
    nmarkers = 100
    
    # Stocker les résultats
    results = Dict[]
    
    # Exécuter les simulations
    for (nx, ny) in mesh_sizes
        result = run_simulation_with_mesh_size(nx, ny, nmarkers)
        push!(results, result)
        println("Mesh $(nx)×$(ny): Rayon = $(result["radius"]), Erreur = $(result["analytical_error"] * 100)%")
    end
    
    # Considérer le maillage le plus fin comme référence
    finest_result = results[end]
    reference_radius = finest_result["radius"]
    
    # Recalculer les erreurs par rapport au maillage le plus fin
    for result in results
        result["mesh_error"] = abs(result["radius"] - reference_radius) / reference_radius
    end
    
    # Créer le répertoire des résultats
    results_dir = joinpath(pwd(), "mesh_convergence_results")
    mkpath(results_dir)

    # 1. Graphique d'erreur de rayon vs taille de maillage
    fig_error = Figure(size=(800, 600))
    ax_error = Axis(fig_error[1, 1], 
                  title="Convergence en Maillage - Erreur de Rayon", 
                  xlabel="Pas d'espace (Δx)", 
                  ylabel="Erreur Relative",
                  xscale=log10, yscale=log10)
    
    dx_values = [r["dx"] for r in results]
    mesh_errors = [r["mesh_error"] for r in results]
    analytical_errors = [r["analytical_error"] for r in results]
    

    # Ajouter les points pour les erreurs par rapport à la solution analytique
    scatter!(ax_error, dx_values, analytical_errors, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=12, 
            color=:red)
    
    lines!(ax_error, dx_values, analytical_errors, 
          color=:red, linewidth=2, linestyle=:dash)

    # Référence d'ordre 1 (Δx) - basée sur l'erreur de maillage
    ref_x = range(minimum(dx_values), maximum(dx_values), length=100)
    ref_y = [mesh_errors[end] * (x / dx_values[end]) for x in ref_x]
    lines!(ax_error, ref_x, ref_y, 
          color=:blue, linewidth=1, linestyle=:dot,
          label="Ordre 1 (Δx)")
          
    
    # Référence d'ordre 2 (Δx²) - basée sur l'erreur analytique
    ref_x = range(minimum(dx_values), maximum(dx_values), length=100)
    ref_y = [analytical_errors[end] * (x / dx_values[end])^2 for x in ref_x]
    lines!(ax_error, ref_x, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 2 (Δx²)")
    
    # Légende
    axislegend(ax_error, position=:rb)

    # 2. Graphique de rayon vs taille de maillage
    fig_radius = Figure(size=(800, 600))
    ax_radius = Axis(fig_radius[1, 1],
                    title="Convergence en Maillage - Valeur du Rayon",
                    xlabel="Pas d'espace (Δx)",
                    ylabel="Rayon Final")
    
    radii = [r["radius"] for r in results]
    nx_values = [r["nx"] for r in results]
    
    scatter!(ax_radius, dx_values, radii,
            marker=:circle,
            markersize=12,
            color=:blue)
    
    lines!(ax_radius, dx_values, radii,
          color=:blue, linewidth=2)
    
    # Ajouter la ligne de référence pour le maillage le plus fin
    hlines!(ax_radius, [reference_radius],
           color=:red, linewidth=2, linestyle=:dash,
           label="Maillage le plus fin ($(nx_values[end])×$(nx_values[end]))")
    
    # Légende
    axislegend(ax_radius, position=:rb)
    
    # 3. Graphique de performance
    fig_perf = Figure(size=(800, 600))
    ax_perf = Axis(fig_perf[1, 1],
                  title="Performance et Coût de Calcul",
                  xlabel="Nombre de Cellules (nx×ny)",
                  ylabel="Temps d'Exécution (s)")
    
    n_cells = [r["nx"] * r["ny"] for r in results]
    runtimes = [r["runtime"] for r in results]
    
    scatter!(ax_perf, n_cells, runtimes,
            marker=:circle,
            markersize=12,
            color=:blue)
    
    lines!(ax_perf, n_cells, runtimes,
          color=:blue, linewidth=2)
    
    # Ajouter les labels de taille de maillage
    for i in 1:length(n_cells)
        nx, ny = mesh_sizes[i]
        text!(ax_perf, n_cells[i], runtimes[i] * 1.05,
             text="$(nx)×$(ny)",
             fontsize=12)
    end
    
    # Référence d'ordre N²
    if length(n_cells) >= 2
        ref_factor = runtimes[end-1] / n_cells[end-1]^2
        ref_x_cells = range(minimum(n_cells), maximum(n_cells), length=10)
        ref_y_runtime = [ref_factor * x^2 for x in ref_x_cells]
        lines!(ax_perf, ref_x_cells, ref_y_runtime,
              color=:red, linestyle=:dash, linewidth=2,
              label="Ordre O(N²)")
    end
    
    # Légende
    axislegend(ax_perf, position=:rb)
    
    # 4. Tableau récapitulatif
    println("\nRésumé des Résultats:")
    println("==========================================")
    println("Maillage | Rayon  |  Erreur vs. Fin | Erreur vs. Analytic | Runtime (s)")
    println("------------------------------------------")
    
    for i in 1:length(results)
        r = results[i]
        nx, ny = mesh_sizes[i]
        @printf("%3d×%-3d | %6.4f | %10.6f%% | %10.6f%% | %7.2f\n", 
                nx, ny, r["radius"], r["mesh_error"]*100, r["analytical_error"]*100, r["runtime"])
    end
    println("==========================================")
    
    # 5. Visualisation des interfaces pour différentes tailles de maillage
    fig_interfaces = Figure(size=(800, 800))
    ax_interfaces = Axis(fig_interfaces[1, 1],
                       title="Comparaison des Interfaces pour Différentes Tailles de Maillage",
                       xlabel="x", ylabel="y",
                       aspect=DataAspect())
    
    # Générer des couleurs pour chaque taille de maillage
    colors = cgrad(:viridis, length(mesh_sizes))
    
    # Tracer l'interface pour chaque maillage
    for (i, result) in enumerate(results)
        markers = result["markers"]
        nx, ny = mesh_sizes[i]
        
        # Extraire les coordonnées pour le tracé
        marker_x = [m[1] for m in markers]
        marker_y = [m[2] for m in markers]
        
        # Tracer l'interface comme une ligne fermée
        lines!(ax_interfaces, marker_x, marker_y, 
              color=colors[i], 
              linewidth=2,
              label="$(nx)×$(ny)")
    end
    
    # Ajouter la légende
    axislegend(ax_interfaces, position=:rt)
    
    # Sauvegarder les figures
    save(joinpath(results_dir, "mesh_error_convergence.png"), fig_error)
    save(joinpath(results_dir, "mesh_radius_convergence.png"), fig_radius)
    save(joinpath(results_dir, "mesh_performance.png"), fig_perf)
    save(joinpath(results_dir, "mesh_interfaces.png"), fig_interfaces)
    
    # Afficher les figures
    display(fig_error)
    display(fig_radius)
    display(fig_perf)
    display(fig_interfaces)
    
    println("\nFigures sauvegardées dans: $results_dir")
    
    return results
end

# Exécuter l'étude de convergence
results = analyze_mesh_convergence()