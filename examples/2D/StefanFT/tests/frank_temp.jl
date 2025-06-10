using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Statistics
using DataFrames
using Printf
using Interpolations

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
        "runtime" => runtime,
        "centroids" => centroids
    )
end

"""
    create_interpolated_field(result, target_nx, target_ny)

Interpole le champ de température sur un maillage plus fin pour la comparaison.
"""
function create_interpolated_field(result, target_nx, target_ny)
    # Extraire les données du champ initial
    temp_data = result["temperature_field"]
    temps = temp_data.temperatures
    x = temp_data.x_coords
    y = temp_data.y_coords
    nx, ny = temp_data.nx, temp_data.ny
    
    # Reshaper pour l'interpolation
    temps_2d = reshape(temps, (nx+1, ny+1))
    
    # Créer l'interpolation
    itp = LinearInterpolation((x, y), temps_2d)
    
    # Générer le maillage cible
    x_fine = range(minimum(x), maximum(x), length=target_nx+1)
    y_fine = range(minimum(y), maximum(y), length=target_ny+1)
    
    # Interpoler sur le maillage fin
    temps_fine = [itp(xi, yi) for xi in x_fine, yi in y_fine]
    
    return temps_fine, x_fine, y_fine
end

"""
    calculate_interpolation_error(coarse_result, fine_result)

Calcule l'erreur entre un résultat grossier interpolé et un résultat fin.
"""
function calculate_interpolation_error(coarse_result, fine_result)
    # Extraire les données du champ fin
    fine_data = fine_result["temperature_field"]
    fine_temps = fine_data.temperatures
    fine_nx, fine_ny = fine_data.nx, fine_data.ny
    fine_capacity = fine_result["capacity"]
    
    # Interpoler le champ grossier sur le maillage fin
    interp_temps, _, _ = create_interpolated_field(coarse_result, fine_nx, fine_ny)
    
    # Calculer l'erreur
    fine_temps_2d = reshape(fine_temps, (fine_nx+1, fine_ny+1))
    error = vec(interp_temps - fine_temps_2d)
    
    # Utiliser la norme LP pondérée
    npts = length(fine_temps)
    indices = 1:npts
    l2_error = lp_norm(error, indices, 2, fine_capacity)
    linf_error = lp_norm(error, indices, Inf, fine_capacity)
    
    # Pour l'erreur relative, normaliser par la norme pondérée
    l2_norm_fine = sqrt(sum((fine_temps.^2) .* diag(fine_capacity.V)[1:npts]) / sum(diag(fine_capacity.V)[1:npts]))
    l2_rel_error = l2_error / l2_norm_fine
    
    return l2_error, linf_error, l2_rel_error
end


"""
    analyze_temperature_convergence()

Réalise une étude de convergence du champ de température et génère des graphiques comparatifs.
"""
function analyze_temperature_convergence()
    # Tailles de maillage à tester
    mesh_sizes = [(24, 24), (32, 32), (48, 48), (64, 64)]
    
    println("Étude de convergence du champ de température pour le problème de Stefan")
    println("===================================================================")
    println("Tailles de maillage à tester : ", mesh_sizes)
    
    # Nombre de marqueurs constant
    nmarkers = 100
    
    # Stocker les résultats
    results = Dict[]
    
    # Exécuter les simulations
    for (nx, ny) in mesh_sizes
        result = run_simulation_with_mesh_size(nx, ny, nmarkers)
        push!(results, result)
        println("Mesh $(nx)×$(ny): L2 relative error = $(result["l2_rel_error"] * 100)%, L∞ error = $(result["linf_error"]))")
    end
    
    # Considérer le maillage le plus fin comme référence
    finest_result = results[end]
    
    # Calculer les erreurs par rapport au maillage le plus fin
    for (i, result) in enumerate(results[1:end-1])
        l2_error, linf_error, l2_rel_error = calculate_interpolation_error(result, finest_result)
        result["mesh_l2_error"] = l2_error
        result["mesh_linf_error"] = linf_error
        result["mesh_l2_rel_error"] = l2_rel_error
    end
    
    # Ajouter des entrées pour le maillage le plus fin (erreurs nulles par définition)
    finest_result["mesh_l2_error"] = 0.0
    finest_result["mesh_linf_error"] = 0.0
    finest_result["mesh_l2_rel_error"] = 0.0
    
    # Créer le répertoire des résultats
    results_dir = joinpath(pwd(), "temperature_convergence_results")
    mkpath(results_dir)
    
    # 1. Graphique d'erreur L2 vs taille de maillage
    fig_error = Figure(size=(900, 600))
    ax_error = Axis(fig_error[1, 1], 
                   title="Convergence du Champ de Température - Erreur L2", 
                   xlabel="Pas d'espace (Δx)", 
                   ylabel="Erreur Relative L2",
                   xscale=log10, yscale=log10)
    
    dx_values = [r["dx"] for r in results]
    mesh_l2_errors = [r["mesh_l2_rel_error"] for r in results]
    analytic_l2_errors = [r["l2_rel_error"] for r in results]
    
    # Ajouter les points pour les erreurs par rapport au maillage fin
    scatter!(ax_error, dx_values, mesh_l2_errors, 
            label="Vs. Maillage Fin", 
            marker=:circle, 
            markersize=12, 
            color=:blue)
    
    lines!(ax_error, dx_values, mesh_l2_errors, 
          color=:blue, linewidth=2)
    
    # Ajouter les points pour les erreurs par rapport à la solution analytique
    scatter!(ax_error, dx_values, analytic_l2_errors, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=12, 
            color=:red)
    
    lines!(ax_error, dx_values, analytic_l2_errors, 
          color=:red, linewidth=2, linestyle=:dash)
    
    # Référence d'ordre 2 (Δx²)
    # Utiliser les premiers points pour établir la référence
    idx = findfirst(e -> e > 0, mesh_l2_errors)
    ref_x = [minimum(dx_values), maximum(dx_values)]
    ref_factor = mesh_l2_errors[idx] / dx_values[idx]^2
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
    
    mesh_linf_errors = [r["mesh_linf_error"] for r in results]
    analytic_linf_errors = [r["linf_error"] for r in results]
    
    scatter!(ax_linf, dx_values, mesh_linf_errors, 
            label="Vs. Maillage Fin", 
            marker=:circle, 
            markersize=12, 
            color=:blue)
    
    lines!(ax_linf, dx_values, mesh_linf_errors, 
          color=:blue, linewidth=2)
    
    scatter!(ax_linf, dx_values, analytic_linf_errors, 
            label="Vs. Solution Analytique", 
            marker=:rect,
            markersize=12, 
            color=:red)
    
    lines!(ax_linf, dx_values, analytic_linf_errors, 
          color=:red, linewidth=2, linestyle=:dash)
    
    # Référence d'ordre 2 (Δx²)
    idx = findfirst(e -> e > 0, mesh_linf_errors)
    ref_factor = mesh_linf_errors[idx] / dx_values[idx]^2
    ref_y = [ref_factor * x^2 for x in ref_x]
    
    lines!(ax_linf, ref_x, ref_y, 
          color=:black, linewidth=1, linestyle=:dot,
          label="Ordre 2 (Δx²)")
    
    # Légende
    axislegend(ax_linf, position=:rb)
    
    # 3. Visualisation des champs de température pour différentes tailles de maillage
    fig_temps = Figure(size=(1200, 900))
    
    # Générer un colormap cohérent pour toutes les visualisations
    all_temps = Float64[]
    for result in results
        temps = result["temperature_field"].temperatures
        push!(all_temps, minimum(temps), maximum(temps))
    end
    temp_range = (minimum(all_temps), maximum(all_temps))
    
    # Tracer les champs de température
    for (i, result) in enumerate(results)
        row = (i-1) ÷ 2 + 1
        col = ((i-1) % 2) + 1
        
        temp_data = result["temperature_field"]
        temps = temp_data.temperatures
        x = temp_data.x_coords
        y = temp_data.y_coords
        nx, ny = temp_data.nx, temp_data.ny
        
        # Reshaper pour la visualisation
        temps_2d = reshape(temps, (nx+1, ny+1))
        
        # Créer le subplot
        ax = Axis(fig_temps[row, col], 
                 title="Maillage $(nx)×$(ny)", 
                 xlabel="x", ylabel="y",
                 aspect=DataAspect())
        
        hm = heatmap!(ax, x, y, temps_2d, 
                     colormap=:thermal,
                     colorrange=temp_range)
        
        # Ajouter une colorbar pour le dernier subplot seulement
        if i == length(results)
            Colorbar(fig_temps[row, col+1], hm, label="Température")
        end
    end
    
    # 4. Cartes d'erreur par rapport au maillage le plus fin
    fig_error_maps = Figure(size=(1200, 900))
    
    # Préparer les données du maillage fin pour la comparaison
    fine_data = finest_result["temperature_field"]
    fine_temps = fine_data.temperatures
    fine_nx, fine_ny = fine_data.nx, fine_data.ny
    
    # Tracer les cartes d'erreur
    error_max = 0.0  # Pour normaliser l'échelle de couleur
    
    for (i, result) in enumerate(results[1:end-1])
        # Interpoler sur le maillage fin
        interp_temps, x_fine, y_fine = create_interpolated_field(result, fine_nx, fine_ny)
        
        # Calculer l'erreur
        fine_temps_2d = reshape(fine_temps, (fine_nx+1, fine_ny+1))
        error = abs.(interp_temps - fine_temps_2d)
        
        # Mettre à jour l'erreur maximale
        error_max = max(error_max, maximum(error))
    end
    
    for (i, result) in enumerate(results[1:end-1])
        row = (i-1) ÷ 2 + 1
        col = ((i-1) % 2) + 1
        
        # Interpoler sur le maillage fin
        interp_temps, x_fine, y_fine = create_interpolated_field(result, fine_nx, fine_ny)
        
        # Calculer l'erreur
        fine_temps_2d = reshape(fine_temps, (fine_nx+1, fine_ny+1))
        error = abs.(interp_temps - fine_temps_2d)
        
        # Créer le subplot
        ax = Axis(fig_error_maps[row, col], 
                 title="Erreur: Maillage $(result["nx"])×$(result["ny"]) vs $(fine_nx)×$(fine_ny)", 
                 xlabel="x", ylabel="y",
                 aspect=DataAspect())
        
        hm = heatmap!(ax, x_fine, y_fine, error, 
                     colormap=:plasma,
                     colorrange=(0, error_max))
        
        # Ajouter une colorbar pour le dernier subplot seulement
        if i == length(results) - 1
            Colorbar(fig_error_maps[row, col+1], hm, label="Erreur Absolue")
        end
    end
    
    # 5. Tableau récapitulatif
    println("\nRésumé des Résultats:")
    println("=================================================================")
    println("Maillage | L2 vs Fin | L∞ vs Fin | L2 vs Analytique | Runtime (s)")
    println("-----------------------------------------------------------------")
    
    for i in 1:length(results)
        r = results[i]
        nx, ny = r["nx"], r["ny"]
        @printf("%3d×%-3d | %9.6f | %9.6f | %9.6f | %7.2f\n", 
                nx, ny, r["mesh_l2_rel_error"]*100, r["mesh_linf_error"], 
                r["l2_rel_error"]*100, r["runtime"])
    end
    println("=================================================================")
    
    # Sauvegarder les figures
    save(joinpath(results_dir, "temperature_l2_convergence.png"), fig_error)
    save(joinpath(results_dir, "temperature_linf_convergence.png"), fig_linf)
    save(joinpath(results_dir, "temperature_fields.png"), fig_temps)
    save(joinpath(results_dir, "temperature_error_maps.png"), fig_error_maps)
    
    # Afficher les figures
    display(fig_error)
    display(fig_linf)
    display(fig_temps)
    display(fig_error_maps)
    
    println("\nFigures sauvegardées dans: $results_dir")
    
    return results
end

# Exécuter l'étude de convergence
results = analyze_temperature_convergence()