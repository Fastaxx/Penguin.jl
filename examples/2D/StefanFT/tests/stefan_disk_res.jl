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
    Newton_params = (2, 1e-7, 1e-7, 0.8)  # max_iter, tol, reltol, α
    
    # Mesurer le temps d'exécution
    start_time = time()
    
    # Exécuter la simulation
    solver = StefanMono2D(Fluide, bc_b, bc, Δt, u0, mesh, "BE")
    
    # Initialiser la structure pour suivre les résidus
    all_residuals = Dict{Int, Vector{Float64}}()
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
        "xf_log" => all_xf_log,
        "timestep_history" => timestep_history,
        "runtime" => runtime
    )
end

"""
    compare_mesh_residuals(mesh_sizes::Vector{Int}, n_timesteps::Int=10)

Compare les résidus de simulation pour différentes tailles de maillage et génère 
des graphiques de comparaison.
"""
function compare_mesh_residuals(mesh_sizes::Vector{Int}, n_timesteps::Int=10)
    # Stocker les résultats
    results = Dict[]
    
    # Exécuter les simulations pour chaque taille de maillage
    for nx in mesh_sizes
        result = run_stefan_simulation(nx, n_timesteps=n_timesteps)
        push!(results, result)
    end
    
    # Créer le répertoire des résultats
    results_dir = joinpath(pwd(), "residuals_comparison")
    mkpath(results_dir)
    
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
        save(joinpath(results_dir, "residuals_timestep_$(step).png"), fig)
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
    
    save(joinpath(results_dir, "residuals_all_meshes.png"), fig_composite)
    
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
    save(joinpath(results_dir, "convergence_rates.png"), fig_rates)
    
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
    
    println("\nVisualisations sauvegardées dans: $results_dir")
    
    return results_dir
end

# Exécuter la comparaison pour différentes tailles de maillage
mesh_sizes = [24, 32, 48, 64]
n_timesteps = 10               # Nombre de pas de temps à simuler

results_dir = compare_mesh_residuals(mesh_sizes, n_timesteps)