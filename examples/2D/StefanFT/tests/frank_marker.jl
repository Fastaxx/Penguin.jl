using Penguin
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using SpecialFunctions
using CairoMakie
using Statistics

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
    nx, ny = 32, 32
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

# Liste des nombres de marqueurs à tester
markers_to_test = [50, 70, 100, 200]

# Exécuter les simulations pour chaque nombre de marqueurs
println("Exécution des simulations pour différents nombres de marqueurs...")
results = Dict{Int, Vector{Float64}}()

for n_markers in markers_to_test
    println("Test avec $n_markers marqueurs...")
    residuals = run_single_timestep(n_markers)
    results[n_markers] = residuals
end

# Créer un graphique de comparaison
fig = Figure(size=(900, 600))
ax = Axis(fig[1, 1], 
          title="Convergence des résidus pour différents nombres de marqueurs", 
          xlabel="Itération", 
          ylabel="Résidu (échelle log)",
          yscale=log10)

# Palette de couleurs distinctes - utiliser des couleurs fixes au lieu d'un gradient
distinct_colors = [:royalblue, :crimson, :darkgreen, :darkorange, :purple, :teal]
# S'assurer qu'on a assez de couleurs
if length(distinct_colors) < length(markers_to_test)
    # Étendre la palette si nécessaire
    append!(distinct_colors, rand(ColorSchemes.rainbow, length(markers_to_test) - length(distinct_colors)))
end

# Tracer les résidus pour chaque nombre de marqueurs
for (i, n_markers) in enumerate(markers_to_test)
    residuals = results[n_markers]
    
    # Tracer la courbe de résidus avec couleur distincte
    lines!(ax, 1:length(residuals), residuals, 
           label="$n_markers marqueurs",
           linewidth=2,
           color=distinct_colors[i])
    
    # Marquer les points de données avec la même couleur
    scatter!(ax, 1:length(residuals), residuals,
             markersize=5,
             color=distinct_colors[i])
end

# Ajouter la légende
Legend(fig[1, 2], ax)

# Analyser la convergence
fig2 = Figure(size=(900, 600))
ax2 = Axis(fig2[1, 1],
           title="Analyse de convergence", 
           xlabel="Nombre de marqueurs", 
           ylabel="Nombre d'itérations jusqu'à convergence")

# Calculer le nombre d'itérations nécessaires pour atteindre un seuil de convergence
convergence_threshold = 1e-6
iterations_to_converge = Int[]
final_residuals = Float64[]

for n_markers in markers_to_test
    residuals = results[n_markers]
    
    # Trouver l'itération où le résidu passe sous le seuil
    converged_at = findfirst(r -> r < convergence_threshold, residuals)
    
    if converged_at === nothing
        # Utiliser le nombre total d'itérations si la convergence n'est pas atteinte
        converged_at = length(residuals)
    end
    
    push!(iterations_to_converge, converged_at)
    push!(final_residuals, residuals[end])
end

# Tracer le nombre d'itérations nécessaires avec couleurs distinctes
barplot!(ax2, markers_to_test, iterations_to_converge, 
         width=5.0, color=distinct_colors[1:length(markers_to_test)])

# Tracer le résidu final
fig3 = Figure(size=(900, 600))
ax3 = Axis(fig3[1, 1],
           title="Résidu final après 20 itérations", 
           xlabel="Nombre de marqueurs", 
           ylabel="Résidu final (échelle log)",
           yscale=log10)

scatter!(ax3, markers_to_test, final_residuals, 
         markersize=15, color=distinct_colors[1:length(markers_to_test)])
lines!(ax3, markers_to_test, final_residuals, 
       linewidth=2, color=:gray, linestyle=:dash)

# Afficher les graphiques
display(fig)
display(fig2)
display(fig3)

# Sauvegarder les graphiques
results_dir = joinpath(pwd(), "marker_convergence_results")
mkpath(results_dir)
save(joinpath(results_dir, "residuals_comparison.png"), fig)
save(joinpath(results_dir, "iterations_to_converge.png"), fig2)
save(joinpath(results_dir, "final_residuals.png"), fig3)

println("\nAnalyse de convergence terminée. Résultats enregistrés dans: $results_dir")