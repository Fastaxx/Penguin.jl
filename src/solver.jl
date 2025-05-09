@enum TimeType begin
    Steady  # ∂ₜT = 0
    Unsteady # ∂ₜT ≠ 0
end

@enum PhaseType begin
    Monophasic  # Single phase
    Diphasic    # Two phases
end

@enum EquationType begin
    Diffusion           # ∂ₜT = ∇·(∇T) + S
    Advection           # ∂ₜT = -∇·(uT) + S
    DiffusionAdvection  # ∂ₜT = ∇·(D∇T) - ∇·(uT) + S
end

"""
    mutable struct Solver{TT<:TimeType, PT<:PhaseType, ET<:EquationType}

The `Solver` struct represents a solver for a specific type of problem.

# Fields
- `time_type::TT`: The type of time used in the solver : `Steady` or `Unsteady`.
- `phase_type::PT`: The type of phase used in the solver : `Monophasic` or `Diphasic`.
- `equation_type::ET`: The type of equation used in the solver : `Diffusion`, `Advection` or `DiffusionAdvection`.
- `A::Union{SparseMatrixCSC{Float64, Int}, Nothing}`: The coefficient matrix A of the equation system, if applicable.
- `b::Union{Vector{Float64}, Nothing}`: The right-hand side vector b of the equation system, if applicable.
- `x::Union{Vector{Float64}, Nothing}`: The solution vector x of the equation system, if applicable.
- `states::Vector{Any}`: The states of the system at different times, if applicable.

"""
mutable struct Solver{TT<:TimeType, PT<:PhaseType, ET<:EquationType}
    time_type::TT
    phase_type::PT
    equation_type::ET
    A::Union{SparseMatrixCSC{Float64, Int}, Nothing}
    b::Union{Vector{Float64}, Nothing}
    x::Union{Vector{Float64}, Nothing}
    ch::IterativeSolvers.ConvergenceHistory
    states::Vector{Any}
end

"""
    remove_zero_rows_cols!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})

Remove zero rows and columns from the coefficient matrix `A` and the right-hand side vector `b`.

# Arguments
- `A::SparseMatrixCSC{Float64, Int}`: The coefficient matrix A of the equation system.
- `b::Vector{Float64}`: The right-hand side vector b of the equation system.

# Returns
- `A::SparseMatrixCSC{Float64, Int}`: The reduced coefficient matrix A.
- `b::Vector{Float64}`: The reduced right-hand side vector b.
- `rows_idx::Vector{Int}`: The indices of the non-zero rows.
- `cols_idx::Vector{Int}`: The indices of the non-zero columns.
"""
function remove_zero_rows_cols!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64})
    # Compute sums of absolute values along rows and columns
    row_sums = vec(sum(abs.(A), dims=2))
    col_sums = vec(sum(abs.(A), dims=1))

    # Find indices of non-zero rows and columns
    rows_idx = findall(row_sums .!= 0.0)
    cols_idx = findall(col_sums .!= 0.0)

    # Create new matrix and RHS vector
    A = A[rows_idx, cols_idx]
    b = b[rows_idx]

    return A, b, rows_idx, cols_idx
end

"""
    solve_system!(s::Solver; method::Function=gmres, kwargs...)

Solve the system of equations stored in the `Solver` struct `s` using the specified method.

# Arguments
- `s::Solver`: The `Solver` struct containing the system of equations to solve.
- `method::Function=gmres`: The method to use to solve the system of equations. Default is `gmres`.
- `kwargs...`: Additional keyword arguments to pass to the solver.
"""
function solve_system!(s::Solver; method::Function=gmres, kwargs...)
    # Compute the problem size
    n = size(s.A, 1)

    # Choose between using a direct solver (\) or an iterative solver
    if method === Base.:\
        # Remove zero rows and columns for direct solver
        A_reduced, b_reduced, rows_idx, cols_idx = remove_zero_rows_cols!(s.A, s.b)
        # Solve the reduced system
        x_reduced = A_reduced \ b_reduced
        # Reconstruct the full solution vector
        s.x = zeros(n)
        s.x[cols_idx] = x_reduced
    else
        # Use iterative solver directly
        kwargs_nt = (; kwargs...)
        log = get(kwargs_nt, :log, false)
        if log
            # If logging is enabled, we store the convergence history
            s.x, s.ch = method(s.A, s.b; kwargs...)
        else
            s.x = method(s.A, s.b; kwargs...)
        end
    end
end

"""
    build_I_bc(operator::AbstractOperators, bc::AbstractBoundary)

Build the boundary conditions matrices Iₐ and Iᵦ for the given operator and boundary conditions.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `bc::AbstractBoundary`: The boundary conditions of the problem.

# Returns
- `Iₐ::SparseMatrixCSC{Float64, Int}`: The matrix Iₐ for the boundary conditions.
- `Iᵦ::SparseMatrixCSC{Float64, Int}`: The matrix Iᵦ for the boundary conditions.
"""
function build_I_bc(operator::AbstractOperators,bc::AbstractBoundary)
    n = prod(operator.size)
    Iᵦ = spzeros(n, n)
    Iₐ = spzeros(n, n)

    if bc isa Dirichlet
        Iₐ = I(n)
    elseif bc isa GibbsThomson
        Iₐ = I(n)
    elseif bc isa Neumann
        Iᵦ = I(n)
    elseif bc isa Robin
        if bc.α isa Function
            Iₐ = bc.α(I(n))
        else
            Iₐ = bc.α * I(n)
            Iᵦ = bc.β * I(n)
        end 
    end
    return Iₐ, Iᵦ
end

"""
    build_I_D(operator::AbstractOperators, D::Union{Float64,Function}, capacite::Capacity)

Build the diffusion matrix Id for the given operator and diffusion coefficient.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `D::Union{Float64,Function}`: The diffusion coefficient of the problem.
- `capacite::Capacity`: The capacity of the problem.

# Returns
- `Id::SparseMatrixCSC{Float64, Int}`: The diffusion matrix Id.
"""
function build_I_D(operator::AbstractOperators, D::Union{Float64,Function}, capacite::Capacity)
    n = prod(operator.size)
    Id = spdiagm(0 => ones(n))

    if D isa Function
        for i in 1:n
            x, y, z = get_coordinates(i, capacite.C_ω)
            Id[i, i] = D(x, y, z)
        end
    else
        Id = D * Id
    end
    return Id
end

"""
    build_source(operator::AbstractOperators, f::Function, capacite::Capacity)

Build the source term vector fₒ for the given operator and source term function.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `f::Function`: The source term function of the problem.
- `capacite::Capacity`: The capacity of the problem.

# Returns
- `fₒ::Vector{Float64}`: The source term vector fₒ.
"""
function build_source(operator::AbstractOperators, f::Function, capacite::Capacity)
    N = prod(operator.size)
    fₒ = zeros(N)

    # Compute the source term
    for i in 1:N
        x, y, z = get_coordinates(i, capacite.C_ω)
        fₒ[i] = f(x, y, z)
    end

    return fₒ
end

"""
    build_source(operator::AbstractOperators, f, t, capacite::Capacity)

Build the source term vector fₒ for the given operator, source term function and time t.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `f`: The source term function of the problem.
- `t::Float64`: The time at which to evaluate the source term.
- `capacite::Capacity`: The capacity of the problem.

# Returns
- `fₒ::Vector{Float64}`: The source term vector fₒ.
"""
function build_source(operator::AbstractOperators, f, t, capacite::Capacity)
    N = prod(operator.size)
    fₒ = zeros(N)

    # Compute the source term
    for i in 1:N
        x, y, z = get_coordinates(i, capacite.C_ω)
        fₒ[i] = f(x, y, z, t)
    end

    return fₒ
end

function get_coordinates(i, C_ω)
    if length(C_ω[1]) == 1
        x = C_ω[i][1]
        return x, 0., 0.
    elseif length(C_ω[1]) == 2
        x, y = C_ω[i][1], C_ω[i][2]
        return x, y, 0.
    else
        x, y, z = C_ω[i][1], C_ω[i][2], C_ω[i][3]
        return x, y, z
    end
end

"""
    build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity)

Build the boundary conditions vector gᵧ for the given operator and boundary conditions.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `bc::Union{AbstractBoundary, AbstractInterfaceBC}`: The boundary conditions of the problem.
- `capacite::Capacity`: The capacity of the problem.

# Returns
- `gᵧ::Vector{Float64}`: The boundary conditions vector gᵧ.
"""
function build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity)
    n = prod(operator.size)
    gᵧ = ones(n)

    if bc.value isa Function
        for i in 1:n
            x, y, z = get_coordinates(i, capacite.C_γ)
            gᵧ[i] = bc.value(x, y, z)
        end
    else
        gᵧ = bc.value * gᵧ
    end
    return gᵧ
end

"""
    build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity)

Build the boundary conditions vector gᵧ for the given operator and boundary conditions.

# Arguments
- `operator::AbstractOperators`: The operators of the problem.
- `bc::Union{AbstractBoundary, AbstractInterfaceBC}`: The boundary conditions of the problem.
- `capacite::Capacity`: The capacity of the problem.
- `t::Float64`: The time at which to evaluate the boundary conditions.

# Returns
- `gᵧ::Vector{Float64}`: The boundary conditions vector gᵧ.
"""
function build_g_g(operator::AbstractOperators, bc::Union{AbstractBoundary, AbstractInterfaceBC}, capacite::Capacity, t::Float64)
    n = prod(operator.size)
    gᵧ = ones(n)

    if bc.value isa Function
        for i in 1:n
            x, y, z = get_coordinates(i, capacite.C_γ)
            gᵧ[i] = bc.value(x, y, z, t)
        end
    else
        gᵧ = bc.value * gᵧ
    end
    return gᵧ
end

function build_g_g(operator::AbstractOperators, bc::GibbsThomson, capacite::Capacity)
    n = prod(operator.size)
    gᵧ = bc.Tm*ones(n) .- bc.ϵᵥ .* bc.vᵞ
    return gᵧ
end

"""
    BC_border_mono!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, bc_b::BorderConditions, mesh::AbstractMesh)

Apply the border conditions to the coefficient matrix `A` and the right-hand side vector `b`.

# Arguments
- `A::SparseMatrixCSC{Float64, Int}`: The coefficient matrix A of the equation system.
- `b::Vector{Float64}`: The right-hand side vector b of the equation system.
- `bc_b::BorderConditions`: The border conditions of the problem.
- `mesh::AbstractMesh`: The mesh of the problem.
"""
function BC_border_mono!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, bc_b::BorderConditions, mesh::AbstractMesh)
    # Identify border cells for each boundary key
    left_cells = Vector{CartesianIndex}()
    right_cells = Vector{CartesianIndex}()
    top_cells = Vector{CartesianIndex}()
    bottom_cells = Vector{CartesianIndex}()
    forward_cells = Vector{CartesianIndex}()
    backward_cells = Vector{CartesianIndex}()

    # Collect sets of cell indices for each boundary
    for (key, bc) in bc_b.borders
        if key == :left
            left_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[2] == 1]
        elseif key == :right
            right_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[2] == length(mesh.centers[2])]
        elseif key == :top
            top_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[1] == length(mesh.centers[1])]
        elseif key == :bottom
            bottom_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[1] == 1]
        elseif key == :forward
            forward_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[3] == length(mesh.centers[3])]
        elseif key == :backward
            backward_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[3] == 1]
        end
    end

    # Apply boundary conditions
    for (ci, pos) in mesh.tag.border_cells
        condition = nothing
        current_key = nothing

        if ci in left_cells
            condition = bc_b.borders[:left]
            current_key = :left
        elseif ci in right_cells
            condition = bc_b.borders[:right]
            current_key = :right
        elseif ci in top_cells
            condition = bc_b.borders[:top]
            current_key = :top
        elseif ci in bottom_cells
            condition = bc_b.borders[:bottom]
            current_key = :bottom
        elseif ci in forward_cells
            condition = bc_b.borders[:forward]
            current_key = :forward
        elseif ci in backward_cells
            condition = bc_b.borders[:backward]
            current_key = :backward
        end

        # Convert CartesianIndex to a linear index
        li = cell_to_index(mesh, ci)

        if condition isa Dirichlet
            # Dirichlet: fix A[li, li] = 1 and set b[li] = boundary value
            A[li, :] .= 0.0
            A[li, li] = 1.0
            b[li] = isa(condition.value, Function) ? condition.value(pos...) : condition.value

        elseif condition isa Periodic
            # Find the opposite boundary
            opposite_key = get_opposite_boundary(current_key)
            if !haskey(bc_b.borders, opposite_key)
                error("Periodic boundary requires both boundaries to be specified")
            end
            corresponding_ci = find_corresponding_cell(ci, current_key, opposite_key, mesh)
            corresponding_idx = cell_to_index(mesh, corresponding_ci)

            # Enforce x_li - x_corresponding = 0
            A[li, li] += 1.0
            A[li, corresponding_idx] -= 1.0
            b[li] = 0.0

        elseif condition isa Neumann
            # Not implemented yet
        elseif condition isa Robin
            # Not implemented yet
        end
    end
end

# Helper function to get the opposite boundary
function get_opposite_boundary(key::Symbol)
    if key == :left
        return :right
    elseif key == :right
        return :left
    elseif key == :bottom
        return :top
    elseif key == :top
        return :bottom
    elseif key == :backward
        return :forward
    elseif key == :forward
        return :backward
    else
        error("Unknown boundary key: $key")
    end
end

# Helper function to find the corresponding cell on the opposite boundary
function find_corresponding_cell(cell::CartesianIndex{N}, key::Symbol, opposite_key::Symbol, mesh::AbstractMesh) where {N}
    if key == :left || key == :right
        new_cell = CartesianIndex(key == :left ? length(mesh.centers[1]) : 1, cell[2])
    elseif key == :bottom || key == :top
        new_cell = CartesianIndex(cell[1], key == :bottom ? length(mesh.centers[2]) : 1)
    elseif key == :backward || key == :forward
        new_cell = CartesianIndex(cell[1], cell[2], key == :backward ? length(mesh.centers[3]) : 1)
    end
    return new_cell
end

function cell_to_index(mesh::Union{Mesh{1}, SpaceTimeMesh{1}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1,))[cell]
end

function cell_to_index(mesh::Union{Mesh{2}, SpaceTimeMesh{2}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1, length(mesh.centers[2])+1))[cell]
end

function cell_to_index(mesh::Union{Mesh{3}, SpaceTimeMesh{3}}, cell::CartesianIndex)
    return LinearIndices((length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1))[cell]
end

"""
    BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, bc_b::BorderConditions, mesh::AbstractMesh)

Apply the border conditions to the coefficient matrix `A` and the right-hand side vector `b` for diphasic problems.

# Arguments
- `A::SparseMatrixCSC{Float64, Int}`: The coefficient matrix A of the equation system.
- `b::Vector{Float64}`: The right-hand side vector b of the equation system.
- `bc_b::BorderConditions`: The border conditions of the problem.
- `mesh::AbstractMesh`: The mesh of the problem.
"""
function BC_border_diph!(A::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, bc_b::BorderConditions, mesh::AbstractMesh)
    # Collect boundary cells by side
    left_cells = Vector{CartesianIndex}()
    right_cells = Vector{CartesianIndex}()
    top_cells = Vector{CartesianIndex}()
    bottom_cells = Vector{CartesianIndex}()
    forward_cells = Vector{CartesianIndex}()
    backward_cells = Vector{CartesianIndex}()

    # For each boundary key, find matching cells
    for (key, bc) in bc_b.borders
        if key == :left
            left_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[2] == 1]
        elseif key == :right
            right_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[2] == length(mesh.centers[2])]
        elseif key == :top
            top_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[1] == length(mesh.centers[1])]
        elseif key == :bottom
            bottom_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[1] == 1]
        elseif key == :forward
            forward_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[3] == length(mesh.centers[3])]
        elseif key == :backward
            backward_cells = [ci for (ci, pos) in mesh.tag.border_cells if ci[3] == 1]
        end
    end

    # Calculate the size of a single phase block
    # In diphasic problems, the structure is typically:
    # [u1ₒ, u1ᵧ, u2ₒ, u2ᵧ] where each block has the same size
    phase_size = size(A, 1) ÷ 4
    
    # Apply boundary conditions to both phases
    for phase_idx in 0:1  # 0 = first phase, 1 = second phase
        for (ci, pos) in mesh.tag.border_cells
            condition = nothing
            if ci in left_cells
                condition = bc_b.borders[:left]
            elseif ci in right_cells
                condition = bc_b.borders[:right]
            elseif ci in top_cells
                condition = bc_b.borders[:top]
            elseif ci in bottom_cells
                condition = bc_b.borders[:bottom]
            elseif ci in forward_cells
                condition = bc_b.borders[:forward]
            elseif ci in backward_cells
                condition = bc_b.borders[:backward]
            end

            # Skip if no condition found
            isnothing(condition) && continue

            # Convert cell index to linear index
            li = cell_to_index(mesh, ci)
            
            # Apply phase offset for the bulk field (ω)
            li_ω = li + phase_idx * 2 * phase_size
            
            # Apply phase offset for the interface field (γ)
            li_γ = li + phase_size + phase_idx * 2 * phase_size

            if condition isa Dirichlet
                # Apply to bulk field (ω)
                A[li_ω, :] .= 0.0
                A[li_ω, li_ω] = 1.0
                b[li_ω] = isa(condition.value, Function) ? condition.value(pos...) : condition.value
                
                # Apply to interface field (γ)
                A[li_γ, :] .= 0.0
                A[li_γ, li_γ] = 1.0
                b[li_γ] = isa(condition.value, Function) ? condition.value(pos...) : condition.value

            elseif condition isa Neumann
                # Not implemented yet
                # Would need to modify rows corresponding to both bulk and interface
            elseif condition isa Robin
                # Not implemented yet
            elseif condition isa Periodic
                # Would need to implement similarly to the monophasic case
                # but applying to both phases
            end
        end
    end
end

function cfl_restriction(mesh::Penguin.Mesh, cfl::Float64, w::Float64)
    # Compute the spatial mesh spacing
    dx = (mesh.nodes[1][end] - mesh.nodes[1][1]) / length(mesh.centers[1])
    δt = cfl * dx / w
    return δt
end


"""
    adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max; 
                  growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)

Adapte le pas de temps en fonction du critère CFL basé sur la vitesse de l'interface.

Paramètres:
- `velocity_field`: Vitesses à l'interface [m/s]
- `mesh`: Maillage de calcul
- `cfl_target`: Nombre CFL cible (typiquement entre 0.1 et 1.0)
- `Δt_current`: Pas de temps actuel [s]
- `Δt_min`: Pas de temps minimum autorisé [s]
- `Δt_max`: Pas de temps maximum autorisé [s]
- `growth_factor`: Facteur maximum d'augmentation du pas de temps (par défaut 1.1)
- `shrink_factor`: Facteur minimum de réduction du pas de temps (par défaut 0.8)
- `safety_factor`: Facteur de sécurité pour le CFL (par défaut 0.9)

Retourne:
- `Δt_new`: Nouveau pas de temps [s]
- `cfl_actual`: Nombre CFL qui sera obtenu avec le nouveau pas de temps
"""
function adapt_timestep(velocity_field, mesh, cfl_target, Δt_current, Δt_min, Δt_max;
                       growth_factor=1.1, shrink_factor=0.8, safety_factor=0.9)
    # 1. Calcul de la vitesse maximale de l'interface
    v_max = maximum(abs.(velocity_field))
    
    # Éviter la division par zéro si l'interface est statique
    if v_max < 1e-10
        # Si la vitesse est très faible, on peut augmenter le pas de temps
        Δt_new = min(Δt_current * growth_factor, Δt_max)
        return Δt_new, 0.0
    end
    
    # 2. Calcul de la taille de maille minimale dans chaque direction
    if length(mesh.nodes) == 1
        Δx = minimum(diff(mesh.nodes[1]))
        Δh_min = Δx
    elseif length(mesh.nodes) == 2
        Δx = minimum(diff(mesh.nodes[1]))
        Δy = minimum(diff(mesh.nodes[2]))
        Δh_min = min(Δx, Δy)
    elseif length(mesh.nodes) == 3
        Δx = minimum(diff(mesh.nodes[1]))
        Δy = minimum(diff(mesh.nodes[2]))
        Δz = minimum(diff(mesh.nodes[3]))
        Δh_min = min(Δx, Δy, Δz)
    else
        error("Unsupported mesh dimension")
    end
    
    # 3. Calcul du CFL actuel
    cfl_current = v_max * Δt_current / Δh_min
    
    # 4. Calcul du pas de temps optimal pour le CFL cible avec un facteur de sécurité
    Δt_optimal = safety_factor * cfl_target * Δh_min / v_max
    
    # 5. Décision d'augmenter ou diminuer le pas de temps
    if Δt_optimal > Δt_current
        # La vitesse a diminué, on peut augmenter le pas de temps (limité par growth_factor)
        Δt_new = min(Δt_optimal, Δt_current * shrink_factor)
    else
        # La vitesse a augmenté, on doit diminuer le pas de temps (mais pas trop brusquement)
        Δt_new = max(Δt_optimal, Δt_current * growth_factor)
    end
    
    # 6. Application des contraintes min/max
    Δt_new = clamp(Δt_new, Δt_min, Δt_max)
    
    # 7. Calcul du CFL effectif avec le nouveau pas de temps
    cfl_actual = v_max * Δt_new / Δh_min
    
    return Δt_new, cfl_actual
end

