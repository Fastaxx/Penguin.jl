"""
    NavierStokesHeat2D

Monolithic time integrator for a two-dimensional Navier–Stokes / heat system
under the Boussinesq approximation. The velocity/pressure block reuses the
staggered Navier–Stokes assembly while the temperature block leverages the
advection–diffusion operators. Buoyancy forcing is obtained from the current
temperature field and injected in the momentum equations.
"""
mutable struct NavierStokesHeat2D
    momentum::NavierStokesMono{2}
    temperature_capacity::Capacity{2}
    thermal_diffusivity::Union{Float64,Function}
    heat_source::Function
    bc_temperature::BorderConditions
    bc_temperature_cut::AbstractBoundary
    β::Float64
    gravity::SVector{2,Float64}
    T_ref::Float64
    temperature::Vector{Float64}
    velocity_states::Vector{Vector{Float64}}
    temperature_states::Vector{Vector{Float64}}
    times::Vector{Float64}
    scalar_nodes::NTuple{2,Vector{Float64}}
    interp_ux::SparseMatrixCSC{Float64,Int}
    interp_uy::SparseMatrixCSC{Float64,Int}
end

function NavierStokesHeat2D(momentum::NavierStokesMono{2},
                            temperature_capacity::Capacity{2},
                            κ::Union{Float64,Function},
                            heat_source::Function,
                            bc_temperature::BorderConditions,
                            bc_temperature_cut::AbstractBoundary;
                            β::Float64=1.0,
                            gravity::NTuple{2,Float64}=(0.0, -1.0),
                            T_ref::Float64=0.0,
                            T0::Union{Nothing,Vector{Float64}}=nothing)
    node_counts = ntuple(i -> length(temperature_capacity.mesh.nodes[i]), 2)
    N = prod(node_counts)
    T_init = if T0 === nothing
        zeros(2N)
    else
        length(T0) == 2N || error("Initial temperature vector must have length $(2N).")
        copy(T0)
    end

    gravity_vec = SVector{2,Float64}(gravity)
    scalar_nodes = ntuple(i -> copy(temperature_capacity.mesh.nodes[i]), 2)

    interp_ux = build_temperature_interpolation(temperature_capacity, momentum.fluid.capacity_u[1])
    interp_uy = build_temperature_interpolation(temperature_capacity, momentum.fluid.capacity_u[2])

    return NavierStokesHeat2D(momentum,
                              temperature_capacity,
                              κ,
                              heat_source,
                              bc_temperature,
                              bc_temperature_cut,
                              β,
                              gravity_vec,
                              T_ref,
                              T_init,
                              Vector{Vector{Float64}}(),
                              Vector{Vector{Float64}}(),
                              Float64[],
                              scalar_nodes,
                              interp_ux,
                              interp_uy)
end

@inline function nearest_index(vec::AbstractVector{<:Real}, val::Real)
    idx = searchsortedfirst(vec, val)
    if idx <= 1
        return 1
    elseif idx > length(vec)
        return length(vec)
    else
        prev_val = vec[idx - 1]
        curr_val = vec[idx]
        return abs(val - prev_val) <= abs(curr_val - val) ? idx - 1 : idx
    end
end

@inline function _split_velocity_components(data, state::AbstractVector{<:Real})
    nu_x = data.nu_x
    nu_y = data.nu_y
    uωx = @view state[1:nu_x]
    uγx = @view state[nu_x+1:2nu_x]
    uωy = @view state[2nu_x+1:2nu_x+nu_y]
    uγy = @view state[2nu_x+nu_y+1:2*(nu_x+nu_y)]
    return uωx, uγx, uωy, uγy
end

function project_field_between_nodes(values::AbstractVector{<:Real},
                                     src_nodes::NTuple{2,Vector{Float64}},
                                     dst_nodes::NTuple{2,Vector{Float64}})
    Nx_src = length(src_nodes[1])
    Ny_src = length(src_nodes[2])
    field = reshape(Vector{Float64}(values), (Nx_src, Ny_src))

    Nx_dst = length(dst_nodes[1])
    Ny_dst = length(dst_nodes[2])
    projected = Vector{Float64}(undef, Nx_dst * Ny_dst)

    for j in 1:Ny_dst
        y = dst_nodes[2][j]
        iy = nearest_index(src_nodes[2], y)
        for i in 1:Nx_dst
            x = dst_nodes[1][i]
            ix = nearest_index(src_nodes[1], x)
            projected[i + (j - 1) * Nx_dst] = field[ix, iy]
        end
    end

    return projected
end

function build_temperature_interpolation(temp_cap::Capacity{2},
                                         vel_cap::Capacity{2})
    coords_temp = temp_cap.C_ω
    coords_vel = vel_cap.C_ω

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (i, pos) in enumerate(coords_vel)
        best_idx = 1
        best_dist = Inf
        for (j, center) in enumerate(coords_temp)
            dx = pos[1] - center[1]
            dy = pos[2] - center[2]
            dist = dx * dx + dy * dy
            if dist < best_dist
                best_dist = dist
                best_idx = j
            end
        end
        push!(rows, i)
        push!(cols, best_idx)
        push!(vals, 1.0)
    end

    return sparse(rows, cols, vals, length(coords_vel), length(coords_temp))
end

function add_sparse_block!(A::SparseMatrixCSC{Float64,Int},
                           block::SparseMatrixCSC{Float64,Int},
                           row_offset::Int,
                           col_offset::Int,
                           scale::Float64)
    nnz_block = nnz(block)
    nnz_block == 0 && return A

    rows = Vector{Int}(undef, nnz_block)
    cols = Vector{Int}(undef, nnz_block)
    vals = Vector{Float64}(undef, nnz_block)

    idx = 1
    for j in 1:size(block, 2)
        for ptr in block.colptr[j]:(block.colptr[j+1]-1)
            rows[idx] = row_offset + block.rowval[ptr]
            cols[idx] = col_offset + j
            vals[idx] = scale * block.nzval[ptr]
            idx += 1
        end
    end

    A .+= sparse(rows, cols, vals, size(A, 1), size(A, 2))
    return A
end

function _scheme_string(scheme::Symbol)
    s = lowercase(String(scheme))
    if s in ("cn", "crank_nicolson", "cranknicolson")
        return "CN"
    elseif s in ("be", "backward_euler", "implicit_euler")
        return "BE"
    else
        error("Unsupported time scheme $(scheme). Use :CN or :BE.")
    end
end

function solve_linear_system(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64};
                             method=Base.:\, algorithm=nothing, kwargs...)
    if algorithm !== nothing
        prob = LinearSolve.LinearProblem(A, b)
        sol = LinearSolve.solve(prob, algorithm; kwargs...)
        return Vector{Float64}(sol.u)
    elseif method === Base.:\
        try
            return A \ b
        catch err
            if err isa SingularException
                @warn "Direct solve failed with SingularException; falling back to bicgstabl" sizeA=size(A)
                return IterativeSolvers.bicgstabl(A, b)
            else
                rethrow(err)
            end
        end
    else
        return method(A, b; kwargs...)
    end
end

function build_temperature_system(s::NavierStokesHeat2D,
                                  data,
                                  velocity_state::AbstractVector{<:Real},
                                  Δt::Float64,
                                  t_prev::Float64,
                                  scheme::Symbol,
                                  T_prev::Vector{Float64})
    scheme_str = _scheme_string(scheme)

    nodes_scalar = s.scalar_nodes
    mesh_ux = s.momentum.fluid.mesh_u[1]
    mesh_uy = s.momentum.fluid.mesh_u[2]

    uωx, _, uωy, _ = _split_velocity_components(data, velocity_state)
    u_bulk_x = project_field_between_nodes(uωx, (mesh_ux.nodes[1], mesh_ux.nodes[2]), nodes_scalar)
    u_bulk_y = project_field_between_nodes(uωy, (mesh_uy.nodes[1], mesh_uy.nodes[2]), nodes_scalar)

    N = length(nodes_scalar[1]) * length(nodes_scalar[2])
    u_interface = zeros(Float64, 2 * N)
    operator = ConvectionOps(s.temperature_capacity, (u_bulk_x, u_bulk_y), u_interface)

    A_T = A_mono_unstead_advdiff(operator, s.temperature_capacity, s.thermal_diffusivity,
                                 s.bc_temperature_cut, Δt, scheme_str)
    b_T = b_mono_unstead_advdiff(operator, s.heat_source, s.temperature_capacity,
                                 s.bc_temperature_cut, T_prev, Δt, t_prev, scheme_str)

    BC_border_mono!(A_T, b_T, s.bc_temperature, s.temperature_capacity.mesh)

    return A_T, b_T
end

function solve_NavierStokesHeat2D_unsteady!(s::NavierStokesHeat2D;
                                            Δt::Float64,
                                            T_end::Float64,
                                            scheme::Symbol=:CN,
                                            method=Base.:\,
                                            algorithm=nothing,
                                            store_states::Bool=true,
                                            kwargs...)
    θ = scheme_to_theta(scheme)
    ns = s.momentum
    data = navierstokes2D_blocks(ns)
    p_offset = 2 * (data.nu_x + data.nu_y)
    np = data.np

    x_prev = length(ns.x) == p_offset + np ? copy(ns.x) : zeros(p_offset + np)
    p_half_prev = zeros(Float64, np)
    if length(ns.x) == p_offset + np && !isempty(ns.x)
        p_half_prev .= ns.x[p_offset+1:p_offset+np]
    end

    T_prev = copy(s.temperature)

    s.velocity_states = store_states ? Vector{Vector{Float64}}([copy(x_prev)]) : Vector{Vector{Float64}}()
    s.temperature_states = store_states ? Vector{Vector{Float64}}([copy(T_prev)]) : Vector{Vector{Float64}}()
    s.times = Float64[0.0]

    conv_prev = ns.prev_conv
    if conv_prev !== nothing && length(conv_prev) != 2
        conv_prev = nothing
    end

    kwargs_nt = (; kwargs...)

    t = 0.0
    while t < T_end - 1e-12 * max(1.0, T_end)
        dt_step = min(Δt, T_end - t)
        t_next = t + dt_step

        conv_curr = assemble_navierstokes2D_unsteady!(ns, data, dt_step,
                                                      x_prev, p_half_prev,
                                                      t, t_next, θ, conv_prev)
        A_mom = copy(ns.A)
        b_mom = copy(ns.b)

        A_T, b_T = build_temperature_system(s, data, x_prev, dt_step, t, scheme, T_prev)

        rows_mom = size(A_mom, 1)
        Ntemp_total = length(T_prev)
        total_size = rows_mom + Ntemp_total

        A_c = [A_mom  spzeros(Float64, rows_mom, Ntemp_total);
               spzeros(Float64, Ntemp_total, rows_mom)  A_T]
        b_c = vcat(b_mom, b_T)

        ρ = ns.fluid.ρ
        ρ_val = ρ isa Function ? 1.0 : ρ
        β = s.β
        g = s.gravity
        scale_x = ρ_val * β * g[1]
        scale_y = ρ_val * β * g[2]

        off_Tω = rows_mom
        row_uωx = 0
        row_uωy = 2 * data.nu_x

        if !iszero(scale_x)
            Bu_x = data.Vx * s.interp_ux
            add_sparse_block!(A_c, Bu_x, row_uωx, off_Tω, -scale_x)
            if !iszero(s.T_ref)
                ones_x = ones(Float64, data.nu_x)
                b_c[row_uωx+1:row_uωx+data.nu_x] .-= scale_x * s.T_ref .* (data.Vx * ones_x)
            end
        end

        if !iszero(scale_y)
            Bu_y = data.Vy * s.interp_uy
            add_sparse_block!(A_c, Bu_y, row_uωy, off_Tω, -scale_y)
            if !iszero(s.T_ref)
                ones_y = ones(Float64, data.nu_y)
                b_c[row_uωy+1:row_uωy+data.nu_y] .-= scale_y * s.T_ref .* (data.Vy * ones_y)
            end
        end

        Ared, bred, _, keep_idx_cols = remove_zero_rows_cols!(A_c, b_c)
        x_red = solve_linear_system(Ared, bred; method=method, algorithm=algorithm, kwargs_nt...)

        full_state = zeros(Float64, total_size)
        full_state[keep_idx_cols] = x_red

        x_new = full_state[1:rows_mom]
        T_next = full_state[rows_mom+1:end]

        ns.x .= x_new
        s.temperature .= T_next

        x_prev = copy(x_new)
        p_half_prev .= x_new[p_offset+1:p_offset+np]
        conv_prev = ntuple(Val(2)) do i
            copy(conv_curr[Int(i)])
        end

        T_prev = copy(T_next)
        ns.prev_conv = conv_prev

        t = t_next
        push!(s.times, t)
        if store_states
            push!(s.velocity_states, copy(x_new))
            push!(s.temperature_states, copy(T_next))
        end
        max_state = maximum(abs, x_new)
        println("[NavierStokesHeat2D] t=$(round(t; digits=6)) max|state|=$(max_state)")
    end

    return s.times, s.velocity_states, s.temperature_states
end
