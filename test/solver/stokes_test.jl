using Penguin
using Test
using LinearAlgebra

@testset "Stokes Poiseuille" begin
    @testset "1D Poiseuille" begin
        nx = 128
        Lx = 1.0
        x0 = 0.0

        mesh_p = Penguin.Mesh((nx,), (Lx,), (x0,))
        dx = Lx / nx
        mesh_u = Penguin.Mesh((nx,), (Lx,), (x0 - 0.5 * dx,))

        body = (x, _=0.0) -> -1.0
        capacity_u = Capacity(body, mesh_u)
        capacity_p = Capacity(body, mesh_p)
        operator_u = DiffusionOps(capacity_u)
        operator_p = DiffusionOps(capacity_p)

        u_left = Dirichlet(0.0)
        u_right = Dirichlet(0.0)
        bc_u = BorderConditions(Dict(:bottom => u_left, :top => u_right))

        p_left = Dirichlet(1.0)
        p_right = Dirichlet(0.0)
        bc_p = BorderConditions(Dict(:bottom => p_left, :top => p_right))

        bc_cut = Dirichlet(0.0)

        fᵤ = (x, y=0.0, z=0.0) -> 0.0
        fₚ = (x, y=0.0, z=0.0) -> 0.0

        μ = 1.0
        ρ = 1.0

        fluid = Fluid(mesh_u, capacity_u, operator_u, mesh_p, capacity_p, operator_p, μ, ρ, fᵤ, fₚ)

        solver = StokesMono(fluid, bc_u, bc_p, bc_cut)
        solve_StokesMono!(solver; method=Base.:\)

        nu = prod(operator_u.size)
        np = prod(operator_p.size)

        uω = solver.x[1:nu]
        pω = solver.x[2nu + 1:2nu + np]

        xs_u = mesh_u.nodes[1]
        xp = mesh_p.nodes[1]

        x_left = first(xs_u)
        x_right = last(xs_u)
        x_left_p = first(xp)
        x_right_p = last(xp)
        dpdx = (p_right.value - p_left.value) / (x_right_p - x_left_p)

        u_exact = [-dpdx / (2μ) * (x - x_left) * (x_right - x) for x in xs_u]

        p_exact = [p_left.value + dpdx * (x - x_left_p) for x in xp]

        Atrim, btrim, _, idx_cols = remove_zero_rows_cols!(solver.A, solver.b)
        xtrim = solver.x[idx_cols]
        r = Atrim * xtrim - btrim
        @test norm(r, Inf) ≤ 1e-10
    end

    @testset "2D Poiseuille" begin
        nx, ny = 64, 64
        Lx, Ly = 2.0, 1.0
        x0, y0 = 0.0, 0.0

        mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
        dx, dy = Lx / nx, Ly / ny
        mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
        mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

        body = (x, y, _=0.0) -> -1.0
        capacity_ux = Capacity(body, mesh_ux)
        capacity_uy = Capacity(body, mesh_uy)
        capacity_p = Capacity(body, mesh_p)
        operator_ux = DiffusionOps(capacity_ux)
        operator_uy = DiffusionOps(capacity_uy)
        operator_p = DiffusionOps(capacity_p)

        Umax = 1.0
        parabola = (x, y) -> 4Umax * (y - y0) * (Ly - (y - y0)) / Ly^2

        ux_left = Dirichlet(parabola)
        ux_right = Dirichlet(parabola)
        ux_bot = Dirichlet((x, y) -> 0.0)
        ux_top = Dirichlet((x, y) -> 0.0)
        bc_ux = BorderConditions(Dict(
            :left => ux_left,
            :right => ux_right,
            :bottom => ux_bot,
            :top => ux_top,
        ))

        uy_zero = Dirichlet((x, y) -> 0.0)
        bc_uy = BorderConditions(Dict(
            :left => uy_zero,
            :right => uy_zero,
            :bottom => uy_zero,
            :top => uy_zero,
        ))

        bc_p = BorderConditions(Dict{Symbol, AbstractBoundary}())
        bc_cut = Dirichlet(0.0)

        fᵤ = (x, y, z=0.0) -> 0.0
        fₚ = (x, y, z=0.0) -> 0.0
        μ = 1.0
        ρ = 1.0

        fluid = Fluid((mesh_ux, mesh_uy),
                      (capacity_ux, capacity_uy),
                      (operator_ux, operator_uy),
                      mesh_p, capacity_p, operator_p,
                      μ, ρ, fᵤ, fₚ)

        solver = StokesMono(fluid, bc_ux, bc_uy; bc_p=bc_p, bc_cut=bc_cut)
        solve_StokesMono!(solver; method=Base.:\)

        nu_x = prod(operator_ux.size)
        nu_y = prod(operator_uy.size)
        np = prod(operator_p.size)

        uωx = solver.x[1:nu_x]
        uωy = solver.x[2nu_x + 1:2nu_x + nu_y]
        pω = solver.x[2(nu_x + nu_y) + 1:2(nu_x + nu_y) + np]

        xs = mesh_ux.nodes[1]
        ys = mesh_ux.nodes[2]
        size_ux = operator_ux.size
        ux_field = reshape(uωx, size_ux)
        ux_exact = [parabola(xs[i], ys[j]) for i in 1:length(xs), j in 1:length(ys)]
        # remove first second and n-1 last and n row (boundary)
        ux_field = ux_field[2:end-1, 2:end-1]
        ux_exact = ux_exact[2:end-1, 2:end-1]

        @test sum(abs.(ux_field .- ux_exact)) / length(ux_field) ≤ 1e-2


        xp = mesh_p.nodes[1]
        yp = mesh_p.nodes[2]
        size_p = operator_p.size
        p_field = reshape(pω, size_p)
        dpdx = -8 * μ * Umax / Ly^2
        p_exact = [dpdx * (x - xp[1]) for x in xp, y in yp]
        p_diff = p_field .- p_exact
        shift = sum(p_diff) / length(p_diff)
        p_error = p_diff .- shift

        Atrim, btrim, _, idx_cols = remove_zero_rows_cols!(solver.A, solver.b)
        xtrim = solver.x[idx_cols]
        r = Atrim * xtrim - btrim
        @test norm(r, Inf) ≤ 1e-10
    end
end