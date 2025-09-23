using Penguin
using Test

@testset "Stokes Diphasic Poiseuille" begin
    @testset "1D constructor keeps provided state" begin
        nx = 64
        Lx = 1.0
        x0_domain = 0.0

        mesh_p = Penguin.Mesh((nx,), (Lx,), (x0_domain,))
        dx = Lx / nx
        mesh_u = Penguin.Mesh((nx,), (Lx,), (x0_domain - 0.5 * dx,))

        body = (x, _=0.0) -> -1.0
        capacity_u = Capacity(body, mesh_u)
        capacity_p = Capacity(body, mesh_p)
        operator_u = DiffusionOps(capacity_u)
        operator_p = DiffusionOps(capacity_p)

        μ₁ = 1.0
        μ₂ = 1.0
        ρ = 1.0
        fᵤ = (x, y=0.0, z=0.0) -> 1.0
        fₚ = (x, y=0.0, z=0.0) -> 0.0

        fluid₁ = Fluid(mesh_u, capacity_u, operator_u,
                       mesh_p, capacity_p, operator_p,
                       μ₁, ρ, fᵤ, fₚ)
        fluid₂ = Fluid(mesh_u, capacity_u, operator_u,
                       mesh_p, capacity_p, operator_p,
                       μ₂, ρ, fᵤ, fₚ)

        bc_u = BorderConditions(Dict(:bottom => Dirichlet(0.0),
                                     :top => Dirichlet(0.0)))
        bc_p = BorderConditions(Dict())
        interface = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0),
                                        FluxJump(1.0, 1.0, 0.0))

        nu = prod(operator_u.size)
        np = prod(operator_p.size)
        expected_size = 4 * nu + 2 * np
        initial_state = collect(Float64.(1:expected_size))

        solver = StokesDiph(fluid₁, fluid₂, (bc_u,), (bc_u,), bc_p,
                             interface, Dirichlet(0.0); x0=initial_state)

        @test solver.x === initial_state
        @test size(solver.A) == (expected_size, expected_size)
        @test length(solver.b) == expected_size

        solve_StokesDiph!(solver; method=Base.:\)

        @test length(solver.x) == expected_size
    end

    @testset "2D continuity across interface" begin
        nx, ny = 12, 8
        Lx, Ly = 2.0, 1.0
        x0_domain, y0_domain = 0.0, 0.0

        mesh_p = Penguin.Mesh((nx, ny), (Lx, Ly), (x0_domain, y0_domain))
        dx, dy = Lx / nx, Ly / ny
        mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0_domain - 0.5 * dx, y0_domain))
        mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0_domain, y0_domain - 0.5 * dy))

        y_mid = y0_domain + Ly / 2
        body_top = (x, y, _=0.0) -> y - y_mid
        body_bot = (x, y, _=0.0) -> y_mid - y

        cap_ux_top = Capacity(body_top, mesh_ux)
        cap_uy_top = Capacity(body_top, mesh_uy; compute_centroids=false)
        cap_p_top  = Capacity(body_top, mesh_p)

        cap_ux_bot = Capacity(body_bot, mesh_ux)
        cap_uy_bot = Capacity(body_bot, mesh_uy; compute_centroids=false)
        cap_p_bot  = Capacity(body_bot, mesh_p)

        op_ux_top = DiffusionOps(cap_ux_top)
        op_uy_top = DiffusionOps(cap_uy_top)
        op_p_top  = DiffusionOps(cap_p_top)

        op_ux_bot = DiffusionOps(cap_ux_bot)
        op_uy_bot = DiffusionOps(cap_uy_bot)
        op_p_bot  = DiffusionOps(cap_p_bot)

        μ_top = 2.0
        μ_bot = 1.0
        ρ = 1.0
        fᵤ = (x, y, z=0.0) -> 0.0
        fₚ = (x, y, z=0.0) -> 0.0

        fluid_top = Fluid((mesh_ux, mesh_uy), (cap_ux_top, cap_uy_top), (op_ux_top, op_uy_top),
                          mesh_p, cap_p_top, op_p_top, μ_top, ρ, fᵤ, fₚ)
        fluid_bot = Fluid((mesh_ux, mesh_uy), (cap_ux_bot, cap_uy_bot), (op_ux_bot, op_uy_bot),
                          mesh_p, cap_p_bot, op_p_bot, μ_bot, ρ, fᵤ, fₚ)

        ux_wall = Dirichlet((x, y) -> 0.0)
        uy_wall = Dirichlet((x, y) -> 0.0)

        bc_ux_top = BorderConditions(Dict(:bottom => ux_wall, :top => ux_wall))
        bc_uy_top = BorderConditions(Dict(:bottom => uy_wall, :top => uy_wall))

        bc_ux_bot = BorderConditions(Dict(:bottom => ux_wall, :top => ux_wall))
        bc_uy_bot = BorderConditions(Dict(:bottom => uy_wall, :top => uy_wall))

        Δp = 1.0
        p_in = Δp
        p_out = 0.0
        bc_p = BorderConditions(Dict(:left => Dirichlet(p_in),
                                     :right => Dirichlet(p_out)))

        interface = InterfaceConditions(ScalarJump(1.0, 1.0, 0.0),
                                        FluxJump(1.0, 1.0, 0.0))

        nu_x = prod(op_ux_top.size)
        nu_y = prod(op_uy_top.size)
        np = prod(op_p_top.size)
        expected_size = 4 * (nu_x + nu_y) + 2 * np
        initial_state = collect(Float64.(1:expected_size))

        solver = StokesDiph(fluid_top, fluid_bot,
                             (bc_ux_top, bc_uy_top),
                             (bc_ux_bot, bc_uy_bot),
                             bc_p, interface, Dirichlet(0.0); x0=initial_state)

        @test solver.x === initial_state
        @test size(solver.A) == (expected_size, expected_size)
        @test length(solver.b) == expected_size

        solve_StokesDiph!(solver; method=Base.:\)

        @test length(solver.x) == expected_size

        sum_nu = nu_x + nu_y
        off_p1 = 2 * sum_nu
        off_u2ωx = off_p1 + np
        off_u2γx = off_u2ωx + nu_x
        off_u2ωy = off_u2γx + nu_x
        off_u2γy = off_u2ωy + nu_y

        u1γx = solver.x[nu_x + 1:2 * nu_x]
        u2γx = solver.x[off_u2γx + 1:off_u2γx + nu_x]
        u1γy = solver.x[2 * nu_x + nu_y + 1:2 * nu_x + 2 * nu_y]
        u2γy = solver.x[off_u2γy + 1:off_u2γy + nu_y]

        println(u1γx)
        println(u2γx)
        @test u1γx ≈ u2γx atol=1e-10 rtol=1e-10
        @test u1γy ≈ u2γy atol=1e-10 rtol=1e-10
    end
end
