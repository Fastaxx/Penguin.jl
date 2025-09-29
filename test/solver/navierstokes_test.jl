using Penguin
using Test
using LinearAlgebra

function build_simple_navierstokes(; nx=6, ny=5)
    Lx = 1.0; Ly = 1.0
    x0 = -0.5; y0 = -0.5

    mesh_p  = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0))
    dx = mesh_p.nodes[1][2] - mesh_p.nodes[1][1]
    dy = mesh_p.nodes[2][2] - mesh_p.nodes[2][1]
    mesh_ux = Penguin.Mesh((nx, ny), (Lx, Ly), (x0 - 0.5 * dx, y0))
    mesh_uy = Penguin.Mesh((nx, ny), (Lx, Ly), (x0, y0 - 0.5 * dy))

    body = (x, y, _=0.0) -> 0.3^2 - ((x - 0.1)^2 + (y + 0.05)^2)

    cap_ux = Capacity(body, mesh_ux)
    cap_uy = Capacity(body, mesh_uy)
    cap_p  = Capacity(body, mesh_p)

    op_ux = DiffusionOps(cap_ux)
    op_uy = DiffusionOps(cap_uy)
    op_p  = DiffusionOps(cap_p)

    bc_zero = Dirichlet((x, y, t=0.0) -> 0.0)
    bc_ux = BorderConditions(Dict(:left=>bc_zero, :right=>bc_zero, :bottom=>bc_zero, :top=>bc_zero))
    bc_uy = BorderConditions(Dict(:left=>bc_zero, :right=>bc_zero, :bottom=>bc_zero, :top=>bc_zero))
    bc_p  = BorderConditions(Dict{Symbol,AbstractBoundary}())
    bc_cut = Dirichlet(0.0)

    μ = 1.0
    ρ = 1.0
    fᵤ = (x, y, z=0.0) -> 0.0
    fₚ = (x, y, z=0.0) -> 0.0

    fluid = Fluid((mesh_ux, mesh_uy),
                  (cap_ux, cap_uy),
                  (op_ux, op_uy),
                  mesh_p,
                  cap_p,
                  op_p,
                  μ, ρ, fᵤ, fₚ)

    solver = NavierStokesMono(fluid, (bc_ux, bc_uy), bc_p, bc_cut)
    data = Penguin.navierstokes2D_blocks(solver)
    return solver, data
end

@testset "Navier–Stokes convection operator" begin
    solver, data = build_simple_navierstokes()
    nu_x = data.nu_x; nu_y = data.nu_y; np = data.np
    Ntot = 2 * (nu_x + nu_y) + np

    advecting_state = zeros(Float64, Ntot)
    # Fill with deterministic patterns (not divergence-free, but sufficient to exercise operator)
    advecting_state[1:nu_x] .= range(-0.3, 0.25; length=nu_x)
    advecting_state[nu_x+1:2nu_x] .= range(0.15, -0.1; length=nu_x)
    advecting_state[2nu_x+1:2nu_x+nu_y] .= range(-0.2, 0.4; length=nu_y)
    advecting_state[2nu_x+nu_y+1:2*(nu_x+nu_y)] .= range(0.05, -0.35; length=nu_y)

    conv_x, conv_y = Penguin.compute_convection_vectors!(solver, data, advecting_state)
    ops = solver.last_conv_ops

    uωx = advecting_state[1:nu_x]
    uωy = advecting_state[2nu_x+1:2nu_x+nu_y]
    calc_x = ops.Cx * uωx - ops.Kx * uωx
    calc_y = ops.Cy * uωy - ops.Ky * uωy

    @test isapprox(conv_x, calc_x; atol=1e-12, rtol=1e-10)
    @test isapprox(conv_y, calc_y; atol=1e-12, rtol=1e-10)

    # Free-stream preservation: constant fields should yield zero convection
    const_state = copy(advecting_state)
    const_state .= 1.0
    conv_x_fs, conv_y_fs = Penguin.compute_convection_vectors!(solver, data, const_state)
    @test maximum(abs, conv_x_fs) ≤ 1e-10
    @test maximum(abs, conv_y_fs) ≤ 1e-10
end
