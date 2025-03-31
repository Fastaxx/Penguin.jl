using Penguin
using CairoMakie
using Test

nx, ny = 20, 20
lx, ly = 1., 1.
x0, y0 = 0., 0.
dx, dy = lx/(nx), ly/(ny)

mesh = (collect(0:dx:lx), collect(0:dy:ly))
mesh_center = (collect(dx/2:dx:lx-dx/2), collect(dy/2:dy:ly-dy/2))

H_values = [0.25230897722466805, 0.3361875379944873, 0.7283705636359846, 0.6249949877617716, 0.03991558949506924, 0.7487243664584315, 0.0581283426924355, 0.13365575925682194, 0.274020585970562, 0.13098356221707774, 0.6321076875342545, 0.9486866746242493, 0.018965911275247604, 0.6562450063275221, 0.14273882165867402, 0.5549944892640787, 0.9030429430588073, 0.5096905667782294, 0.4005560888800397, 0.4395835174524384]
x_mesh = mesh[1]

# Interpolation
h_tilde = lin_interpol(x_mesh, H_values)
h_tilde_q = quad_interpol(x_mesh, H_values)
h_tilde_c = cubic_interpol(x_mesh, H_values)

# Plotting
x_plot = range(-dx, stop = lx+dx, length = 1000)
h_plot = h_tilde.(x_plot)
h_plot_q = h_tilde_q.(x_plot)
h_plot_c = h_tilde_c.(x_plot)


fig = Figure()
ax = Axis(fig[1, 1], title = "Linear Interpolation")
lines!(ax, x_plot, h_plot, color = :blue)
lines!(ax, x_plot, h_plot_q, color = :green)
lines!(ax, x_plot, h_plot_c, color = :orange)
scatter!(ax, mesh_center[1], H_values, color = :red, markersize = 5)
display(fig)

@testset "Interpolation Tests" begin
    # Test for extrapolation
    @test h_tilde(-dx) ≈ -4.360700574992466
    @test h_tilde(lx + dx) ≈ 5.066296084808194

    # Test for interpolation
    x_test = 0.5 * (mesh_center[1][1] + mesh_center[1][2])
    @test h_tilde(x_test) ≈ 1.7899788279637125

    # Test for quadratic interpolation
    @test h_tilde_q(x_test) ≈ 0.23193057917030827
    @test h_tilde_q(-dx) ≈ 0.5820704634937888
    @test h_tilde_q(lx + dx) ≈ 0.7860905563679295


end