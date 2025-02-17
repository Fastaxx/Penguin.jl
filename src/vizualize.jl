function plot_solution(solver, mesh::Mesh{1}, body::Function, capacity::Capacity; state_i=1)
    # Z_sdf = [body(x) for x in mesh.nodes[1]]

    # Déterminer le type de problème
    is_steady = solver.time_type == Steady # Problème stationnaire
    is_monophasic = solver.phase_type == Monophasic # Problème monophasique

    # Tracer selon la dimension et le type de problème
    if is_steady
        if is_monophasic # Monophasic
            uₒ = solver.x[1:length(solver.x) ÷ 2]
            uᵧ = solver.x[length(solver.x) ÷ 2 + 1:end]
            x = mesh.centers[1]

            # Désactiver les cellules désactivées
            cell_types = capacity.cell_types
            uₒ[cell_types .== 0] .= NaN
            #uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN
            
            fig = Figure()
            ax = Axis(fig[1, 1], title="Monophasic Steady Solution", xlabel="x", ylabel="u")
            scatter!(ax, uₒ, color=:blue, label="Bulk")
            scatter!(ax, uᵧ, color=:green, label="Interface")
            axislegend(ax)
            display(fig)
        else # Diphasic
            u1ₒ = solver.x[1:length(solver.x) ÷ 4]
            u1ᵧ = solver.x[length(solver.x) ÷ 4 + 1:2*length(solver.x) ÷ 4]
            u2ₒ = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            u2ᵧ = solver.x[3*length(solver.x) ÷ 4 + 1:end]

            # Désactiver les cellules désactivées
            cell_types = capacity.cell_types
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            x = mesh.centers[1]
            fig = Figure()
            ax = Axis(fig[1, 1], title="Diphasic Steady Solutions", xlabel="x", ylabel="u")
            scatter!(ax, u1ₒ, color=:blue, label="Phase 1 - Bulk")
            scatter!(ax, u1ᵧ, color=:green, label="Phase 1 - Interface")
            scatter!(ax, u2ₒ, color=:red, label="Phase 2 - Bulk")
            scatter!(ax, u2ᵧ, color=:purple, label="Phase 2 - Interface")
            axislegend(ax, position=:rb)
            display(fig)
        end
    else
        # Tracé unsteady en 1D
        if is_monophasic # Monophasic
            states = solver.states
            fig = Figure()
            ax = Axis(fig[1, 1], title="Monophasic Unsteady Solutions", xlabel="x", ylabel="u")
            for state in states
                lines!(ax, state[1:length(state) ÷ 2], color=:blue, alpha=0.3, label="Bulk")
                lines!(ax, state[length(state) ÷ 2 + 1:end], color=:green, alpha=0.3, label="Interface")
            end
            #axislegend(ax)
            display(fig)
        else # Diphasic

            states1ₒ = solver.states[state_i][1:length(solver.states[state_i]) ÷ 4]  # Phase 1 - Bulk
            states1ᵧ = solver.states[state_i][length(solver.states[state_i]) ÷ 4 + 1:2*length(solver.states[state_i]) ÷ 4]  # Phase 1 - Interface
            states2ₒ = solver.states[state_i][2*length(solver.states[state_i]) ÷ 4 + 1:3*length(solver.states[state_i]) ÷ 4]  # Phase 2 - Bulk
            states2ᵧ = solver.states[state_i][3*length(solver.states[state_i]) ÷ 4 + 1:end]  # Phase 2 - Interface

            cell_types = capacity.cell_types
            states1ₒ[cell_types .== 0] .= NaN
            states1ᵧ[cell_types .== 0] .= NaN
            states1ᵧ[cell_types .== 1] .= NaN
            states2ₒ[cell_types .== 1] .= NaN
            states2ᵧ[cell_types .== 1] .= NaN
            states2ᵧ[cell_types .== 0] .= NaN

            fig = Figure(size=(800, 600))
            ax1 = Axis(fig[1, 1], title="Diphasic Unsteady - Phase 1", xlabel="x", ylabel="u1")
            ax2 = Axis(fig[2, 1], title="Diphasic Unsteady - Phase 2", xlabel="x", ylabel="u2")
            lines!(ax1, states1ₒ, color=:blue, label="Bulk")
            scatter!(ax1, states1ᵧ, color=:green, label="Interface")
            lines!(ax2, states2ₒ, color=:blue, label="Bulk")
            scatter!(ax2, states2ᵧ, color=:green, label="Interface")
            #axislegend(ax1)
            #axislegend(ax2)
            display(fig)
        end
    end
end

function plot_solution(solver, mesh::Mesh{2}, body::Function, capacity::Capacity; state_i=1)
    #Z_sdf = [body(x,y) for x in mesh.nodes[1], y in mesh.nodes[2]]
    is_steady = solver.time_type == Steady
    is_monophasic = solver.phase_type == Monophasic

    # Tracé en 2D
    if is_steady
        fig = Figure(size=(800, 600))
        if is_monophasic # Monophasic
            cell_types = capacity.cell_types
            uₒ = solver.x[1:length(solver.x) ÷ 2]
            uᵧ = solver.x[length(solver.x) ÷ 2 + 1:end]

            # Désactiver les cellules désactivées
            uₒ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN

            reshaped_uₒ = reshape(uₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_uᵧ = reshape(uᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            ax1 = Axis(fig[1, 1], title="Monophasic Steady Solution - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            ax2 = Axis(fig[1, 3], title="Monophasic Steady Solution - Interface", xlabel="x", ylabel="y", aspect = DataAspect())

            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_uₒ, colormap=:viridis)
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_uᵧ, colormap=:viridis)

            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            Colorbar(fig[1, 2], hm1, label="Bulk Temperature")
            Colorbar(fig[1, 4], hm2, label="Interface Temperature")
        else # Diphasic
            cell_types = capacity.cell_types
            u1ₒ = solver.x[1:length(solver.x) ÷ 4]
            u1ᵧ = solver.x[length(solver.x) ÷ 4 + 1:2*length(solver.x) ÷ 4]
            u2ₒ = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            u2ᵧ = solver.x[3*length(solver.x) ÷ 4+1:end]

            # Désactiver les cellules désactivées
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            reshaped_u1ₒ = reshape(u1ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u1ᵧ = reshape(u1ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ₒ = reshape(u2ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ᵧ = reshape(u2ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            
            ax1 = Axis(fig[1, 1], title="Diphasic Steady - Phase 1 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_u1ₒ, colormap=:viridis)
            cb1 = Colorbar(fig[1, 2], hm1, label="Phase 1 Bulk Temperature")
            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax2 = Axis(fig[1, 3], title="Diphasic Steady - Phase 1 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_u1ᵧ, colormap=:viridis)
            cb2 = Colorbar(fig[1, 4], hm2, label="Phase 1 Interface Temperature")
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax3 = Axis(fig[2, 1], title="Diphasic Steady - Phase 2 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm3 = heatmap!(ax3, mesh.centers[1], mesh.centers[2], reshaped_u2ₒ, colormap=:viridis)
            cb3 = Colorbar(fig[2, 2], hm3, label="Phase 2 Bulk Temperature")
            #contour!(ax3, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax4 = Axis(fig[2, 3], title="Diphasic Steady - Phase 2 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm4 = heatmap!(ax4, mesh.centers[1], mesh.centers[2], reshaped_u2ᵧ, colormap=:viridis)
            cb4 = Colorbar(fig[2, 4], hm4, label="Phase 2 Interface Temperature")
            #contour!(ax4, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

        end
        display(fig)
    else
        # Tracé unsteady en 2D
        if is_monophasic
            cell_types = capacity.cell_types
            uₒ = solver.states[state_i][1:length(solver.states[1]) ÷ 2]
            uᵧ = solver.states[state_i][length(solver.states[1]) ÷ 2 + 1:end]

            # Désactiver les cellules désactivées
            uₒ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN

            reshaped_uₒ = reshape(uₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_uᵧ = reshape(uᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            fig = Figure(size=(800, 600))

            ax1 = Axis(fig[1, 1], title="Monophasic Unsteady Diffusion - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_uₒ, colormap=:viridis)
            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            Colorbar(fig[1, 2], hm1, label="Bulk Temperature")

            ax2 = Axis(fig[1, 3], title="Monophasic Unsteady Diffusion - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_uᵧ, colormap=:viridis)
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            Colorbar(fig[1, 4], hm2, label="Interface Temperature")

            display(fig)  
        else
            # Plot Last State
            cell_types = capacity.cell_types
            u1ₒ = solver.states[state_i][1:length(solver.states[1]) ÷ 4]
            u1ᵧ = solver.states[state_i][length(solver.states[1]) ÷ 4 + 1:2*length(solver.states[1]) ÷ 4]
            u2ₒ = solver.states[state_i][2*length(solver.states[1]) ÷ 4 + 1:3*length(solver.states[1]) ÷ 4]
            u2ᵧ = solver.states[state_i][3*length(solver.states[1]) ÷ 4 + 1:end]

            # Désactiver les cellules désactivées
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            reshaped_u1ₒ = reshape(u1ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u1ᵧ = reshape(u1ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ₒ = reshape(u2ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ᵧ = reshape(u2ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            fig = Figure(size=(800, 600))

            x, y = range(mesh.x0[1], stop=mesh.x0[1]+mesh.h[1][1]*length(mesh.h[1]), length=length(mesh.h[1])+1), range(mesh.x0[2], stop=mesh.x0[2]+mesh.h[2][1]*length(mesh.h[2]), length=length(mesh.h[2])+1)

            ax1 = Axis(fig[1, 1], title="Diphasic Unsteady - Phase 1 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, x, y, reshaped_u1ₒ, colormap=:viridis)
            Colorbar(fig[1, 2], hm1, label="Phase 1 Bulk Temperature")

            ax2 = Axis(fig[1, 3], title="Diphasic Unsteady - Phase 1 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, x, y, reshaped_u1ᵧ, colormap=:viridis)
            Colorbar(fig[1, 4], hm2, label="Phase 1 Interface Temperature")

            ax3 = Axis(fig[2, 1], title="Diphasic Unsteady - Phase 2 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm3 = heatmap!(ax3, x, y, reshaped_u2ₒ, colormap=:viridis)
            Colorbar(fig[2, 2], hm3, label="Phase 2 Bulk Temperature")

            ax4 = Axis(fig[2, 3], title="Diphasic Unsteady - Phase 2 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm4 = heatmap!(ax4, x, y, reshaped_u2ᵧ, colormap=:viridis)
            Colorbar(fig[2, 4], hm4, label="Phase 2 Interface Temperature")

            display(fig)
        end
    end
end


function plot_solution(solver, mesh::Mesh{3}, body::Function, capacity::Capacity; state_i=1)
    #Z_sdf = [body.sdf(xi, yi, zi) for zi in mesh.nodes[3], yi in mesh.nodes[2], xi in mesh.nodes[1]]

    is_steady = solver.time_type == Steady
    is_monophasic = solver.phase_type == Monophasic

    # Tracé en 3D
    if is_steady
        if is_monophasic
            fig = Figure()
            ax = LScene(fig[1, 1], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol = solver.x[1:length(solver.x) ÷ 2]
            vol = reshape(vol, nx+1, ny+1, nz+1)
            plt = volumeslices!(ax, x, y, z, vol)
            Colorbar(fig[1, 2], plt)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v; plt[:update_yz][](v) end
            on(sl_xz.value) do v; plt[:update_xz][](v) end
            on(sl_xy.value) do v; plt[:update_xy][](v) end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax.scene, projectiontype=Makie.Orthographic)

            display(fig)
        else # Diphasic
            fig = Figure()
            ax1 = LScene(fig[1, 1], show_axis=false)
            ax2 = LScene(fig[1, 3], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol1 = solver.x[1:length(solver.x) ÷ 4]
            vol1 = reshape(vol1, nx+1, ny+1, nz+1)
            plt1 = volumeslices!(ax1, x, y, z, vol1)
            Colorbar(fig[1, 2], plt1)

            vol2 = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            vol2 = reshape(vol2, nx+1, ny+1, nz+1)
            plt2 = volumeslices!(ax2, x, y, z, vol2)
            Colorbar(fig[1, 4], plt2)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v
                plt1[:update_yz][](v) 
                plt2[:update_yz][](v)
            end
            on(sl_xz.value) do v
                plt1[:update_xz][](v)
                plt2[:update_xz][](v)
            end
            on(sl_xy.value) do v
                plt1[:update_xy][](v)
                plt2[:update_xy][](v)
            end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt1[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax1.scene, projectiontype=Makie.Orthographic)
            # cam3d!(ax2.scene, projectiontype=Makie.Orthographic)
            display(fig)
        end
    else # Unsteady
        if is_monophasic
            fig = Figure()
            ax = LScene(fig[1, 1], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol = solver.states[state_i][1:length(solver.states[state_i]) ÷ 2]
            vol = reshape(vol, nx+1, ny+1, nz+1)
            plt = volumeslices!(ax, x, y, z, vol)
            Colorbar(fig[1, 2], plt)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v; plt[:update_yz][](v) end
            on(sl_xz.value) do v; plt[:update_xz][](v) end
            on(sl_xy.value) do v; plt[:update_xy][](v) end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax.scene, projectiontype=Makie.Orthographic)  

            display(fig)

        else # Diphasic
            fig = Figure()
            ax1 = LScene(fig[1, 1], show_axis=false)
            ax2 = LScene(fig[1, 3], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol1 = solver.states[state_i][1:length(solver.states[state_i]) ÷ 4]
            vol1 = reshape(vol1, nx+1, ny+1, nz+1)
            plt1 = volumeslices!(ax1, x, y, z, vol1)
            Colorbar(fig[1, 2], plt1)

            vol2 = solver.states[state_i][2*length(solver.states[state_i]) ÷ 4 + 1:3*length(solver.states[state_i]) ÷ 4]
            vol2 = reshape(vol2, nx+1, ny+1, nz+1)
            plt2 = volumeslices!(ax2, x, y, z, vol2)
            Colorbar(fig[1, 4], plt2)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v
                plt1[:update_yz][](v) 
                plt2[:update_yz][](v)
            end
            on(sl_xz.value) do v
                plt1[:update_xz][](v)
                plt2[:update_xz][](v)
            end
            on(sl_xy.value) do v
                plt1[:update_xy][](v)
                plt2[:update_xy][](v)
            end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt1[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax1.scene, projectiontype=Makie.Orthographic)
            # cam3d!(ax2.scene, projectiontype=Makie.Orthographic)
            display(fig)
        end
    end
end