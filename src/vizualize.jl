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