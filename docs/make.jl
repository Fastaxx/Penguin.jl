using Documenter, Penguin

makedocs(sitename="Penguin.jl", remotes=nothing, modules = [Penguin],
        pages = [
            "index.md",
            "Simulation key blocks" => [
                "blocks/mesh.md",
                "blocks/body.md",
                "blocks/capacity.md",
                "blocks/operators.md",
                "blocks/boundary.md",
                "blocks/phase.md",
            ],
        ])