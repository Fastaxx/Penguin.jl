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
                "blocks/solver.md",
                "blocks/vizualize.md",
            ],
            "Examples" => [
                "tests/poisson.md",
                "tests/poisson_2ph.md",
                "tests/heat.md",
            ],
            "Benchmark" => [
                "benchmark/poisson.md",
                "benchmark/heat.md"
            ],
        ])