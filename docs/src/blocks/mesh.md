# Mesh Usage Guide

This page describes how to create and work with a `Mesh` in the **Penguin.jl** package.  
A `Mesh` represents a discretized domain in one or multiple dimensions.

## Overview

A `Mesh` is constructed from one or more coordinate vectors. Each vector defines cell centers along a specific dimension. The constructor calculates:

1. **nodes** - The boundary positions of each cell.  
2. **centers** - The coordinate vectors you provided.  
3. **sizes** - Each cell’s size, adjusted for the first and last cells.

## Creating a 1D Mesh

```julia
using Penguin

x = range(0.0, stop=1.0, length=5)
mesh1D = Mesh((x,))

println(mesh1D.centers) # ([0.0, 0.25, 0.5, 0.75, 1.0],)
println(mesh1D.sizes)   # ([0.125, 0.25, 0.25, 0.25, 0.125],)
println(nC(mesh1D))     # 5 total cells
```

- **centers**: Cell center positions.  
- **sizes**: Cell sizes, with half-sizes at the extremes.  
- **nC(mesh)**: Total number of cells.

## Creating a 2D Mesh

```julia
using Penguin

x = range(0.0, stop=1.0, length=5)
y = range(0.0, stop=1.0, length=5)
mesh2D = Mesh((x, y))

println(mesh2D.centers) # ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0])
println(nC(mesh2D))     # 25 total cells
```

## Border Cells

Cells on the boundary can be identified with `get_border_cells`. A cell is at the border if its index is 1 or the last index along any dimension.

```julia
borders2D = get_border_cells(mesh2D)
println(length(borders2D)) # Number of border cells (16 for this 5x5 grid)
```

Each returned element is a tuple containing a `CartesianIndex` and the cell’s center coordinate:
```julia
(CartesianIndex((1, 1)), (0.0, 0.0))
```

## Summary

1. Construct a `Mesh` by passing coordinate vectors for each dimension.  
2. Use `nC(mesh)` to get the total number of cells.  
3. Use `get_border_cells(mesh)` to find which cells are located on the boundary.  