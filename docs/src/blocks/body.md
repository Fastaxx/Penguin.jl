# Defining a Body

A body can be represented with a signed distance function (SDF). Penguin.jl supports both vectorized and non-vectorized SDFs. You can switch between the two styles depending on whether you need a single function call per coordinate (`LS`) or a function that accepts a vector (`Φ`).

## Example

```julia
using Penguin

# 1) Vectorized version
Φ(X) = sqrt(X[1]^2 + X[2]^2) - 0.5  # ϕ(x, y) = Φ([x, y]) to switch from vectorized to non-vectorized

# 2) Non-vectorized version
LS(x, y, _=0) = sqrt(x^2 + y^2) - 0.5 # LS(X) = LS(X[1], X[2]) to switch from non-vectorized to vectorized

# Create a mesh
x = range(-1.0, stop=1.0, length=50)
y = range(-1.0, stop=1.0, length=50)
mesh = Mesh((x, y))

# Define the body using either approach
# Option A: VOFI
capacity = Capacity(LS, mesh, method="VOFI")

# Option B: ImplicitIntegration
capacity = Capacity(Φ, mesh, method="ImplicitIntegration")
```

- **Φ(X)**: A vectorized function that expects a coordinate array, e.g. `[x, y]`.  
- **LS(x, y, _=0)**: A non-vectorized function that can be easily used by VOFI.  


With these definitions, you can build simulation code that distinguishes inside vs. outside regions of a body. Both vectorized and non-vectorized forms are interchangeable, thanks to the bridging functions used in VOFI vs. ImplicitIntegration.