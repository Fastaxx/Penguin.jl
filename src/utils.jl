
## Temperature initialization functions

# Initialize temperature uniformly across the domain
function initialize_temperature_uniform!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, value::Float64)
    fill!(T0ₒ, value)
    fill!(T0ᵧ, value)
end

# Initialize temperature within a square region centered at 'center' with half-width 'half_width'
function initialize_temperature_square!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, center::Tuple{Float64, Float64}, half_width::Int, value::Float64, nx::Int, ny::Int)
    center_i = findfirst(x -> x >= center[1], x_coords)
    center_j = findfirst(y -> y >= center[2], y_coords)

    i_min = max(center_i - half_width, 1)
    i_max = min(center_i + half_width, nx + 1)
    j_min = max(center_j - half_width, 1)
    j_max = min(center_j + half_width, ny + 1)
    for j in j_min:j_max
        for i in i_min:i_max
            idx = i + (j - 1) * (nx + 1)
            T0ₒ[idx] = value
            T0ᵧ[idx] = value
        end
    end
end

# Initialize temperature within a circular region centered at 'center' with radius 'radius'
function initialize_temperature_circle!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, center::Tuple{Float64, Float64}, radius::Float64, value::Float64, nx::Int, ny::Int)
    for j in 1:(ny )
        for i in 1:(nx)
            idx = i + (j - 1) * (nx + 1)
            x_i = x_coords[i]
            y_j = y_coords[j]
            distance = sqrt((x_i - center[1])^2 + (y_j - center[2])^2)
            if distance <= radius
                T0ₒ[idx] = value
                T0ᵧ[idx] = value
            end
        end
    end
end

# Initialize temperature using a custom function 'func(x, y)'
function initialize_temperature_function!(T0ₒ::Vector{Float64}, T0ᵧ::Vector{Float64}, x_coords::Vector{Float64}, y_coords::Vector{Float64}, func::Function, nx::Int, ny::Int)
    for j in 1:(ny)
        for i in 1:(nx)
            idx = i + (j - 1) * (nx + 1)
            x = x_coords[i]
            y = y_coords[j]
            T_value = func(x, y)
            T0ₒ[idx] = T_value
            T0ᵧ[idx] = T_value
        end
    end
end


## Velocity field initialization functions

# Initialize the velocity field with a Rotational flow
function initialize_rotating_velocity_field(nx, ny, lx, ly, x0, y0, magnitude)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Define the center of rotation
    center_x, center_y = lx / 2, ly / 2

    # Calculate rotating velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            x = x0 + i * (lx / nx)
            y = y0 + j * (ly / ny)
            uₒx[idx] = -(y - center_y) * magnitude
            uₒy[idx] = (x - center_x) * magnitude
        end
    end
    return uₒx, uₒy
end


# Initialize the velocity field with a Poiseuille flow
function initialize_poiseuille_velocity_field(nx, ny, lx, ly, x0, y0)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Calculate Poiseuille velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            y = y0 + j * (ly / ny)
            x = x0 + i * (lx / nx)
            uₒx[idx] =  x * (1 - x) 
            uₒy[idx] =  0.0  # No velocity in the y direction for Poiseuille flow
        end
    end
    return uₒx, uₒy
end

# Initialize the velocity field with a radial flow
function initialize_radial_velocity_field(nx, ny, lx, ly, x0, y0, center, magnitude)
    # Number of nodes
    N = (nx + 1) * (ny + 1)

    # Initialize velocity components
    uₒx = zeros(N)
    uₒy = zeros(N)

    # Calculate radial velocity components
    for j in 0:ny
        for i in 0:nx
            idx = i + j * (nx + 1) + 1
            y = y0 + j * (ly / ny)
            x = x0 + i * (lx / nx)
            r = sqrt((x - center[1])^2 + (y - center[2])^2)
            uₒx[idx] = (x - center[1]) / r * magnitude
            uₒy[idx] = (y - center[2]) / r * magnitude
        end
    end
    return uₒx, uₒy
end

function initialize_stokes_velocity_field(nx, ny, nz, lx, ly, lz, x0, y0, z0, a, u)
    N = (nx + 1) * (ny + 1) * (nz + 1)
    uₒx = zeros(N)
    uₒy = zeros(N)
    uₒz = zeros(N)
    for k in 0:nz
        z = z0 + k * (lz / nz)
        for j in 0:ny
            y = y0 + j * (ly / ny)
            for i in 0:nx
                x = x0 + i * (lx / nx)
                idx = i + j * (nx + 1) + k * (nx + 1) * (ny + 1) + 1
                w = x^2 + y^2 + z^2
                sqrt_w = sqrt(w)
                term_common = (3 * a * u * z) / (4 * sqrt_w)
                term_factor = (a^2 / w^2) - (1 / w)
                uₒx[idx] = term_common * term_factor * x
                uₒy[idx] = term_common * term_factor * y
                term_z = ((2 * a^2 + 3 * x^2 + 3 * y^2) / (3 * w)) - ((a^2 * (x^2 + y^2)) / w^2) - 2
                uₒz[idx] = u + term_common * term_z
            end
        end
    end
    return uₒx, uₒy, uₒz
end

# Volume redefinition
function volume_redefinition!(capacity::Capacity{1}, operator::AbstractOperators)
    pₒ = [capacity.C_ω[i][1] for i in 1:length(capacity.C_ω)]
    pᵧ = [capacity.C_γ[i][1] for i in 1:length(capacity.C_ω)]
    p = vcat(pₒ, pᵧ)
    grad = ∇(operator, p)
    W_new = [grad[i] * capacity.W[1][i,i] for i in eachindex(grad)]
    W_new = spdiagm(0 => W_new)

    pₒ = [(capacity.C_ω[i][1]^2)/2 for i in 1:length(capacity.C_ω)]
    pᵧ = [(capacity.C_γ[i][1]^2)/2 for i in 1:length(capacity.C_ω)]

    p = vcat(pₒ, pᵧ)
    grad = ∇(operator, p)

    qω = vcat(grad)
    qγ = vcat(grad)

    div = ∇₋(operator, qω, qγ)
    V_new = spdiagm(0 => div)

    capacity.W = (W_new,)
    capacity.V = V_new

    println("Don't forget to rebuild the operator")
end