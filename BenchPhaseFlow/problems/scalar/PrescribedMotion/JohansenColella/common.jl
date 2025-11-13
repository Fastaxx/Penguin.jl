module JohansenColellaCommon

using ..Penguin

const γ = sqrt(2) / 15
const ELLIPSES = [
    (-6γ, -5γ, 3γ, 2γ),
    (10γ, -7γ, 2γ, 1γ),
    (7γ, 3γ, 1.5γ, 2γ)
]
const VELOCITIES = [
    (-0.10, 0.20),
    (-0.15, 0.15),
    (-0.20, 0.20)
]

φ_exact(x,y,t) = 4.0 / (5π * (t + 1)) * exp(-(x^2 + y^2) / (5 * (t + 1)))
function source_term(x,y,z,t)
    r2 = x^2 + y^2
    return 4.0 * (r2 - 5 * (t + 1)) / (125π * (t + 1)^3) * exp(-r2 / (5 * (t + 1)))
end

function ellipse_level(x,y,px,qx,ax,bx)
    return (x - px)^2 / ax^2 + (y - qx)^2 / bx^2 - 1
end

interval_min(a,b) = 0.5 * (a + b - abs(a - b))

function union_level(x,y, centers_axes)
    iterator = iterate(centers_axes)
    iterator === nothing && return Inf
    (px,qx,ax,bx), state = iterator
    min_level = ellipse_level(x,y,px,qx,ax,bx)
    while true
        nxt = iterate(centers_axes, state)
        nxt === nothing && break
        (px,qx,ax,bx), state = nxt
        lvl = ellipse_level(x,y,px,qx,ax,bx)
        min_level = interval_min(min_level, lvl)
    end
    return min_level
end

function centers_at_time(t, moving::Bool)
    if !moving
        return ELLIPSES
    end
    [(px + vx * t, qx + vy * t, ax, bx) for ((px,qx,ax,bx), (vx,vy)) in zip(ELLIPSES, VELOCITIES)]
end

function body_function(moving::Bool)
    if moving
        return (x,y,t=0.0)->begin
            centers = centers_at_time(t, true)
            union_level(x,y, centers)
        end
    else
        centers = centers_at_time(0.0, false)
        return (x,y,t=0.0)->-union_level(x,y, centers)
    end
end

function neumann_flux(x,y,t, moving::Bool)
    centers = centers_at_time(t, moving)
    min_level = Inf
    best = nothing
    for params in centers
        lvl = ellipse_level(x,y, params...)
        if lvl < min_level
            min_level = lvl
            best = params
        end
    end
    px,qx,ax,bx = best
    gx = 2*(x - px) / ax^2
    gy = 2*(y - qx) / bx^2
    norm = sqrt(gx^2 + gy^2) + eps()
    nx, ny = gx / norm, gy / norm
    coeff = -2 / (5 * (t + 1)) * φ_exact(x,y,t)
    gradx = coeff * x
    grady = coeff * y
    return gradx * nx + grady * ny
end

export φ_exact, source_term, body_function, neumann_flux, ELLIPSES

end # module
