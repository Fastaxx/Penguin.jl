module GibouFedkiwCommon

using ..Penguin

const sqrt5 = sqrt(5.0)

star_radius(θ) = 0.02 * sqrt5 + (0.5 + 0.2 * sin(5θ))

function star_levelset(x,y)
    r = sqrt(x^2 + y^2) + eps()
    θ = atan(y, x)
    return r - star_radius(θ)
end

function sphere_levelset(x,y,z, center, R)
    return sqrt((x-center[1])^2 + (y-center[2])^2 + (z-center[3])^2) - R
end

export star_levelset, sphere_levelset

end
