
# Check Convergence
# Weighted Lp or L∞ norm helper
function lp_norm(errors, indices, pval, capacity)
    if pval == Inf
        return maximum(abs.(errors[indices]))
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V))^(1/pval)
    end
end

# Relative Lp norm helper
function relative_lp_norm(errors, indices, pval, capacity, u_ana)
    if pval == Inf
        return maximum(abs.(errors[indices]/u_ana[indices]))
    else
        part_sum = 0.0
        for i in indices
            Vi = capacity.V[i,i]
            part_sum += (abs(errors[i]/u_ana[i])^pval) * Vi
        end
        return (part_sum / sum(capacity.V))^(1/pval)
    end
end

function check_convergence(u_analytical::Function, solver, capacity::Capacity{1}, p::Real, relative::Bool=false)
    # 1) Compute pointwise error
    cell_centroids = capacity.C_ω
    u_ana = map(c -> u_analytical(c[1]), cell_centroids)
    u_num = solver.x[1:end÷2]
    err   = u_ana .- u_num

    # 2) Retrieve cell types and separate full, cut, empty
    cell_types = capacity.cell_types
    idx_all    = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full   = findall(cell_types .== 1)
    idx_cut    = findall(cell_types .== -1)
    idx_empty  = findall(cell_types .== 0)

    # 4) Compute norms (relative or not)
    if relative
        global_err = relative_lp_norm(err, idx_all, p, capacity, u_ana)
        full_err   = relative_lp_norm(err, idx_full,  p, capacity, u_ana)
        cut_err    = relative_lp_norm(err, idx_cut,   p, capacity, u_ana)
        empty_err  = relative_lp_norm(err, idx_empty, p, capacity, u_ana)
    else
        global_err = lp_norm(err, idx_all, p, capacity)
        full_err   = lp_norm(err, idx_full,  p, capacity)
        cut_err    = lp_norm(err, idx_cut,   p, capacity)
        empty_err  = lp_norm(err, idx_empty, p, capacity)
    end

    println("All cells L$p norm        = $global_err")
    println("Full cells L$p norm   = $full_err")
    println("Cut cells L$p norm    = $cut_err")
    println("Empty cells L$p norm  = $empty_err")

    return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
end



function check_convergence(u_analytical::Function, solver, capacity::Capacity{2}, p::Real=2, relative::Bool=false)
    # 1) Compute pointwise error
    cell_centroids = capacity.C_ω
    u_ana = map(c -> u_analytical(c[1], c[2]), cell_centroids)
    
    u_num = solver.x[1:end÷2]
    err   = u_ana .- u_num

    # 2) Retrieve cell types and separate full, cut, empty
    cell_types = capacity.cell_types
    idx_all    = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full   = findall(cell_types .== 1)
    idx_cut    = findall(cell_types .== -1)
    idx_empty  = findall(cell_types .== 0)

    # 4) Compute norms (relative or not)
    if relative
        global_err = relative_lp_norm(err, idx_all, p, capacity, u_ana)
        full_err   = relative_lp_norm(err, idx_full,  p, capacity, u_ana)
        cut_err    = relative_lp_norm(err, idx_cut,   p, capacity, u_ana)
        empty_err  = relative_lp_norm(err, idx_empty, p, capacity, u_ana)
    else
        global_err = lp_norm(err, idx_all, p, capacity)
        full_err   = lp_norm(err, idx_full,  p, capacity)
        cut_err    = lp_norm(err, idx_cut,   p, capacity)
        empty_err  = lp_norm(err, idx_empty, p, capacity)
    end

    println("All cells L$p norm        = $global_err")
    println("Full cells L$p norm   = $full_err")
    println("Cut cells L$p norm    = $cut_err")
    println("Empty cells L$p norm  = $empty_err")

    return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
end

function check_convergence(u_analytical::Function, solver, capacity::Capacity{3}, p::Real=2, relative::Bool=false)
    # 1) Compute pointwise error
    cell_centroids = capacity.C_ω
    u_ana = map(c -> u_analytical(c[1], c[2], c[3]), cell_centroids)
    
    u_num = solver.x[1:end÷2]
    err   = u_ana .- u_num

    # 2) Retrieve cell types and separate full, cut, empty
    cell_types = capacity.cell_types
    idx_all   = findall((cell_types .== 1) .| (cell_types .== -1))
    idx_full   = findall(cell_types .== 1)
    idx_cut    = findall(cell_types .== -1)
    idx_empty  = findall(cell_types .== 0)

    # 4) Compute norms (relative or not)
    if relative
        global_err = relative_lp_norm(err, idx_all, p, capacity, u_ana)
        full_err   = relative_lp_norm(err, idx_full,  p, capacity, u_ana)
        cut_err    = relative_lp_norm(err, idx_cut,   p, capacity, u_ana)
        empty_err  = relative_lp_norm(err, idx_empty, p, capacity, u_ana)
    else
        global_err = lp_norm(err, idx_all, p, capacity)
        full_err   = lp_norm(err, idx_full,  p, capacity)
        cut_err    = lp_norm(err, idx_cut,   p, capacity)
        empty_err  = lp_norm(err, idx_empty, p, capacity)
    end

    println("All cells L$p norm        = $global_err")
    println("Full cells L$p norm   = $full_err")
    println("Cut cells L$p norm    = $cut_err")
    println("Empty cells L$p norm  = $empty_err")

    return (u_ana, u_num, global_err, full_err, cut_err, empty_err)
end