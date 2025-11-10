using LsqFit
using DataFrames

"""
    compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals; use_last=3)

Return rounded convergence-rate estimates for the global, full-cell, and cut-cell
errors. Fits are performed on `log(err)` vs `log(h)` using Linear Least Squares.
"""
function compute_orders(h_vals, err_vals, err_full_vals, err_cut_vals)
    function fit_model(x, p)
        p[1] .* x .+ p[2]
    end

    function safe_fit(h, err, use_last_n)
        mask = err .> 0
        if count(mask) < 2
            return NaN
        end
        h_pos = h[mask]
        err_pos = err[mask]
        log_h = log.(h_pos)
        log_err = log.(err_pos)

        n = min(use_last_n, length(log_h))
        idx = length(log_h) - n + 1 : length(log_h)
        fit_result = curve_fit(fit_model, log_h[idx], log_err[idx], [-1.0, 0.0])
        return fit_result.param[1]
    end

    p_all_all  = safe_fit(h_vals, err_vals, length(h_vals))
    p_full_all = safe_fit(h_vals, err_full_vals, length(h_vals))
    p_cut_all  = safe_fit(h_vals, err_cut_vals, length(h_vals))

    p_all  = safe_fit(h_vals, err_vals, 3)
    p_full = safe_fit(h_vals, err_full_vals, 3)
    p_cut  = safe_fit(h_vals, err_cut_vals, 3)

    round_or_nan(x) = isnan(x) ? NaN : round(x, digits=1)

    return (
        all = round_or_nan(p_all),
        full = round_or_nan(p_full),
        cut = round_or_nan(p_cut),
        all_all = round_or_nan(p_all_all),
        full_all = round_or_nan(p_full_all),
        cut_all = round_or_nan(p_cut_all)
    )
end

"""
    make_convergence_dataframe(method_name, data)

Create a `DataFrame` with convergence information for a given method.
"""
function make_convergence_dataframe(method_name, data)
    n = length(data.h_vals)
    lp_label = Vector{Union{Missing,String}}(undef, n)
    norm_value = haskey(data, :norm) ? data.norm : nothing

    if isnothing(norm_value)
        fill!(lp_label, missing)
    else
        fill!(lp_label, "L^$(norm_value)")
    end

    inside_cells = haskey(data, :inside_cells) ? data.inside_cells : fill(missing, n)

    return DataFrame(
        method = fill(method_name, length(data.h_vals)),
        h = data.h_vals,
        lp_norm = lp_label,
        inside_cells = inside_cells,
        all_err = data.err_vals,
        full_err = data.err_full_vals,
        cut_err = data.err_cut_vals,
        empty_err = data.err_empty_vals
    )
end


"""
    count_inside_cells(capacity)

Return the number of fully inside cells (cell type == 1).
"""
count_inside_cells(capacity) = count(x -> x == 1, capacity.cell_types)

"""
    count_cells_by_dim_inside_body(capacity)

Return the number of indices per dimension that contain at least one
fully inside cell.
"""
function count_cells_by_dim_inside_body(capacity)
    dims = capacity.mesh.dims .+1
    reshaped = reshape(capacity.cell_types, dims...)
    inside_mask = reshaped .== 1
    nd = length(dims)
    counts = Vector{Int}(undef, nd)

    for d in 1:nd
        c = 0
        for idx in 1:dims[d]
            slicer = ntuple(i -> i == d ? idx : Colon(), nd)
            if any(view(inside_mask, slicer...))
                c += 1
            end
        end
        counts[d] = c
    end
    return counts
end
