using LsqFit
using DataFrames

"""
    count_inside_cells(capacity)

Return the number of fully inside cells (cell type == 1).
"""
count_inside_cells(capacity) = count(x -> x == 1, capacity.cell_types)

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
    compute_pairwise_orders(h_vals, err_vals)

Return per-level convergence orders computed between consecutive mesh levels:
`p_i = log(e_{i-1}/e_i) / log(h_{i-1}/h_i)` for `i >= 2`. Entries with
insufficient data are filled with `NaN`.
"""
function compute_pairwise_orders(h_vals, err_vals)
    n = length(h_vals)
    orders = fill(NaN, n)
    for i in 2:n
        h_prev = h_vals[i-1]
        h_curr = h_vals[i]
        e_prev = err_vals[i-1]
        e_curr = err_vals[i]
        if h_prev > 0 && h_curr > 0 && e_prev > 0 && e_curr > 0 && h_prev != h_curr
            orders[i] = log(e_prev / e_curr) / log(h_prev / h_curr)
        end
    end
    return orders
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

    pair_all = compute_pairwise_orders(data.h_vals, data.err_vals)
    pair_full = compute_pairwise_orders(data.h_vals, data.err_full_vals)
    pair_cut = compute_pairwise_orders(data.h_vals, data.err_cut_vals)

    df = DataFrame(
        method = fill(method_name, n),
        h = data.h_vals,
        lp_norm = lp_label,
        inside_cells = data.inside_cells,
        inside_cells_by_dim = data.inside_cells_by_dim,
        all_err = data.err_vals,
        full_err = data.err_full_vals,
        cut_err = data.err_cut_vals,
        empty_err = data.err_empty_vals,
        pair_order_all = pair_all,
        pair_order_full = pair_full,
        pair_order_cut = pair_cut
    )

    return df
end

"""
    make_diphasic_convergence_dataframe(method_name, data)

Create a `DataFrame` with per-phase and combined convergence information for a
two-phase configuration. `data` must provide the same fields as the monophasic
case plus per-phase errors and cell counts.
"""
function make_diphasic_convergence_dataframe(method_name, data)
    n = length(data.h_vals)
    lp_label = Vector{Union{Missing,String}}(undef, n)
    norm_value = haskey(data, :norm) ? data.norm : nothing

    if isnothing(norm_value)
        fill!(lp_label, missing)
    else
        fill!(lp_label, "L^$(norm_value)")
    end

    pair_all = compute_pairwise_orders(data.h_vals, data.err_vals)
    pair_full = compute_pairwise_orders(data.h_vals, data.err_full_vals)
    pair_cut = compute_pairwise_orders(data.h_vals, data.err_cut_vals)

    df = DataFrame(
        method = fill(method_name, n),
        h = data.h_vals,
        lp_norm = lp_label,
        inside_cells = data.inside_cells,
        inside_cells_phase1 = data.inside_cells_phase1,
        inside_cells_phase2 = data.inside_cells_phase2,
        inside_cells_by_dim = data.inside_cells_by_dim,
        all_err = data.err_vals,
        full_err = data.err_full_vals,
        cut_err = data.err_cut_vals,
        empty_err = data.err_empty_vals,
        phase1_all_err = data.phase1_all_errs,
        phase1_full_err = data.phase1_full_errs,
        phase1_cut_err = data.phase1_cut_errs,
        phase1_empty_err = data.phase1_empty_errs,
        phase2_all_err = data.phase2_all_errs,
        phase2_full_err = data.phase2_full_errs,
        phase2_cut_err = data.phase2_cut_errs,
        phase2_empty_err = data.phase2_empty_errs,
        pair_order_all = pair_all,
        pair_order_full = pair_full,
        pair_order_cut = pair_cut
    )

    return df
end
