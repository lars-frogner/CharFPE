module Solvers

export propagate_lagrange, propagate_semi_lagrange_2d, advance_E_μ

using Interpolations
using UnzipLoops

import ..RealArr1D, ..RealArr2D
using ..Const: mₑ, K, KEV_TO_ERG
using ..Numerical: trapz
using ..Atmosphere: CoulombLog, HybridCoulombLog
using ..Beams: Beam, compute_E_mean
using ..Transport

function advance_E_μ(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
    max_E_fac::Real,
)::Tuple{Float64,Float64}
    tr = LowEnergyTransport(hcl, nH, dlnBdN, ℰ)
    result = compute_E_μ(tr, E₀, μ₀, ΔN)
    return (true || result === nothing) ?
           advance_E_μ_euler(E₀, μ₀, hcl, nH, dlnBdN, ℰ, ΔN, max_E_fac) : result
end

function advance_E_μ_euler(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
    max_E_fac::Real,
)::Tuple{Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)

    @assert dμdN < 0

    ΔE_max = max_E_fac * E₀
    ΔE = abs(dEdN * ΔN)
    n_steps = Int(ceil(ΔE / ΔE_max))

    ΔN_sub = ΔN / n_steps

    E = E₀ + dEdN * ΔN_sub
    μ = μ₀ + dμdN * ΔN_sub

    if E <= 0.0 || μ <= 0
        return 0.0, 0.0
    end

    for _ = 2:n_steps
        dEdN = compute_dEdN.(E, μ, hcl, nH, ℰ)
        dμdN = compute_dμdN.(E, μ, hcl, nH, dlnBdN, ℰ)
        E += dEdN * ΔN_sub
        μ += dμdN * ΔN_sub
        if E <= 0.0 || μ <= 0
            E = 0
            μ = 0
            break
        end
    end

    return E, μ
end

function advance_E_μ_euler(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
    max_E_fac::Real,
)::Tuple{Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)

    @assert dμdN < 0

    ΔE_max = max_E_fac * E₀
    ΔE = abs(dEdN * ΔN)
    n_steps = Int(ceil(ΔE / ΔE_max))

    ΔN_sub = ΔN / n_steps

    E = E₀ + dEdN * ΔN_sub
    μ = μ₀ + dμdN * ΔN_sub

    if E <= 0.0 || μ <= 0
        return 0.0, 0.0
    end

    for _ = 2:n_steps
        dEdN = compute_dEdN.(E, μ, hcl, nH, ℰ)
        dμdN = compute_dμdN.(E, μ, hcl, nH, dlnBdN, ℰ)
        E += dEdN * ΔN_sub
        μ += dμdN * ΔN_sub
        if E <= 0.0 || μ <= 0
            E = 0
            μ = 0
            break
        end
    end

    return E, μ
end

function propagate_lagrange(
    beam::Beam,
    s::RealArr1D,
    log₁₀E_range::AbstractRange,
    μ₀::Real,
    n₀::RealArr1D,
    nH::Function,
    x::Function,
    dlnBds::Function,
    ℰ::Function,
    max_E_fac::Real,
)
    n_distances = length(s)
    n_energies = length(log₁₀E_range)

    E = collect(10 .^ log₁₀E_range)

    μ = zeros(n_energies, n_distances)
    μ[:, 1] .= μ₀

    n = zeros(n_energies, n_distances)
    n[:, 1] = n₀

    E₀ = zeros(n_energies, n_distances)
    E₀[:, 1] = E

    s_prev = s[1]
    nH_prev = nH(s_prev)
    x_prev = x(s_prev)
    dlnBdN_prev = dlnBds(s_prev) / nH_prev
    ℰ_prev = ℰ(s_prev)

    Q = zeros(n_distances)

    E_mean = compute_E_mean(beam)
    coulomb_log = CoulombLog(nH_prev, x_prev, E_mean)

    hcl_prev = HybridCoulombLog(coulomb_log, x_prev)

    offset = 1

    for k in eachindex(s)[2:end]
        s_curr = s[k]
        nH_curr = nH(s_curr)
        x_curr = x(s_curr)
        dlnBdN_curr = dlnBds(s_curr) / nH_curr
        ℰ_curr = ℰ(s_curr)
        hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

        ΔN = (s[k] - s[k-1]) * nH_curr

        result = step_lagrange(
            hcl_prev,
            nH_prev,
            dlnBdN_prev,
            ℰ_prev,
            hcl_curr,
            nH_curr,
            ℰ_curr,
            E[offset:end],
            μ[offset:end, k-1],
            n[offset:end, k-1],
            E₀[offset:end, k-1],
            ΔN,
            max_E_fac,
        )

        if result === nothing
            break
        end

        μ_curr, n_curr, E₀_curr, Q_curr, i_min = result

        offset += i_min - 1

        μ[offset:end, k] = μ_curr
        n[offset:end, k] = n_curr
        E₀[offset:end, k] = E₀_curr

        Q[k] = Q_curr

        hcl_prev = hcl_curr
        dlnBdN_prev = dlnBdN_curr
    end

    return μ, n, E₀, Q
end

function step_lagrange(
    hcl_prev,
    nH_prev,
    dlnBdN_prev,
    ℰ_prev,
    hcl_curr,
    nH_curr,
    ℰ_curr,
    E,
    μ_prev,
    n_prev,
    E₀_prev,
    ΔN,
    max_E_fac,
)
    @assert all(E .> 0.0)
    @assert all(μ_prev .> 0.0)

    E_points, μ_points = broadcast_unzip(
        advance_E_μ,
        E,
        μ_prev,
        hcl_prev,
        nH_prev,
        dlnBdN_prev,
        ℰ_prev,
        ΔN,
        max_E_fac,
    )

    i_min_prev = searchsortedlast(E_points, 0.0) + 1

    if i_min_prev >= length(E_points)
        return nothing
    end

    Q = 0.0

    if i_min_prev > 1
        valid_E_ = E[1:i_min_prev]
        μ_prev_ = μ_prev[1:i_min_prev]
        n_prev_ = n_prev[1:i_min_prev]
        E₀_prev_ = E₀_prev[1:i_min_prev]

        dEdN_ = compute_dEdN.(valid_E_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)
        dE₀dN_ = compute_dEdN.(E₀_prev_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)

        dQdE₀_ = @. (dEdN_ / dE₀dN_) *
           n_prev_ *
           (-dEdN_ * nH_prev * μ_prev_ * sqrt(2 * max(0, valid_E_) / mₑ))

        Q += trapz(E₀_prev_, dQdE₀_)
    end

    valid_E_points = E_points[i_min_prev:end]
    log₁₀valid_E_points = log10.(valid_E_points)
    valid_μ_points = μ_points[i_min_prev:end]

    @assert all(valid_E_points .> 0)

    i_min = searchsortedlast(E, valid_E_points[1]) + 1

    valid_E = E[i_min:end]
    log₁₀valid_E = log10.(valid_E)

    μ_interp = extrapolate(
        interpolate((log₁₀valid_E_points,), valid_μ_points, Gridded(Linear())),
        Line(),
    )
    μ_curr = μ_interp.(log₁₀valid_E)

    dndN_prev =
        compute_dndN.(
            E[i_min_prev:end],
            μ_prev[i_min_prev:end],
            n_prev[i_min_prev:end],
            hcl_prev,
        )
    n_points = @. n_prev[i_min_prev:end] + dndN_prev * ΔN

    n_interp =
        extrapolate(interpolate((log₁₀valid_E_points,), n_points, Gridded(Linear())), 0.0)
    n_curr = n_interp.(log₁₀valid_E)

    E₀_interp = extrapolate(
        interpolate((log₁₀valid_E_points,), E₀_prev[i_min_prev:end], Gridded(Linear())),
        Line(),
    )
    E₀_curr = E₀_interp.(log₁₀valid_E)

    dEdN = compute_dEdN.(valid_E, μ_curr, hcl_curr, nH_curr, ℰ_curr)
    dE₀dN = compute_dEdN.(E₀_curr, μ_curr, hcl_curr, nH_curr, ℰ_curr)

    dQdE₀ = @. (dEdN / dE₀dN) *
       n_curr *
       (-dEdN * nH_curr * μ_curr * sqrt(2 * max(0, valid_E) / mₑ))

    Q += trapz(E₀_curr, dQdE₀)

    return μ_curr, n_curr, E₀_curr, Q, i_min
end

function step_semi_lagrange(
    hcl_prev,
    nH_prev,
    dlnBdN_prev,
    ℰ_prev,
    hcl_curr,
    nH_curr,
    dlnBdN_curr,
    ℰ_curr,
    E,
    μ_prev,
    n_prev,
    E₀_prev,
    ΔN,
    max_E_fac,
)
    @assert all(E .> 0.0)
    @assert all(μ_prev .> 0.0)

    E_points, μ_points = broadcast_unzip(
        advance_E_μ,
        E,
        μ_prev,
        hcl_prev,
        nH_prev,
        dlnBdN_prev,
        ℰ_prev,
        ΔN,
        max_E_fac,
    )

    i_min_prev = searchsortedlast(E_points, 0.0) + 1

    if i_min_prev >= length(E_points)
        return nothing
    end

    Q = 0.0

    if i_min_prev > 1
        valid_E_ = E[1:i_min_prev]
        μ_prev_ = μ_prev[1:i_min_prev]
        n_prev_ = n_prev[1:i_min_prev]
        E₀_prev_ = E₀_prev[1:i_min_prev]

        dEdN_ = compute_dEdN.(valid_E_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)
        dE₀dN_ = compute_dEdN.(E₀_prev_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)

        dQdE₀_ = @. (dEdN_ / dE₀dN_) *
           n_prev_ *
           (-dEdN_ * nH_prev * μ_prev_ * sqrt(2 * max(0, valid_E_) / mₑ))

        Q += trapz(E₀_prev_, dQdE₀_)
    end

    valid_E_points = E_points[i_min_prev:end]
    valid_μ_points = μ_points[i_min_prev:end]

    @assert all(valid_E_points .> 0)

    i_min = searchsortedlast(E, valid_E_points[1]) + 1

    valid_E = E[i_min:end]
    log₁₀valid_E = log10.(valid_E)
    valid_n_prev = n_prev[i_min:end]
    valid_E₀_prev = E₀_prev[i_min:end]

    μ_interp = extrapolate(
        interpolate((log10.(valid_E_points),), valid_μ_points, Gridded(Linear())),
        Line(),
    )
    μ_curr = μ_interp.(log₁₀valid_E)

    E_DP, μ_DP = broadcast_unzip(
        advance_E_μ,
        valid_E,
        μ_curr,
        hcl_curr,
        nH_curr,
        dlnBdN_curr,
        ℰ_curr,
        -ΔN,
        max_E_fac,
    )

    n_interp =
        extrapolate(interpolate((log₁₀valid_E,), valid_n_prev, Gridded(Linear())), 0.0)
    n_DP = n_interp.(log10.(E_DP))
    dndN_over_n_DP = compute_dndN_over_n.(E_DP, μ_DP, hcl_prev)
    n_curr = @. n_DP + dndN_over_n_DP * n_DP * ΔN

    E₀_interp =
        extrapolate(interpolate((log₁₀valid_E,), valid_E₀_prev, Gridded(Linear())), Line())
    E₀_curr = E₀_interp.(log10.(E_DP))

    dEdN = compute_dEdN.(valid_E, μ_curr, hcl_curr, nH_curr, ℰ_curr)
    dE₀dN = compute_dEdN.(E₀_curr, μ_curr, hcl_curr, nH_curr, ℰ_curr)

    dQdE₀ = @. (dEdN / dE₀dN) *
       n_curr *
       (-dEdN * nH_curr * μ_curr * sqrt(2 * max(0, valid_E) / mₑ))

    Q += trapz(E₀_curr, dQdE₀)

    return μ_curr, n_curr, E₀_curr, Q, i_min
end

function propagate_semi_lagrange_2d(
    beam::Beam,
    s::RealArr1D,
    log₁₀E_range::AbstractRange,
    μ_range::AbstractRange,
    n₀::RealArr2D,
    nH::Function,
    x::Function,
    dlnBds::Function,
    ℰ::Function,
    max_E_fac::Real,
    E_min::Real,
)
    n_distances = length(s)
    n_energies = length(log₁₀E_range)
    n_pitch_angles = length(μ_range)

    E = collect(10 .^ log₁₀E_range)
    μ = collect(μ_range)

    E_DP_all = zeros(n_distances, n_energies, n_pitch_angles)
    μ_DP_all = zeros(n_distances, n_energies, n_pitch_angles)
    n_DP_all = zeros(n_distances, n_energies, n_pitch_angles)

    n = zeros(n_distances, n_energies, n_pitch_angles)
    n[1, :, :] = n₀

    Q = zeros(n_distances)

    s_prev = s[1]
    nH_prev = nH(s_prev)
    x_prev = x(s_prev)
    dlnBdN_prev = dlnBds(s_prev) / nH_prev
    ℰ_prev = ℰ(s_prev)

    E_mean = compute_E_mean(beam)
    coulomb_log = CoulombLog(nH_prev, x_prev, E_mean)

    hcl_prev = HybridCoulombLog(coulomb_log, x_prev)

    for k in eachindex(s)[2:end]
        s_curr = s[k]
        nH_curr = nH(s_curr)
        x_curr = x(s_curr)
        dlnBdN_curr = dlnBds(s_curr) / nH_curr
        ℰ_curr = ℰ(s_curr)
        hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

        ΔN = (s[k] - s[k-1]) * nH_curr

        # E_min[k] = E_min[k-1] + compute_dEds(E_min[k-1], μ_min[k-1], hcl_curr, nH_curr) * Δs
        # μ_min[k] =
        #     μ_min[k-1] +
        #     compute_dμds(E_min[k-1], μ_min[k-1], hcl_curr, nH_curr, dlnBds_curr) * Δs

        # if E_min[k] < 0 || μ_min[k] < 0
        #     E_min[k] = NaN
        #     μ_min[k] = NaN
        # end

        i_min = 1 #isnan(E_min[k]) ? 1 : searchsortedfirst(E, E_min[k])
        j_min = 1 #isnan(μ_min[k]) ? 1 : searchsortedfirst(μ, μ_min[k])

        Δlog₁₀E = log₁₀E_range[2] - log₁₀E_range[1]
        Δμ = μ_range[2] - μ_range[1]

        log₁₀E_center = log₁₀E_range[i_min:end] .+ 0.5 * Δlog₁₀E
        μ_center = μ_range[j_min:end] .+ 0.5 * Δμ
        E_center = 10 .^ log₁₀E_center

        # E_DP_fwd, μ_DP_fwd = broadcast_unzip(
        #     advance_E_μ,
        #     E_center,
        #     μ_center',
        #     hcl_prev,
        #     nH_prev,
        #     dlnBdN_prev,
        #     ℰ_prev,
        #     ΔN,
        #     E_min,
        #     max_E_fac,
        # )
        # dndN_over_n_DP = compute_dndN_over_n.(E_center, μ_center', hcl_prev)
        # n2 = @. n[k-1, i_min:end, j_min:end] +
        #    dndN_over_n_DP * n[k-1, i_min:end, j_min:end] * ΔN
        # mask = E_DP_fwd[:] .> 0
        # n2_interp = Spline2D(
        #     log10.(E_DP_fwd[mask]),
        #     μ_DP_fwd[mask],
        #     n2[mask],
        #     kx = 1,
        #     ky = 1,
        #     s = 0.0,
        # )
        # n2 = evalgrid(n2_interp, log₁₀E_center, μ_center)

        E_DP, μ_DP = broadcast_unzip(
            advance_E_μ,
            E[i_min:end],
            μ[j_min:end]',
            hcl_curr,
            nH_curr,
            dlnBdN_curr,
            ℰ_curr,
            -ΔN,
            E_min,
            max_E_fac,
        )

        E_DP_all[k, i_min:end, j_min:end] = E_DP
        μ_DP_all[k, i_min:end, j_min:end] = μ_DP

        n_interp = linear_interpolation(
            (log₁₀E_center, μ_center),
            n[k-1, i_min:end, j_min:end],
            extrapolation_bc = 0.0,
        )
        n_DP = n_interp.(log10.(E_DP), μ_DP)

        n_DP_all[k, i_min:end, j_min:end] = n_DP

        dndN_over_n_DP = compute_dndN_over_n.(E_DP, μ_DP, hcl_prev)

        n1 = @. n_DP + dndN_over_n_DP * n_DP * ΔN #@. n_DP * (2 + Δs * dnds_over_n_DP) / (2 - Δs * dnds_over_n)

        n[k, i_min:end, j_min:end] = n1

        dEdN = compute_dEdN.(E[i_min:end], μ[j_min:end]', hcl_curr, nH_curr, ℰ_curr)

        dQdEdμ = @. n[k, i_min:end, j_min:end] *
           (-dEdN * nH_curr * μ[j_min:end]' * sqrt(2 * E[i_min:end] / mₑ))

        # Issue: for low energies and pitch angles, grid cells that
        # are adjacent in pitch angle will have quite different μ_DP,
        # meaning that it is easy to miss the narrow band in pitch angle
        # for which the initial distribution is non-zero. The heating
        # then gets underestimated. Possible mitigation: account for
        # the values of he distribution in grid cells lying between the
        # μ_DPs of grid cells that are adjacent in pitch angle. Maybe
        # specify E and μ on grid cell edges and n in center?

        # Possible optimization: change of variables (μE?) so that the box
        # aligns better with the direction the distribution typically evolves.

        Q[k] = trapz(E[i_min:end], μ[j_min:end], dQdEdμ)

        s_prev = s_curr
        nH_prev = nH_curr
        x_prev = x_curr
        dlnBdN_prev = dlnBdN_curr
        ℰ_prev = ℰ_curr
        hcl_prev = hcl_curr
    end

    return n, Q, E_DP_all, μ_DP_all, n_DP_all
end

end
