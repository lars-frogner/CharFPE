module Solvers

export propagate_lagrange, propagate_lagrange_discrete, propagate_semi_lagrange_2d

using Interpolations
using UnzipLoops
using Dierckx

import ..RealArr1D, ..RealArr2D
using ..Const: mₑ, K, KEV_TO_ERG
using ..Numerical: trapz
using ..Atmosphere: CoulombLog, HybridCoulombLog
using ..Beams: Beam, compute_E_mean, compute_n₀
using ..Transport

function propagate_lagrange_discrete(
    beam::Beam,
    s::RealArr1D,
    log₁₀E_range::AbstractRange,
    μ₀::Real,
    nH::RealArr1D,
    x::RealArr1D,
    dlnBds::RealArr1D,
    ℰ::RealArr1D,
    use_analytical::Bool;
    min_n_steps_to_thermalize::Int = 2,
    max_n_steps_to_thermalize::Int = 10,
)
    n_distances = length(s)
    n_energies = length(log₁₀E_range)

    Δlog₁₀E = log₁₀E_range[2] - log₁₀E_range[1]

    E = zeros(n_energies, n_distances)
    E[:, 1] = collect(10 .^ log₁₀E_range)

    μ = zeros(n_energies, n_distances)
    μ[:, 1] .= μ₀

    n = zeros(n_energies, n_distances)
    n[:, 1] = compute_n₀.(beam, E[:, 1])

    E₀ = zeros(n_energies, n_distances)
    E₀[:, 1] = E[:, 1]

    high_energy_μ = μ₀
    ℰ_energy_loss = 0

    nH_prev = nH[1]
    x_prev = x[1]
    dlnBdN_prev = dlnBds[1] / nH_prev
    ℰ_prev = ℰ[1]

    Q = zeros(n_distances)

    E_mean = compute_E_mean(beam)
    coulomb_log = CoulombLog(nH_prev, x_prev, E_mean)

    hcl_prev = HybridCoulombLog(coulomb_log, x_prev)

    ΔN = (s[2] - s[1]) * nH_prev

    dEcdN = compute_dEdN(beam.Ec, μ₀, hcl_prev, nH_prev, ℰ_prev)
    n_steps_to_thermalize = beam.Ec / (-dEcdN * ΔN)
    n_initial_steps = min(
        max_n_steps_to_thermalize,
        Int(ceil(min_n_steps_to_thermalize / n_steps_to_thermalize)),
    )
    ΔN_new = ΔN / n_initial_steps

    E_sub = zeros(n_energies, n_initial_steps + 1)
    E_sub[:, 1] = E[:, 1]
    μ_sub = zeros(n_energies, n_initial_steps + 1)
    μ_sub[:, 1] = μ[:, 1]
    n_sub = zeros(n_energies, n_initial_steps + 1)
    n_sub[:, 1] = n[:, 1]
    E₀_sub = zeros(n_energies, n_initial_steps + 1)
    E₀_sub[:, 1] = E₀[:, 1]
    Q_sub = 0.0

    nH_curr = nH[2]
    x_curr = x[2]
    dlnBdN_curr = dlnBds[2] / nH_curr
    ℰ_curr = ℰ[2]
    hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

    for k = 2:n_initial_steps+1
        result = step_lagrange(
            μ₀,
            hcl_prev,
            nH_prev,
            dlnBdN_prev,
            ℰ_prev,
            hcl_curr,
            nH_curr,
            ℰ_curr,
            E_sub[:, k-1],
            μ_sub[:, k-1],
            n_sub[:, k-1],
            E₀_sub[:, k-1],
            ΔN_new,
            use_analytical,
        )

        if result === nothing
            @goto terminate
        end

        high_energy_μ = advance_high_energy_μ(high_energy_μ, dlnBdN_prev, ΔN_new)
        ℰ_energy_loss = advance_ℰ_energy_loss(ℰ_energy_loss, ℰ_prev, nH_prev, ΔN_new)

        if high_energy_μ == 0.0
            @goto terminate
        end

        μ_curr, n_curr, E₀_curr, Q_curr, i_therm = result

        shift = i_therm - 1

        E_sub[1:end-shift, k] = E_sub[i_therm:end, k-1]
        μ_sub[1:end-shift, k] = μ_curr
        n_sub[1:end-shift, k] = n_curr
        E₀_sub[1:end-shift, k] = E₀_curr

        if shift > 0
            E_sub[end+1-shift:end, k] =
                10.0 .^ (log10(E_sub[end, k-1]) .+ [i * Δlog₁₀E for i = 1:shift])
            E₀_sub[end+1-shift:end, k] = E_sub[end+1-shift:end, k] .+ ℰ_energy_loss
            μ_sub[end+1-shift:end, k] .= high_energy_μ
            n_sub[end+1-shift:end, k] = compute_n₀.(beam, E₀_sub[end+1-shift:end, k])
        end

        Q_sub = Q_curr
    end

    E[:, 2] = E_sub[:, end]
    μ[:, 2] = μ_sub[:, end]
    n[:, 2] = n_sub[:, end]
    E₀[:, 2] = E₀_sub[:, end]
    Q[2] += Q_sub

    for k in eachindex(s)[3:end]
        nH_curr = nH[k]
        x_curr = x[k]
        dlnBdN_curr = dlnBds[k] / nH_curr
        ℰ_curr = ℰ[k]
        hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

        ΔN = (s[k] - s[k-1]) * nH_curr

        result = step_lagrange(
            μ₀,
            hcl_prev,
            nH_prev,
            dlnBdN_prev,
            ℰ_prev,
            hcl_curr,
            nH_curr,
            ℰ_curr,
            E[:, k-1],
            μ[:, k-1],
            n[:, k-1],
            E₀[:, k-1],
            ΔN,
            use_analytical,
        )

        if result === nothing
            break
        end

        high_energy_μ = advance_high_energy_μ(high_energy_μ, dlnBdN_prev, ΔN)
        ℰ_energy_loss = advance_ℰ_energy_loss(ℰ_energy_loss, ℰ_prev, nH_prev, ΔN)

        if high_energy_μ == 0.0
            break
        end

        μ_curr, n_curr, E₀_curr, Q_curr, i_therm = result

        shift = i_therm - 1

        E[1:end-shift, k] = E[i_therm:end, k-1]
        μ[1:end-shift, k] = μ_curr
        n[1:end-shift, k] = n_curr
        E₀[1:end-shift, k] = E₀_curr

        if shift > 0
            E[end+1-shift:end, k] =
                10.0 .^ (log10(E[end, k-1]) .+ [i * Δlog₁₀E for i = 1:shift])
            E₀[end+1-shift:end, k] = E[end+1-shift:end, k] .+ ℰ_energy_loss
            μ[end+1-shift:end, k] .= high_energy_μ
            n[end+1-shift:end, k] = compute_n₀.(beam, E₀[end+1-shift:end, k])
        end

        Q[k] = Q_curr

        hcl_prev = hcl_curr
        dlnBdN_prev = dlnBdN_curr
    end
    @label terminate

    return E, μ, n, E₀, Q, E_sub, μ_sub, n_sub
end

function step_lagrange(
    μ₀,
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
    use_analytical,
)
    @assert all(E .> 0.0)
    @assert all(μ_prev .> 0.0)

    E_points, μ_points, n_points = broadcast_unzip(
        advance_E_μ_n,
        E,
        μ_prev,
        n_prev,
        hcl_prev,
        nH_prev,
        dlnBdN_prev,
        ℰ_prev,
        ΔN,
        use_analytical,
    )

    i_therm_prev = searchsortedlast(E_points, 0.0) + 1

    if i_therm_prev >= length(E_points)
        return nothing
    end

    Q = 0.0

    if i_therm_prev > 1
        valid_E_ = E[1:i_therm_prev]
        μ_prev_ = μ_prev[1:i_therm_prev]
        n_prev_ = n_prev[1:i_therm_prev]
        E₀_prev_ = E₀_prev[1:i_therm_prev]

        dEdN_ = compute_dEdN.(valid_E_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)
        dE₀dN_ = compute_dEdN.(E₀_prev_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)

        dQdE₀_ = @. (dEdN_ / dE₀dN_) *
           n_prev_ *
           (-dEdN_ * nH_prev * μ_prev_ * sqrt(2 * max(0, valid_E_) / mₑ))

        Q += trapz(E₀_prev_, dQdE₀_)
    end

    valid_E_points = E_points[i_therm_prev:end]
    log₁₀valid_E_points = log10.(valid_E_points)
    valid_μ_points = μ_points[i_therm_prev:end]
    valid_n_points = n_points[i_therm_prev:end]

    @assert all(valid_E_points .> 0)

    i_therm = searchsortedlast(E, valid_E_points[1]) + 1

    valid_E = E[i_therm:end]
    log₁₀valid_E = log10.(valid_E)

    μ_interp = Spline1D(log₁₀valid_E_points, valid_μ_points, k = 1, bc = "extrapolate")
    μ_curr = min.(μ₀, μ_interp.(log₁₀valid_E))

    n_interp = Spline1D(log₁₀valid_E_points, valid_n_points, k = 1, bc = "extrapolate")
    n_curr = max.(0.0, n_interp.(log₁₀valid_E))

    E₀_interp =
        Spline1D(log₁₀valid_E_points, E₀_prev[i_therm_prev:end], k = 1, bc = "extrapolate")
    E₀_curr = E₀_interp.(log₁₀valid_E)

    dEdN = compute_dEdN.(valid_E, μ_curr, hcl_curr, nH_curr, ℰ_curr)
    dE₀dN = compute_dEdN.(E₀_curr, μ_curr, hcl_curr, nH_curr, ℰ_curr)

    dQdE₀ = @. (dEdN / dE₀dN) *
       n_curr *
       (-dEdN * nH_curr * μ_curr * sqrt(2 * max(0, valid_E) / mₑ))

    Q += trapz(E₀_curr, dQdE₀)

    return μ_curr, n_curr, E₀_curr, Q, i_therm
end

# function propagate_lagrange(
#     beam::Beam,
#     s::RealArr1D,
#     log₁₀E_range::AbstractRange,
#     μ₀::Real,
#     n₀::RealArr1D,
#     nH::Function,
#     x::Function,
#     dlnBds::Function,
#     ℰ::Function,
# )
#     n_distances = length(s)
#     n_energies = length(log₁₀E_range)

#     E = collect(10 .^ log₁₀E_range)

#     μ = zeros(n_energies, n_distances)
#     μ[:, 1] .= μ₀

#     n = zeros(n_energies, n_distances)
#     n[:, 1] = n₀

#     E₀ = zeros(n_energies, n_distances)
#     E₀[:, 1] = E

#     s_prev = s[1]
#     nH_prev = nH(s_prev)
#     x_prev = x(s_prev)
#     dlnBdN_prev = dlnBds(s_prev) / nH_prev
#     ℰ_prev = ℰ(s_prev)

#     Q = zeros(n_distances)

#     E_mean = compute_E_mean(beam)
#     coulomb_log = CoulombLog(nH_prev, x_prev, E_mean)

#     hcl_prev = HybridCoulombLog(coulomb_log, x_prev)

#     offset = 1

#     for k in eachindex(s)[2:end]
#         s_curr = s[k]
#         nH_curr = nH(s_curr)
#         x_curr = x(s_curr)
#         dlnBdN_curr = dlnBds(s_curr) / nH_curr
#         ℰ_curr = ℰ(s_curr)
#         hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

#         ΔN = (s[k] - s[k-1]) * nH_curr

#         result = step_lagrange(
#             μ₀,
#             hcl_prev,
#             nH_prev,
#             dlnBdN_prev,
#             ℰ_prev,
#             hcl_curr,
#             nH_curr,
#             ℰ_curr,
#             E[offset:end],
#             μ[offset:end, k-1],
#             n[offset:end, k-1],
#             E₀[offset:end, k-1],
#             ΔN,
#         )

#         if result === nothing
#             break
#         end

#         μ_curr, n_curr, E₀_curr, Q_curr, i_therm = result

#         offset += i_therm - 1

#         μ[offset:end, k] = μ_curr
#         n[offset:end, k] = n_curr
#         E₀[offset:end, k] = E₀_curr

#         Q[k] = Q_curr

#         hcl_prev = hcl_curr
#         dlnBdN_prev = dlnBdN_curr
#     end

#     return μ, n, E₀, Q
# end

# function step_semi_lagrange(
#     hcl_prev,
#     nH_prev,
#     dlnBdN_prev,
#     ℰ_prev,
#     hcl_curr,
#     nH_curr,
#     dlnBdN_curr,
#     ℰ_curr,
#     E,
#     μ_prev,
#     n_prev,
#     E₀_prev,
#     ΔN,
# )
#     @assert all(E .> 0.0)
#     @assert all(μ_prev .> 0.0)

#     E_points, μ_points =
#         broadcast_unzip(advance_E_μ, E, μ_prev, hcl_prev, nH_prev, dlnBdN_prev, ℰ_prev, ΔN)

#     i_min_prev = searchsortedlast(E_points, 0.0) + 1

#     if i_min_prev >= length(E_points)
#         return nothing
#     end

#     Q = 0.0

#     if i_min_prev > 1
#         valid_E_ = E[1:i_min_prev]
#         μ_prev_ = μ_prev[1:i_min_prev]
#         n_prev_ = n_prev[1:i_min_prev]
#         E₀_prev_ = E₀_prev[1:i_min_prev]

#         dEdN_ = compute_dEdN.(valid_E_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)
#         dE₀dN_ = compute_dEdN.(E₀_prev_, μ_prev_, hcl_prev, nH_prev, ℰ_prev)

#         dQdE₀_ = @. (dEdN_ / dE₀dN_) *
#            n_prev_ *
#            (-dEdN_ * nH_prev * μ_prev_ * sqrt(2 * max(0, valid_E_) / mₑ))

#         Q += trapz(E₀_prev_, dQdE₀_)
#     end

#     valid_E_points = E_points[i_min_prev:end]
#     valid_μ_points = μ_points[i_min_prev:end]

#     @assert all(valid_E_points .> 0)

#     i_min = searchsortedlast(E, valid_E_points[1]) + 1

#     valid_E = E[i_min:end]
#     log₁₀valid_E = log10.(valid_E)
#     valid_n_prev = n_prev[i_min:end]
#     valid_E₀_prev = E₀_prev[i_min:end]

#     μ_interp = extrapolate(
#         interpolate((log10.(valid_E_points),), valid_μ_points, Gridded(Linear())),
#         Line(),
#     )
#     μ_curr = μ_interp.(log₁₀valid_E)

#     E_DP, μ_DP = broadcast_unzip(
#         advance_E_μ,
#         valid_E,
#         μ_curr,
#         hcl_curr,
#         nH_curr,
#         dlnBdN_curr,
#         ℰ_curr,
#         -ΔN,
#     )

#     n_interp =
#         extrapolate(interpolate((log₁₀valid_E,), valid_n_prev, Gridded(Linear())), 0.0)
#     n_DP = n_interp.(log10.(E_DP))
#     dndN_over_n_DP = compute_dndN_over_n.(E_DP, μ_DP, hcl_prev)
#     n_curr = @. n_DP + dndN_over_n_DP * n_DP * ΔN

#     E₀_interp =
#         extrapolate(interpolate((log₁₀valid_E,), valid_E₀_prev, Gridded(Linear())), Line())
#     E₀_curr = E₀_interp.(log10.(E_DP))

#     dEdN = compute_dEdN.(valid_E, μ_curr, hcl_curr, nH_curr, ℰ_curr)
#     dE₀dN = compute_dEdN.(E₀_curr, μ_curr, hcl_curr, nH_curr, ℰ_curr)

#     dQdE₀ = @. (dEdN / dE₀dN) *
#        n_curr *
#        (-dEdN * nH_curr * μ_curr * sqrt(2 * max(0, valid_E) / mₑ))

#     Q += trapz(E₀_curr, dQdE₀)

#     return μ_curr, n_curr, E₀_curr, Q, i_min
# end

# function propagate_semi_lagrange_2d(
#     beam::Beam,
#     s::RealArr1D,
#     log₁₀E_range::AbstractRange,
#     μ_range::AbstractRange,
#     n₀::RealArr2D,
#     nH::Function,
#     x::Function,
#     dlnBds::Function,
#     ℰ::Function,
#     E_min::Real,
# )
#     n_distances = length(s)
#     n_energies = length(log₁₀E_range)
#     n_pitch_angles = length(μ_range)

#     E = collect(10 .^ log₁₀E_range)
#     μ = collect(μ_range)

#     E_DP_all = zeros(n_distances, n_energies, n_pitch_angles)
#     μ_DP_all = zeros(n_distances, n_energies, n_pitch_angles)
#     n_DP_all = zeros(n_distances, n_energies, n_pitch_angles)

#     n = zeros(n_distances, n_energies, n_pitch_angles)
#     n[1, :, :] = n₀

#     Q = zeros(n_distances)

#     s_prev = s[1]
#     nH_prev = nH(s_prev)
#     x_prev = x(s_prev)
#     dlnBdN_prev = dlnBds(s_prev) / nH_prev
#     ℰ_prev = ℰ(s_prev)

#     E_mean = compute_E_mean(beam)
#     coulomb_log = CoulombLog(nH_prev, x_prev, E_mean)

#     hcl_prev = HybridCoulombLog(coulomb_log, x_prev)

#     for k in eachindex(s)[2:end]
#         s_curr = s[k]
#         nH_curr = nH(s_curr)
#         x_curr = x(s_curr)
#         dlnBdN_curr = dlnBds(s_curr) / nH_curr
#         ℰ_curr = ℰ(s_curr)
#         hcl_curr = HybridCoulombLog(coulomb_log, x_curr)

#         ΔN = (s[k] - s[k-1]) * nH_curr

#         # E_min[k] = E_min[k-1] + compute_dEds(E_min[k-1], μ_min[k-1], hcl_curr, nH_curr) * Δs
#         # μ_min[k] =
#         #     μ_min[k-1] +
#         #     compute_dμds(E_min[k-1], μ_min[k-1], hcl_curr, nH_curr, dlnBds_curr) * Δs

#         # if E_min[k] < 0 || μ_min[k] < 0
#         #     E_min[k] = NaN
#         #     μ_min[k] = NaN
#         # end

#         i_min = 1 #isnan(E_min[k]) ? 1 : searchsortedfirst(E, E_min[k])
#         j_min = 1 #isnan(μ_min[k]) ? 1 : searchsortedfirst(μ, μ_min[k])

#         Δlog₁₀E = log₁₀E_range[2] - log₁₀E_range[1]
#         Δμ = μ_range[2] - μ_range[1]

#         log₁₀E_center = log₁₀E_range[i_min:end] .+ 0.5 * Δlog₁₀E
#         μ_center = μ_range[j_min:end] .+ 0.5 * Δμ
#         E_center = 10 .^ log₁₀E_center

#         # E_DP_fwd, μ_DP_fwd = broadcast_unzip(
#         #     advance_E_μ,
#         #     E_center,
#         #     μ_center',
#         #     hcl_prev,
#         #     nH_prev,
#         #     dlnBdN_prev,
#         #     ℰ_prev,
#         #     ΔN,
#         #     E_min,
#         # )
#         # dndN_over_n_DP = compute_dndN_over_n.(E_center, μ_center', hcl_prev)
#         # n2 = @. n[k-1, i_min:end, j_min:end] +
#         #    dndN_over_n_DP * n[k-1, i_min:end, j_min:end] * ΔN
#         # mask = E_DP_fwd[:] .> 0
#         # n2_interp = Spline2D(
#         #     log10.(E_DP_fwd[mask]),
#         #     μ_DP_fwd[mask],
#         #     n2[mask],
#         #     kx = 1,
#         #     ky = 1,
#         #     s = 0.0,
#         # )
#         # n2 = evalgrid(n2_interp, log₁₀E_center, μ_center)

#         E_DP, μ_DP = broadcast_unzip(
#             advance_E_μ,
#             E[i_min:end],
#             μ[j_min:end]',
#             hcl_curr,
#             nH_curr,
#             dlnBdN_curr,
#             ℰ_curr,
#             -ΔN,
#             E_min,
#         )

#         E_DP_all[k, i_min:end, j_min:end] = E_DP
#         μ_DP_all[k, i_min:end, j_min:end] = μ_DP

#         n_interp = linear_interpolation(
#             (log₁₀E_center, μ_center),
#             n[k-1, i_min:end, j_min:end],
#             extrapolation_bc = 0.0,
#         )
#         n_DP = n_interp.(log10.(E_DP), μ_DP)

#         n_DP_all[k, i_min:end, j_min:end] = n_DP

#         dndN_over_n_DP = compute_dndN_over_n.(E_DP, μ_DP, hcl_prev)

#         n1 = @. n_DP + dndN_over_n_DP * n_DP * ΔN #@. n_DP * (2 + Δs * dnds_over_n_DP) / (2 - Δs * dnds_over_n)

#         n[k, i_min:end, j_min:end] = n1

#         dEdN = compute_dEdN.(E[i_min:end], μ[j_min:end]', hcl_curr, nH_curr, ℰ_curr)

#         dQdEdμ = @. n[k, i_min:end, j_min:end] *
#            (-dEdN * nH_curr * μ[j_min:end]' * sqrt(2 * E[i_min:end] / mₑ))

#         # Issue: for low energies and pitch angles, grid cells that
#         # are adjacent in pitch angle will have quite different μ_DP,
#         # meaning that it is easy to miss the narrow band in pitch angle
#         # for which the initial distribution is non-zero. The heating
#         # then gets underestimated. Possible mitigation: account for
#         # the values of he distribution in grid cells lying between the
#         # μ_DPs of grid cells that are adjacent in pitch angle. Maybe
#         # specify E and μ on grid cell edges and n in center?

#         # Possible optimization: change of variables (μE?) so that the box
#         # aligns better with the direction the distribution typically evolves.

#         Q[k] = trapz(E[i_min:end], μ[j_min:end], dQdEdμ)

#         s_prev = s_curr
#         nH_prev = nH_curr
#         x_prev = x_curr
#         dlnBdN_prev = dlnBdN_curr
#         ℰ_prev = ℰ_curr
#         hcl_prev = hcl_curr
#     end

#     return n, Q, E_DP_all, μ_DP_all, n_DP_all
# end

end
