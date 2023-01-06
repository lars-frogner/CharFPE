module Transport

export compute_dEdN,
    compute_dμdN, compute_dndN_over_n, compute_dndN, LowEnergyTransport, compute_E_μ
using ..Const: K, mₑ, e
using ..Atmosphere: HybridCoulombLog

using NLsolve: nlsolve, converged

compute_dEdN(E, μ, hcl::HybridCoulombLog, nH, ℰ) = -K * hcl.γ / (μ * E) - e * ℰ / nH
compute_dμdN(E, μ, hcl::HybridCoulombLog, nH, dlnBdN, ℰ) =
    -K * hcl.γ′ / (2 * E^2) - (dlnBdN + e * ℰ / (nH * E)) * (1 - μ^2) / (2μ)
compute_dndN_over_n(E, μ, hcl::HybridCoulombLog) = K * hcl.γ′′ / (2 * μ * E^2)
compute_dndN(E, μ, n, hcl::HybridCoulombLog) = compute_dndN_over_n(E, μ, hcl) * n

struct LowEnergyTransport
    hcl::HybridCoulombLog
    cB::Real
    cℰ::Real
    max_ℰ_ratio::Real
    max_μ::Real
    max_cB_E₀²_div_μ₀::Real
    max_cℰ_E₀_div_μ₀::Real

    function LowEnergyTransport(
        hcl::HybridCoulombLog,
        nH,
        dlnBdN,
        ℰ;
        max_ℰ_ratio = 1e-3,
        max_μ = 1e-1,
        max_cB_E₀²_div_μ₀ = 1e-2,
        max_cℰ_E₀_div_μ₀ = 1e-2,
    )
        cB = dlnBdN / (K * hcl.γ)
        cℰ = e * ℰ / (K * hcl.γ * nH)
        return new(hcl, cB, cℰ, max_ℰ_ratio, max_μ, max_cB_E₀²_div_μ₀, max_cℰ_E₀_div_μ₀)
    end
end

function compute_E_μ(
    tr::LowEnergyTransport,
    E₀::Real,
    μ₀::Real,
    N::Real,
)::Union{Tuple{Float64,Float64},Nothing}
    @assert E₀ > 0
    @assert μ₀ > 0

    cB_E₀²_div_μ₀ = tr.cB * E₀^2 / μ₀
    cℰ_E₀_div_μ₀ = tr.cℰ * E₀ / μ₀

    if (E₀ * μ₀ * tr.cℰ > tr.max_ℰ_ratio || μ₀ > tr.max_μ) && (
        abs(cB_E₀²_div_μ₀) > tr.max_cB_E₀²_div_μ₀ ||
        abs(cℰ_E₀_div_μ₀) > tr.max_cℰ_E₀_div_μ₀
    )
        return nothing
    end

    β = tr.hcl.β
    β_not_2 = min(1.9999, β)

    offset =
        K * tr.hcl.γ * N / (μ₀ * E₀^2) -
        (1 - cB_E₀²_div_μ₀ / 8 - cℰ_E₀_div_μ₀ / 6) / (2 + β / 2)

    if offset >= 0
        return 0.0, 0.0
    end

    function f(log_x)
        x = exp(log_x)
        return x^(2 + β / 2) *
               (1 - cB_E₀²_div_μ₀ / (4 - β) - cℰ_E₀_div_μ₀ / (2 - β_not_2)) / (2 + β / 2) +
               x^4 * cB_E₀²_div_μ₀ / (4 * (4 - β)) +
               x^3 * cℰ_E₀_div_μ₀ / (3 * (2 - β_not_2)) +
               offset
    end

    function dfdlogx(log_x)
        x = exp(log_x)
        return x * (
            x^(1 + β / 2) * (1 - cB_E₀²_div_μ₀ / (4 - β) - cℰ_E₀_div_μ₀ / (2 - β_not_2)) +
            x^3 * cB_E₀²_div_μ₀ / (4 - β) +
            x^2 * cℰ_E₀_div_μ₀ / (2 - β_not_2)
        )
    end

    result = solve(f, dfdlogx, 0.0)
    @assert converged(result)
    log_x = result.zero[1]

    E = exp(log_x) * E₀

    μ = compute_μ(tr, E, E₀, μ₀)

    return E, μ
end

function compute_μ(tr::LowEnergyTransport, E::Real, E₀::Real, μ₀::Real)
    β = tr.hcl.β
    β_not_2 = min(1.9999, β)
    return (E / E₀)^(β / 2) * (μ₀ - E₀^2 * tr.cB / (4 - β) - E₀ * tr.cℰ / (2 - β_not_2)) +
           E^2 * tr.cB / (4 - β) +
           E * tr.cℰ / (2 - β_not_2)
end

function solve(f, dfdx, x₀::Real)
    function f!(F, x)
        F[1] = f(x[1])
    end
    function dfdx!(dFdx, x)
        dFdx[1] = dfdx(x[1])
    end
    return nlsolve(f!, dfdx!, [x₀], ftol = 1e-5, xtol = 1e-8)
end

end
