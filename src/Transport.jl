module Transport

export advance_E_μ_n, advance_high_energy_μ, advance_ℰ_energy_loss, compute_dEdN
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

function advance_E_μ_n(
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
    use_analytical::Bool,
)::Tuple{Float64,Float64,Float64}
    if use_analytical
        tr = LowEnergyTransport(hcl, nH, dlnBdN, ℰ)
        result = compute_E_μ_n(tr, E₀, μ₀, n₀, ΔN)
        return (result === nothing) ?
               advance_E_μ_n_Heun2(E₀, μ₀, n₀, hcl, nH, dlnBdN, ℰ, ΔN) : result
    else
        return advance_E_μ_n_Heun2(E₀, μ₀, n₀, hcl, nH, dlnBdN, ℰ, ΔN)
    end
end

function advance_high_energy_μ(μ₀::Real, dlnBdN::Real, ΔN::Real)::Float64
    sqrt(max(0, 1 - (1 - μ₀^2) * exp(dlnBdN * ΔN)))
end

function advance_ℰ_energy_loss(ℰ_energy_loss::Real, ℰ::Real, nH::Real, ΔN::Real)::Float64
    ℰ_energy_loss + (e * ℰ / nH) * ΔN
end

function advance_E_μ_RK4(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
)::Tuple{Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN₁ = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN₁ = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)

    E₁ = E₀ + dEdN₁ * 0.5 * ΔN
    μ₁ = μ₀ + dμdN₁ * 0.5 * ΔN

    if E₁ <= 0.0 || μ₁ <= 0
        return 0.0, 0.0
    end

    dEdN₂ = compute_dEdN.(E₁, μ₁, hcl, nH, ℰ)
    dμdN₂ = compute_dμdN.(E₁, μ₁, hcl, nH, dlnBdN, ℰ)

    E₂ = E₀ + dEdN₂ * 0.5 * ΔN
    μ₂ = μ₀ + dμdN₂ * 0.5 * ΔN

    if E₂ <= 0.0 || μ₂ <= 0
        return 0.0, 0.0
    end

    dEdN₃ = compute_dEdN.(E₂, μ₂, hcl, nH, ℰ)
    dμdN₃ = compute_dμdN.(E₂, μ₂, hcl, nH, dlnBdN, ℰ)

    E₃ = E₀ + dEdN₃ * ΔN
    μ₃ = μ₀ + dμdN₃ * ΔN

    if E₃ <= 0.0 || μ₃ <= 0
        return 0.0, 0.0
    end

    dEdN₄ = compute_dEdN.(E₃, μ₃, hcl, nH, ℰ)
    dμdN₄ = compute_dμdN.(E₃, μ₃, hcl, nH, dlnBdN, ℰ)

    E = E₀ + (dEdN₁ + 2 * dEdN₂ + 2 * dEdN₃ + dEdN₄) / 6.0 * ΔN
    μ = μ₀ + (dμdN₁ + 2 * dμdN₂ + 2 * dμdN₃ + dμdN₄) / 6.0 * ΔN

    if E <= 0.0 || μ <= 0
        return 0.0, 0.0
    end

    return E, μ
end

function advance_E_μ_Heun3(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
)::Tuple{Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN₁ = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN₁ = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)

    E₁ = E₀ + dEdN₁ * ΔN / 3.0
    μ₁ = μ₀ + dμdN₁ * ΔN / 3.0

    if E₁ <= 0.0 || μ₁ <= 0
        return 0.0, 0.0
    end

    dEdN₂ = compute_dEdN.(E₁, μ₁, hcl, nH, ℰ)
    dμdN₂ = compute_dμdN.(E₁, μ₁, hcl, nH, dlnBdN, ℰ)

    E₂ = E₀ + dEdN₂ * ΔN * 2.0 / 3.0
    μ₂ = μ₀ + dμdN₂ * ΔN * 2.0 / 3.0

    if E₂ <= 0.0 || μ₂ <= 0
        return 0.0, 0.0
    end

    dEdN₃ = compute_dEdN.(E₂, μ₂, hcl, nH, ℰ)
    dμdN₃ = compute_dμdN.(E₂, μ₂, hcl, nH, dlnBdN, ℰ)

    E = E₀ + (0.25 * dEdN₁ + 0.75 * dEdN₃) * ΔN
    μ = μ₀ + (0.25 * dμdN₁ + 0.75 * dμdN₃) * ΔN

    if E <= 0.0 || μ <= 0
        return 0.0, 0.0
    end

    return E, μ
end

function advance_E_μ_Heun2(
    E₀::Real,
    μ₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
)::Tuple{Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN₁ = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN₁ = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)

    E₁ = E₀ + dEdN₁ * ΔN
    μ₁ = μ₀ + dμdN₁ * ΔN

    if E₁ <= 0.0 || μ₁ <= 0
        return 0.0, 0.0
    end

    dEdN₂ = compute_dEdN.(E₁, μ₁, hcl, nH, ℰ)
    dμdN₂ = compute_dμdN.(E₁, μ₁, hcl, nH, dlnBdN, ℰ)

    E = E₀ + 0.5 * (dEdN₁ + dEdN₂) * ΔN
    μ = μ₀ + 0.5 * (dμdN₁ + dμdN₂) * ΔN

    if E <= 0.0 || μ <= 0
        return 0.0, 0.0
    end

    return E, μ
end

function advance_E_μ_n_Heun2(
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    hcl::HybridCoulombLog,
    nH::Real,
    dlnBdN::Real,
    ℰ::Real,
    ΔN::Real,
)::Tuple{Float64,Float64,Float64}
    @assert E₀ > 0
    @assert μ₀ > 0

    dEdN₁ = compute_dEdN.(E₀, μ₀, hcl, nH, ℰ)
    dμdN₁ = compute_dμdN.(E₀, μ₀, hcl, nH, dlnBdN, ℰ)
    dndN₁ = compute_dndN.(E₀, μ₀, n₀, hcl)

    E₁ = E₀ + dEdN₁ * ΔN
    μ₁ = μ₀ + dμdN₁ * ΔN
    n₁ = n₀ + dndN₁ * ΔN

    if E₁ <= 0.0 || μ₁ <= 0 || n₁ <= 0
        return 0.0, 0.0, 0.0
    end

    dEdN₂ = compute_dEdN.(E₁, μ₁, hcl, nH, ℰ)
    dμdN₂ = compute_dμdN.(E₁, μ₁, hcl, nH, dlnBdN, ℰ)
    dndN₂ = compute_dndN.(E₁, μ₁, n₁, hcl)

    E = E₀ + 0.5 * (dEdN₁ + dEdN₂) * ΔN
    μ = μ₀ + 0.5 * (dμdN₁ + dμdN₂) * ΔN
    n = n₀ + 0.5 * (dndN₁ + dndN₂) * ΔN

    if E <= 0.0 || μ <= 0 || n <= 0
        return 0.0, 0.0, 0.0
    end

    return E, μ, n
end

function compute_E_μ_n(
    tr::LowEnergyTransport,
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    N::Real,
)::Union{Tuple{Float64,Float64,Float64},Nothing}
    @assert E₀ > 0
    @assert μ₀ > 0
    @assert n₀ >= 0

    cB_E₀²_div_μ₀ = tr.cB * E₀^2 / μ₀
    cℰ_E₀_div_μ₀ = tr.cℰ * E₀ / μ₀

    if (E₀ * μ₀ * tr.cℰ > tr.max_ℰ_ratio || μ₀ > tr.max_μ) && (
        abs(cB_E₀²_div_μ₀) > tr.max_cB_E₀²_div_μ₀ ||
        abs(cℰ_E₀_div_μ₀) > tr.max_cℰ_E₀_div_μ₀
    )
        return nothing
    end

    if 2.0 - tr.hcl.β < 0.05
        return compute_E_μ_n_ionized(tr, E₀, μ₀, n₀, N, cB_E₀²_div_μ₀, cℰ_E₀_div_μ₀)
    else
        return compute_E_μ_n_unionized(tr, E₀, μ₀, n₀, N, cB_E₀²_div_μ₀, cℰ_E₀_div_μ₀)
    end
end

function compute_E_μ_n_ionized(
    tr::LowEnergyTransport,
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    N::Real,
    cB_E₀²_div_μ₀::Real,
    cℰ_E₀_div_μ₀::Real,
)::Tuple{Float64,Float64,Float64}
    offset = K * tr.hcl.γ * N / (μ₀ * E₀^2) - (1 - cB_E₀²_div_μ₀ / 8 - cℰ_E₀_div_μ₀ / 6) / 3

    if offset >= 0
        return 0.0, 0.0, 0.0
    end

    function f(log_x)
        x = exp(log_x)
        return x^3 * (1 - cB_E₀²_div_μ₀ / 2 + cℰ_E₀_div_μ₀ * (log_x - 1 / 3) / 2) / 3 +
               x^4 * cB_E₀²_div_μ₀ / 8 +
               offset
    end

    function dfdlogx(log_x)
        x = exp(log_x)
        return x^3 * (1 + cB_E₀²_div_μ₀ * (x - 1) / 2 - cℰ_E₀_div_μ₀ * log_x / 2)
    end

    result = solve(f, dfdlogx, 0.0)
    @assert converged(result)
    log_x = result.zero[1]

    x = exp(log_x)
    E = x * E₀

    μ = μ₀ * x + tr.cB * E^2 * (1 - 1 / x) / 2 + tr.cℰ * E * log_x

    n = n₀ / sqrt(x)

    return E, μ, n
end

function compute_E_μ_n_unionized(
    tr::LowEnergyTransport,
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    N::Real,
    cB_E₀²_div_μ₀::Real,
    cℰ_E₀_div_μ₀::Real,
)::Tuple{Float64,Float64,Float64}
    β = tr.hcl.β

    offset =
        K * tr.hcl.γ * N / (μ₀ * E₀^2) -
        (1 - cB_E₀²_div_μ₀ / 8 - cℰ_E₀_div_μ₀ / 6) / (2 + β / 2)

    if offset >= 0
        return 0.0, 0.0, 0.0
    end

    function f(log_x)
        x = exp(log_x)
        return x^(2 + β / 2) * (1 - cB_E₀²_div_μ₀ / (4 - β) - cℰ_E₀_div_μ₀ / (2 - β)) /
               (2 + β / 2) +
               x^4 * cB_E₀²_div_μ₀ / (4 * (4 - β)) +
               x^3 * cℰ_E₀_div_μ₀ / (3 * (2 - β)) +
               offset
    end

    function dfdlogx(log_x)
        x = exp(log_x)
        return x * (
            x^(1 + β / 2) * (1 - cB_E₀²_div_μ₀ / (4 - β) - cℰ_E₀_div_μ₀ / (2 - β)) +
            x^3 * cB_E₀²_div_μ₀ / (4 - β) +
            x^2 * cℰ_E₀_div_μ₀ / (2 - β)
        )
    end

    result = solve(f, dfdlogx, 0.0)
    @assert converged(result)
    log_x = result.zero[1]

    x = exp(log_x)
    E = x * E₀

    μ =
        x^(β / 2) * (μ₀ - E₀^2 * tr.cB / (4 - β) - E₀ * tr.cℰ / (2 - β)) +
        E^2 * tr.cB / (4 - β) +
        E * tr.cℰ / (2 - β)

    n = n₀ * x^(-tr.hcl.γ′′ / (2 * tr.hcl.γ))

    return E, μ, n
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
