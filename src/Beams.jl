module Beams

export Beam,
    compute_F₀,
    compute_n₀,
    compute_E_mean,
    compute_analytical_heating,
    compute_analytical_heating2,
    compute_analytical_evolution

using SpecialFunctions: beta_inc, beta

import ..RealArr1D
using ..Numerical: cumtrapz, trapz
using ..Const: mₑ, K, KEV_TO_ERG
using ..Atmosphere: CoulombLog, HybridCoulombLog

struct Beam
    Ec::Real
    δ::Real
    ℱ_tot::Real
end

Base.Broadcast.broadcastable(beam::Beam) = Ref(beam)

compute_F₀(beam::Beam, μ₀::Real, Δμ::Real, E, μ) = (μ == μ₀) ? compute_F₀(beam, E) / Δμ : 0
compute_F₀(beam::Beam, E) =
    (E >= beam.Ec) ? beam.ℱ_tot * ((beam.δ - 2) / beam.Ec^2) * (beam.Ec / E)^beam.δ : 0

compute_n₀(beam::Beam, E) = compute_F₀(beam, E) / sqrt(2E / mₑ)
compute_n₀(beam::Beam, μ₀::Real, Δμ::Real, E, μ) =
    compute_F₀(beam, μ₀, Δμ, E, μ) / sqrt(2E / mₑ)

compute_E_mean(beam::Beam) = beam.Ec * (beam.δ - 0.5) / (beam.δ - 1.5)

Β(x, a, b) = beta_inc(a, b, x)[1] * beta(a, b)

function compute_analytical_heating(
    beam::Beam,
    μ₀::Real,
    s::RealArr1D,
    nH::RealArr1D,
    x::Real,
)
    E_mean = compute_E_mean(beam)
    cl = CoulombLog(nH[1], x, E_mean)
    hcl = HybridCoulombLog(cl, x)

    N = cumtrapz(s, nH)
    Nc = (μ₀ * beam.Ec^2) / ((2 + hcl.β / 2) * K * hcl.γ)

    Q = @. (0.5 * K * (beam.δ - 2) * beam.ℱ_tot / (beam.Ec^2)) *
       nH *
       hcl.γ *
       Β(min(1, N / Nc), beam.δ / 2, 2 / (4 + hcl.β)) *
       (N / Nc)^(-beam.δ / 2)

    return Q
end

function compute_analytical_heating(
    beam::Beam,
    μ₀::Real,
    s::RealArr1D,
    nH::RealArr1D,
    x::RealArr1D,
)
    E_mean = compute_E_mean(beam)
    cl = CoulombLog(nH[1], x[1], E_mean)
    hcl = HybridCoulombLog.(cl, x)
    γ = getfield.(hcl, :γ)

    N = cumtrapz(s, nH)
    Nc = @. (μ₀ * beam.Ec^2) / (3K * γ)

    N_star = cumtrapz(s, @. nH * γ / cl.lnΛ)
    Nc_star = μ₀ * beam.Ec^2 / (3K * cl.lnΛ)

    Q = @. (0.5 * K * (beam.δ - 2) * beam.ℱ_tot / (μ₀ * beam.Ec^2)) *
       nH *
       γ *
       Β(min(1, N / Nc), beam.δ / 2, 1 / 3) *
       (N_star / Nc_star)^(-beam.δ / 2)

    return Q
end

function compute_analytical_evolution(E₀::Real, μ₀::Real, hcl::HybridCoulombLog, N::Real)
    common = (1 - min.(1, ((2 + hcl.β / 2) * K * hcl.γ / (μ₀ * E₀^2)) * N))
    E = E₀ * common^(2 / (4 + hcl.β))
    μ = μ₀ * common^(hcl.β / (4 + hcl.β))
    return (E, μ)
end

function compute_analytical_evolution(
    E₀::Real,
    μ₀::Real,
    n₀::Real,
    hcl::HybridCoulombLog,
    N::Real,
)
    common = (1 - min.(1, ((2 + hcl.β / 2) * K * hcl.γ / (μ₀ * E₀^2)) * N))
    E = E₀ * common^(2 / (4 + hcl.β))
    μ = μ₀ * common^(hcl.β / (4 + hcl.β))
    n = n₀ * common^((1 - hcl.β) / (4 + hcl.β))
    return (E, μ, n)
end

function compute_analytical_evolution(
    E₀::Real,
    μ₀::Real,
    nH::RealArr1D,
    x::Real,
    s::RealArr1D,
)
    cl = CoulombLog(nH[1], x, E₀)
    hcl = HybridCoulombLog.(cl, x)
    N = cumtrapz(s, nH)
    common = @. (1 - min.(1, ((2 + hcl.β / 2) * K * hcl.γ / (μ₀ * E₀^2)) * N))
    E = @. E₀ * common^(2 / (4 + hcl.β))
    μ = @. μ₀ * common^(hcl.β / (4 + hcl.β))
    return (E, μ)
end

end
