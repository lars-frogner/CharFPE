module Atmosphere

export CoulombLog, HybridCoulombLog

using ..Const

const lnΛ_OFFSET = -0.5 * log(2π * e^6)
const lnΛ′_OFFSET = log(2 / (1.105 * χ * 1e-3 * KEV_TO_ERG))
const lnΛ″_OFFSET = log(sqrt(2 / mₑ) / (α * c))
const MIN_lnΛ_E = 1e-3 * KEV_TO_ERG

struct CoulombLog
    lnΛ::Real
    lnΛ′::Real
    lnΛ″::Real

    function CoulombLog(nH, x, E)
        nₑ = x * nH
        lnΛ = compute_lnΛ(E, nₑ)
        lnΛ′ = compute_lnΛ′(E)
        lnΛ″ = compute_lnΛ″(E)
        return new(lnΛ, lnΛ′, lnΛ″)
    end
end

struct HybridCoulombLog
    coulomb_log::CoulombLog
    γ::Real
    γ′::Real
    γ′′::Real
    β::Real

    function HybridCoulombLog(cl::CoulombLog, x)
        γ = compute_γ(x, cl.lnΛ, cl.lnΛ′)
        γ′ = compute_γ′(x, cl.lnΛ, cl.lnΛ″)
        γ′′ = compute_γ′′(x, cl.lnΛ, cl.lnΛ′, cl.lnΛ″)
        β = compute_β(x, cl.lnΛ, cl.lnΛ′, cl.lnΛ″)
        return new(cl, γ, γ′, γ′′, β)
    end
end

Base.Broadcast.broadcastable(coulomb_log::CoulombLog) = Ref(coulomb_log)
Base.Broadcast.broadcastable(coulomb_log::HybridCoulombLog) = Ref(coulomb_log)

compute_lnΛ(E, nₑ) = lnΛ_OFFSET + 0.5 * log(max(E, MIN_lnΛ_E)^3 / nₑ)
compute_lnΛ′(E) = lnΛ′_OFFSET + log(max(E, MIN_lnΛ_E))
compute_lnΛ″(E) = lnΛ″_OFFSET + 0.5 * log(max(E, MIN_lnΛ_E))

compute_γ(x, lnΛ, lnΛ′) = x * lnΛ + (1 - x) * lnΛ′
compute_γ′(x, lnΛ, lnΛ″) = 2x * lnΛ + (1 - x) * lnΛ″
compute_γ′′(x, lnΛ, lnΛ′, lnΛ″) = x * lnΛ + (1 - x) * (lnΛ″ - lnΛ′)
compute_β(x, lnΛ, lnΛ′, lnΛ″) = compute_γ′(x, lnΛ, lnΛ″) / compute_γ(x, lnΛ, lnΛ′)

end
