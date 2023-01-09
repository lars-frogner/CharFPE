### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ ab99e056-6007-11ed-0ec5-27044884ba2c
begin
    using Revise # Don't move this line, or Revise won't work
    import Pkg
    Pkg.activate(".")

    using CharFPE.Const
    using CharFPE.Numerical
    using CharFPE.Atmosphere
    using CharFPE.Beams
    using CharFPE.Solvers
    using CharFPE.Transport
end

# ╔═╡ ebc04802-c456-4085-8a38-3b8b22af5a3c
begin
    using CairoMakie
    set_theme!(theme_dark())
end

# ╔═╡ cb35b024-30b2-4795-9469-d3bedf8bd0ed
using SpecialFunctions

# ╔═╡ 47dbf79d-a78b-4386-ba9c-730b13e46c0d
using Printf

# ╔═╡ 4bba2be0-9e74-4fb4-b6dd-86206a2a3e61
import PlutoUI

# ╔═╡ 487aa234-4f78-485c-ac57-f02938341ad7
function plotdistr(E, μ, n; axis = (;), colorrange = nothing)
    limits = (nothing, nothing, -0.1, 1.0)
    axis = (; xscale = log10, xlabel = "E [keV]", ylabel = "μ", limits = limits, axis...)
    fig, ax, sc = scatter(
        E ./ KEV_TO_ERG,
        μ,
        color = n,
        markersize = 6,
        colorrange = colorrange,
        axis = axis,
    )
    Colorbar(fig[1, 2], sc; label = "log₁₀(n)")
    return fig, ax
end

# ╔═╡ 63954b95-eed0-4508-b7e1-78c74a8a6a6c
begin
    nH(s) = 1e10
    x(s) = 0.4
    dlnBds(s) = 0*1e-9
    ℰ(s) = 0*8e-9
end

# ╔═╡ 4271b9f9-5174-44f5-93d2-05ed6787a59b
begin
    s_max = 5e8
    Δs = 3e6
    s = 0.0:Δs:s_max
end

# ╔═╡ 592b6144-a4fe-4c82-b57d-f2e8d662ec6c
begin
    Ec = 1.0 * KEV_TO_ERG
    δ = 6.0
    ℱ_tot = 1.0
    beam = Beam(Ec, δ, ℱ_tot)
end

# ╔═╡ 6b248a0c-bb97-461a-8495-2ffe1362b9ae
begin
	E_min = 0.1 * Ec
	E_max = 10.0 * Ec
    log₁₀E = range(log10(E_min), log10(E_max), 100)
    E = 10 .^ log₁₀E
end

# ╔═╡ 67affd08-62ae-471f-9b4d-9451dbb3df82
begin
    μ₀ = 0.8
    n₀ = compute_n₀.(beam, E)
end

# ╔═╡ 2163f87c-9d96-452e-badc-be339ecf281a
begin
    s_an = range(0, s_max, 200)
    Q_an = compute_analytical_heating(beam, μ₀, s_an, nH.(s_an), x(s_an[1]))
end

# ╔═╡ 62b5e343-968c-4470-83d4-58a460e28e9f
μ_l, n_l, E₀_l, Q_l = propagate_lagrange(beam, s, log₁₀E, μ₀, n₀, nH, x, dlnBds, ℰ)

# ╔═╡ e8804911-1896-47d1-bcc0-b37b6c889791
@bind k PlutoUI.Slider(1:length(s))

# ╔═╡ 974bb463-08fd-4d4a-afe4-766e9b42a99e
let
	n2 = copy(n_l)
	n2[n2.<0.0] .= NaN
	fig, ax = plotdistr(E, μ_l[:, k], log10.(n2[:, k]), colorrange=(-1.0, 8.0))
	fig
end

# ╔═╡ 488de288-f4a9-484a-b6ab-667d393d04f4
let
    fig, ax, _ = lines(s_an .* 1e-8, Q_an, axis = (; yscale = log10))
	lines!(ax, s[Q_l .> 0] .* 1e-8, Q_l[Q_l.>0])
    vlines!(ax, s[k] .* 1e-8, color = :gray)
    fig
end

# ╔═╡ 1cc3ce85-233f-4035-90ad-ee65d1dba95a
E[1]

# ╔═╡ 1259c087-fbea-404d-9649-b5222589babf
let
    E₀ = 1.602177330000003e-10
    μ₀ = 0.7890285575636417

    dlnBdN = 0
    ℰ = 0
    ΔN = 1e14

    hcl = HybridCoulombLog(CoulombLog(nH(0.0), x(0.0), compute_E_mean(beam)), x(0.0))
    tr = LowEnergyTransport(hcl, nH(0.0), dlnBdN, ℰ)

    @show tr
    @show is_valid(tr, E₀, μ₀)

    f, E_an = compute_E(tr, E₀, μ₀, ΔN)

    @show E_an

    log_x = 10 .^ range(-10, 0, 100)
    fig, ax, _ = lines(log_x * E₀, f.(log_x), axis = (; xscale = log10))
    fig
end

# ╔═╡ 87954508-6161-4300-a9a3-e2439f3a7512
vₜ(m, T) = sqrt(kB*T/m)

# ╔═╡ f6cb8aa1-d8ff-46f0-8ac4-fabb377694cc
E_to_v(E) = sqrt(2E/mₑ)

# ╔═╡ 8301ede4-b41f-4206-b645-e44baff12c8b
E_to_u(E, m, T) = E_to_v(E)/(sqrt(2)*vₜ(m, T))

# ╔═╡ f4afb753-c8ad-4563-9297-c4a99af7714e
erf′(u) = (2/sqrt(π))*exp(-u^2)

# ╔═╡ aa3a990e-8f32-404e-9e16-95946b01aef0
G(u) = (erf(u) - u*erf′(u))/(2u^2)

# ╔═╡ 8c413bd4-eb16-4b15-af92-8f6fa2e4ed11
Γ(lnΛ) = 2K*lnΛ/mₑ^2

# ╔═╡ 4f12647a-e4b8-4d68-b5de-ba5e185bf236
let
    E = 10 .^range(log10(1e-3 * Ec), log10(1e2 * Ec), 500)
	T = 1e5
	u = E_to_u.(E, mₑ, T)
	curve = @. (erf(u) + 2*u*erf′(u) + 2*G(u))/E
	curve = @. (erf(u) - G(u))/E^2
    fig, ax, _ = lines(E./KEV_TO_ERG, curve, axis = (; xscale = log10, xlabel = "E [keV]"))
	lines!(ax, E./KEV_TO_ERG, 1 ./E.^2)
	vlines!(ax, (3/2)*kB*T/KEV_TO_ERG)
	fig
end

# ╔═╡ 5bce3ac2-7a41-42de-b5a5-2b2db338886b
let
    E = 10 .^range(log10(1e-3 * Ec), log10(2e0 * Ec), 500)
	T = 1e5
	u = E_to_u.(E, mₑ, T)
	curve = @. (erf(u) + 2*u*erf′(u) + 2*G(u))
	approx = @. 22*u/(3*sqrt(π))
    fig, ax, _ = lines(u, curve, axis = (; xscale = log10, xlabel = "u"))
	lines!(ax, u, approx)
	fig
end

# ╔═╡ 31fba526-6b2f-4e2b-bb1f-633cbb4b77ba
let
    E = 10 .^range(log10(1e-3 * Ec), log10(1e2 * Ec), 500)
	T = 1e5
	uₑ = u.(E, mₑ, T)
	uₚ = u.(E, mₚ, T)
    fig, ax, _ = lines(E./KEV_TO_ERG, erf.(uₑ) - G.(uₑ), label="e", axis = (; xscale = log10, xlabel = "E [keV]"))
	lines!(ax, E./KEV_TO_ERG, erf.(uₚ) - G.(uₚ), label="p")
	vlines!(ax, (3/2)*kB*T/KEV_TO_ERG)
	axislegend(ax)
	fig
end

# ╔═╡ Cell order:
# ╠═ab99e056-6007-11ed-0ec5-27044884ba2c
# ╠═ebc04802-c456-4085-8a38-3b8b22af5a3c
# ╠═cb35b024-30b2-4795-9469-d3bedf8bd0ed
# ╠═47dbf79d-a78b-4386-ba9c-730b13e46c0d
# ╠═4bba2be0-9e74-4fb4-b6dd-86206a2a3e61
# ╠═487aa234-4f78-485c-ac57-f02938341ad7
# ╠═63954b95-eed0-4508-b7e1-78c74a8a6a6c
# ╠═4271b9f9-5174-44f5-93d2-05ed6787a59b
# ╠═592b6144-a4fe-4c82-b57d-f2e8d662ec6c
# ╠═6b248a0c-bb97-461a-8495-2ffe1362b9ae
# ╠═67affd08-62ae-471f-9b4d-9451dbb3df82
# ╠═2163f87c-9d96-452e-badc-be339ecf281a
# ╠═62b5e343-968c-4470-83d4-58a460e28e9f
# ╠═e8804911-1896-47d1-bcc0-b37b6c889791
# ╠═974bb463-08fd-4d4a-afe4-766e9b42a99e
# ╠═488de288-f4a9-484a-b6ab-667d393d04f4
# ╠═1cc3ce85-233f-4035-90ad-ee65d1dba95a
# ╠═1259c087-fbea-404d-9649-b5222589babf
# ╠═87954508-6161-4300-a9a3-e2439f3a7512
# ╠═f6cb8aa1-d8ff-46f0-8ac4-fabb377694cc
# ╠═8301ede4-b41f-4206-b645-e44baff12c8b
# ╠═f4afb753-c8ad-4563-9297-c4a99af7714e
# ╠═aa3a990e-8f32-404e-9e16-95946b01aef0
# ╠═8c413bd4-eb16-4b15-af92-8f6fa2e4ed11
# ╠═4f12647a-e4b8-4d68-b5de-ba5e185bf236
# ╠═5bce3ac2-7a41-42de-b5a5-2b2db338886b
# ╠═31fba526-6b2f-4e2b-bb1f-633cbb4b77ba
